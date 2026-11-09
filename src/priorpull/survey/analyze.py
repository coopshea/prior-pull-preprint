"""Prespecified estimates for the survey. Everything here is reported."""
from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from ..worlds import World
from .assign import Row
from .run import hint_location

MAIN = "C(labels) + C(hint_kind) + C(strategy) + C(rationale) + C(setup) + C(feedback) + C(budget) + C(model) + C(world)"
INTERACTIONS = "C(labels):C(feedback) + C(labels):C(hint_kind) + C(labels):C(strategy) + C(setup):C(hint_kind)"


def quality(record: dict, world: World) -> dict:
    grid = np.sort(world.y.ravel())
    pct = lambda v: float(np.searchsorted(grid, v, side="right") / grid.size)
    best = np.maximum.accumulate(record["true"])
    series = [pct(b) for b in best]
    return {"quality": float(np.mean(series)), "final_percentile": series[-1]}


def search_table(rows: list[Row], records: list[dict], worlds: dict[str, World]) -> pd.DataFrame:
    by_id = {r["id"]: r for r in records}
    out = []
    for row in rows:
        rec = by_id[row.id]
        w = worlds[row.world]
        f, level = hint_location(w, row.id)
        share = float(np.mean([p[f] == level for p in rec["picks"]]))
        out.append({
            "id": row.id, "world": row.world, "source": w.source, "labels": row.effective_labels,
            "hint_kind": "none" if row.hint == "none" else "hint", "hint": row.hint, "strategy": row.strategy, "wording": row.wording,
            "rationale": row.rationale, "setup": row.setup, "feedback": row.feedback, "budget": f"{row.rounds}x{row.q}", "model": row.model,
            "prompt": row.prompt, "hint_level_share": share, "random_fills": rec["random_fills"], "refusals": rec["refusals"], "retries": rec["retries"],
            **quality(rec, w),
        })
    return pd.DataFrame(out)


FACTORS = ("labels", "hint_kind", "strategy", "rationale", "setup", "feedback", "budget", "model")


def _randomization_p(df: pd.DataFrame, formula: str, n_perm: int, seed: int) -> dict[str, float]:
    """Randomization p-values: reshuffle each factor among trajectories within a world, then refit.

    Assignment was random, so this is valid without relying on large-sample error formulas.
    """
    rng = np.random.default_rng(seed)
    observed = smf.ols(formula, data=df).fit().params
    y = df["quality"].to_numpy()
    results: dict[str, float] = {}
    for factor in FACTORS:
        if f"C({factor})" not in formula or df[factor].nunique() < 2:
            continue
        terms = [t for t in observed.index if t.startswith(f"C({factor})") and ":" not in t]
        extreme = np.zeros(len(terms))
        for _ in range(n_perm):
            shuffled = df.copy()
            shuffled[factor] = df.groupby("world")[factor].transform(lambda c: rng.permutation(c.to_numpy()))
            X = patsy.dmatrix(formula.split("~", 1)[1], shuffled, return_type="dataframe")
            beta = pd.Series(np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0], index=X.columns)
            extreme += np.abs(beta.reindex(terms).fillna(0).to_numpy()) >= np.abs(observed[terms].to_numpy()) - 1e-12
        results.update({t: float((extreme[i] + 1) / (n_perm + 1)) for i, t in enumerate(terms)})
    return results


def _fit(df: pd.DataFrame, formula: str, scope: str, n_perm: int, seed: int) -> pd.DataFrame:
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
    ci = model.conf_int()
    t = pd.DataFrame({"estimate": model.params, "se": model.bse, "lo": ci[0], "hi": ci[1], "p_hc3": model.pvalues})
    t = t[~t.index.str.contains("C\\(world\\)|Intercept")]
    t.insert(0, "scope", scope)
    t = t.reset_index(names="term")
    if n_perm:
        permuted = _randomization_p(df, formula, n_perm, seed)
        t["p"] = t["term"].map(permuted).fillna(t["p_hc3"])
    else:
        t["p"] = t["p_hc3"]
    return t


def main_effects(df: pd.DataFrame, n_perm: int = 499, seed: int = 0) -> pd.DataFrame:
    frames = [_fit(df, f"quality ~ {MAIN}", "all", n_perm, seed)]
    frames.append(_fit(df, f"quality ~ {MAIN} + {INTERACTIONS}", "all_with_interactions", 0, seed))
    for source, sub in df.groupby("source"):
        terms = [t for t in MAIN.split(" + ") if sub[t[2:-1]].nunique() > 1]
        frames.append(_fit(sub, "quality ~ " + " + ".join(terms), source, n_perm, seed))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["p"])
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    return out


def prompt_variance(df: pd.DataFrame) -> dict:
    """Between-prompt variance of residual quality, by the method of moments."""
    resid = smf.ols(f"quality ~ {MAIN}", data=df).fit().resid
    groups = pd.Series(resid.values, index=df["prompt"].values).groupby(level=0)
    means, sizes, within = groups.mean(), groups.size(), groups.var(ddof=1).mean()
    return {"between_prompt_var": max(float(means.var(ddof=1) - within / sizes.mean()), 0.0), "residual_var": float(resid.var(ddof=1))}


def _level_hist(picks, world: World) -> list[np.ndarray]:
    return [np.bincount([p[f] for p in picks], minlength=n) / len(picks) for f, n in enumerate(world.shape)]


def replay_table(results: list[dict], worlds: dict[str, World]) -> pd.DataFrame:
    frame = {}
    for r in results:
        frame.setdefault(r["pair"], {})[r["side"]] = r
    rows = []
    for pair, sides in frame.items():
        a, b = sides.get("a"), sides.get("b")
        if not a or not b or not a["picks"] or not b["picks"]:
            continue
        w = worlds[a["world"]]
        ha, hb = _level_hist(a["picks"], w), _level_hist(b["picks"], w)
        tv = float(np.mean([0.5 * np.abs(x - y).sum() for x, y in zip(ha, hb)]))
        rows.append({"pair": pair, "factor": a["factor"], "k": a["k"], "world": a["world"], "source": w.source, "shift": tv})
    return pd.DataFrame(rows)


def replay_summary(t: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for factor, g in t.groupby("factor"):
        worlds = g["world"].unique()
        means = []
        for _ in range(n_boot):
            sample = rng.choice(worlds, size=len(worlds), replace=True)
            means.append(np.mean(np.concatenate([g.loc[g.world == w, "shift"].to_numpy() for w in sample])))
        out.append({"factor": factor, "pairs": len(g), "mean_shift": float(g["shift"].mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))})
    return pd.DataFrame(out)
