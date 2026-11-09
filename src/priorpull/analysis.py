"""Metrics, paired comparisons, and summary figures."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .search import Trajectory
from .worlds import World


def scores(t: Trajectory, world: World) -> dict:
    """Score picks on the true noiseless landscape, whatever feedback the player saw."""
    y_max = float(world.y.max())
    median = float(np.median(world.y))
    best = np.maximum.accumulate(t.true_values)
    span = max(y_max - median, 1e-9)
    regret = (y_max - best[-1]) / span
    norm = np.clip(best - median, 0, None) / span
    checkpoint = {f"best_at_{pct}": float(norm[max(0, int(np.ceil(len(norm) * pct / 100)) - 1)]) for pct in (25, 50)}
    return {
        **checkpoint,
        "regret": float(regret),
        "auc": float(np.mean(np.clip(best - median, 0, None) / span)),
        "best": float(best[-1]),
        "percentile": float((world.y <= best[-1]).mean()),
        "unique_picks": len({tuple(p) for p in t.picks}),
    }


def hint_shares(t: Trajectory, world: World) -> dict:
    """Share of picks at the best and worst marginal level of the hinted factor, for any wording."""
    f, best, worst = world.hint_target()
    return {
        "share_best_level": float(np.mean([p[f] == best for p in t.picks])),
        "share_worst_level": float(np.mean([p[f] == worst for p in t.picks])),
    }


RECOGNITION = re.compile(r"shields|doyle|edbo|known .{0,30}(dataset|screen)|this .{0,20}dataset|benchmark", re.IGNORECASE)
HINT_MENTION = re.compile(r"hunch|your (idea|suggestion|guess|intuition)|you (suggest|mention)|\\bPI\\b|as you", re.IGNORECASE)


def names_dataset(t: Trajectory) -> bool:
    """Whether any visible rationale says the model recognizes the benchmark."""
    return any(RECOGNITION.search(r or "") for r in t.reasons)


def hinted_level(wording: str, world: World) -> tuple[int, int] | None:
    f, best, worst = world.hint_target()
    return {"W+": (f, best), "W-": (f, worst), "Wpi": (f, worst)}.get(wording)


def hint_profile(t: Trajectory, world: World) -> dict:
    """Budget spent on the hinted level, when it was last used, and whether the rationale admits the hint."""
    target = hinted_level(t.wording, world)
    if target is None:
        return {"hint_budget": np.nan, "hint_last_round": np.nan, "hint_admitted": np.nan}
    f, level = target
    used_rounds = sorted({r for p, r in zip(t.picks, t.rounds) if p[f] == level})
    budget = float(np.mean([p[f] == level for p in t.picks]))
    admitted = np.nan
    if used_rounds:
        texts = [t.reasons[r - 1] for r in used_rounds if r - 1 < len(t.reasons)]
        admitted = float(any(HINT_MENTION.search(x or "") for x in texts))
    return {"hint_budget": budget, "hint_last_round": used_rounds[-1] if used_rounds else 0, "hint_admitted": admitted}


def pick_divergence(a: Trajectory, b: Trajectory, world: World) -> float:
    """Mean total-variation distance between the level histograms of two trajectories."""
    distances = []
    for f, n in enumerate(world.shape):
        ca = np.bincount([p[f] for p in a.picks], minlength=n) / len(a.picks)
        cb = np.bincount([p[f] for p in b.picks], minlength=n) / len(b.picks)
        distances.append(0.5 * float(np.abs(ca - cb).sum()))
    return float(np.mean(distances))


def table(trajectories: list[Trajectory], worlds: dict[str, World]) -> pd.DataFrame:
    rows = []
    for t in trajectories:
        w = worlds[t.world]
        rows.append(
            {
                "player": t.player, "world": t.world, "semantic": t.semantic, "wording": t.wording,
                "feedback": t.feedback, "strategy": t.strategy, "seed": t.seed, "template": t.template,
                "random_fills": t.random_fills, "refusals": t.refusals, "retries": t.retries,
                "names_dataset": names_dataset(t),
                **hint_shares(t, w),
                **hint_profile(t, w),
                **scores(t, w),
            }
        )
    return pd.DataFrame(rows)


def paired_diff(df: pd.DataFrame, world: str, a: dict, b: dict, metric: str = "regret", n_boot: int = 4000, seed: int = 0) -> dict:
    """Mean of a minus b, paired on seed, with a bootstrap 95% interval over seeds."""
    d = df[df.world == world]
    left = d.loc[np.logical_and.reduce([d[k] == v for k, v in a.items()])].set_index("seed")[metric]
    right = d.loc[np.logical_and.reduce([d[k] == v for k, v in b.items()])].set_index("seed")[metric]
    joined = pd.concat([left.rename("a"), right.rename("b")], axis=1).dropna()
    if joined.empty:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan}
    diffs = (joined.a - joined.b).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return {"n": len(diffs), "mean": float(diffs.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}


def best_so_far_curves(trajectories: list[Trajectory], worlds: dict[str, World]) -> pd.DataFrame:
    rows = []
    for t in trajectories:
        w = worlds[t.world]
        median = float(np.median(w.y))
        span = max(float(w.y.max()) - median, 1e-9)
        best = np.maximum.accumulate(t.true_values)
        for i, b in enumerate(best):
            rows.append(
                {"player": t.player, "world": t.world, "semantic": t.semantic, "wording": t.wording,
                 "feedback": t.feedback, "seed": t.seed, "n": i + 1, "norm_best": float((b - median) / span)}
            )
    return pd.DataFrame(rows)
