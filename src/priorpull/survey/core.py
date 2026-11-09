"""The core study: one history, a reference request, and one change at a time."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .. import prompts as P
from ..llm import Request, Store, execute
from ..search import parse_reply
from ..worlds import World, _stable_int
from .assign import HINT_CELLS, Row
from .run import base_prompt, hint_location, hint_sentence, random_history, schema_for
from .views import view

MODEL = "claude-opus-5-5"
SOURCES = ("shields", "suzuki", "efb", "cernak")
HISTORY_SIZES = (0, 3, 6, 12)
ADVICE = ("evidence", "audit", "caution")
ARMS = ("A1", "A2", "scrambled", "coded", "swapped", "prompt", "hunch", "evidence", "audit", "caution", "no_rationale")
ROUNDS, Q = 8, 3


@dataclass
class Unit:
    unit: int
    world: str
    source: str
    k: int
    prompt: int
    other_prompt: int
    hint_cell: str
    wording: int


def build_units(worlds: dict[str, World], per_cell: int, n_prompts: int, seed: int = 0) -> list[Unit]:
    """Every source and history size gets the same number of units."""
    rng = np.random.default_rng(seed)
    by_source = defaultdict(list)
    for w in worlds.values():
        if w.source in SOURCES:
            by_source[w.source].append(w.id)
    units: list[Unit] = []
    for source in SOURCES:
        for k in HISTORY_SIZES:
            for i in range(per_cell):
                prompt = int(rng.integers(n_prompts))
                units.append(Unit(
                    len(units), by_source[source][i % len(by_source[source])], source, k, prompt,
                    (prompt + 1 + int(rng.integers(n_prompts - 1))) % n_prompts,
                    HINT_CELLS[int(rng.integers(len(HINT_CELLS)))], int(rng.integers(5)),
                ))
    return units


def units_digest(units: list[Unit]) -> str:
    return hashlib.sha256(json.dumps([asdict(u) for u in units], sort_keys=True).encode()).hexdigest()


def row_for(unit: Unit, arm: str) -> Row:
    """The reference row, with only the named arm's input changed."""
    labels, feedback, hint, strategy, rationale, prompt = "real", "true", "none", "none", True, unit.prompt
    if arm == "scrambled":
        feedback = "scrambled"
    elif arm == "coded":
        labels = "coded"
    elif arm == "swapped":
        labels = "perm"
    elif arm == "prompt":
        prompt = unit.other_prompt
    elif arm == "hunch":
        hint = unit.hint_cell
    elif arm in ADVICE:
        strategy = arm
    elif arm == "no_rationale":
        rationale = False
    return Row(unit.unit, unit.world, labels, labels, prompt, hint, strategy, unit.wording, rationale, "single", feedback, ROUNDS, Q, MODEL)


def arms_for(unit: Unit) -> list[str]:
    return [a for a in ARMS if not (a == "scrambled" and unit.k == 0)]


def run_units(units: list[Unit], worlds: dict[str, World], pools: dict, backend, store: Store, model: str = MODEL, attempts: int = 3) -> list[dict]:
    work = []
    for u in units:
        world = worlds[u.world]
        for arm in arms_for(u):
            row = row_for(u, arm)
            pres = view(world, row.effective_labels, row.id)
            history = random_history(world, row, u.k)
            round_no = 1 + u.k // Q
            prompt = base_prompt(pres, row, history, round_no, pools, hint_sentence(pres, row, pools))
            work.append({"unit": u, "arm": arm, "row": row, "pres": pres, "history": history, "prompt": prompt, "schema": schema_for(pres, row)})
    results: dict[tuple[int, str], dict] = {}
    pending = work
    for attempt in range(attempts):
        if not pending:
            break
        requests = [
            Request(model, w["prompt"], w["schema"], attempt=attempt, salt=f"core|{w['unit'].unit}|{w['arm']}",
                    tag={"study": "core", "unit": w["unit"].unit, "arm": w["arm"], "q": Q, "k": w["unit"].k, "world": w["unit"].world})
            for w in pending
        ]
        replies = execute(requests, backend, store)
        still = []
        for w, req in zip(pending, requests):
            reply = replies.get(req.key)
            key = (w["unit"].unit, w["arm"])
            if reply is not None and reply.stop_reason == "refusal":
                results[key] = {"picks": None, "status": "refusal"}
                continue
            parsed = parse_reply(reply, w["pres"], Q, False) if reply else None
            if parsed is None:
                still.append(w)
                results[key] = {"picks": None, "status": "unparsed"}
            else:
                results[key] = {"picks": [list(p) for p in parsed[0]], "status": "ok", "reason": parsed[1]}
        pending = still
    out = []
    for w in work:
        u, arm = w["unit"], w["arm"]
        out.append({**asdict(u), "arm": arm, "labels_shown": w["row"].effective_labels, "shown": w["pres"].labels,
                    "history": [[list(ix), y] for ix, y in w["history"]], **results[(u.unit, arm)]})
    return out


# ---------- zero-shot rankings, used for "follows prior"

def ranking_requests(worlds: dict[str, World], reps: int, model: str = MODEL) -> list[Request]:
    reqs = []
    for w in worlds.values():
        if w.source not in SOURCES:
            continue
        pres = view(w, "real", 0)
        settings = "\n".join(f"- {n}: {', '.join(pres.options(f))}" for f, n in enumerate(pres.names))
        text = (
            f"{pres.context} I want the highest {pres.objective} I can get. I haven't run anything yet.\n\nSettings:\n{settings}\n\n"
            "From what you already know, rank every option of every setting from best to worst."
        )
        schema = {
            "type": "object",
            "properties": {P.key_for(n): {"type": "array", "items": {"type": "string", "enum": pres.options(f)}} for f, n in enumerate(pres.names)},
            "required": [P.key_for(n) for n in pres.names],
            "additionalProperties": False,
        }
        for r in range(reps):
            reqs.append(Request(model, text, schema, salt=f"core-rank-{w.id}-{r}", tag={"probe": "rank", "world": w.id, "rep": r}))
    return reqs


def prior_scores(requests: list[Request], replies, worlds: dict[str, World]) -> dict:
    """For each world and factor, a score from 0 (ranked last) to 1 (ranked first), keyed by level name."""
    collected: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in requests:
        reply = replies.get(r.key)
        if reply is None:
            continue
        world = worlds[r.tag["world"]]
        pres = view(world, "real", 0)
        try:
            data = json.loads(reply.text)
        except json.JSONDecodeError:
            continue
        for f, name in enumerate(pres.names):
            order = list(dict.fromkeys(x for x in data.get(P.key_for(name), []) if x in pres.labels[f]))
            if len(order) < 2:
                continue  # long option lists often come back incomplete, so only the listed options are scored
            for pos, label in enumerate(order):
                collected[world.id][f][label].append(1 - pos / (len(order) - 1))
    return {w: {f: {l: float(np.mean(v)) for l, v in levels.items()} for f, levels in factors.items()} for w, factors in collected.items()}


# ---------- measures

def _histogram(picks, shape) -> list[np.ndarray]:
    return [np.bincount([p[f] for p in picks], minlength=n) / len(picks) for f, n in enumerate(shape)]


def change_in_picks(a, b, world: World) -> float:
    ha, hb = _histogram(a, world.shape), _histogram(b, world.shape)
    return float(np.mean([0.5 * np.abs(x - y).sum() for x, y in zip(ha, hb)]))


def true_rank(picks, world: World) -> float:
    grid = np.sort(world.y.ravel())
    return float(np.mean([np.searchsorted(grid, world.y[tuple(p)], side="right") / grid.size for p in picks]))


def follows_results(picks, history, world: World) -> float | None:
    """Share of setting choices matching the level with the best mean in the results shown."""
    if not history:
        return None
    hits = []
    for f in range(len(world.shape)):
        by_level = defaultdict(list)
        for ix, y in history:
            by_level[ix[f]].append(y)
        best = max(by_level, key=lambda level: np.mean(by_level[level]))
        hits += [p[f] == best for p in picks]
    return float(np.mean(hits))


def follows_prior(picks, shown, world_prior: dict | None) -> float | None:
    """Mean zero-shot score of the names shown for the picks. Undefined for coded names."""
    if not world_prior:
        return None
    values = []
    for p in picks:
        for f, level in enumerate(p):
            label = shown[f][level]
            scores = world_prior.get(f) if f in world_prior else world_prior.get(str(f))
            if scores and label in scores:
                values.append(scores[label])
    return float(np.mean(values)) if values else None


def measure(results: list[dict], worlds: dict[str, World], priors: dict) -> pd.DataFrame:
    """One row per unit and arm, measured against that unit's reference request."""
    refs = {r["unit"]: r for r in results if r["arm"] == "A1" and r["picks"]}
    rows = []
    for r in results:
        ref = refs.get(r["unit"])
        if not r["picks"] or ref is None:
            continue
        w = worlds[r["world"]]
        history = [(tuple(ix), y) for ix, y in r["history"]]
        rows.append({
            "unit": r["unit"], "world": r["world"], "source": r["source"], "k": r["k"], "arm": r["arm"],
            "change": change_in_picks(ref["picks"], r["picks"], w),
            "true_rank": true_rank(r["picks"], w),
            "follows_results": follows_results(r["picks"], history, w),
            "follows_prior": follows_prior(r["picks"], r["shown"], priors.get(r["world"])) if r["labels_shown"] != "coded" else None,
            "ref_true_rank": true_rank(ref["picks"], w),
        })
    return pd.DataFrame(rows)


GROUP = {"A2": "Ask again (noise)", "scrambled": "Scrambled data", "coded": "Coded names", "swapped": "Swapped names",
         "prompt": "Different prompt", "hunch": "Added hunch", "evidence": "Added advice", "audit": "Added advice", "caution": "Added advice"}
CATEGORY = {"scrambled": "Data", "coded": "Priors", "swapped": "Priors", "prompt": "Wording", "hunch": "Wording"}


def bootstrap(values: np.ndarray, rng, n: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def source_balanced(frame: pd.DataFrame, column: str, rng) -> dict:
    """Mean with sources weighted equally, and a bootstrap interval over units within source."""
    groups = [g[column].dropna().to_numpy() for _, g in frame.groupby("source")]
    groups = [g for g in groups if len(g)]
    if not groups:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    draws = np.mean([rng.choice(g, size=(2000, len(g)), replace=True).mean(axis=1) for g in groups], axis=0)
    return {"mean": float(np.mean([g.mean() for g in groups])), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)), "n": int(sum(map(len, groups)))}


def figure2_table(m: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for name in dict.fromkeys(GROUP.values()):
        arms = [a for a, g in GROUP.items() if g == name]
        out.append({"change": name, **source_balanced(m[m.arm.isin(arms)], "change", rng)})
    return pd.DataFrame(out)


def figure3_table(m: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    labels = {**{a: c for a, c in CATEGORY.items()}, "A2": "Noise floor", "evidence": "Wording", "audit": "Wording", "caution": "Wording"}
    frame = m[m.arm.isin(labels)].assign(category=lambda d: d.arm.map(labels))
    for (category, k), g in frame.groupby(["category", "k"]):
        out.append({"category": category, "k": k, **source_balanced(g, "change", rng)})
    return pd.DataFrame(out)


def figure4_table(m: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Each strategy minus the reference on the same units, for the three measures."""
    rng = np.random.default_rng(seed)
    names = {"evidence": "Evidence first", "audit": "Audit assumptions", "caution": "Caution", "no_rationale": "No rationale"}
    ref = m[m.arm == "A1"].set_index("unit")
    out = []
    for arm, name in names.items():
        sub = m[m.arm == arm].set_index("unit")
        joined = sub.join(ref, rsuffix="_ref", how="inner")
        row = {"strategy": name}
        for col in ("true_rank", "follows_results", "follows_prior"):
            frame = pd.DataFrame({"source": joined["source"], "diff": joined[col] - joined[f"{col}_ref"]})
            stats = source_balanced(frame.dropna(), "diff", rng)
            row.update({col: stats["mean"], f"{col}_lo": stats["lo"], f"{col}_hi": stats["hi"]})
        out.append(row)
    return pd.DataFrame(out)
