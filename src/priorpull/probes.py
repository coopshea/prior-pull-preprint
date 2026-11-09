"""No-feedback probes: what does the model already believe about a dataset?"""
from __future__ import annotations

import json
import random

import numpy as np
from scipy.stats import spearmanr

from . import prompts
from .llm import Reply, Request
from .worlds import World, _stable_int, present

SOURCE_SCHEMA = {
    "type": "object",
    "properties": {
        "recognized": {"type": "boolean"},
        "source_guess": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["recognized", "source_guess", "confidence"],
    "additionalProperties": False,
}
ANOMALY_SCHEMA = {
    "type": "object",
    "properties": {
        "looks_wrong": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": ["looks_wrong", "explanation"],
    "additionalProperties": False,
}
CONTINUE_SCHEMA = {
    "type": "object",
    "properties": {"rows": {"type": "array", "items": {"type": "string"}}},
    "required": ["rows"],
    "additionalProperties": False,
}
SHIELDS_TERMS = ("shields", "doyle", "edbo", "bayesian reaction optimization")


def _table(pres, idxs: list[tuple[int, ...]], values: list[float], objective: str | None = None) -> str:
    head = " | ".join(pres.names + [objective or pres.objective])
    rows = [" | ".join([pres.labels[f][i] for f, i in enumerate(ix)] + [f"{v:g}"]) for ix, v in zip(idxs, values)]
    return head + "\n" + "\n".join(rows)


def _sample(world: World, n: int, salt: str) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(_stable_int(world.id, salt))
    flat = rng.choice(world.size, size=n, replace=False)
    return [tuple(int(v) for v in np.unravel_index(int(f), world.shape)) for f in flat]


def rank_schema(pres) -> dict:
    return {
        "type": "object",
        "properties": {
            "rankings": {
                "type": "object",
                "properties": {
                    prompts.key_for(n): {"type": "array", "items": {"type": "string", "enum": pres.options(f)}}
                    for f, n in enumerate(pres.names)
                },
                "required": [prompts.key_for(n) for n in pres.names],
                "additionalProperties": False,
            },
            "predictions": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["rankings", "predictions"],
        "additionalProperties": False,
    }


def build_p1(model, world, reps=10):
    idxs = _sample(world, 20, "p1")
    pres = present(world, "S0")
    table = _table(pres, idxs, [float(world.y[i]) for i in idxs])
    text = (
        "Here are some rows from a dataset.\n\n" + table +
        "\n\nDo you recognize this dataset? Say what paper, group, or source it might come from, "
        "and how confident you are from 0 to 1. If you don't recognize it, say so."
    )
    return [Request(model, text, SOURCE_SCHEMA, salt=f"p1-{world.id}-{r}", tag={"probe": "P1", "world": world.id, "rep": r}) for r in range(reps)]


def build_p2(model, world, reps=10):
    idxs = _sample(world, 20, "p1")
    pres = present(world, "S1")
    table = _table(pres, idxs, [float(world.y[i]) for i in idxs], "value")
    text = (
        "Here are some rows from a dataset, with the column names and categories replaced.\n\n" + table +
        "\n\nDo you recognize where these numbers might come from? Say what paper, group, or source, "
        "and how confident you are from 0 to 1. If you don't recognize it, say so."
    )
    return [Request(model, text, SOURCE_SCHEMA, salt=f"p2-{world.id}-{r}", tag={"probe": "P2", "world": world.id, "rep": r}) for r in range(reps)]


def _source_lines(world: World) -> list[str]:
    """The table in its published order and format, with no header."""
    if world.raw_lines:
        return world.raw_lines[1:]
    pres = present(world, "S0")
    rows = []
    for i in range(world.size):
        ix = tuple(int(v) for v in np.unravel_index(i, world.shape))
        rows.append(",".join([pres.labels[f][j] for f, j in enumerate(ix)] + [f"{float(world.y[ix]):g}"]))
    return rows


def _source_header(world: World) -> str:
    if world.raw_lines:
        return world.raw_lines[0]
    return ",".join(f.name for f in world.factors) + "," + world.objective


def build_p3(model, world, reps=5, shown=30, asked=20):
    lines = _source_lines(world)
    table = "\n".join([_source_header(world)] + lines[:shown])
    text = (
        "Here is the start of a data file.\n\n" + table +
        f"\n\nPlease continue the file with the next {asked} lines, exactly as you think they appear. "
        "Give each line as one string."
    )
    return [Request(model, text, CONTINUE_SCHEMA, salt=f"p3-{world.id}-{r}", tag={"probe": "P3", "world": world.id, "rep": r, "shown": shown, "asked": asked}) for r in range(reps)]


def p4_conditions(world: World, n=40) -> list[tuple[int, ...]]:
    return _sample(world, n, "p4")


def build_p4(model, world, semantic, reps=10):
    pres = present(world, semantic)
    settings = "\n".join(f"- {name}: {', '.join(pres.options(f))}" for f, name in enumerate(pres.names))
    conds = p4_conditions(world)
    listing = "\n".join(
        f"{k + 1}. " + ", ".join(f"{pres.names[f]}={pres.labels[f][i]}" for f, i in enumerate(ix))
        for k, ix in enumerate(conds)
    )
    text = (
        f"{pres.context} I want the highest {pres.objective} I can get. I haven't run anything yet.\n\n"
        f"Settings:\n{settings}\n\n"
        "From what you already know, rank every option of every setting from best to worst. "
        f"Then predict the {pres.objective} for each of these conditions, in order, as a number from 0 to 100:\n{listing}"
    )
    return [Request(model, text, rank_schema(pres), salt=f"p4-{world.id}-{semantic}-{r}", tag={"probe": "P4", "world": world.id, "semantic": semantic, "rep": r}) for r in range(reps)]


def build_p5(model, world, reps=10):
    out = []
    idxs = _sample(world, 40, "p5")
    for semantic in ("S0", "S2"):
        pres = present(world, semantic)
        table = _table(pres, idxs, [float(world.y[i]) for i in idxs])
        text = (
            f"{pres.context} A colleague sent me these results.\n\n" + table +
            "\n\nDoes anything look wrong or inconsistent with what you'd expect? Answer yes or no and explain briefly."
        )
        out += [Request(model, text, ANOMALY_SCHEMA, salt=f"p5-{world.id}-{semantic}-{r}", tag={"probe": "P5", "world": world.id, "semantic": semantic, "rep": r}) for r in range(reps)]
    return out


DECOY_BASES = ["K3-Mevate", "Cs-Ortholate", "Na-Tribenzate", "Rb-Quinolate"]
DECOY_LIGANDS = ["VentaPhos", "OrbiPhos", "KelvaPhos", "TetraMorph-P", "CyLoxaPhos", "NeoDalPhos"]
DECOY_SOLVENTS = ["Tetrazene-Me", "CPME-3", "Fluorocumene", "DiBuOx"]


def decoy_table(seed: int = 5, n: int = 20) -> str:
    rng = random.Random(seed)
    rows = ["Base | Ligand | Solvent | Temperature (°C) | reaction yield (%)"]
    for _ in range(n):
        rows.append(" | ".join([rng.choice(DECOY_BASES), rng.choice(DECOY_LIGANDS), rng.choice(DECOY_SOLVENTS),
                                str(rng.choice([80, 100, 120])), f"{rng.uniform(0, 95):.2f}"]))
    return "\n".join(rows)


def build_decoy(model, reps=10):
    text = (
        "Here are some rows from a dataset.\n\n" + decoy_table() +
        "\n\nDo you recognize this dataset? Say what paper, group, or source it might come from, "
        "and how confident you are from 0 to 1. If you don't recognize it, say so."
    )
    return [Request(model, text, SOURCE_SCHEMA, salt=f"decoy-{r}", tag={"probe": "decoy", "world": "decoy", "rep": r}) for r in range(reps)]


def build_all(model, world: World, reps=10) -> list[Request]:
    reqs = build_p1(model, world, reps) + build_p2(model, world, reps) + build_p3(model, world, max(3, reps // 2))
    for s in ("S0", "S1", "S2"):
        if s in world.semantics:
            reqs += build_p4(model, world, s, reps)
    if "S2" in world.semantics:
        reqs += build_p5(model, world, reps)
    return reqs


def score_source(reply: Reply, world_id: str) -> dict:
    try:
        d = json.loads(reply.text)
    except json.JSONDecodeError:
        return {"parsed": False}
    guess = str(d.get("source_guess", "")).lower()
    return {
        "parsed": True,
        "recognized": bool(d.get("recognized")),
        "confidence": d.get("confidence"),
        "correct_source": world_id == "shields" and any(t in guess for t in SHIELDS_TERMS),
        "guess": d.get("source_guess"),
    }


def score_continuation(reply: Reply, world: World, shown: int, asked: int) -> dict:
    """Compare claimed lines with the true next lines of the file."""
    try:
        rows = json.loads(reply.text)["rows"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"parsed": False}
    truth = _source_lines(world)[shown : shown + asked]
    norm = lambda s: "".join(str(s).split()).lower()
    exact = sum(norm(a) == norm(b) for a, b in zip(rows, truth))
    # a zero is the most common value in the data, so only a matching nonzero value is evidence of recall
    exact_nonzero = sum(norm(a) == norm(b) and float(b.rsplit(",", 1)[1]) != 0 for a, b in zip(rows, truth))
    # the last field is the measured value; matching everything else only shows the grid pattern was followed
    conditions = sum(norm(a).rsplit(",", 1)[0] == norm(b).rsplit(",", 1)[0] for a, b in zip(rows, truth))
    return {"parsed": True, "exact_rows": exact, "exact_nonzero": exact_nonzero, "true_nonzero": sum(float(b.rsplit(",", 1)[1]) != 0 for b in truth), "conditions_only": conditions, "asked": asked}


def score_zero_shot(reply: Reply, world: World, semantic: str) -> dict:
    """Spearman of predicted values against truth, plus ranking agreement with true marginals."""
    pres = present(world, semantic)
    try:
        d = json.loads(reply.text)
        preds = d["predictions"]
        ranks = d["rankings"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"parsed": False}
    conds = p4_conditions(world)
    out: dict = {"parsed": True}
    if len(preds) == len(conds):
        truth = [float(world.y[c]) for c in conds]
        out["pred_spearman"] = float(spearmanr(preds, truth).statistic)
    marg_rhos = []
    for f, name in enumerate(pres.names):
        order = ranks.get(prompts.key_for(name), [])
        if sorted(order) != sorted(pres.labels[f]):
            continue
        canon = [pres.label_to_index(f, label) for label in order]
        true_marg = world.marginals()[f]
        model_score = [-canon.index(i) for i in range(len(canon))]
        marg_rhos.append(float(spearmanr(model_score, true_marg).statistic))
    out["marginal_spearman"] = float(np.nanmean(marg_rhos)) if marg_rhos else None
    return out


def score_anomaly(reply: Reply) -> dict:
    try:
        d = json.loads(reply.text)
    except json.JSONDecodeError:
        return {"parsed": False}
    return {"parsed": True, "looks_wrong": bool(d.get("looks_wrong"))}
