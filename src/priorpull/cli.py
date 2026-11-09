"""Command line: baselines, pilot, and full experiments.

    python -m priorpull.cli baselines --seeds 30
    python -m priorpull.cli pilot --cap 12
    python -m priorpull.cli pilot --mock
    python -m priorpull.cli full --model claude-opus-5-5 --cap 80
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd

from . import analysis, llm, probes, search
from .search import Condition
from .worlds import load_all

ROOT = Path(__file__).resolve().parents[2]
REAL = ("shields", "efb")
FORMULA = ("enzyme", "resin")


def pilot_conditions() -> list[Condition]:
    return [
        Condition("shields", "S0"),
        Condition("shields", "S1"),
        Condition("shields", "S2"),
        Condition("shields", "S0", "W-"),
        Condition("shields", "S0", "W0", "Fs"),
    ]


def full_plan() -> list[tuple[Condition, list[int]]]:
    """Fresh seeds from 100, so the confirmatory run is separate from the pilot."""
    wide, narrow = list(range(100, 130)), list(range(100, 112))
    plan: list[tuple[Condition, list[int]]] = []
    for world, seeds in [(w, wide) for w in REAL] + [(w, narrow) for w in FORMULA]:
        plan += [(Condition(world, s), seeds) for s in ("S0", "S1", "S2")]
        plan += [(Condition(world, s, "W0", "Fs"), seeds) for s in ("S0", "S1")]
        plan += [(Condition(world, s, "W-"), seeds) for s in ("S0", "S1")]
    plan += [(Condition("gmm", s), narrow) for s in ("S1", "S3")] + [(Condition("gmm", "S1", "W0", "Fs"), narrow)]
    for world in REAL + FORMULA:
        plan.append((Condition(world, "S3"), narrow))
        plan += [(Condition(world, s, w), narrow) for s, w in product(("S0", "S1"), ("W+", "Wpi", "Wexp"))]
    for world, strat in product(REAL, ("M1", "M2", "M3", "NR", "OPEN", "ORCH")):
        plan += [(Condition(world, s, strategy=strat), narrow) for s in ("S0", "S2")]
        if strat in ("M1", "M2", "M3", "ORCH"):
            plan.append((Condition(world, "S0", "W-", strategy=strat), narrow))
    return plan


def calibration_plan() -> list[tuple[Condition, list[int]]]:
    return [(Condition(w, s, calibrate=True), list(range(100, 112))) for w, s in product(REAL + FORMULA, ("S0", "S2"))]


def _baseline_job(args):
    wid, seed, name, feedback = args
    worlds = load_all()
    fn = {"random": search.run_random, "ofat": search.run_ofat, "bayes": search.run_bayes}[name]
    return fn(worlds[wid], seed, feedback=feedback)


def cmd_baselines(a):
    seeds = range(a.first_seed, a.first_seed + a.seeds)
    jobs = [(w, s, n, "F0") for w in load_all() for s in seeds for n in a.players.split(",")]
    jobs += [(w, s, n, "Fs") for w in load_all() for s in seeds for n in ("bayes", "random")]  # positive control
    out = ROOT / "runs" / "baselines"
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        trajs = list(pool.map(_baseline_job, jobs))
    search.save_trajectories(out / "trajectories.jsonl", trajs)
    df = analysis.table(trajs, load_all())
    df.to_csv(out / "scores.csv", index=False)
    print(df.groupby(["world", "player", "feedback"])[["regret", "auc", "percentile"]].mean().round(3))


def _backend(a):
    if a.mock:
        return llm.MockBackend()
    return llm.BatchBackend() if a.batch else llm.OnlineBackend(concurrency=a.concurrency)


def _run_probes(a, worlds, world_ids, backend, store):
    reqs = probes.build_decoy(a.model, a.reps)
    for w in world_ids:
        reqs += probes.build_all(a.model, worlds[w], a.reps)
    replies = llm.execute(reqs, backend, store)
    rows = []
    for r in reqs:
        reply = replies.get(r.key)
        if reply is None:
            continue
        p, w = r.tag["probe"], r.tag["world"]
        if p in ("P1", "P2", "decoy"):
            s = probes.score_source(reply, w)
        elif p == "P3":
            s = probes.score_continuation(reply, worlds[w], r.tag["shown"], r.tag["asked"])
        elif p == "P4":
            s = probes.score_zero_shot(reply, worlds[w], r.tag["semantic"])
        else:
            s = probes.score_anomaly(reply)
        rows.append({**r.tag, "stop": reply.stop_reason, **s})
    return pd.DataFrame(rows)


def _finish(a, run_dir, trajs, worlds, store):
    search.save_trajectories(run_dir / "trajectories.jsonl", trajs)
    df = analysis.table(trajs, worlds)
    df.to_csv(run_dir / "scores.csv", index=False)
    print(df.groupby(["world", "semantic", "wording", "feedback", "strategy"])[["regret", "auc", "percentile", "random_fills", "refusals", "retries"]].mean().round(3))
    print(f"spent ${store.spent:.2f} of ${a.cap:.2f}")


def cmd_pilot(a):
    worlds = load_all()
    name = "pilot_mock" if a.mock else "pilot"
    if a.mock:
        a.model = "mock"
    run_dir = ROOT / "runs" / name
    store = llm.Store(run_dir, a.cap)
    backend = _backend(a)
    pdf = _run_probes(a, worlds, ["shields"], backend, store)
    pdf.to_csv(run_dir / "probes.csv", index=False)
    trajs = search.run_llm(a.model, worlds, pilot_conditions(), list(range(a.seeds)), backend, store)
    _finish(a, run_dir, trajs, worlds, store)


def cmd_full(a):
    worlds = load_all()
    if a.mock:
        a.model = "mock"
    run_dir = ROOT / "runs" / f"full_{a.model}"
    store = llm.Store(run_dir, a.cap)
    backend = _backend(a)
    pdf = _run_probes(a, worlds, list(worlds), backend, store)
    pdf.to_csv(run_dir / "probes.csv", index=False)
    plan = full_plan() + (calibration_plan() if a.calibration else [])
    trajs = search.run_llm(a.model, worlds, plan, None, backend, store)
    _finish(a, run_dir, trajs, worlds, store)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    b = sub.add_parser("baselines")
    b.add_argument("--seeds", type=int, default=30)
    b.add_argument("--first-seed", type=int, default=100)
    b.add_argument("--players", default="random,ofat,bayes")
    b.add_argument("--workers", type=int, default=12)
    b.set_defaults(fn=cmd_baselines)
    for name, fn, seeds, reps in (("pilot", cmd_pilot, 6, 5), ("full", cmd_full, 12, 10)):
        c = sub.add_parser(name)
        c.add_argument("--model", default="claude-opus-5-5")
        c.add_argument("--cap", type=float, required=True, help="hard spend cap in USD")
        c.add_argument("--seeds", type=int, default=seeds)
        c.add_argument("--reps", type=int, default=reps)
        c.add_argument("--mock", action="store_true")
        c.add_argument("--batch", action="store_true")
        c.add_argument("--concurrency", type=int, default=8)
        c.add_argument("--calibration", action="store_true")
        c.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
