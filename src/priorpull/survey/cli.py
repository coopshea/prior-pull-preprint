"""Build the assignment table, run the survey, and analyze it.

    python -m priorpull.survey.cli table
    python -m priorpull.survey.cli run --cap 40 --batch
    python -m priorpull.survey.cli run --mock --cap 1 --n 40 --pairs 30
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import product
from pathlib import Path

import pandas as pd

from .. import llm, probes as old_probes, search
from ..worlds import load_survey
from . import analyze, assign, inputs, probes, run

ROOT = Path(__file__).resolve().parents[3]


def cmd_table(a):
    worlds = load_survey()
    pools = inputs.load()
    rows = assign.build(worlds, a.n, len(pools["base"]))
    path = ROOT / "runs" / "survey" / "assignment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows]))
    pairs = run.build_pairs(worlds, a.pairs, rows)
    (ROOT / "runs" / "survey" / "pairs.json").write_text(json.dumps(pairs))
    print("rows", len(rows), "assignment sha256", assign.table_digest(rows), "pools sha256", inputs.digest())
    print(pd.DataFrame([asdict(r) for r in rows])[["labels", "setup", "feedback", "hint", "strategy", "model"]].melt().value_counts().sort_index().to_string())


def _baseline(args):
    wid, seed, name, feedback, rounds, q = args
    worlds = load_survey()
    fn = {"random": search.run_random, "ofat": search.run_ofat, "bayes": search.run_bayes}[name]
    t = fn(worlds[wid], seed, rounds=rounds, q=q, feedback=feedback)
    return {"world": wid, "seed": seed, "player": name, "feedback": feedback, "budget": f"{rounds}x{q}", **analyze.quality({"true": t.true_values}, worlds[wid])}


def cmd_baselines(a):
    worlds = load_survey()
    jobs = []
    for wid, seed, (rounds, q) in product(worlds, range(a.seeds), assign.BUDGETS):
        jobs += [(wid, seed, n, "F0", rounds, q) for n in ("random", "ofat", "bayes")]
        jobs += [(wid, seed, n, "Fs", rounds, q) for n in ("random", "bayes")]
    out = ROOT / "runs" / "survey_baselines"
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(_baseline, jobs, chunksize=4))
    t = pd.DataFrame(rows)
    t.to_csv(out / "baselines.csv", index=False)
    t["source"] = t["world"].map(lambda w: worlds[w].source)
    print(t.groupby(["source", "player", "feedback"])["quality"].mean().round(3).to_string())


def backend(a):
    if a.mock:
        return llm.MockBackend()
    return llm.BatchBackend() if a.batch else llm.OnlineBackend(concurrency=a.concurrency)


def cmd_run(a):
    worlds = load_survey()
    pools = inputs.load()
    rows = assign.build(worlds, a.n, len(pools["base"]))
    pairs = run.build_pairs(worlds, a.pairs, rows)
    out = ROOT / "runs" / ("survey_mock" if a.mock else a.name)
    store = llm.Store(out, a.cap)
    be = backend(a)
    override = "mock" if a.mock else None

    records = run.run_search(rows, worlds, pools, be, store, model_override=override)
    run.save_jsonl(out / "search.jsonl", records)
    replay = run.run_replay(pairs, {r.id: r for r in rows}, worlds, pools, be, store, model_override=override)
    run.save_jsonl(out / "replay.jsonl", replay)
    reqs = probes.build(override or "claude-opus-5-5", worlds, a.reps)
    replies = llm.execute(reqs, be, store)
    probe_rows = []
    for r in reqs:
        x = replies.get(r.key)
        if x is None:
            continue
        p, w = r.tag["probe"], r.tag["world"]
        if p == "P4":
            s = probes.score_ranking(x, worlds[w], r.tag["labels"])
        elif p == "P3":
            s = old_probes.score_continuation(x, worlds[w], r.tag["shown"], r.tag["asked"])
        else:
            s = old_probes.score_source(x, w)
        probe_rows.append({**r.tag, "stop": x.stop_reason, **s})
    pd.DataFrame(probe_rows).to_csv(out / "probes.csv", index=False)
    report(rows, records, replay, worlds, out)
    print(f"spent ${store.spent:.2f} of ${a.cap:.2f}")


def report(rows, records, replay, worlds, out: Path):
    t = analyze.search_table(rows, records, worlds)
    t.to_csv(out / "search_table.csv", index=False)
    effects = analyze.main_effects(t)
    effects.to_csv(out / "effects.csv", index=False)
    json.dump(analyze.prompt_variance(t), (out / "prompt_variance.json").open("w"))
    rt = analyze.replay_table(replay, worlds)
    rt.to_csv(out / "replay_table.csv", index=False)
    summary = analyze.replay_summary(rt)
    summary.to_csv(out / "replay_summary.csv", index=False)
    print(effects[effects.scope == "all"][["term", "estimate", "lo", "hi", "p_bh"]].round(3).to_string())
    print(summary.round(3).to_string())
    print("failed rows", int((t.random_fills > 0).sum()), "of", len(t), "; replay pairs scored", len(rt))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    t = sub.add_parser("table")
    t.add_argument("--n", type=int, default=600)
    t.add_argument("--pairs", type=int, default=750)
    t.set_defaults(fn=cmd_table)
    b = sub.add_parser("baselines")
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--workers", type=int, default=12)
    b.set_defaults(fn=cmd_baselines)
    r = sub.add_parser("run")
    r.add_argument("--cap", type=float, required=True)
    r.add_argument("--n", type=int, default=600)
    r.add_argument("--pairs", type=int, default=750)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--name", default="survey")
    r.add_argument("--mock", action="store_true")
    r.add_argument("--batch", action="store_true")
    r.add_argument("--concurrency", type=int, default=8)
    r.set_defaults(fn=cmd_run)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
