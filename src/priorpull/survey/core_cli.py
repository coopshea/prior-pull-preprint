"""Run the core study and draw its three result figures.

    python -m priorpull.survey.core_cli run --mock --cap 1 --per-cell 3
    python -m priorpull.survey.core_cli run --cap 30 --per-cell 20 --batch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .. import llm, probes as old_probes
from ..worlds import load_survey
from . import core, inputs

ROOT = Path(__file__).resolve().parents[3]


def draw(m: pd.DataFrame, out: Path) -> None:
    t2, t3, t4 = core.figure2_table(m), core.figure3_table(m), core.figure4_table(m)
    t2.to_csv(out / "figure2.csv", index=False)
    t3.to_csv(out / "figure3.csv", index=False)
    t4.to_csv(out / "figure4.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3))
    ordered = t2.iloc[::-1]
    ax.errorbar(ordered["mean"], ordered["change"], xerr=[ordered["mean"] - ordered["lo"], ordered["hi"] - ordered["mean"]], fmt="o", color="#2a78d6", capsize=2)
    noise = t2.loc[t2["change"] == "Ask again (noise)", "mean"].iloc[0]
    ax.axvline(noise, color="#898781", linestyle="--", linewidth=1)
    ax.set_xlabel("Change in picks")
    fig.tight_layout()
    fig.savefig(out / "figure2.png", dpi=200)
    fig, ax = plt.subplots(figsize=(5, 3))
    styles = {"Noise floor": ("#898781", "--"), "Data": ("#2a78d6", "-"), "Priors": ("#eb6834", "-"), "Wording": ("#1baf7a", ":")}
    for category, (color, line) in styles.items():
        g = t3[t3.category == category].sort_values("k")
        ax.errorbar(g["k"], g["mean"], yerr=[g["mean"] - g["lo"], g["hi"] - g["mean"]], color=color, linestyle=line, marker="o", capsize=2, label=category)
    ax.set_xlabel("Results shown")
    ax.set_ylabel("Change in picks")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figure3.png", dpi=200)


def report_sources(requests, replies, worlds) -> pd.DataFrame:
    rows = []
    for r in requests:
        x = replies.get(r.key)
        if x is not None:
            rows.append({**r.tag, **old_probes.score_source(x, r.tag["world"])})
    return pd.DataFrame(rows)


def cmd_run(a):
    worlds = load_survey(0)
    pools = inputs.load()
    out = ROOT / "runs" / ("core_mock" if a.mock else a.name)
    store = llm.Store(out, a.cap)
    backend = llm.MockBackend() if a.mock else (llm.BatchBackend() if a.batch else llm.OnlineBackend(concurrency=a.concurrency))
    model = "mock" if a.mock else core.MODEL
    units = core.build_units(worlds, a.per_cell, len(pools["base"]))
    print("units", len(units), "digest", core.units_digest(units), "pools", inputs.digest())

    rank_requests = core.ranking_requests(worlds, a.rank_reps, model)
    rank_replies = llm.execute(rank_requests, backend, store)
    priors = core.prior_scores(rank_requests, rank_replies, worlds)
    (out / "priors.json").write_text(json.dumps(priors))

    source_requests = old_probes.build_decoy(model, a.source_reps)
    for w in worlds.values():
        if w.source in core.SOURCES:
            source_requests += old_probes.build_p1(model, w, a.source_reps)
    source_replies = llm.execute(source_requests, backend, store)
    report_sources(source_requests, source_replies, worlds).to_csv(out / "source_guess.csv", index=False)

    results = core.run_units(units, worlds, pools, backend, store, model=model)
    with (out / "decisions.jsonl").open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    m = core.measure(results, worlds, priors)
    m.to_csv(out / "measures.csv", index=False)
    draw(m, out)
    bad = sum(1 for r in results if not r["picks"])
    print(f"decisions {len(results)}, unusable {bad}, spent ${store.spent:.2f} of ${a.cap:.2f}")
    print(core.figure2_table(m).round(3).to_string())
    print(core.figure3_table(m).round(3).to_string())
    print(core.figure4_table(m).round(3).to_string())


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--cap", type=float, required=True)
    r.add_argument("--per-cell", type=int, default=20)
    r.add_argument("--rank-reps", type=int, default=3)
    r.add_argument("--source-reps", type=int, default=3)
    r.add_argument("--name", default="core")
    r.add_argument("--mock", action="store_true")
    r.add_argument("--batch", action="store_true")
    r.add_argument("--concurrency", type=int, default=8)
    r.set_defaults(fn=cmd_run)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
