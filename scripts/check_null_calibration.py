"""On data with no real effects, how often does the analysis report one? Uses no model calls."""
import sys
import warnings

import numpy as np
import pandas as pd

from priorpull.survey import analyze, assign, inputs
from priorpull.worlds import load_survey

warnings.filterwarnings("ignore")
worlds = load_survey()
pools = inputs.load()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
sims = int(sys.argv[2]) if len(sys.argv) > 2 else 30
rows = assign.build(worlds, n, len(pools["base"]))
base = pd.DataFrame([{
    "id": r.id, "world": r.world, "source": worlds[r.world].source, "labels": r.effective_labels,
    "hint_kind": "none" if r.hint == "none" else "hint", "hint": r.hint, "strategy": r.strategy, "rationale": r.rationale,
    "setup": r.setup, "feedback": r.feedback, "budget": f"{r.rounds}x{r.q}", "model": r.model, "prompt": r.prompt,
} for r in rows])
rng = np.random.default_rng(0)
world_effect = {w: rng.normal(0.7, 0.1) for w in worlds}
hc3, rand, any_bh = [], [], 0
for i in range(sims):
    df = base.copy()
    df["quality"] = df["world"].map(world_effect) + rng.normal(0, 0.15, len(df))
    t = analyze.main_effects(df, n_perm=199, seed=i)
    main = t[t.scope == "all"]
    hc3.append((main["p_hc3"] < 0.05).mean())
    rand.append((main["p"] < 0.05).mean())
    any_bh += int((t["p_bh"] < 0.05).any())
print(f"n={n}, sims={sims}: share of null main effects with p<.05: HC3 {np.mean(hc3):.3f}, randomization {np.mean(rand):.3f}; runs with any BH-significant effect: {any_bh}/{sims}")
