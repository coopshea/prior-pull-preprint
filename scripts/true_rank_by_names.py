"""True rank with real, coded, and swapped names, by source, with paired differences from real names."""
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
m = pd.read_csv(root / "runs/core/measures.csv")
wide = m[m.arm.isin(["A1", "coded", "swapped"])].pivot(index=["source", "unit"], columns="arm", values="true_rank").reset_index()
sources = ["shields", "suzuki", "efb", "cernak"]
rng = np.random.default_rng(0)
B = 10000


def interval(by_source):
    """Resample units within each source, weight sources equally."""
    draws = np.mean([rng.choice(x, size=(B, len(x))).mean(axis=1) for x in by_source], axis=0)
    return np.percentile(draws, [2.5, 97.5])


rows = []
for name, parts in [(s, [s]) for s in sources] + [("all", sources)]:
    data = [wide[wide.source == s] for s in parts]
    row = {"source": name, "n": sum(len(d) for d in data)}
    for arm in ["A1", "coded", "swapped"]:
        row[arm] = float(np.mean([d[arm].mean() for d in data]))
    for arm in ["coded", "swapped"]:
        diffs = [(d[arm] - d["A1"]).to_numpy() for d in data]
        low, high = interval(diffs)
        row[f"{arm}_diff"], row[f"{arm}_low"], row[f"{arm}_high"] = float(np.mean([x.mean() for x in diffs])), float(low), float(high)
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(root / "runs/core/true_rank_names.csv", index=False)
print(out.round(3).to_string(index=False))
