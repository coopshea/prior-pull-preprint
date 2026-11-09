"""Best yield so far for human players, BO from human starts, and Opus pilot runs, all on the Shields grid."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
g = pd.read_csv(root / "data/raw/evml/arylation_game_summary.csv")
b = pd.read_csv(root / "data/raw/evml/arylation_bo_results_GP-EI_bs5_human_init.csv", index_col=0)
raw = g[[f"yield_{i}" for i in range(1, 101)]].to_numpy(float)
played = (~np.isnan(raw)).sum(axis=1)  # players could stop early, and keep the best they found
human = np.fmax.accumulate(raw, axis=1)
bo = np.maximum.accumulate(b.to_numpy(float), axis=1)
points = [5, 10, 15, 20, 25, 50, 100]
rng = np.random.default_rng(0)
ci = lambda a: np.percentile(rng.choice(a, size=(4000, len(a))).mean(axis=1), [2.5, 97.5])
rows = []
for n in points:
    for name, arr in (("Expert players", human), ("BO from human starts", bo)):
        v = arr[:, n - 1]
        lo, hi = ci(v)
        rows.append({"who": name, "n": n, "mean": v.mean(), "lo": lo, "hi": hi, "reach90": float((v >= 90).mean()), "runs": len(v), "still_playing": int((played >= n).sum()) if arr is human else len(v)})
pilot = [json.loads(l) for l in open(root / "runs/pilot/trajectories.jsonl")]
for label, match in (("Opus, real names (pilot)", lambda t: t["semantic"] == "S0" and t["wording"] == "W0" and t["feedback"] == "F0"),):
    arr = np.array([np.maximum.accumulate(t["true_values"]) for t in pilot if match(t)])
    for n in (6, 12, 18, 24):
        v = arr[:, n - 1]
        lo, hi = ci(v)
        rows.append({"who": label, "n": n, "mean": v.mean(), "lo": lo, "hi": hi, "reach90": float((v >= 90).mean()), "runs": len(v)})
t = pd.DataFrame(rows)
t.to_csv(root / "runs/core/shields_baseline.csv", index=False)
print(t.round(2).to_string())
print("experiments played: median", int(np.median(played)), "min", int(played.min()), "max", int(played.max()))
print("players by area", g["area"].value_counts().to_dict(), "expertise", g["expertise"].value_counts().to_dict())
for n in (25, 100):
    print(f"sd of best at {n}: humans {human[:, n-1].std():.1f}, BO {bo[:, n-1].std():.1f}")
