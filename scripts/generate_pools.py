"""Generate the frozen prompt, hint, and strategy pools once. Rerunning replays the stored replies."""
from pathlib import Path

from priorpull import llm
from priorpull.survey import inputs

store = llm.Store(Path("runs/pools"), 1.0)
pools = inputs.generate("claude-opus-5-5", llm.OnlineBackend(concurrency=4), store)
print("sha256", inputs.save(pools))
print("base", len(pools["base"]), "strategy", {k: len(v) for k, v in pools["strategy"].items()}, "hint", {k: len(v) for k, v in pools["hint"].items()})
print("spent", round(store.spent, 3))
