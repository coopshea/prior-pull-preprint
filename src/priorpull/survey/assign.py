"""Seeded, balanced assignment of inputs to trajectories."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from ..worlds import World

LABELS = ("real", "coded", "perm", "story")
SETUPS = ("single", "chat", "orch", "analyst")
BUDGETS = ((4, 3), (8, 3), (12, 1))
MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
HINT_CELLS = tuple(f"{s}_{c}" for s in ("self", "pi", "paper") for c in ("tentative", "confident"))
STRATEGY_KINDS = ("evidence", "audit", "caution")


@dataclass
class Row:
    id: int
    world: str
    labels: str  # as assigned
    effective_labels: str  # after setup and world limits
    prompt: int
    hint: str  # none, or source_certainty
    strategy: str  # none, or kind
    wording: int
    rationale: bool
    setup: str
    feedback: str
    rounds: int
    q: int
    model: str


def _balanced(levels, n: int, rng) -> list:
    repeated = [levels[i % len(levels)] for i in range(n)]
    return [repeated[int(i)] for i in rng.permutation(n)]


def effective(labels: str, setup: str, world: World) -> str:
    if setup == "analyst":
        return "coded"  # the analyst only ever sees coded labels
    if world.source == "surface":
        return {"real": "coded", "perm": "story"}.get(labels, labels)  # surfaces have no real names
    return labels


def build(worlds: dict[str, World], n: int, n_prompts: int, seed: int = 0) -> list[Row]:
    """Each source gets an equal share of trajectories, and each world an equal share of its source."""
    rng = np.random.default_rng(seed)
    by_source: dict[str, list[str]] = defaultdict(list)
    for w in worlds.values():
        by_source[w.source].append(w.id)
    sources = sorted(by_source)
    world_list: list[str] = []
    for i in range(n):
        ids = by_source[sources[i % len(sources)]]
        world_list.append(ids[(i // len(sources)) % len(ids)])
    world_list = [world_list[int(i)] for i in rng.permutation(n)]

    labels = _balanced(LABELS, n, rng)
    prompts = _balanced(list(range(n_prompts)), n, rng)
    hints = _balanced(["none"] * len(HINT_CELLS) + list(HINT_CELLS), n, rng)  # half have no hint
    strategies = _balanced(["none"] * 2 + list(STRATEGY_KINDS) * 1, n, rng)  # 40% no strategy
    wordings = _balanced(list(range(5)), n, rng)
    rationale = _balanced([True, False], n, rng)
    setups = _balanced(SETUPS, n, rng)
    feedback = _balanced(["true", "scrambled"], n, rng)
    budgets = _balanced(BUDGETS, n, rng)
    models = _balanced(MODELS, n, rng)
    rows = []
    for i in range(n):
        world = worlds[world_list[i]]
        rows.append(Row(i, world.id, labels[i], effective(labels[i], setups[i], world), prompts[i], hints[i], strategies[i], wordings[i],
                        rationale[i], setups[i], feedback[i], budgets[i][0], budgets[i][1], models[i]))
    return rows


def table_digest(rows: list[Row]) -> str:
    return hashlib.sha256(json.dumps([asdict(r) for r in rows], sort_keys=True).encode()).hexdigest()
