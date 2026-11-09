"""What the model is shown: labels for a world under each condition. Numbers never change."""
from __future__ import annotations

import numpy as np

from ..prompts import key_for  # noqa: F401
from ..worlds import Presentation, World, _stable_int

STORIES = (
    ("I'm developing a loaf for our bakery.", "loaf score (0 to 100)", ["Flour", "Starter", "Proof", "Oven", "Hydration"]),
    ("I'm dialing in a coffee recipe for our cafe.", "tasting score (0 to 100)", ["Bean", "Roast", "Water", "Grind", "Dose"]),
    ("I'm tuning a garden plot for the summer.", "harvest score (0 to 100)", ["Cultivar", "Mulch", "Feed", "Shade", "Spacing"]),
    ("I'm working out a stock for our restaurant.", "flavor score (0 to 100)", ["Bones", "Aromatics", "Salt", "Simmer", "Skim"]),
)
TONES = ("Amber", "Birch", "Cedar", "Dune", "Ember", "Fern", "Grove", "Hazel", "Iris", "Juniper", "Kestrel", "Linden")
SHAPES = ("Field", "Ridge", "Hollow", "Creek")


def _story_levels(n: int, rng) -> list[str]:
    names = [f"{t} {s}" for s in SHAPES for t in TONES]
    return [names[int(i)] for i in rng.permutation(len(names))[:n]]


def view(world: World, labels: str, seed: int) -> Presentation:
    n_f = len(world.factors)
    if labels == "real":
        names = [f.name for f in world.factors]
        shown = [list(f.levels) for f in world.factors]
        context, objective = world.context, world.objective
    elif labels == "perm":
        rng = np.random.default_rng(_stable_int(world.id, seed, "perm"))
        names = [f.name for f in world.factors]
        shown = [[f.levels[int(j)] for j in rng.permutation(len(f.levels))] for f in world.factors]
        context, objective = world.context, world.objective
    elif labels == "coded":
        rng = np.random.default_rng(_stable_int(world.id, "coded"))
        names = [f"x{i + 1}" for i in range(n_f)]
        shown = []
        for f in world.factors:
            if f.numeric:
                shown.append([str(i + 1) for i in range(len(f.levels))])
            else:
                shown.append([f"L{int(i) + 1}" for i in rng.permutation(len(f.levels))])
        context, objective = "I'm trying to optimize a process with the settings below.", "output score (0 to 100)"
    elif labels == "story":
        rng = np.random.default_rng(_stable_int(world.id, seed, "story"))
        context, objective, pool = STORIES[int(rng.integers(len(STORIES)))]
        names = [pool[i % len(pool)] if i < len(pool) else f"{pool[i % len(pool)]} {i}" for i in range(n_f)]
        shown = [_story_levels(len(f.levels), rng) for f in world.factors]
    else:
        raise ValueError(labels)

    def sort_key(f_idx):
        def key(i):
            text = shown[f_idx][i]
            try:
                return (0, float(text), "")
            except ValueError:
                return (1, 0.0, text.lower())
        return key

    order = [sorted(range(len(f.levels)), key=sort_key(f_idx)) for f_idx, f in enumerate(world.factors)]
    return Presentation(world, labels, context, objective, names, shown, order)
