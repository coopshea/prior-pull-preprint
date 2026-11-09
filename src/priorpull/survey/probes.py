"""No-feedback probes across the worlds, for prior strength and recognition."""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

from .. import probes as old, prompts as P
from ..llm import Reply, Request
from ..worlds import World
from .views import view


def build(model: str, worlds: dict[str, World], reps: int = 3) -> list[Request]:
    reqs = old.build_decoy(model, reps)
    for w in worlds.values():
        if w.source != "surface":
            reqs += old.build_p1(model, w, reps)
            reqs += old.build_p3(model, w, 1)
        for labels in (("real", "coded", "perm") if w.source != "surface" else ("coded", "story")):
            pres = view(w, labels, 0)
            settings = "\n".join(f"- {n}: {', '.join(pres.options(f))}" for f, n in enumerate(pres.names))
            text = (
                f"{pres.context} I want the highest {pres.objective} I can get. I haven't run anything yet.\n\nSettings:\n{settings}\n\n"
                "From what you already know, rank every option of every setting from best to worst."
            )
            schema = {
                "type": "object",
                "properties": {P.key_for(n): {"type": "array", "items": {"type": "string"}} for n in pres.names},
                "required": [P.key_for(n) for n in pres.names],
                "additionalProperties": False,
            }
            for r in range(reps):
                reqs.append(Request(model, text, schema, salt=f"survey-p4-{w.id}-{labels}-{r}", tag={"probe": "P4", "world": w.id, "labels": labels, "rep": r}))
    return reqs


def score_ranking(reply: Reply, world: World, labels: str) -> dict:
    """Spearman between the model's ranking and the true marginal means, per factor, averaged."""
    pres = view(world, labels, 0)
    try:
        data = json.loads(reply.text)
    except json.JSONDecodeError:
        return {"parsed": False}
    rhos = []
    marginals = world.marginals()
    for f, name in enumerate(pres.names):
        order = data.get(P.key_for(name), [])
        if sorted(order) != sorted(pres.labels[f]) or len(order) < 2 or np.ptp(marginals[f]) == 0:
            continue
        canon = [pres.label_to_index(f, x) for x in order]
        scores = [-canon.index(i) for i in range(len(canon))]
        rhos.append(float(spearmanr(scores, marginals[f]).statistic))
    return {"parsed": True, "marginal_spearman": float(np.nanmean(rhos)) if rhos else None}
