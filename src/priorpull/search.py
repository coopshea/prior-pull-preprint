"""Run search trajectories for LLM players and baselines."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import prompts
from .llm import Reply, Request, Store, execute
from .worlds import Oracle, World, _stable_int, present

ROUNDS = 8
Q = 3
TEMPLATE_CYCLE = ("T0", "T1", "T2")


@dataclass(frozen=True)
class Condition:
    world: str
    semantic: str
    wording: str = "W0"
    feedback: str = "F0"
    calibrate: bool = False
    strategy: str = "none"  # none, M1, M2, M3, NR (no rationale), OPEN (plan all at once), ORCH (limited-context agents)


@dataclass
class Trajectory:
    player: str
    world: str
    semantic: str
    wording: str
    feedback: str
    seed: int
    template: str
    picks: list[list[int]] = field(default_factory=list)
    observed: list[float] = field(default_factory=list)
    true_values: list[float] = field(default_factory=list)
    rounds: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    predictions: list[dict] = field(default_factory=list)
    random_fills: int = 0
    refusals: int = 0
    retries: int = 0
    strategy: str = "none"


def parse_reply(reply: Reply, pres, q: int, calibrate: bool):
    """Return (picks, reason, predictions), or None when the reply cannot be used."""
    try:
        data = json.loads(reply.text)
        reason = str(data.get("reason", ""))
        experiments = data["experiments"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if isinstance(experiments, list):
        if len(experiments) < q:
            return None
        rows = experiments[:q]
    elif isinstance(experiments, dict):
        try:
            rows = [experiments[f"experiment_{i + 1}"] for i in range(q)]
        except KeyError:
            return None
    else:
        return None
    picks, preds = [], []
    for row in rows:
        try:
            idx = tuple(
                pres.label_to_index(f, row[prompts.key_for(name)]) for f, name in enumerate(pres.names)
            )
        except (KeyError, ValueError, TypeError):
            return None
        picks.append(idx)
        if calibrate:
            preds.append({"predicted": row.get("predicted"), "p_beats_best": row.get("p_beats_best")})
    return picks, reason, preds


def _random_point(world: World, rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.randrange(n) for n in world.shape)


def _request(state, model, round_no, attempt) -> Request:
    c, t, pres = state["cond"], state["traj"], state["pres"]
    rounds, q = state["rounds"], state["q"]
    if c.strategy == "ORCH":
        prompt = prompts.orchestrator_prompt(
            pres, state["analyst_picks"], state["domain_note"], round_no, rounds, q, c.wording
        )
    else:
        prompt = prompts.build_prompt(
            pres, state["history"], round_no, rounds, q, t.template, c.wording, c.calibrate, c.strategy
        )
    return Request(
        model=model,
        prompt=prompt,
        schema=prompts.pick_schema(pres, q, c.calibrate, reason=c.strategy != "NR"),
        system=prompts.system_for(pres, c.wording),
        attempt=attempt,
        salt=f"{c.world}|{t.seed}|{c.semantic}|{c.wording}|{c.feedback}|{c.strategy}",
        tag={"round": round_no, "seed": t.seed, "q": q, **asdict(c)},
    )


def _prepare_orchestrated(states, model, round_no, backend, store, max_attempts) -> None:
    """Get the coded data analyst's picks each round, and the domain note once."""
    need_domain = [s for s in states if s["domain_note"] is None]
    if need_domain:
        requests = {}
        for s in need_domain:
            c = s["cond"]
            requests[id(s)] = Request(
                model, prompts.domain_prompt(s["pres"]), prompts.DOMAIN_SCHEMA,
                salt=f"domain|{c.world}|{s['traj'].seed}|{c.semantic}|{c.wording}|{c.feedback}",
                tag={"role": "domain", "seed": s["traj"].seed, **asdict(c)},
            )
        replies = execute(list(requests.values()), backend, store)
        for s in need_domain:
            reply = replies.get(requests[id(s)].key)
            try:
                s["domain_note"] = json.loads(reply.text)["note"]
            except (AttributeError, json.JSONDecodeError, KeyError, TypeError):
                s["domain_note"] = "No comment available."
    pending = list(states)
    for attempt in range(max_attempts):
        if not pending:
            break
        requests = {}
        for s in pending:
            c, t, anon = s["cond"], s["traj"], s["anon"]
            requests[id(s)] = Request(
                model,
                prompts.build_prompt(anon, s["history"], round_no, s["rounds"], s["q"], t.template, "W0"),
                prompts.pick_schema(anon, s["q"]),
                attempt=attempt,
                salt=f"analyst|{c.world}|{t.seed}|{c.semantic}|{c.wording}|{c.feedback}",
                tag={"role": "analyst", "round": round_no, "seed": t.seed, "q": s["q"], **asdict(c)},
            )
        replies = execute(list(requests.values()), backend, store)
        still = []
        for s in pending:
            reply = replies.get(requests[id(s)].key)
            parsed = parse_reply(reply, s["anon"], s["q"], False) if reply else None
            if parsed is None:
                still.append(s)
            else:
                s["analyst_picks"] = parsed[0]
        for s in still:
            s["analyst_picks"] = []
        pending = still


def run_llm(
    model: str,
    worlds: dict[str, World],
    conditions,
    seeds: list[int] | None,
    backend,
    store: Store,
    rounds: int | None = None,
    q: int = Q,
    max_attempts: int = 3,
) -> list[Trajectory]:
    """Advance every trajectory in lockstep. Pass seeds=None to give (condition, seeds) pairs instead."""
    plan = conditions if seeds is None else [(c, seeds) for c in conditions]
    active = []
    for cond, cond_seeds in plan:
        for seed in cond_seeds:
            world = worlds[cond.world]
            state_rounds, state_q = rounds or world.rounds, q
            if cond.strategy == "OPEN":
                state_q, state_rounds = state_rounds * q, 1
            template = TEMPLATE_CYCLE[seed % len(TEMPLATE_CYCLE)]
            active.append(
                {
                    "cond": cond,
                    "pres": present(world, cond.semantic),
                    "anon": present(world, "S1"),
                    "oracle": Oracle(world, seed, cond.feedback),
                    "history": [],
                    "rounds": state_rounds,
                    "q": state_q,
                    "analyst_picks": [],
                    "domain_note": None,
                    "traj": Trajectory(
                        model, cond.world, cond.semantic, cond.wording, cond.feedback, seed, template,
                        strategy=cond.strategy,
                    ),
                }
            )
    for round_no in range(1, max(s["rounds"] for s in active) + 1):
        current = [s for s in active if round_no <= s["rounds"]]
        orchestrated = [s for s in current if s["cond"].strategy == "ORCH"]
        if orchestrated:
            _prepare_orchestrated(orchestrated, model, round_no, backend, store, max_attempts)
        pending = list(current)
        decisions: dict[int, tuple] = {}
        for attempt in range(max_attempts):
            if not pending:
                break
            requests = {id(state): _request(state, model, round_no, attempt) for state in pending}
            replies = execute(list(requests.values()), backend, store)
            still = []
            for state in pending:
                reply = replies.get(requests[id(state)].key)
                if reply is not None and reply.stop_reason == "refusal":
                    state["traj"].refusals += 1
                    continue  # a refusal is recorded and filled at random, not retried
                parsed = parse_reply(reply, state["pres"], state["q"], state["cond"].calibrate) if reply else None
                if parsed is None:
                    still.append(state)
                    state["traj"].retries += 1
                else:
                    decisions[id(state)] = parsed
            pending = still

        for state in current:
            t, world = state["traj"], state["oracle"].world
            if id(state) in decisions:
                picks, reason, preds = decisions[id(state)]
            else:
                rng = random.Random(_stable_int(t.world, t.seed, t.semantic, t.wording, t.feedback, t.strategy, round_no))
                picks = [_random_point(world, rng) for _ in range(state["q"])]
                reason, preds = "", []
                t.random_fills += state["q"]
            t.reasons.append(reason)
            t.predictions.append({"round": round_no, "items": preds})
            for idx in picks:
                y = state["oracle"].observe(idx)
                state["history"].append((idx, y))
                t.picks.append(list(idx))
                t.observed.append(y)
                t.true_values.append(float(world.y[idx]))
                t.rounds.append(round_no)
    return [s["traj"] for s in active]


def _baseline_shell(player, world, seed, feedback="F0") -> Trajectory:
    return Trajectory(player, world.id, "blind", "W0", feedback, seed, "none")


def _record(t: Trajectory, world: World, oracle: Oracle, idx, round_no: int) -> float:
    y = oracle.observe(idx)
    t.picks.append([int(i) for i in idx])
    t.observed.append(y)
    t.true_values.append(float(world.y[tuple(idx)]))
    t.rounds.append(round_no)
    return y


def run_random(world: World, seed: int, rounds: int | None = None, q: int = Q, feedback: str = "F0") -> Trajectory:
    rounds = rounds or world.rounds
    t = _baseline_shell("random", world, seed, feedback)
    oracle = Oracle(world, seed, feedback)
    rng = np.random.default_rng(_stable_int(world.id, seed, "random"))
    flat = rng.choice(world.size, size=rounds * q, replace=False)
    for i, f in enumerate(flat):
        _record(t, world, oracle, tuple(int(v) for v in np.unravel_index(int(f), world.shape)), i // q + 1)
    return t


def run_ofat(world: World, seed: int, rounds: int | None = None, q: int = Q, feedback: str = "F0") -> Trajectory:
    """One factor at a time from a random start, holding the best setting found so far."""
    rounds = rounds or world.rounds
    t = _baseline_shell("ofat", world, seed, feedback)
    oracle = Oracle(world, seed, feedback)
    rng = random.Random(_stable_int(world.id, seed, "ofat"))
    current = list(_random_point(world, rng))
    best_y = _record(t, world, oracle, tuple(current), 1)
    total, used = rounds * q, 1
    while used < total:
        improved = False
        for f in rng.sample(range(len(world.shape)), len(world.shape)):
            levels = [i for i in range(world.shape[f]) if i != current[f]]
            rng.shuffle(levels)
            best_level = current[f]
            for level in levels:
                if used >= total:
                    break
                trial = list(current)
                trial[f] = level
                used += 1
                y = _record(t, world, oracle, tuple(trial), (used - 1) // q + 1)
                if y > best_y:
                    best_y, best_level, improved = y, level, True
            current[f] = best_level
            if used >= total:
                break
        if not improved and used < total:
            current = list(_random_point(world, rng))
            used += 1
            best_y = max(best_y, _record(t, world, oracle, tuple(current), (used - 1) // q + 1))
    return t


def run_bayes(world: World, seed: int, rounds: int | None = None, q: int = Q, feedback: str = "F0") -> Trajectory:
    """BayBE on a label-blind encoding, so one run serves every semantic condition."""
    import pandas as pd
    import torch
    from baybe import Campaign
    from baybe.objectives import SingleTargetObjective
    from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter
    from baybe.recommenders import BotorchRecommender, RandomRecommender, TwoPhaseMetaRecommender
    from baybe.searchspace import SearchSpace
    from baybe.targets import NumericalTarget
    from baybe.utils.random import set_random_seed

    rounds = rounds or world.rounds
    torch.set_num_threads(1)
    set_random_seed(_stable_int(world.id, seed, "bayes") % (2**31))
    params = []
    names = [f"f{i}" for i in range(len(world.shape))]
    for name, f in zip(names, world.factors):
        if f.numeric:
            params.append(NumericalDiscreteParameter(name=name, values=list(range(len(f.levels)))))
        else:
            params.append(
                CategoricalParameter(name=name, values=[str(i) for i in range(len(f.levels))], encoding="OHE")
            )
    campaign = Campaign(
        searchspace=SearchSpace.from_product(params),
        objective=SingleTargetObjective(NumericalTarget(name="y")),
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(), recommender=BotorchRecommender(), switch_after=q
        ),
    )
    t = _baseline_shell("bayes", world, seed, feedback)
    oracle = Oracle(world, seed, feedback)
    for round_no in range(1, rounds + 1):
        rec = campaign.recommend(batch_size=q)
        ys = []
        for _, row in rec.iterrows():
            idx = tuple(int(row[n]) for n in names)
            ys.append(_record(t, world, oracle, idx, round_no))
        measured = pd.DataFrame(rec).copy()
        measured["y"] = ys
        campaign.add_measurements(measured)
    return t


def save_trajectories(path: Path, trajectories: list[Trajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for t in trajectories:
            fh.write(json.dumps(asdict(t)) + "\n")


def load_trajectories(path: Path) -> list[Trajectory]:
    return [Trajectory(**json.loads(line)) for line in path.read_text().splitlines()]
