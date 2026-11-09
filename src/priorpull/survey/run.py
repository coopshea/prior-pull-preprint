"""Run the sampled search trajectories and the one-change replay decisions."""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .. import prompts as P
from ..llm import Reply, Request, Store, execute
from ..search import parse_reply
from ..worlds import Oracle, World, _stable_int
from .assign import Row
from .views import view

FORMAT = "\n\nKeep the reason to two sentences or fewer, and give exactly {q} experiments."
NO_REASON = "\n\nGive exactly {q} experiments and nothing else."


def settings_text(pres) -> str:
    return "\n".join(f"- {n}: {', '.join(pres.options(f))}" for f, n in enumerate(pres.names))


def hint_location(world: World, row_id: int) -> tuple[int, int]:
    """A random factor and level, drawn without looking at the outcomes."""
    rng = np.random.default_rng(_stable_int(world.id, row_id, "hint"))
    f = int(rng.integers(len(world.shape)))
    return f, int(rng.integers(world.shape[f]))


def hint_sentence(pres, row: Row, pools: dict) -> str:
    if row.hint == "none" or row.setup == "analyst":
        return ""
    f, level = hint_location(pres.world, row.id)
    options = pools["hint"][row.hint]
    template = options[row.id % len(options)]
    return template.format(phrase=f"{pres.names[f]} = {pres.labels[f][level]}") + " "


def strategy_sentence(row: Row, pools: dict) -> str:
    if row.strategy == "none":
        return ""
    options = pools["strategy"][row.strategy]
    return " " + options[row.wording % len(options)]


def closing(row: Row, q: int) -> str:
    return (FORMAT if row.rationale else NO_REASON).format(q=q)


def base_prompt(pres, row: Row, history, round_no: int, pools: dict, hint: str) -> str:
    template = pools["base"][row.prompt % len(pools["base"])]
    text = template.format(
        context=pres.context, objective=pres.objective, q=row.q, rounds=row.rounds,
        settings=settings_text(pres), history=P.render_history(pres, history).replace("Results so far:\n", ""), round=round_no, hint=hint,
    )
    return text + strategy_sentence(row, pools) + closing(row, row.q)


def schema_for(pres, row: Row) -> dict:
    big = sum(len(o) for o in pres.labels) * row.q > 90  # large enumerations exceed the API's grammar limit in slot form
    return P.pick_schema(pres, row.q, False, reason=row.rationale, use_array=big)


class Search:
    """State of one trajectory."""

    def __init__(self, row: Row, world: World, model: str):
        self.row, self.world, self.model = row, world, model
        self.pres = view(world, row.effective_labels, row.id)
        self.coded = view(world, "coded", row.id)
        self.oracle = Oracle(world, row.id, "F0" if row.feedback == "true" else "Fs")
        self.history: list = []
        self.turns: list = []
        self.domain: str | None = None
        self.analyst: list = []
        self.record: dict = {
            "id": row.id, "picks": [], "observed": [], "true": [], "round": [], "reasons": [],
            "retries": 0, "refusals": 0, "random_fills": 0,
        }

    def tag(self, round_no, role) -> dict:
        return {"study": "search", "id": self.row.id, "round": round_no, "role": role, "q": self.row.q}

    def request(self, round_no: int, attempt: int, pools: dict) -> Request:
        row = self.row
        salt = f"survey|{row.id}"
        if row.setup == "orch":
            proposals = "\n".join(
                "- " + ", ".join(f"{self.pres.names[f]}={self.pres.labels[f][i]}" for f, i in enumerate(ix)) for ix in self.analyst
            ) or "- none available"
            prompt = (
                f"{self.pres.context} I can run {row.q} experiments per round, for {row.rounds} rounds in total, and I want the highest {self.pres.objective} I can get.\n\n"
                f"Settings:\n{settings_text(self.pres)}\n\nTwo assistants have reported to you. You haven't seen the results yourself.\n"
                f"The data analyst has seen every result so far, but with the settings coded so it doesn't know what they mean. It proposes:\n{proposals}\n\n"
                f"The domain expert knows the field but hasn't seen any results. It says: {self.domain}\n\n"
                f"This is round {round_no} of {row.rounds}. {hint_sentence(self.pres, row, pools)}Which {row.q} experiments should I run next?"
                + strategy_sentence(row, pools) + closing(row, row.q)
            )
            return Request(self.model, prompt, schema_for(self.pres, row), attempt=attempt, salt=salt, tag=self.tag(round_no, "orchestrator"))
        if row.setup == "chat" and round_no > 1:
            done = self.history[-row.q:]
            prompt = (
                P.render_history(self.pres, done).replace("Results so far:", "Here are the results from that round:")
                + f"\n\nThis is round {round_no} of {row.rounds}. Which {row.q} experiments should I run next?"
                + closing(row, row.q)
            )
            return Request(self.model, prompt, schema_for(self.pres, row), attempt=attempt, salt=salt, messages=list(self.turns), tag=self.tag(round_no, "chat"))
        pres = self.coded if row.setup == "analyst" else self.pres
        return Request(
            self.model, base_prompt(pres, row, self.history, round_no, pools, hint_sentence(pres, row, pools)),
            schema_for(pres, row), attempt=attempt, salt=salt, tag=self.tag(round_no, row.setup),
        )

    def decider(self):
        return self.coded if self.row.setup == "analyst" else self.pres


def _prepare_orchestrated(items, round_no, pools, backend, store, attempts):
    need = [s for s in items if s.domain is None]
    if need:
        reqs = {id(s): Request(s.model, P.domain_prompt(s.pres), P.DOMAIN_SCHEMA, salt=f"domain|{s.row.id}", tag=s.tag(round_no, "domain")) for s in need}
        replies = execute(list(reqs.values()), backend, store)
        for s in need:
            r = replies.get(reqs[id(s)].key)
            try:
                s.domain = json.loads(r.text)["note"]
            except (AttributeError, json.JSONDecodeError, KeyError, TypeError):
                s.domain = "No comment available."
    pending = list(items)
    for attempt in range(attempts):
        if not pending:
            break
        reqs = {}
        for s in pending:
            analyst_row = Row(**{**asdict(s.row), "hint": "none", "strategy": "none", "rationale": True})
            reqs[id(s)] = Request(
                s.model, base_prompt(s.coded, analyst_row, s.history, round_no, pools, ""), schema_for(s.coded, analyst_row),
                attempt=attempt, salt=f"analyst|{s.row.id}", tag=s.tag(round_no, "analyst"),
            )
        replies = execute(list(reqs.values()), backend, store)
        still = []
        for s in pending:
            r = replies.get(reqs[id(s)].key)
            parsed = parse_reply(r, s.coded, s.row.q, False) if r else None
            if parsed is None:
                still.append(s)
                s.analyst = []
            else:
                s.analyst = parsed[0]
        pending = still


def run_search(rows: list[Row], worlds: dict[str, World], pools: dict, backend, store: Store, model_override: str | None = None, attempts: int = 3) -> list[dict]:
    states = [Search(r, worlds[r.world], model_override or r.model) for r in rows]
    for round_no in range(1, max(r.rounds for r in rows) + 1):
        current = [s for s in states if round_no <= s.row.rounds]
        orch = [s for s in current if s.row.setup == "orch"]
        if orch:
            _prepare_orchestrated(orch, round_no, pools, backend, store, attempts)
        pending, decisions = list(current), {}
        for attempt in range(attempts):
            if not pending:
                break
            requests = {id(s): s.request(round_no, attempt, pools) for s in pending}
            replies = execute(list(requests.values()), backend, store)
            still = []
            for s in pending:
                reply: Reply | None = replies.get(requests[id(s)].key)
                if reply is not None and reply.stop_reason == "refusal":
                    s.record["refusals"] += 1
                    continue
                parsed = parse_reply(reply, s.decider(), s.row.q, False) if reply else None
                if parsed is None:
                    s.record["retries"] += 1
                    still.append(s)
                else:
                    decisions[id(s)] = (parsed, requests[id(s)], reply)
            pending = still
        for s in current:
            if id(s) in decisions:
                (picks, reason, _), req, reply = decisions[id(s)]
                if s.row.setup == "chat":
                    s.turns = [*(req.messages or []), {"role": "user", "content": req.prompt}, {"role": "assistant", "content": reply.text}]
            else:
                rng = random.Random(_stable_int(s.row.id, round_no, "fill"))
                picks = [tuple(rng.randrange(n) for n in s.world.shape) for _ in range(s.row.q)]
                reason = ""
                s.record["random_fills"] += s.row.q
            s.record["reasons"].append(reason)
            for idx in picks:
                y = s.oracle.observe(idx)
                s.history.append((idx, y))
                s.record["picks"].append(list(idx))
                s.record["observed"].append(y)
                s.record["true"].append(float(s.world.y[idx]))
                s.record["round"].append(round_no)
    return [s.record for s in states]


# ---------- replay study

REPLAY_FACTORS = ("labels", "prompt", "hint", "strategy", "rationale", "feedback", "model")


def build_pairs(worlds: dict[str, World], n_pairs: int, base_rows: list[Row], seed: int = 1) -> list[dict]:
    """Each pair shares a world and a random history, and differs in exactly one factor."""
    rng = np.random.default_rng(seed)
    pairs = []
    order = rng.permutation(len(base_rows))
    for i in range(n_pairs):
        row = base_rows[int(order[i % len(base_rows)])]
        factor = REPLAY_FACTORS[i % len(REPLAY_FACTORS)]
        k = int(rng.choice([3, 6, 12])) if factor == "feedback" else int(rng.choice([0, 3, 6, 12]))
        pairs.append({"pair": i, "row": row.id, "factor": factor, "k": k, "world": row.world})
    return pairs


def _variant(row: Row, factor: str, world: World, pools: dict, rng) -> Row:
    d = asdict(row)
    if factor == "labels":
        alternatives = [x for x in ("real", "coded", "perm", "story") if x != row.effective_labels]
        if world.source == "surface":
            alternatives = [x for x in ("coded", "story") if x != row.effective_labels]
        d["effective_labels"] = alternatives[int(rng.integers(len(alternatives)))]
    elif factor == "prompt":
        d["prompt"] = (row.prompt + 1 + int(rng.integers(len(pools["base"]) - 1))) % len(pools["base"])
    elif factor == "hint":
        from .assign import HINT_CELLS
        d["hint"] = "none" if row.hint != "none" else HINT_CELLS[int(rng.integers(len(HINT_CELLS)))]
    elif factor == "strategy":
        from .assign import STRATEGY_KINDS
        d["strategy"] = "none" if row.strategy != "none" else STRATEGY_KINDS[int(rng.integers(3))]
    elif factor == "rationale":
        d["rationale"] = not row.rationale
    elif factor == "feedback":
        d["feedback"] = "scrambled" if row.feedback == "true" else "true"
    elif factor == "model":
        from .assign import MODELS
        alternatives = [m for m in MODELS if m != row.model]
        d["model"] = alternatives[int(rng.integers(2))]
    return Row(**d)


def random_history(world: World, row: Row, k: int) -> list:
    rng = np.random.default_rng(_stable_int(world.id, row.id, "history"))
    flat = rng.choice(world.size, size=min(k, world.size), replace=False)
    idxs = [tuple(int(v) for v in np.unravel_index(int(f), world.shape)) for f in flat]
    oracle = Oracle(world, row.id, "F0" if row.feedback == "true" else "Fs")
    return [(ix, oracle.observe(ix)) for ix in idxs]


def run_replay(pairs: list[dict], rows_by_id: dict[int, Row], worlds: dict[str, World], pools: dict, backend, store: Store, model_override: str | None = None) -> list[dict]:
    requests, meta = [], []
    for p in pairs:
        base = rows_by_id[p["row"]]
        base = Row(**{**asdict(base), "setup": "single", "q": 3})
        world = worlds[p["world"]]
        rng = np.random.default_rng(_stable_int("variant", p["pair"]))
        other = _variant(base, p["factor"], world, pools, rng)
        for side, row in (("a", base), ("b", other)):
            pres = view(world, row.effective_labels, row.id)
            history = random_history(world, row, p["k"])
            prompt = base_prompt(pres, row, history, 1 + p["k"] // 3, pools, hint_sentence(pres, row, pools))
            req = Request(model_override or row.model, prompt, schema_for(pres, row), salt=f"replay|{p['pair']}|{side}", tag={"study": "replay", "q": 3, **p, "side": side})
            requests.append(req)
            meta.append((p, side, pres, row))
    replies = execute(requests, backend, store)
    out = []
    for req, (p, side, pres, row) in zip(requests, meta):
        reply = replies.get(req.key)
        parsed = None
        if reply is not None and reply.stop_reason != "refusal":
            parsed = parse_reply(reply, pres, 3, False)
        out.append({**p, "side": side, "picks": [list(x) for x in parsed[0]] if parsed else None,
                    "labels": row.effective_labels, "hint": row.hint, "strategy": row.strategy, "rationale": row.rationale,
                    "feedback": row.feedback, "model": row.model, "prompt": row.prompt})
    return out


def save_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for x in items:
            fh.write(json.dumps(x) + "\n")
