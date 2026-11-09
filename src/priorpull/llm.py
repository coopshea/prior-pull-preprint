"""Model backends, the append-only response store, and spend tracking."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
API_BASE = "https://api.anthropic.com"  # explicit, so a host-level ANTHROPIC_BASE_URL never receives the project key
PRICES = {  # USD per million tokens, first-party rates, cached 2026-10-06
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "mock": (0.0, 0.0),
    "mock-a": (0.0, 0.0),
    "mock-b": (0.0, 0.0),
}
DEFAULT_EFFORT = {"claude-opus-5-5": "medium", "claude-sonnet-5-5": "high", "claude-haiku-5-5": "medium"}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Request:
    model: str
    prompt: str
    schema: dict | None = None
    system: str | None = None
    attempt: int = 0
    salt: str = ""  # keeps repeated identical prompts (different seeds) as separate samples
    max_tokens: int = 16000
    tag: dict = field(default_factory=dict)
    messages: list | None = None  # earlier turns of a chat, oldest first

    @property
    def key(self) -> str:
        parts = [self.model, self.prompt, self.schema, self.system, self.attempt, self.salt]
        if self.messages:
            parts.append(self.messages)
        blob = json.dumps(parts, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:40]

    def params(self) -> dict:
        p: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "adaptive"},
            "messages": [*(self.messages or []), {"role": "user", "content": self.prompt}],
        }
        config: dict = {"effort": DEFAULT_EFFORT[self.model]}
        if self.schema is not None:
            config["format"] = {"type": "json_schema", "schema": self.schema}
        p["output_config"] = config
        if self.system:
            p["system"] = self.system
        return p


@dataclass
class Reply:
    key: str
    text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    model: str
    batch: bool = False
    error: str | None = None
    refusal_category: str | None = None

    def cost(self) -> float:
        inp, out = PRICES.get(self.model, (0.0, 0.0))
        c = (self.input_tokens * inp + self.output_tokens * out) / 1e6
        return c * 0.5 if self.batch else c


class Store:
    """Append-only JSONL of every reply. Resuming a run replays it at no cost."""

    def __init__(self, run_dir: Path, cap_usd: float):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "responses.jsonl"
        self.cap = cap_usd
        self.replies: dict[str, Reply] = {}
        self.spent = 0.0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                rec = json.loads(line)
                reply = Reply(**rec["reply"])
                if reply.error is None:
                    self.replies[reply.key] = reply
                    self.spent += reply.cost()

    def add(self, request: Request, reply: Reply) -> None:
        with self.path.open("a") as fh:
            rec = {
                "ts": time.time(),
                "tag": request.tag,
                "prompt": request.prompt,
                "system": request.system,
                "reply": reply.__dict__,
            }
            fh.write(json.dumps(rec) + "\n")
        if reply.error is None:
            self.replies[reply.key] = reply
            self.spent += reply.cost()

    def check(self) -> None:
        if self.spent >= self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.2f} of ${self.cap:.2f} cap")


def projected_cost(requests: list[Request], batch: bool) -> float:
    """An upper-side estimate made before sending: about 3 characters per token, and 1,000 output tokens per reply."""
    total = 0.0
    for r in requests:
        inp, out = PRICES.get(r.model, (0.0, 0.0))
        chars = len(r.prompt) + len(json.dumps(r.schema or {})) + len(json.dumps(r.messages or []))
        total += (chars / 3 * inp + 1000 * out) / 1e6
    return total * (0.5 if batch else 1.0)


def execute(requests: list[Request], backend, store: Store) -> dict[str, Reply]:
    """Fetch replies for requests, reusing anything already stored. Refuses to send past the cap."""
    store.check()
    todo: dict[str, Request] = {}
    for r in requests:
        if r.key not in store.replies:
            todo[r.key] = r
    if todo:
        projected = projected_cost(list(todo.values()), isinstance(backend, BatchBackend))
        if store.spent + projected > store.cap:
            raise BudgetExceeded(f"would reach about ${store.spent + projected:.2f} against a ${store.cap:.2f} cap, sent nothing")
        for request, reply in backend.run(list(todo.values())):
            store.add(request, reply)
        store.check()
    return {r.key: store.replies[r.key] for r in requests if r.key in store.replies}


def make_client_kwargs() -> dict:
    load_dotenv(ROOT / ".env", override=True)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(f"Add ANTHROPIC_API_KEY to {ROOT / '.env'}")
    return {"api_key": key, "base_url": API_BASE}


class OnlineBackend:
    def __init__(self, concurrency: int = 8):
        self.concurrency = concurrency

    def run(self, requests: list[Request]):
        return asyncio.run(self._run(requests))

    async def _run(self, requests: list[Request]):
        client = anthropic.AsyncAnthropic(max_retries=5, **make_client_kwargs())
        gate = asyncio.Semaphore(self.concurrency)

        async def one(r: Request):
            async with gate:
                try:
                    m = await client.messages.create(**r.params())
                except anthropic.BadRequestError:
                    raise  # a malformed request is a bug, so stop the run
                except anthropic.AuthenticationError:
                    raise
                except anthropic.APIError as exc:
                    return r, Reply(r.key, "", "error", 0, 0, r.model, error=f"{type(exc).__name__}: {exc}")
            return r, _reply_from_message(r.key, m, batch=False)

        out = await asyncio.gather(*(one(r) for r in requests))
        await client.close()
        return out


class BatchBackend:
    def __init__(self, poll_seconds: int = 30):
        self.poll_seconds = poll_seconds

    def run(self, requests: list[Request]):
        client = anthropic.Anthropic(max_retries=5, **make_client_kwargs())
        by_id = {r.key: r for r in requests}
        batch = client.messages.batches.create(
            requests=[{"custom_id": r.key, "params": r.params()} for r in requests]
        )
        while True:
            state = client.messages.batches.retrieve(batch.id)
            if state.processing_status == "ended":
                break
            time.sleep(self.poll_seconds)
        out = []
        for item in client.messages.batches.results(batch.id):
            r = by_id[item.custom_id]
            if item.result.type == "succeeded":
                out.append((r, _reply_from_message(r.key, item.result.message, batch=True)))
            else:
                detail = getattr(getattr(item.result, "error", None), "type", item.result.type)
                out.append((r, Reply(r.key, "", "error", 0, 0, r.model, batch=True, error=str(detail))))
        return out


def _reply_from_message(key: str, m, batch: bool) -> Reply:
    text = next((b.text for b in m.content if b.type == "text"), "")
    category = None
    if m.stop_reason == "refusal" and getattr(m, "stop_details", None):
        category = getattr(m.stop_details, "category", None)
    return Reply(
        key, text, m.stop_reason, m.usage.input_tokens, m.usage.output_tokens, m.model, batch=batch,
        refusal_category=category,
    )


class MockBackend:
    """Offline stand-in: picks random valid options, and sometimes returns junk to exercise retries."""

    def __init__(self, junk_rate: float = 0.05, refusal_rate: float = 0.0):
        self.junk_rate = junk_rate
        self.refusal_rate = refusal_rate

    def run(self, requests: list[Request]):
        out = []
        for r in requests:
            rng = random.Random(r.key)
            if rng.random() < self.refusal_rate:
                out.append((r, Reply(r.key, "", "refusal", 10, 0, "mock")))
                continue
            if rng.random() < self.junk_rate:
                out.append((r, Reply(r.key, "not json", "end_turn", 10, 5, "mock")))
                continue
            out.append((r, Reply(r.key, json.dumps(self._answer(r.schema, rng, r.tag.get("q", 3))), "end_turn", 100, 50, "mock")))
        return out

    def _answer(self, schema: dict | None, rng: random.Random, q: int):
        if schema is None:
            return {"text": "unknown"}
        return self._fill(schema, rng, q)

    def _fill(self, s: dict, rng: random.Random, q: int):
        if "enum" in s:
            return rng.choice(s["enum"])
        t = s.get("type")
        if t == "object":
            return {k: self._fill(v, rng, q) for k, v in s["properties"].items()}
        if t == "array":
            return [self._fill(s["items"], rng, q) for _ in range(q)]
        if t == "number":
            return round(rng.random(), 3)
        if t == "integer":
            return rng.randint(0, 5)
        if t == "boolean":
            return rng.random() < 0.5
        return "mock text"
