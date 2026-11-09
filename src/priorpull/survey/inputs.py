"""Frozen pools of varied prompts, hints, and asking strategies, generated once by a model."""
from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path

from ..llm import Request, Store, execute

ROOT = Path(__file__).resolve().parents[3]
POOLS = ROOT / "data" / "pools" / "pools.json"
SLOTS = {"context", "objective", "q", "rounds", "settings", "history", "round", "hint"}
STRATEGIES = {
    "evidence": "Tell the assistant to treat general knowledge about this kind of system as unverified, and to base its choices on the user's own results.",
    "audit": "Ask the assistant to separate which of its assumptions come from general knowledge rather than from the user's results, and not to rely on those.",
    "caution": "Warn the assistant that published trends may not apply to this system and that names may not behave as expected.",
}
SOURCES = {"self": "the person asking", "pi": "the person's PI or boss", "paper": "a paper the person has read"}
CERTAINTY = {"tentative": "tentative", "confident": "confident"}


def _fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}


def _schema(key: str) -> dict:
    return {
        "type": "object",
        "properties": {key: {"type": "array", "items": {"type": "string"}}},
        "required": [key],
        "additionalProperties": False,
    }


def generate(model: str, backend, store: Store) -> dict:
    """Ask the model for each pool, in one round of requests."""
    requests: dict[str, Request] = {
        "base": Request(
            model,
            "Write 40 different ways a person might ask an AI assistant which experiments to run next. "
            "Each is a template string containing each of these placeholders exactly once: "
            "{context} {objective} {q} {rounds} {settings} {history} {round} {hint}. "
            "{context} is the person describing their situation, {settings} is a list of options, {history} is a table of past results or a note that none exist, "
            "{objective} is a noun phrase naming the quantity to maximize, such as 'the output score', so the sentence around it must make it the thing being maximized. "
            "Write in the first person, as the person asking. {hint} is an optional sentence that may be empty. {q} is experiments per round, {rounds} is total rounds, {round} is the current round. "
            "Vary tone (casual, terse, formal, anxious, blunt), length, assumed expertise (student to expert), and structure. "
            "Mention no scientific field, material, or number. Give no advice on how to reason.",
            _schema("templates"),
            salt="pool-base-2",
            tag={"pool": "base"},
        )
    }
    for name, meaning in STRATEGIES.items():
        requests[f"strategy_{name}"] = Request(
            model,
            f"Write 5 differently worded sentences or short passages a person could add to a request to an AI assistant. Each one should do this: {meaning} "
            "Vary tone and length. Mention no scientific field, material, or number.",
            _schema("wordings"),
            salt=f"pool-{name}",
            tag={"pool": name},
        )
    for source, who in SOURCES.items():
        for certainty in CERTAINTY:
            requests[f"hint_{source}_{certainty}"] = Request(
                model,
                f"Write 3 differently worded sentences stating that {who} believes a setting choice is the way to go, with a {certainty} level of certainty. "
                "Write in the first person as the person asking, for example 'I' and 'my'. Each sentence contains the placeholder {phrase} exactly once, standing for a setting and its value. Mention no scientific field, material, or number.",
                _schema("wordings"),
                salt=f"pool-hint-2-{source}-{certainty}",
                tag={"pool": "hint"},
            )
    replies = execute(list(requests.values()), backend, store)
    pools: dict = {"strategy": {}, "hint": {}}
    for key, req in requests.items():
        data = json.loads(replies[req.key].text)
        if key == "base":
            pools["base"] = [t for t in data["templates"] if _fields(t) == SLOTS]
        elif key.startswith("strategy_"):
            pools["strategy"][key.split("_", 1)[1]] = data["wordings"]
        else:
            pools["hint"][key.split("_", 1)[1]] = [t for t in data["wordings"] if _fields(t) == {"phrase"}]
    return pools


def save(pools: dict) -> str:
    POOLS.parent.mkdir(parents=True, exist_ok=True)
    POOLS.write_text(json.dumps(pools, indent=1, ensure_ascii=False, sort_keys=True))
    return digest()


def digest() -> str:
    return hashlib.sha256(POOLS.read_bytes()).hexdigest()


def load() -> dict:
    return json.loads(POOLS.read_text())
