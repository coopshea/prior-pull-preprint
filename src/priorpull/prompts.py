"""Plain-language prompts and response schemas."""
from __future__ import annotations

import re

from .worlds import Presentation

TEMPLATES = {
    "T0": (
        "{context} I can run {q} experiments per round, for {rounds} rounds in total. "
        "Each experiment uses one option for each setting below. I want the highest {objective} I can get.\n\n"
        "Settings:\n{settings}\n\n{history}\n\n"
        "This is round {round} of {rounds}. {hint}Which {q} experiments should I run next?"
    ),
    "T1": (
        "{context} My goal is to maximize {objective}. The budget is {rounds} rounds of {q} experiments each, "
        "and every experiment takes exactly one option per setting.\n\n"
        "Available settings:\n{settings}\n\n{history}\n\n"
        "We're at round {round}/{rounds}. {hint}What {q} experiments would you run next?"
    ),
    "T2": (
        "{context} Here is what I can vary, one option per setting in each experiment:\n{settings}\n\n"
        "I get {rounds} rounds with {q} experiments per round, and I care about getting the best possible {objective}.\n\n"
        "{history}\n\nRound {round} of {rounds} is next. {hint}Please suggest the next {q} experiments."
    ),
}

WORDINGS = ("W0", "W+", "W-", "Wpi", "Wexp")
TRANSPLANT_DOMAIN = {
    "shields": "espresso", "enzyme": "sourdough baking", "resin": "greenhouse tomatoes",
    "gmm": "cold brew coffee", "efb": "restaurant stock",
}
STRATEGIES = {
    "none": "",
    "M1": " Treat general knowledge about this kind of system as unverified. Base each choice on my results, and when results are thin, spread the experiments out instead of going with what usually works.",
    "M2": " Before choosing, work out which of your assumptions come from general knowledge rather than from my results, and don't rely on those.",
    "M3": " Published trends may not apply to my system, and the names here may not behave the way you'd expect.",
    "NR": "",
    "OPEN": "",
}
FORMAT_NOTE = " Keep the reason to two sentences or fewer, and give exactly {q} experiments."
MAX_SLOTS = 6
NO_REASON_NOTE = " Give exactly {q} experiments and nothing else."


def key_for(name: str) -> str:
    return re.sub(r"\W+", "_", name, flags=re.UNICODE).strip("_")


def render_history(pres: Presentation, history: list[tuple[tuple[int, ...], float]]) -> str:
    if not history:
        return "No results yet."
    head = " | ".join(pres.names + [pres.objective])
    rows = []
    for idx, y in history:
        cells = [pres.labels[f][i] for f, i in enumerate(idx)]
        rows.append(" | ".join(cells + [f"{y:g}"]))
    return "Results so far:\n" + head + "\n" + "\n".join(rows)


def hint_text(pres: Presentation, wording: str) -> str:
    if wording in ("W0", "Wexp"):
        return ""
    f, best, worst = pres.world.hint_target()
    level = best if wording == "W+" else worst
    phrase = f"{pres.names[f]} = {pres.labels[f][level]}"
    if wording == "Wpi":
        return f"My PI has a hunch that {phrase} is the way to go. "
    return f"I have a hunch that {phrase} is the way to go. "


def hinted_level(pres: Presentation, wording: str) -> tuple[int, int] | None:
    if wording in ("W0", "Wexp"):
        return None
    f, best, worst = pres.world.hint_target()
    return f, best if wording == "W+" else worst


def build_prompt(
    pres: Presentation,
    history,
    round_no: int,
    rounds: int,
    q: int,
    template: str,
    wording: str,
    calibrate: bool = False,
    strategy: str = "none",
) -> str:
    settings = "\n".join(
        f"- {name}: {', '.join(pres.options(f))}" for f, name in enumerate(pres.names)
    )
    text = TEMPLATES[template].format(
        context=pres.context,
        q=q,
        rounds=rounds,
        objective=pres.objective,
        settings=settings,
        history=render_history(pres, history),
        round=round_no,
        hint=hint_text(pres, wording),
    )
    text = text.replace("for 1 rounds", "for 1 round").replace("1 rounds", "1 round")
    text += STRATEGIES[strategy]
    text += NO_REASON_NOTE.format(q=q) if strategy == "NR" else FORMAT_NOTE.format(q=q)
    if calibrate:
        text += (
            " For each experiment also give your predicted result and your probability"
            " that it beats the best result so far."
        )
    return text


def pick_schema(pres: Presentation, q: int, calibrate: bool = False, reason: bool = True, use_array: bool = False) -> dict:
    """One required property per experiment, since the API cannot enforce an array length."""
    props: dict = {key_for(n): {"type": "string", "enum": pres.options(f)} for f, n in enumerate(pres.names)}
    if calibrate:
        props["predicted"] = {"type": "number"}
        props["p_beats_best"] = {"type": "number"}
    row = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    slots = [f"experiment_{i + 1}" for i in range(q)]
    experiments: dict = {
        "type": "object",
        "properties": {slot: row for slot in slots},
        "required": slots,
        "additionalProperties": False,
    }
    if use_array or q > MAX_SLOTS:  # many slots exceed the API's grammar limit, so long plans use an array checked on parsing
        experiments = {"type": "array", "items": row}
    top: dict = {"experiments": experiments}
    if reason:
        top = {"reason": {"type": "string"}, **top}
    return {"type": "object", "properties": top, "required": list(top), "additionalProperties": False}


DOMAIN_SCHEMA = {
    "type": "object",
    "properties": {"note": {"type": "string"}},
    "required": ["note"],
    "additionalProperties": False,
}


def domain_prompt(pres: Presentation) -> str:
    settings = "\n".join(f"- {name}: {', '.join(pres.options(f))}" for f, name in enumerate(pres.names))
    return (
        f"{pres.context} I want the highest {pres.objective} I can get.\n\nSettings:\n{settings}\n\n"
        "I haven't shown you any results. From general knowledge only, say in three sentences or fewer "
        "which options you'd expect to work best and why."
    )


def orchestrator_prompt(pres, analyst_picks, domain_note, round_no, rounds, q, wording, strategy_note="") -> str:
    settings = "\n".join(f"- {name}: {', '.join(pres.options(f))}" for f, name in enumerate(pres.names))
    proposals = "\n".join(
        "- " + ", ".join(f"{pres.names[f]}={pres.labels[f][i]}" for f, i in enumerate(ix)) for ix in analyst_picks
    ) or "- none available"
    return (
        f"{pres.context} I can run {q} experiments per round, for {rounds} rounds in total, and I want the highest "
        f"{pres.objective} I can get.\n\nSettings:\n{settings}\n\n"
        "Two assistants have reported to you. You haven't seen the results yourself.\n"
        f"The data analyst has seen every result so far, but with the settings coded so it doesn't know what they mean. It proposes:\n{proposals}\n\n"
        f"The domain expert knows the field but hasn't seen any results. It says: {domain_note}\n\n"
        f"This is round {round_no} of {rounds}. {hint_text(pres, wording)}Which {q} experiments should I run next?"
        + strategy_note + FORMAT_NOTE.format(q=q)
    )


def system_for(pres: Presentation, wording: str) -> str | None:
    if wording != "Wexp":
        return None
    domain = "process optimization" if pres.semantic == "S1" else pres.world.domain
    if pres.semantic == "S3":
        domain = TRANSPLANT_DOMAIN[pres.world.id]
    return f"You are a world-class expert in {domain}."
