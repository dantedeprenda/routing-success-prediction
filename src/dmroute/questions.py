"""Versioned System One question specs for routing.

All formulations go in one fan-out request per example, so a single call scores every
formulation. Iterate on the dev split only; bump ``VERSION`` and freeze before any
test-split call (the cache key includes the question text, so old answers stay valid
for old versions).
"""

from __future__ import annotations

from dmroute.schema import ModelSpec

VERSION = "v1"  # FROZEN 2026-09-25 before any test-split call. Do not edit; add v2.

# v1 (2026-09-25, dev split only): questions judge the *next assistant response*, not the
# whole task. In agent trajectories the router decides one step; v0's "how hard is this
# task" rated long multi-part requests as hard even when the next step was a few plain
# tool calls.
SUBJECT = "the next assistant response in this conversation"

RUNGS = [
    (
        "Trivial: any small model writes it correctly (a greeting, a direct lookup, or a plainly "
        "specified tool call)."
    ),
    (
        "Routine: a standard step that a capable small model handles reliably, even when the "
        "request has several simple parts."
    ),
    (
        "Moderate: needs multi-step reasoning, careful tracking of earlier results, or domain "
        "knowledge; small models often slip."
    ),
    (
        "Hard: long reasoning, tricky code changes, recovering from errors, or specialist knowledge; "
        "only strong models reliably get it right."
    ),
    "Very hard: research-level or very long-horizon work that only the best models manage.",
]
LADDERS = {2: [0, 3], 3: [0, 2, 3], 4: [0, 1, 2, 3], 5: [0, 1, 2, 3, 4]}

FEATURES = {
    "multistep": f"Does writing {SUBJECT} correctly require a multi-step chain of reasoning, "
    "beyond recall or a single transformation?",
    "math": f"Does {SUBJECT} require non-trivial mathematical calculation or proof?",
    "code": f"Does {SUBJECT} require writing, modifying, or debugging code?",
    "expert": f"Does {SUBJECT} require specialist, expert-level knowledge (for example "
    "graduate-level science, law, or medicine)?",
    "obscure_fact": f"Does a correct {SUBJECT} depend on recalling an obscure or long-tail fact?",
    "long_context": f"Does {SUBJECT} depend on details spread across a long conversation or "
    "document?",
    "state_tracking": f"Does {SUBJECT} depend on exactly tracking results of earlier tool calls "
    "or turns (values, file contents, prior outputs)?",
    "error_recovery": f"Must {SUBJECT} recover from an earlier error, failed test, or wrong "
    "assumption?",
    "strict_format": f"Does {SUBJECT} demand an exact output format or precise constraints that "
    "are easy to violate?",
    "agent_action": f"Is {SUBJECT} a tool call or other action inside an agentic workflow?",
    "routine_tool_use": f"Is {SUBJECT} one or more straightforward tool calls whose arguments "
    "are stated plainly in the request?",
    "ambiguous": f"Is the request behind {SUBJECT} ambiguous or open-ended, so a good answer "
    "requires judgment?",
    "trivial": f"Could almost any language model write {SUBJECT} correctly?",
}

FORMULATIONS = ("choice", "tier_nouls", "difficulty", "features")


def fingerprint(pool: list[ModelSpec]) -> str:
    """Short hash of the exact question map sent for this pool (recorded with results)."""
    import hashlib
    import json

    blob = json.dumps(build_questions(pool), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def ladder(k: int) -> list[str]:
    return [RUNGS[i] for i in LADDERS[k]]


def tier_label(i: int, pool: list[ModelSpec]) -> str:
    return f"tier {i + 1} of {len(pool)}"


def build_questions(pool: list[ModelSpec], formulations: tuple[str, ...] = FORMULATIONS) -> dict:
    """Question map for one request. Ids encode the formulation; they are not sent to the model."""
    k = len(pool)
    tiers = {tier_label(i, pool): m.description or None for i, m in enumerate(pool)}
    q: dict[str, dict] = {}
    if "choice" in formulations:
        q["choice"] = {
            "type": "choice",
            "instructions": "Models are listed from cheapest and weakest to most expensive and "
            f"capable. Which is the cheapest model that would still write {SUBJECT} fully and "
            "correctly?",
            "criteria": tiers,
        }
    if "tier_nouls" in formulations:
        for i, (label, desc) in enumerate(tiers.items()):
            q[f"tier_{i}"] = {
                "type": "noul",
                "instructions": {
                    "model": f"{label} (1 is cheapest and weakest). {desc or ''}".strip(),
                    "question": f"Would `model` write {SUBJECT} fully and correctly?",
                },
            }
    if "difficulty" in formulations:
        q["difficulty"] = {
            "type": "score",
            "instructions": f"How difficult is it for a language model to write {SUBJECT} fully "
            "and correctly?",
            "criteria": ladder(k),
        }
    if "features" in formulations:
        for name, text in FEATURES.items():
            q[f"feat_{name}"] = {"type": "noul", "instructions": text}
    return q
