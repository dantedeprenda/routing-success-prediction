"""Generate the label-blind question sets Q-blind-1..3 (preregistration/v2.md, Part 1).

The prompt is the pre-registered one, verbatim. Each sample is one independent call with default
sampling settings; the raw reply, the returned model id and usage are saved next to the parsed
questions, and each set is checked against the pre-registered format rules. Existing sets are
never overwritten: rerunning only creates missing ones (and, once, the regeneration of a set
that failed its format check).

    uv run python scripts/generate_questions.py            # needs ANTHROPIC_API_KEY
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from dmroute.cli import _load_dotenv

OUT = Path("configs/questions")  # committed: frozen inputs, not data
MODEL = "claude-sonnet-5"
PROMPT = (
    "A router must decide, before any language model answers, whether a given language model will "
    "answer a user request fully and correctly. It sees only the request: a conversation, possibly "
    "including an agent's earlier steps. The requests come from a mix of competition mathematics, "
    "programming problems, graduate-level science questions, general multiple-choice knowledge "
    "questions and software-engineering issues.\n\n"
    "Write exactly 13 yes/no questions about the request that a careful reader could answer from the "
    "request text alone, and whose answers would help predict whether a language model succeeds. Do "
    "not mention any specific model, company or benchmark. Return a JSON object mapping a short "
    "snake_case id to each question."
)
# Names a question must not contain (models, companies, benchmarks in this study).
BANNED = re.compile(
    r"\b(gpt|claude|gemini|deepseek|qwen|kimi|glm|intern|openai|anthropic|google|llama|mistral|"
    r"aime|livemathbench|livecodebench|gpqa|mmlu|swe-?bench|llmrouterbench)\b",
    re.IGNORECASE,
)


def parse(text: str) -> dict[str, str]:
    """The JSON object in the reply (tolerating a code fence around it)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in reply")
    return {str(k): str(v) for k, v in json.loads(match.group(0)).items()}


def problems(questions: dict[str, str]) -> list[str]:
    """Pre-registered format check; an empty list means the set passes."""
    out = []
    if len(questions) != 13:
        out.append(f"{len(questions)} questions, not 13")
    for qid, q in questions.items():
        if not q.rstrip().endswith("?"):
            out.append(f"{qid}: not phrased as a question")
        if BANNED.search(q):
            out.append(f"{qid}: names a model, company or benchmark")
        if len(q.split()) > 50:
            out.append(f"{qid}: over 50 words")
    return out


def generate(client) -> dict:
    msg = client.messages.create(
        model=MODEL, max_tokens=4000, messages=[{"role": "user", "content": PROMPT}]
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "requested_model": MODEL,
        "returned_model": msg.model,
        "usage": {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens},
        "prompt": PROMPT,
        "reply": text,
    }
    try:
        record["questions"] = parse(text)
        record["problems"] = problems(record["questions"])
    except ValueError as e:
        record["questions"], record["problems"] = {}, [str(e)]
    return record


def main() -> int:
    import anthropic

    _load_dotenv()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    OUT.mkdir(parents=True, exist_ok=True)
    for i in (1, 2, 3):
        path = OUT / f"qblind_{i}.json"
        if not path.exists():
            path.write_text(json.dumps(generate(client), indent=1))
        record = json.loads(path.read_text())
        retry = OUT / f"qblind_{i}_regenerated.json"
        if record["problems"] and not retry.exists():  # regenerate once, keep both files
            retry.write_text(json.dumps(generate(client), indent=1))
        final = json.loads(retry.read_text()) if retry.exists() else record
        status = "; ".join(final["problems"]) or "passes"
        print(
            f"Q-blind-{i}: {final['returned_model']}, {len(final['questions'])} questions, {status}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
