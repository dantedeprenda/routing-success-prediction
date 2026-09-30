"""Generate pair-aware question sets (preregistration/v3.md, H3).

One sample of the pre-registered prompt per model pair, with the two model names filled in. As in
scripts/generate_questions.py, the raw reply, returned model id and usage are saved, sets are never
overwritten, and a set that fails the format check is regenerated once (both files kept). Model names
are allowed in these questions; benchmark names are not.

    uv run python scripts/generate_pair_questions.py       # needs ANTHROPIC_API_KEY
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from dmroute.cli import _load_dotenv
from generate_questions import MODEL, parse

OUT = Path("configs/questions/pairs")
PAIRS = [  # pre-registered order
    ("deepseek-v3-0324", "gemini-2.5-flash"),
    ("deepseek-v3-0324", "glm-4.6"),
    ("kimi-k2-0905", "claude-sonnet-4"),
    ("deepseek-v3-0324", "claude-sonnet-4"),
    ("qwen3-235b-a22b-2507", "claude-sonnet-4"),
    ("deepseek-v3.1-terminus", "gpt-5"),
]
PROMPT = (
    "Two language models, {A} and {B}, will each attempt the same user request: a conversation, "
    "possibly including an agent's earlier steps. The requests come from a mix of competition "
    "mathematics, programming problems, graduate-level science questions, general multiple-choice "
    "knowledge questions and software-engineering issues. Both models often succeed or fail together; "
    "what matters here is the requests where exactly one of them is right.\n\n"
    "Write exactly 13 yes/no questions about the request that a careful reader could answer from the "
    "request text alone, and whose answers would help predict which of the two models answers "
    "correctly when they disagree. Do not mention any benchmark. Return a JSON object mapping a short "
    "snake_case id to each question."
)
BENCHMARKS = re.compile(
    r"\b(aime|livemathbench|livecodebench|gpqa|mmlu|swe-?bench|llmrouterbench)\b", re.IGNORECASE
)


def problems(questions: dict[str, str]) -> list[str]:
    out = [] if len(questions) == 13 else [f"{len(questions)} questions, not 13"]
    for qid, q in questions.items():
        if not q.rstrip().endswith("?"):
            out.append(f"{qid}: not phrased as a question")
        if BENCHMARKS.search(q):
            out.append(f"{qid}: names a benchmark")
        if len(q.split()) > 50:
            out.append(f"{qid}: over 50 words")
    return out


def generate(client, cheap: str, strong: str) -> dict:
    prompt = PROMPT.format(A=cheap, B=strong)
    msg = client.messages.create(
        model=MODEL, max_tokens=4000, messages=[{"role": "user", "content": prompt}]
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "pair": [cheap, strong],
        "requested_model": MODEL,
        "returned_model": msg.model,
        "usage": {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens},
        "prompt": prompt,
        "reply": text,
    }
    try:
        record["questions"] = parse(text)
        record["problems"] = problems(record["questions"])
    except ValueError as e:
        record["questions"], record["problems"] = {}, [str(e)]
    return record


def path_for(cheap: str, strong: str, suffix: str = "") -> Path:
    return OUT / f"{cheap}__{strong}{suffix}.json"


def final_set(cheap: str, strong: str) -> dict:
    retry = path_for(cheap, strong, "_regenerated")
    return json.loads((retry if retry.exists() else path_for(cheap, strong)).read_text())


def main() -> int:
    import anthropic

    _load_dotenv()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    OUT.mkdir(parents=True, exist_ok=True)
    for cheap, strong in PAIRS:
        path = path_for(cheap, strong)
        if not path.exists():
            path.write_text(json.dumps(generate(client, cheap, strong), indent=1))
        record = json.loads(path.read_text())
        retry = path_for(cheap, strong, "_regenerated")
        if record["problems"] and not retry.exists():
            retry.write_text(json.dumps(generate(client, cheap, strong), indent=1))
        final = final_set(cheap, strong)
        status = "; ".join(final["problems"]) or "passes"
        print(f"{cheap} -> {strong}: {len(final['questions'])} questions, {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
