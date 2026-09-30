"""TwinRouterBench static track (arXiv 2605.18859, Apache-2.0).

``question_bank.jsonl`` from https://huggingface.co/datasets/Amorph/TwinRouterBench.
Each row is one agent step with the router-visible ``messages`` prefix and the cheapest
sufficient tier (``target_tier_id``: 0 low, 1 mid, 2 mid_high, 3 high), verified by
downgrade search. Rows carry no per-model outcomes, so tier m "succeeds" iff
m >= target. Cost is an approximation (input chars / 4, fixed output tokens) at the
benchmark's locked OpenRouter prices; the tier metrics (RowPass/RowExact/TrajPass)
do not depend on it.

Note: 71% of rows are ``low``, and SWE-bench rows are flagged as weak labels upstream.
"""

from __future__ import annotations

import json
from pathlib import Path

from dmroute.schema import Dataset, Example, ModelSpec

# data/dynamic/tier_to_model.json + model_pricing.json (fetched 2026-04-23) in the upstream repo.
POOL = [
    ModelSpec("low", "DeepSeek-V3.2: cheap open model for routine steps.", 0.252, 0.378),
    ModelSpec("mid", "MiniMax-M2.7: mid-tier agentic model.", 0.30, 1.20),
    ModelSpec("mid_high", "Gemini-3-Flash: strong fast model.", 0.50, 3.00),
    ModelSpec("high", "Claude Opus 4.6: frontier model for the hardest steps.", 5.00, 25.0),
]


def render_messages(messages: list[dict]) -> str:
    """Flatten an OpenAI-style message list into plain text for a text-only router."""
    lines = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        text = content or ""
        if msg.get("tool_calls"):
            text += "\n[tool_calls] " + json.dumps(msg["tool_calls"])[:500]
        lines.append(f"{msg.get('role', '?')}: {text}")
    return "\n\n".join(lines)


def load_twinrouterbench(
    path: str,
    output_tokens: int = 400,
    benchmarks: list[str] | None = None,
    name: str = "twinrouterbench",
) -> Dataset:
    rows = [json.loads(line) for line in Path(path).expanduser().read_text().splitlines() if line]
    if benchmarks:
        rows = [r for r in rows if r["benchmark"] in set(benchmarks)]
    examples = []
    for r in rows:
        messages = r["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        text = render_messages(messages)
        target = int(r["target_tier_id"])
        in_tok = len(text) / 4
        examples.append(
            Example(
                id=r["id"],
                dataset=r["benchmark"],
                text=text,
                quality={m.name: float(i >= target) for i, m in enumerate(POOL)},
                cost={
                    m.name: (in_tok * m.price_in + output_tokens * m.price_out) / 1e6 for m in POOL
                },
                category=r["benchmark"],
                group=f"{r['benchmark']}:{r['instance_id']}",
                meta={
                    "target_tier_id": target,
                    "instance_id": r["instance_id"],
                    "step_index": r.get("step_index"),
                    "total_steps": r.get("total_steps"),
                },
            )
        )
    return Dataset(name, POOL, examples)
