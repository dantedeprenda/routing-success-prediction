"""Four-case outcome table per model pair: the value of escalating to the stronger model.

    uv run python scripts/escalation_cases.py    # -> results/evidence/escalation_cases.{md,json}

Descriptive, offline, $0: model outcomes only (no router output), on the LLMRouterBench test split.
For a default (cheap) and a fallback (strong) model, every test prompt falls in one of four cases:
both right, only the default right, only the fallback right, both wrong. Escalating *every* prompt
changes accuracy by P(only fallback right) - P(only default right): the fallback repairs some errors
and introduces others. A perfect selective policy escalates exactly the "only fallback right" prompts.
Neither model is treated as an oracle; correctness is the benchmark's own grading.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import extensions as ext
from generate_pair_questions import PAIRS
from second_pool import costs

REFERENCE = ("deepseek-v3-0324", "gpt-5-chat")


def main() -> int:
    s = ext.Setup(offline=True)
    te = s.test
    rows, payload = [], {}
    for cheap, strong in [*PAIRS, REFERENCE]:
        a, b = s.labels(cheap)[te], s.labels(strong)[te]
        both, only_a = float(np.mean((a == 1) & (b == 1))), float(np.mean((a == 1) & (b == 0)))
        only_b, neither = float(np.mean((a == 0) & (b == 1))), float(np.mean((a == 0) & (b == 0)))
        ratio = float(costs(s, strong)[te].mean() / costs(s, cheap)[te].mean())
        payload[f"{cheap} -> {strong}"] = {
            "both_right": both,
            "only_default_right": only_a,
            "only_fallback_right": only_b,
            "both_wrong": neither,
            "escalate_all_value": only_b - only_a,
            "perfect_selective_gain_over_default": only_b,
            "cost_ratio": ratio,
            "reference_only": (cheap, strong) == REFERENCE,
        }
        rows.append(
            f"| {cheap} → {strong} | {both:.3f} | {only_a:.3f} | {only_b:.3f} | {neither:.3f} | "
            f"{only_b - only_a:+.3f} | {ratio:.1f} |"
        )
    lines = [
        "# Four-case outcomes per model pair (descriptive; LLMRouterBench test split, 1,328 prompts)",
        "",
        "Default = cheap model, fallback = strong model. Escalating every prompt changes accuracy by",
        "P(only fallback right) - P(only default right). Cost ratio = strong / cheap mean cost.",
        "",
        (
            "| pair | both right | only default | only fallback | both wrong | escalate-all value | "
            "cost ratio |"
        ),
        "|---|---|---|---|---|---|---|",
        *rows,
    ]
    out = Path("results/evidence")
    (out / "escalation_cases.md").write_text("\n".join(lines) + "\n")
    (out / "escalation_cases.json").write_text(json.dumps(payload, indent=1))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
