"""LLM-as-router baseline: Omnigent's ``LLMRoutingClient`` prompt, near-verbatim.

Source: omnigent/server/smart_routing.py ``_JUDGE_SYSTEM_TEMPLATE`` (omnigent-ai/omnigent,
main @ e913aa993, unchanged at 16e9c00df). Differences: one pseudo-harness ("default")
holding the benchmark pool, and the harness-description block dropped because benchmark
models are not tied to coding harnesses. As in Omnigent: the message is cut to 4000
chars (head + latest step, as for the decision routers, via ``fit_text``), the reply is constrained to ``_VERDICT_SCHEMA`` with strict structured output (so the
judge cannot answer the task instead of routing it), and unknown models are clamped to the
cheapest. Omnigent's 9 s judge timeout (``ROUTING_REQUEST_TIMEOUT_S``) is applied at analysis
time from recorded latency, so calls are never cut short here.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dmroute.routers.base import Router, fit_text
from dmroute.schema import RouterOutput, step_p

TEMPLATE = """\
You are a model router for a coding assistant. Given the user's message,
pick the harness and model best suited for the task.

Available harnesses and their models:
{harness_menu}

Each harness list is ordered by provider-relative cost when the catalog reports
it, with provider catalog order as the tie-breaker. Classify the task using
stable model intents and choose the corresponding relative position:

  SIMPLE   → fast: first available model
             Examples: greetings, quick lookups, one-line fixes, trivial Q&A.

  MODERATE → balanced: middle available model
             Examples: single-file edits, debugging a known issue, brief explanations.

  COMPLEX  → powerful: last available model
             Examples: multi-file refactors, architecture decisions, security analysis,
             long reasoning chains, tasks requiring high accuracy or broad context.

The rationale field must follow this exact pattern so the explanation is consistent
with the model chosen:
  "This is a [SIMPLE/MODERATE/COMPLEX] task ([brief reason]); \
selected [cheapest/mid-range/most capable] model [model-id]."

Return **strict JSON only**:
{{"harness": "<harness-id>", "model": "<model-id>", "rationale": "<sentence>"}}
"""

MAX_MESSAGE_CHARS = 4000  # smart_routing.py: message[:4000]
ROUTING_TIMEOUT_S = 9.0  # smart_routing.py: ROUTING_REQUEST_TIMEOUT_S

# smart_routing.py _VERDICT_SCHEMA, sent as strict structured output.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "harness": {"type": "string"},
        "model": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["harness", "model", "rationale"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "routing_verdict", "strict": True, "schema": VERDICT_SCHEMA},
}


def system_prompt(model_names: list[str]) -> str:
    lines = "\n".join(f"    - {m}" for m in model_names)
    return TEMPLATE.format(harness_menu=f"  harness: default\n{lines}")


def parse_choice(text: str, model_names: list[str]) -> tuple[int, bool]:
    """Return (model index, parsed_ok). Unknown or unparseable -> cheapest, like Omnigent."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        model = json.loads(match.group(0))["model"] if match else None
    except (json.JSONDecodeError, KeyError, TypeError):
        model = None
    if model in model_names:
        return model_names.index(model), True
    return 0, False


class LLMJudge(Router):
    def __init__(
        self,
        backend,
        label: str = "llm_judge",
        workers: int = 8,
        model_ids: list[str] | None = None,
        structured: bool = True,
    ):
        """``model_ids``: the names the judge sees, in pool order (default: pool names).

        Omnigent's judge sees real catalog ids; pass them when a benchmark names its pool
        by tier (TwinRouterBench: low / mid / mid_high / high).
        """
        self.backend = backend
        self.label = label
        self.workers = workers
        self.model_ids = model_ids
        self.structured = structured

    @property
    def name(self):
        return self.label

    def predict(self, ds, idx):
        names = self.model_ids or ds.model_names
        assert len(names) == len(ds.model_names), "one model id per pool model"
        system = system_prompt(names)
        k = len(names)

        def one(i):
            # Same router-visible text as the decision routers (head + latest step), within
            # Omnigent's 4000-char budget; a bare prefix hides the step being routed.
            text = fit_text(ds.examples[i].text, MAX_MESSAGE_CHARS)
            try:
                fmt = RESPONSE_FORMAT if self.structured else None
                reply, lat, cost = self.backend.complete(system, text, response_format=fmt)
            except Exception as e:  # noqa: BLE001 - recorded and counted in metrics
                # A failed judge means no routing: the session keeps its default (strongest) model.
                err = f"{type(e).__name__}: {e}"
                return RouterOutput(p=step_p(k - 1, k), choice=k - 1, raw={"error": err})
            choice, ok = parse_choice(reply, names)
            return RouterOutput(
                p=step_p(choice, k),
                choice=choice,
                router_cost=cost,
                router_latency_s=lat,
                raw={"parsed": ok, "reply": reply[:500]},
            )

        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(one, np.asarray(idx)))
