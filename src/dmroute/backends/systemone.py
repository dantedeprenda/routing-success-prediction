"""System One decision-model backends speaking ``POST /v1/systemone``.

Jev (TypeSafe) and Laya (``laya-serve``) share the request/response shape:
``{state, model, questions}`` -> ``{model, answers, usage}``. The stub backend answers
from crude text heuristics so the pipeline runs without keys; its numbers mean nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from typing import Any

import httpx

from dmroute.backends.cache import CallCache


class SystemOneBackend:
    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        price_per_m_input: float = 0.0,
        timeout_s: float = 30.0,
        max_retries: int = 4,
        cache: CallCache | None = None,
    ):
        self.name = name
        self.base_url = os.path.expandvars(base_url).rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.price_per_m_input = price_per_m_input
        self.max_retries = max_retries
        self.cache = cache or CallCache()
        self._client = httpx.Client(timeout=timeout_s)

    @property
    def backend_id(self) -> str:
        return f"systemone:{self.name}:{self.model}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def ask(self, state: Any, questions: dict[str, dict]) -> tuple[dict, float, float]:
        """Return (response, latency_s, cost_usd). Latency is the successful attempt's."""
        request = {"state": state, "model": self.model, "questions": questions}
        cached = self.cache.get(self.backend_id, request)
        if cached is None:
            response, latency = self._post(request)
            self.cache.put(self.backend_id, request, response, latency)
        else:
            response, latency = cached
        tokens = response.get("usage", {}).get("input_tokens", 0)
        return response, latency, tokens * self.price_per_m_input / 1e6

    def _post(self, request: dict) -> tuple[dict, float]:
        url = f"{self.base_url}/v1/systemone"
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            resp = self._client.post(url, json=request, headers=self._headers())
            latency = time.perf_counter() - start
            retryable = resp.status_code in (429, 529) or resp.status_code >= 500
            if retryable and attempt < self.max_retries:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return resp.json(), latency
        raise RuntimeError("unreachable")


_HARD = re.compile(
    r"prove|derive|multi-step|refactor|optimi[sz]e|edge case|concurren|formal|"
    r"architecture|security|traceback|debug",
    re.IGNORECASE,
)
_EASY = re.compile(r"what is|define|quick|hello|list|in one word|thanks", re.IGNORECASE)


class StubSystemOne:
    """Deterministic fake decision model. Reads question ids by convention
    (``choice``, ``difficulty``, ``tier_<i>``, ``feat_<name>``)."""

    name = "stub"
    model = "stub-0"
    backend_id = "systemone:stub"

    def __init__(self, noise: float = 0.15, **_: Any):
        self.noise = noise

    def _difficulty(self, text: str) -> float:
        hard = len(_HARD.findall(text))
        easy = len(_EASY.findall(text))
        length = min(len(text) / 400, 1.0)
        raw = 0.25 * hard - 0.2 * easy + 0.5 * length
        jitter = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF - 0.5
        return min(max(0.2 + raw + self.noise * jitter, 0.0), 1.0)

    def ask(self, state: Any, questions: dict[str, dict]) -> tuple[dict, float, float]:
        text = state if isinstance(state, str) else str(state)
        d = self._difficulty(text)
        n_tiers = max(sum(qid.startswith("tier_") for qid in questions), 1)
        answers: dict[str, dict] = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                if qid.startswith("tier_"):
                    cap = (int(qid.split("_")[1]) + 0.5) / n_tiers
                    p = 1 / (1 + math.exp(-(cap - d) * 8))
                else:
                    salt = int(hashlib.sha256((qid + text).encode()).hexdigest()[:6], 16)
                    p = min(max(d + (salt / 0xFFFFFF - 0.5) * 0.6, 0.01), 0.99)
                    if qid == "feat_trivial":
                        p = 1 - p
                answers[qid] = {"type": "noul", "noul": p}
            else:
                options = (
                    list(q["criteria"])
                    if q["type"] == "choice"
                    else list(range(len(q["criteria"])))
                )
                n = len(options)
                logits = [-(((i + 0.5) / n - d) ** 2) * 40 for i in range(n)]
                z = sum(math.exp(v) for v in logits)
                probs = [math.exp(v) / z for v in logits]
                best = max(range(n), key=probs.__getitem__)
                conf = (n * probs[best] - 1) / (n - 1)
                if q["type"] == "choice":
                    answers[qid] = {
                        "type": "choice",
                        "choice": options[best],
                        "probabilities": dict(zip(options, probs)),
                        "confidence": conf,
                    }
                else:
                    answers[qid] = {
                        "type": "score",
                        "score": sum(i * p for i, p in enumerate(probs)),
                        "probabilities": {str(i): p for i, p in enumerate(probs)},
                        "confidence": conf,
                    }
        usage = {"input_tokens": len(text) // 4, "output_tokens": 0}
        return {"model": self.model, "answers": answers, "usage": usage}, 0.0, 0.0


class LayaLocal:
    """Laya (convaiinnovations/laya, Apache-2.0) run in-process via ``laya.Router``.

    Same question map and answer shape as Jev. Two documented adaptations:
    - ``noul_as_choice``: Laya's Noul can follow its false/true labels instead of the state
      (laya issue #156); the vendor's workaround asks a two-option Choice with neutral keys.
      We map P(yes) back into a Noul answer so downstream code is unchanged.
    - ``checkpoint`` / ``max_len``: the English checkpoint reads ~320 state tokens; the
      multilingual one reads up to 8k with ``max_len=8192``.
    Probabilities ship over-confident; recalibration (policy.Recalibrator) handles that.
    """

    price_per_m_input = 0.0

    def __init__(
        self,
        checkpoint: str = "english",
        max_len: int | None = None,
        device: str | None = None,
        noul_as_choice: bool = True,
        batch_size: int = 1,  # >1 only on CUDA: on an M2 (MPS) batching was 7x slower
        cache: CallCache | None = None,
        name: str = "laya",
    ):
        os.environ.setdefault("USE_TF", "0")  # transformers' TF probe can deadlock laya.load
        from laya import Router

        self.checkpoint = checkpoint
        self.max_len = max_len
        self.noul_as_choice = noul_as_choice
        self.batch_size = batch_size
        self.cache = cache or CallCache()
        self.name = name
        self.model = f"laya-{checkpoint}"
        self._router = Router(device=device)
        self._lock = __import__("threading").Lock()  # one forward pass at a time

    @property
    def backend_id(self) -> str:
        return f"laya:{self.checkpoint}:{self.max_len}:{'nc' if self.noul_as_choice else 'noul'}"

    def _convert(self, questions: dict[str, dict]) -> dict[str, dict]:
        if not self.noul_as_choice:
            return questions
        out = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                out[qid] = {
                    "type": "choice",
                    "instructions": q["instructions"],
                    "criteria": {"A": "yes", "B": "no"},
                }
            else:
                out[qid] = q
        return out

    def _response(self, result: dict, questions: dict[str, dict], routed: str) -> dict:
        answers = {}
        for qid, q in questions.items():
            a = result["answers"][qid]
            if q["type"] == "noul" and self.noul_as_choice:
                answers[qid] = {"type": "noul", "noul": float(a["probabilities"]["A"])}
            else:
                answers[qid] = a
        response = {
            "model": f"{self.model}@{routed}",
            "answers": answers,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return json.loads(json.dumps(response, default=float))  # numpy -> plain JSON

    def ask(self, state: Any, questions: dict[str, dict]) -> tuple[dict, float, float]:
        request = {"state": state, "questions": questions}
        cached = self.cache.get(self.backend_id, request)
        if cached is not None:
            return cached[0], cached[1], 0.0
        with self._lock:
            start = time.perf_counter()
            result = self._router.predict(
                state, self._convert(questions), model=self.checkpoint, max_len=self.max_len
            )
            latency = time.perf_counter() - start
        routed = result.get("routing", {}).get("model", self.checkpoint)
        response = self._response(result, questions, routed)
        self.cache.put(self.backend_id, request, response, latency)
        return response, latency, 0.0

    def ask_many(
        self, states: list[Any], questions: dict[str, dict]
    ) -> list[tuple[dict, float, float]]:
        """Batched ``ask``: uncached states share forward passes via ``Agent.predict_batch``.

        Latency stored per state is the batch time divided by the batch size (throughput, not
        single-request latency; measure that separately with ``ask``).
        """
        out: list[tuple[dict, float, float] | None] = [None] * len(states)
        misses = []
        for i, state in enumerate(states):
            cached = self.cache.get(self.backend_id, {"state": state, "questions": questions})
            if cached is None:
                misses.append(i)
            else:
                out[i] = (cached[0], cached[1], 0.0)
        if misses:
            agent = self._router.load(self.checkpoint)
            with self._lock:
                start = time.perf_counter()
                results = agent.predict_batch(
                    [states[i] for i in misses],
                    self._convert(questions),
                    batch_size=self.batch_size,
                    max_len=self.max_len,
                    sort_by_length=len(misses) > self.batch_size,
                )
                per = (time.perf_counter() - start) / len(misses)
            for i, result in zip(misses, results):
                response = self._response(result, questions, self.checkpoint)
                self.cache.put(
                    self.backend_id, {"state": states[i], "questions": questions}, response, per
                )
                out[i] = (response, per, 0.0)
        return out


def make_backend(spec: dict[str, Any], cache: CallCache) -> SystemOneBackend | StubSystemOne:
    spec = dict(spec)
    kind = spec.pop("kind", "http")
    if kind == "stub":
        return StubSystemOne(**spec)
    if kind == "laya_local":
        return LayaLocal(cache=cache, **spec)
    return SystemOneBackend(cache=cache, **spec)
