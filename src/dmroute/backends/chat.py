"""OpenAI-compatible chat backend for the LLM-router baseline, plus a stub."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

import httpx

from dmroute.backends.cache import CallCache

_log = logging.getLogger(__name__)


class ChatBackend:
    def __init__(
        self,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        price_in: float = 0.0,
        price_out: float = 0.0,
        timeout_s: float = 60.0,
        max_retries: int = 8,
        rpm: float | None = None,
        cache: CallCache | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.price_in = price_in
        self.price_out = price_out
        self.max_retries = max_retries
        self.cache = cache or CallCache()
        self._client = httpx.Client(timeout=timeout_s)
        # Requests-per-minute cap shared by all threads using this backend. OpenRouter limits
        # new accounts to 20/min per model and a 429 that exhausts retries would be scored
        # as a routing error, so the cap is enforced client-side rather than left to retries.
        self._interval = 60.0 / rpm if rpm else 0.0
        self._next_slot = 0.0
        self._slot_lock = threading.Lock()

    def _wait_turn(self) -> None:
        if not self._interval:
            return
        with self._slot_lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + self._interval
        time.sleep(start - now)

    @property
    def backend_id(self) -> str:
        return f"chat:{self.model}"

    def complete(
        self, system: str, user: str, response_format: dict | None = None
    ) -> tuple[str, float, float]:
        """Return (text, latency_s, cost_usd). ``response_format``: OpenAI-style json_schema."""
        request = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if response_format is not None:  # absent key keeps older cache entries addressable
            request["response_format"] = response_format
        cached = self.cache.get(self.backend_id, request)
        if cached is None:
            response, latency = self._post(request)
            self.cache.put(self.backend_id, request, response, latency)
        else:
            response, latency = cached
        usage = response.get("usage", {})
        cost = (
            usage.get("prompt_tokens", 0) * self.price_in
            + usage.get("completion_tokens", 0) * self.price_out
        ) / 1e6
        return response["choices"][0]["message"]["content"], latency, cost

    def _post(self, request: dict) -> tuple[dict, float]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        headers = {"Authorization": f"Bearer {key}"}
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            start = time.perf_counter()
            resp = self._client.post(
                f"{self.base_url}/chat/completions", json=request, headers=headers
            )
            latency = time.perf_counter() - start
            if (resp.status_code == 429 or resp.status_code >= 500) and attempt < self.max_retries:
                wait = max(_retry_after(resp), min(2**attempt, 60))
                _log.warning(
                    "%s: HTTP %s, retry %d in %.0fs",
                    self.model,
                    resp.status_code,
                    attempt + 1,
                    wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json(), latency
        raise RuntimeError("unreachable")


def _retry_after(resp: httpx.Response) -> float:
    """Seconds the server asked us to wait (Retry-After header), 0 when absent or unparseable."""
    try:
        return max(float(resp.headers.get("retry-after", 0)), 0.0)
    except ValueError:
        return 0.0


class AnthropicDirect(ChatBackend):
    """The same judge through Anthropic's Messages API instead of OpenRouter.

    Shares ChatBackend's cache key (backend id and request dict), so calls made through either
    provider form one pool. The request is translated: system messages go to ``system``,
    ``temperature`` is dropped (Sonnet 5 rejects sampling parameters, and OpenRouter drops it
    too), and ``thinking`` is left unset, which runs adaptive thinking as OpenRouter's default
    did. Replies are stored in the OpenAI shape ``complete`` reads, tagged with the provider.
    """

    def __init__(
        self,
        model: str,
        anthropic_model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 16000,
        **kwargs: Any,
    ):
        super().__init__(model, api_key_env=api_key_env, **kwargs)
        self.anthropic_model = anthropic_model
        self.max_tokens = max_tokens
        self._anthropic = None

    def _post(self, request: dict) -> tuple[dict, float]:
        import anthropic

        if self._anthropic is None:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} is not set")
            # Retries are handled below so latency covers only the successful attempt.
            self._anthropic = anthropic.Anthropic(api_key=key, max_retries=0)
        system = "\n\n".join(m["content"] for m in request["messages"] if m["role"] == "system")
        messages = [m for m in request["messages"] if m["role"] != "system"]
        extra = {}
        if request.get("response_format", {}).get("type") == "json_schema":
            schema = request["response_format"]["json_schema"]["schema"]
            extra["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            start = time.perf_counter()
            try:
                msg = self._anthropic.messages.create(
                    model=self.anthropic_model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=messages,
                    **extra,
                )
            except anthropic.RateLimitError as e:
                if attempt == self.max_retries:
                    raise
                wait = max(_retry_after(e.response), min(2**attempt, 60))
                _log.warning("%s: 429, retry %d in %.0fs", self.anthropic_model, attempt + 1, wait)
                time.sleep(wait)
                continue
            except (anthropic.InternalServerError, anthropic.APIConnectionError) as e:
                if attempt == self.max_retries:
                    raise
                wait = min(2**attempt, 60)
                _log.warning(
                    "%s: %s, retry %d in %.0fs",
                    self.anthropic_model,
                    type(e).__name__,
                    attempt + 1,
                    wait,
                )
                time.sleep(wait)
                continue
            latency = time.perf_counter() - start
            text = "".join(b.text for b in msg.content if b.type == "text")
            response = {
                "id": msg.id,
                "provider": "anthropic-direct",
                "model": msg.model,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": msg.stop_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": msg.usage.input_tokens,
                    "completion_tokens": msg.usage.output_tokens,
                },
            }
            return response, latency
        raise RuntimeError("unreachable")


class StubChat:
    """Picks a tier from message length and a few keywords. For pipeline tests only."""

    backend_id = "chat:stub"

    def __init__(self, **_: Any):
        pass

    def complete(self, system: str, user: str, **_: Any) -> tuple[str, float, float]:
        models = re.findall(r"^\s+- (\S+)$", system, re.MULTILINE)
        hard = len(re.findall(r"prove|refactor|derive|multi-step|security", user, re.IGNORECASE))
        level = min(hard + (len(user) > 300), 2)
        pick = models[[0, len(models) // 2, len(models) - 1][level]]
        return f'{{"harness": "default", "model": "{pick}", "rationale": "stub"}}', 0.0, 0.0


def make_chat(spec: dict[str, Any], cache: CallCache) -> ChatBackend | StubChat:
    spec = dict(spec)
    kind = spec.pop("kind", "http")
    if kind == "stub":
        return StubChat(**spec)
    if kind == "anthropic":
        return AnthropicDirect(cache=cache, **spec)
    return ChatBackend(cache=cache, **spec)
