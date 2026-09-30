"""Common data format shared by every benchmark, router and metric.

A benchmark becomes a :class:`Dataset`: an ordered model pool (cheapest -> strongest)
and a list of :class:`Example`, each carrying the router-visible text plus the realized
quality and cost of every model in the pool. Step-level agentic data (TwinRouterBench)
fits the same shape by deriving per-tier outcomes from the cheapest-sufficient-tier label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str = ""
    price_in: float = 0.0  # USD per 1M input tokens
    price_out: float = 0.0  # USD per 1M output tokens


@dataclass
class Example:
    id: str
    dataset: str
    text: str
    quality: dict[str, float]  # model name -> score in [0, 1]
    cost: dict[str, float]  # model name -> USD for answering this example
    category: str = ""
    group: str = ""  # examples sharing a group never straddle a split (e.g. one trajectory)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    pool: list[ModelSpec]
    examples: list[Example]

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.pool]

    def quality_matrix(self) -> np.ndarray:
        """(n_examples, n_models) realized quality."""
        names = self.model_names
        return np.array([[ex.quality[m] for m in names] for ex in self.examples], dtype=float)

    def cost_matrix(self) -> np.ndarray:
        """(n_examples, n_models) realized cost in USD."""
        names = self.model_names
        return np.array([[ex.cost[m] for m in names] for ex in self.examples], dtype=float)

    def subset(self, idx: list[int] | np.ndarray, name: str | None = None) -> Dataset:
        return Dataset(name or self.name, self.pool, [self.examples[i] for i in idx])


@dataclass
class RouterOutput:
    """What a router says about one example.

    ``p[m]`` is the router's estimate that model ``m`` (pool order) is good enough.
    Discrete routers (always-X, LLM judge) set ``choice`` and leave ``p`` as a step
    function. Policies in :mod:`dmroute.policy` turn ``p`` into a pick offline, so a
    threshold sweep never needs new API calls.
    """

    p: list[float]
    choice: int | None = None
    confidence: float | None = None
    router_cost: float = 0.0  # USD spent by the router itself
    router_latency_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


def step_p(choice: int, n_models: int) -> list[float]:
    """P vector for a router that only names a model: models >= choice suffice."""
    return [1.0 if m >= choice else 0.0 for m in range(n_models)]
