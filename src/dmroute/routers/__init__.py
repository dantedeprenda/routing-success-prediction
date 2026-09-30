"""Router registry: build any router from an experiment-config entry."""

from __future__ import annotations

from typing import Any

from dmroute.backends.cache import CallCache
from dmroute.backends.chat import make_chat
from dmroute.backends.systemone import make_backend
from dmroute.routers import baselines
from dmroute.routers.base import Router
from dmroute.routers.decision import DecisionModelRouter
from dmroute.routers.llm_judge import LLMJudge
from dmroute.schema import Dataset


def build_routers(specs: list[dict[str, Any]], ds: Dataset, cache: CallCache) -> list[Router]:
    """Expand config entries into routers. A ``decision`` entry yields one router per formulation."""
    routers: list[Router] = []
    for spec in specs:
        spec = dict(spec)
        kind = spec.pop("type")
        if kind == "decision":
            backend = make_backend(spec.pop("backend"), cache)
            label = spec.pop("label")
            for formulation in spec.pop("formulations"):
                routers.append(DecisionModelRouter(backend, formulation, label, **spec))
        elif kind == "llm_judge":
            routers.append(LLMJudge(make_chat(spec.pop("backend"), cache), **spec))
        elif kind == "embedding":
            from dmroute.routers.embedding import EmbeddingKNN, EmbeddingLogistic, EmbeddingStore

            store = EmbeddingStore(**spec.pop("store", {}))
            label = spec.pop("label", "emb")
            tune = spec.pop("tune", False)
            routers.append(EmbeddingLogistic(store, label=f"{label}_logreg", tune=tune))
            routers.append(EmbeddingKNN(store, k=spec.pop("k", 10), label=f"{label}_knn"))
        elif kind == "always_each":
            routers.extend(baselines.Always(m) for m in range(len(ds.pool)))
        else:
            routers.append(baselines.build(kind, spec, ds))
    return routers
