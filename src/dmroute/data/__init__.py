"""Benchmark loaders. Each returns a :class:`dmroute.schema.Dataset`."""

from __future__ import annotations

from typing import Any

from dmroute.schema import Dataset


def load(spec: dict[str, Any]) -> Dataset:
    kind = spec["loader"]
    params = {k: v for k, v in spec.items() if k != "loader"}
    if kind == "synthetic":
        from dmroute.data.synthetic import load_synthetic

        return load_synthetic(**params)
    if kind == "llmrouterbench":
        from dmroute.data.llmrouterbench import load_llmrouterbench

        return load_llmrouterbench(**params)
    if kind == "twinrouterbench":
        from dmroute.data.twinrouterbench import load_twinrouterbench

        return load_twinrouterbench(**params)
    raise ValueError(f"unknown loader {kind!r}")
