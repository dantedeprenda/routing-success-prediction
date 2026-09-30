"""Offline policies: turn cached router outputs into model picks.

Threshold policy: pick the cheapest model whose P(suffices) >= tau, else the strongest.
Sweeping tau traces a router's cost-quality curve without any new calls. Recalibration
and the confidence-gated cascade are also applied here, on cached outputs.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from dmroute.schema import RouterOutput


def p_matrix(outputs: list[RouterOutput]) -> np.ndarray:
    return np.array([o.p for o in outputs], dtype=float)


def pick(p: np.ndarray, tau: float) -> np.ndarray:
    """Vectorized threshold policy over an (n, k) P matrix."""
    ok = p >= tau
    first = ok.argmax(axis=1)
    return np.where(ok.any(axis=1), first, p.shape[1] - 1)


def tau_grid(p: np.ndarray, max_points: int = 201) -> np.ndarray:
    """Thresholds hitting every distinct operating point (subsampled), plus both extremes."""
    vals = np.unique(p)
    if len(vals) > max_points:
        vals = np.quantile(vals, np.linspace(0, 1, max_points))
    return np.unique(np.concatenate([[-np.inf], vals, [np.inf]]))


def router_costs(outputs: list[RouterOutput]) -> np.ndarray:
    return np.array([o.router_cost for o in outputs], dtype=float)


class Recalibrator:
    """Per-model Platt or isotonic map from raw P to calibrated P, fit on labeled outputs."""

    def __init__(self, method: str = "platt"):
        self.method = method

    def fit(self, p: np.ndarray, q: np.ndarray) -> Recalibrator:
        y = (q >= 0.5).astype(int)
        self.maps_ = []
        for m in range(p.shape[1]):
            if y[:, m].min() == y[:, m].max():
                self.maps_.append(float(y[0, m]))
            elif self.method == "isotonic":
                self.maps_.append(IsotonicRegression(out_of_bounds="clip").fit(p[:, m], y[:, m]))
            else:
                x = _logit(p[:, m])[:, None]
                self.maps_.append(LogisticRegression().fit(x, y[:, m]))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        cols = []
        for m, f in enumerate(self.maps_):
            if isinstance(f, float):
                cols.append(np.full(len(p), f))
            elif isinstance(f, IsotonicRegression):
                cols.append(f.predict(p[:, m]))
            else:
                cols.append(f.predict_proba(_logit(p[:, m])[:, None])[:, 1])
        return np.maximum.accumulate(np.column_stack(cols), axis=1)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def cascade(
    primary: list[RouterOutput], fallback: list[RouterOutput], conf_threshold: float
) -> list[RouterOutput]:
    """Use ``primary`` unless its confidence < threshold, then defer to ``fallback``.

    The primary call is always paid; the fallback only when used. Latencies add on
    fallback because the second call waits on the first.
    """
    out = []
    for a, b in zip(primary, fallback):
        conf = a.confidence if a.confidence is not None else 1.0
        if conf >= conf_threshold:
            out.append(a)
        else:
            out.append(
                RouterOutput(
                    p=b.p,
                    choice=b.choice,
                    confidence=conf,
                    router_cost=a.router_cost + b.router_cost,
                    router_latency_s=a.router_latency_s + b.router_latency_s,
                    raw={"deferred": True},
                )
            )
    return out
