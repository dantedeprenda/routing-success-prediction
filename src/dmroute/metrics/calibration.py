"""Calibration and discrimination of P(model suffices).

Binned ECE is biased and bin-dependent (Roelofs et al., arXiv 2012.08668), so the
default is equal-mass bins; always report the estimator and bin count with the number.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _bins(p: np.ndarray, n_bins: int, scheme: str) -> list[np.ndarray]:
    if scheme == "equal_mass":
        order = np.argsort(p, kind="stable")
        return [b for b in np.array_split(order, n_bins) if len(b)]
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    return [np.flatnonzero(ids == b) for b in range(n_bins) if (ids == b).any()]


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15, scheme: str = "equal_mass") -> float:
    bins = _bins(p, n_bins, scheme)
    return float(sum(len(b) * abs(p[b].mean() - y[b].mean()) for b in bins) / len(p))


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10, scheme: str = "equal_mass"):
    """Rows of (mean predicted, observed frequency, count) for a reliability diagram."""
    return [(float(p[b].mean()), float(y[b].mean()), len(b)) for b in _bins(p, n_bins, scheme)]


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve: keep the most confident first, average the risk."""
    order = np.argsort(-conf, kind="stable")
    errors = 1 - correct[order]
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(risk.mean())


MIN_CLASS = 10  # fewer positives or negatives than this and a per-category AUROC is noise


def by_category(p: np.ndarray, q: np.ndarray, cats: list[str], model: int = 0) -> dict:
    """Discrimination of P(model suffices) inside each category.

    Pooled AUROC can come entirely from telling categories apart (a per-category mean
    already does that). ``within`` is the n-weighted mean of per-category AUROCs, counting
    only categories with at least ``MIN_CLASS`` examples of each outcome.
    """
    from dmroute.metrics.stats import bootstrap_auroc

    y = (q[:, model] >= 0.5).astype(float)
    cats_arr = np.array(cats)
    per = {}
    for cat in sorted(set(cats)):
        m = cats_arr == cat
        per[cat] = {**bootstrap_auroc(p[m, model], y[m]), "base_rate": float(y[m].mean())}
    scored = [
        (v["auroc"], v["n"]) for v in per.values() if min(v["n_pos"], v["n_neg"]) >= MIN_CLASS
    ]
    within = (
        float(sum(a * n for a, n in scored) / sum(n for _, n in scored)) if scored else float("nan")
    )
    return {"model": model, "pooled": bootstrap_auroc(p[:, model], y), "within": within, "per": per}


def calibration_report(p: np.ndarray, q: np.ndarray) -> dict:
    """Per-model and pooled calibration of an (n, k) P matrix against realized quality."""
    y = (q >= 0.5).astype(float)
    per_model = []
    for m in range(p.shape[1]):
        per_model.append(
            {
                "brier": brier(p[:, m], y[:, m]),
                "ece_mass15": ece(p[:, m], y[:, m]),
                "auroc": auroc(p[:, m], y[:, m]),
                "base_rate": float(y[:, m].mean()),
            }
        )
    flat_p, flat_y = p.ravel(), y.ravel()
    return {
        "per_model": per_model,
        "pooled": {
            "brier": brier(flat_p, flat_y),
            "ece_mass15": ece(flat_p, flat_y),
            "ece_width15": ece(flat_p, flat_y, scheme="equal_width"),
            "auroc": auroc(flat_p, flat_y),
        },
        # Selective prediction for "keep it on the cheapest model".
        "aurc_cheapest": aurc(p[:, 0], y[:, 0]),
        "reliability_cheapest": reliability(p[:, 0], y[:, 0]),
    }
