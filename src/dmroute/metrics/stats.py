"""Paired significance for router comparisons on the same examples."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, rankdata


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """Mean of (a - b) with a 95% percentile CI, resampling examples with replacement."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)]}


def _auroc_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise AUROC via the Mann-Whitney rank formula (ties get average ranks)."""
    ranks = rankdata(p, axis=1)
    pos = y.sum(axis=1)
    neg = y.shape[1] - pos
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((ranks * y).sum(axis=1) - pos * (pos + 1) / 2) / (pos * neg)


def bootstrap_auroc(p: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """AUROC with a 95% percentile CI; resamples lacking both classes are skipped."""
    p = np.asarray(p, float)
    y = (np.asarray(y) >= 0.5).astype(float)
    counts = {"n": len(y), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    if y.all() or not y.any():
        return {"auroc": float("nan"), "ci95": [float("nan"), float("nan")], **counts}
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n, len(y)))
    vals = _auroc_rows(p[idx], y[idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    point = float(_auroc_rows(p[None, :], y[None, :])[0])
    return {"auroc": point, "ci95": [float(lo), float(hi)], **counts}


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact McNemar test on discordant pairs of binary outcomes."""
    a = np.asarray(a_correct) >= 0.5
    b = np.asarray(b_correct) >= 0.5
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "p_value": float(p)}
