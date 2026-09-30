"""Grouped train/test splits and k-shot label budgets."""

from __future__ import annotations

import numpy as np

from dmroute.schema import Dataset


def grouped_split(ds: Dataset, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split by ``Example.group`` so one trajectory (or prompt) never lands on both sides."""
    groups = sorted({ex.group or ex.id for ex in ds.examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test = round(len(groups) * test_frac)
    test_groups = set(groups[:n_test])
    is_test = np.array([(ex.group or ex.id) in test_groups for ex in ds.examples])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def k_shot(train_idx: np.ndarray, k: int | None, seed: int) -> np.ndarray:
    """The labeled examples a router may learn from. ``None`` means the full train split."""
    if k is None or k >= len(train_idx):
        return train_idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(train_idx, k, replace=False))
