"""Trained routers on sentence embeddings: the modern text baseline for few-shot routing.

TF-IDF only counts words; embedding routers are what current trained routers build on.
Each prompt is embedded once with a local open model (cut to ``max_chars`` with the same
head + latest-step rule as the decision routers), cached on disk, and fed to the same
per-model logistic head and kNN rule the TF-IDF baselines use, so only the representation
differs between them and the decision-model feature router.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import NearestNeighbors

from dmroute.routers.base import Router, fit_text
from dmroute.routers.baselines import _PerModelLogistic
from dmroute.schema import Dataset, RouterOutput

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


class EmbeddingStore:
    """Unit-norm embeddings for every example of a dataset, computed once and cached."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_chars: int = 2000,
        cache_dir: str | Path = ".cache/embeddings",
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.model = model
        self.max_chars = max_chars
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.batch_size = batch_size
        self._vectors: dict[str, np.ndarray] = {}

    def matrix(self, ds: Dataset, idx: np.ndarray) -> np.ndarray:
        if ds.name not in self._vectors:
            self._vectors[ds.name] = self._load_or_embed(ds)
        return self._vectors[ds.name][np.asarray(idx)]

    def _load_or_embed(self, ds: Dataset) -> np.ndarray:
        texts = [fit_text(ex.text, self.max_chars) for ex in ds.examples]
        digest = hashlib.sha256("\x00".join([self.model, *texts]).encode()).hexdigest()[:16]
        path = self.cache_dir / f"{self.model.replace('/', '__')}_{digest}.npy"
        if path.exists():
            return np.load(path)
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(self.model, device=self.device)
        vectors = encoder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors


class _TunedPerModelLogistic(_PerModelLogistic):
    """Per-model logistic heads with C chosen by stratified CV on the labeled set only.

    Guards against a handicapped baseline: with 768-1024 dims and 25 labels the default C=1
    may be far from the best regularization. Falls back to C=1 when a class has < 2 examples.
    """

    CS = (0.01, 0.1, 1.0, 10.0, 100.0)

    def fit(self, x, q):
        self.heads_ = []
        for m in range(q.shape[1]):
            y = (q[:, m] >= 0.5).astype(int)
            minority = int(min(y.sum(), len(y) - y.sum()))
            if minority == 0:
                self.heads_.append(float(y[0]))
            elif minority < 2:
                self.heads_.append(LogisticRegression(max_iter=2000, C=1.0).fit(x, y))
            else:
                self.heads_.append(
                    LogisticRegressionCV(
                        Cs=list(self.CS), cv=min(5, minority), scoring="roc_auc", max_iter=2000
                    ).fit(x, y)
                )
        return self


class EmbeddingLogistic(Router):
    """Per-model success classifier on embeddings (the TF-IDF logistic router, re-represented)."""

    needs_labels = True

    def __init__(self, store: EmbeddingStore, label: str = "emb_logreg", tune: bool = False):
        self.store = store
        self.label = label
        self.tune = tune

    @property
    def name(self):
        return self.label

    def fit(self, ds, idx):
        x = self.store.matrix(ds, idx)
        head = _TunedPerModelLogistic() if self.tune else _PerModelLogistic()
        self.model_ = head.fit(x, ds.quality_matrix()[idx])

    def predict(self, ds, idx):
        return [
            RouterOutput(p=row.tolist()) for row in self.model_.predict(self.store.matrix(ds, idx))
        ]


class EmbeddingKNN(Router):
    """kNN over embeddings: P(m) = mean quality of m on the k nearest labeled prompts."""

    needs_labels = True

    def __init__(self, store: EmbeddingStore, k: int = 10, label: str | None = None):
        self.store = store
        self.k = k
        self.label = label or f"emb_knn{k}"

    @property
    def name(self):
        return self.label

    def fit(self, ds, idx):
        x = self.store.matrix(ds, idx)
        self.nn_ = NearestNeighbors(n_neighbors=min(self.k, len(idx)), metric="cosine").fit(x)
        self.q_ = ds.quality_matrix()[idx]

    def predict(self, ds, idx):
        _, nbrs = self.nn_.kneighbors(self.store.matrix(ds, idx))
        return [RouterOutput(p=self.q_[n].mean(axis=0).tolist()) for n in nbrs]
