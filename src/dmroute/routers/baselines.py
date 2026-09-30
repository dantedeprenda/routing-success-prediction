"""Reference routers every router paper is expected to beat (or bound by)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from dmroute.routers.base import Router
from dmroute.schema import Dataset, RouterOutput, step_p


class Oracle(Router):
    """Knows the outcomes. Upper bound only; never a target."""

    name = "oracle"

    def predict(self, ds, idx):
        q = ds.quality_matrix()[idx]
        out = []
        for row in q:
            best = row.max()
            choice = int(np.flatnonzero(row >= best)[0])  # cheapest model reaching the best score
            out.append(RouterOutput(p=row.tolist(), choice=choice, raw={"oracle": True}))
        return out


class Always(Router):
    def __init__(self, model: int | str):
        self.model = model

    @property
    def name(self):
        return f"always:{self.model}"

    def predict(self, ds, idx):
        m = self.model if isinstance(self.model, int) else ds.model_names.index(self.model)
        m = m % len(ds.pool)
        return [RouterOutput(p=step_p(m, len(ds.pool)), choice=m) for _ in idx]


class RandomScores(Router):
    """Uniform random P per model; its sweep traces a matched-cost random router."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def predict(self, ds, idx):
        rng = np.random.default_rng(self.seed)
        return [RouterOutput(p=rng.random(len(ds.pool)).tolist()) for _ in idx]


class CategoryBest(Router):
    """Per-category mean quality of each model on the labeled set (the 'dataset oracle' idea)."""

    name = "category_best"
    needs_labels = True

    def fit(self, ds, idx):
        q = ds.quality_matrix()
        self.global_ = q[idx].mean(axis=0)
        self.table_ = {}
        cats = np.array([ex.category for ex in ds.examples])
        for c in set(cats[idx]):
            rows = idx[cats[idx] == c]
            self.table_[c] = q[rows].mean(axis=0)

    def predict(self, ds, idx):
        return [
            RouterOutput(p=self.table_.get(ds.examples[i].category, self.global_).tolist())
            for i in idx
        ]


class KNN(Router):
    """kNN over TF-IDF: P(m) = mean quality of m on the k nearest labeled prompts."""

    needs_labels = True

    def __init__(self, k: int = 10):
        self.k = k

    @property
    def name(self):
        return f"knn{self.k}"

    def fit(self, ds, idx):
        self.vec_ = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
        x = self.vec_.fit_transform([ds.examples[i].text for i in idx])
        self.nn_ = NearestNeighbors(n_neighbors=min(self.k, len(idx)), metric="cosine").fit(x)
        self.q_ = ds.quality_matrix()[idx]

    def predict(self, ds, idx):
        x = self.vec_.transform([ds.examples[i].text for i in idx])
        _, nbrs = self.nn_.kneighbors(x)
        return [RouterOutput(p=self.q_[n].mean(axis=0).tolist()) for n in nbrs]


class _PerModelLogistic:
    """One logistic head per model; constant fallback when a model's labels are all one class."""

    def fit(self, x, q):
        self.heads_ = []
        for m in range(q.shape[1]):
            y = (q[:, m] >= 0.5).astype(int)
            if y.min() == y.max():
                self.heads_.append(float(y[0]))
            else:
                self.heads_.append(LogisticRegression(max_iter=2000, C=1.0).fit(x, y))
        return self

    def predict(self, x) -> np.ndarray:
        cols = [
            np.full(x.shape[0], h) if isinstance(h, float) else h.predict_proba(x)[:, 1]
            for h in self.heads_
        ]
        return np.column_stack(cols)


class TfidfLogistic(Router):
    """A standard trained router: TF-IDF features, per-model success classifier."""

    name = "tfidf_logreg"
    needs_labels = True

    def fit(self, ds, idx):
        self.vec_ = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
        x = self.vec_.fit_transform([ds.examples[i].text for i in idx])
        self.model_ = _PerModelLogistic().fit(x, ds.quality_matrix()[idx])

    def predict(self, ds, idx):
        x = self.vec_.transform([ds.examples[i].text for i in idx])
        return [RouterOutput(p=row.tolist()) for row in self.model_.predict(x)]


def build(kind: str, params: dict, ds: Dataset) -> Router:
    if kind == "oracle":
        return Oracle()
    if kind == "always":
        return Always(**params)
    if kind == "random":
        return RandomScores(**params)
    if kind == "category_best":
        return CategoryBest()
    if kind == "knn":
        return KNN(**params)
    if kind == "tfidf_logreg":
        return TfidfLogistic()
    raise ValueError(kind)
