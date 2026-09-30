"""Embedding routers: same heads as the TF-IDF baselines, fed by a cached vector store."""

import numpy as np

from dmroute.data.synthetic import load_synthetic
from dmroute.routers.embedding import EmbeddingKNN, EmbeddingLogistic, EmbeddingStore


def _store_with(ds, vectors: np.ndarray) -> EmbeddingStore:
    store = EmbeddingStore(model="stub")
    store._vectors[ds.name] = vectors  # skip the encoder: tests must not download a model
    return store


def test_embedding_routers_learn_a_separable_signal():
    ds = load_synthetic(n=200, seed=1)
    q = ds.quality_matrix()
    cheap_ok = (q[:, 0] >= 0.5).astype(float)
    rng = np.random.default_rng(0)
    # One informative direction (the cheap model's outcome) plus noise.
    vectors = np.column_stack([cheap_ok * 2 - 1, rng.normal(size=(len(q), 7))])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    store = _store_with(ds, vectors)
    train, test = np.arange(0, 150), np.arange(150, 200)
    for router in (EmbeddingLogistic(store), EmbeddingKNN(store, k=5)):
        router.fit(ds, train)
        out = router.predict(ds, test)
        assert len(out) == len(test) and len(out[0].p) == len(ds.pool)
        p0 = np.array([o.p[0] for o in out])
        # The informative direction must separate the cheap model's successes from failures.
        assert p0[cheap_ok[test] == 1].mean() > p0[cheap_ok[test] == 0].mean() + 0.2


def test_store_caches_vectors_per_dataset(tmp_path, monkeypatch):
    ds = load_synthetic(n=20, seed=2)
    calls = []
    store = EmbeddingStore(model="stub", cache_dir=tmp_path)

    def fake_embed(d):
        calls.append(d.name)
        return np.eye(len(d.examples), dtype=np.float32)

    monkeypatch.setattr(store, "_load_or_embed", fake_embed)
    a = store.matrix(ds, np.array([0, 1]))
    b = store.matrix(ds, np.array([2]))
    assert calls == [ds.name]
    assert a.shape == (2, 20) and b[0, 2] == 1.0


def test_tuned_head_selects_c_on_labels_only():
    ds = load_synthetic(n=120, seed=3)
    q = ds.quality_matrix()
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(len(q), 16))
    vectors[:, 0] = (q[:, 0] >= 0.5) * 2 - 1 + rng.normal(scale=0.3, size=len(q))
    store = _store_with(ds, vectors)
    router = EmbeddingLogistic(store, tune=True)
    router.fit(ds, np.arange(60))
    out = router.predict(ds, np.arange(60, 120))
    assert len(out) == 60
    heads = [h for h in router.model_.heads_ if not isinstance(h, float)]
    assert heads and all(hasattr(h, "C_") for h in heads if type(h).__name__.endswith("CV"))
