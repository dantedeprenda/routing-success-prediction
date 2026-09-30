"""Pre-registered experiments T, D and E2/E3 (preregistration/v1.md). No API calls.

    uv run python scripts/extensions.py reproduce   # sanity: original target matches metrics.json
    uv run python scripts/extensions.py T           # success prediction for 9 unseen models
    uv run python scripts/extensions.py D           # few-shot adaptation to held-out domains
    uv run python scripts/extensions.py E           # 13-dim PCA embeddings, question subsets

Every method is re-implemented here on precomputed representations (cached Jev answers, cached
embeddings, TF-IDF refit per labeled set) so a new target only swaps the label vector. The
`reproduce` step checks the re-implementation against the main pipeline's saved results for the
original target before any new target is scored. Writes results/evidence/extensions_<exp>.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from dmroute.backends.cache import CallCache
from dmroute.data import load
from dmroute.data.splits import k_shot
from dmroute.questions import FEATURES
from dmroute.routers import build_routers
from dmroute.routers.baselines import _PerModelLogistic
from dmroute.routers.embedding import EmbeddingStore, _TunedPerModelLogistic
from evidence import SEEDS, splits, stratified_draws, within_rows

CONFIG = "configs/llmrouterbench_test.yaml"
KS = [25, 50, 100, 200, None]
B = 1000
TARGETS = [  # pre-registered order (preregistration/v1.md)
    "qwen3-235b-a22b-2507",
    "deepseek-v3-0324",
    "kimi-k2-0905",
    "intern-s1",
    "gemini-2.5-flash",
    "gpt-5-chat",
    "glm-4.6",
    "claude-sonnet-4",
    "gemini-2.5-pro",
]
BASELINES = ["tfidf_logreg", "knn10", "bge_logreg", "bge_knn", "bgeL_logreg", "bgeL_knn"]


class Setup:
    """Dataset, split, and every representation, built once."""

    def __init__(self, config: str = CONFIG, jev: str = "jev", offline: bool = False):
        self.cfg = yaml.safe_load(Path(config).read_text())
        self.ds = load(self.cfg["dataset"])
        self.train, self.test = splits(self.ds, self.cfg["split"])
        self.texts = [ex.text for ex in self.ds.examples]
        self.cats = np.array([ex.category for ex in self.ds.examples])
        self.domains = np.array([ex.dataset for ex in self.ds.examples])
        spec = next(s for s in self.cfg["routers"] if s["type"] == "decision" and s["label"] == jev)
        if offline:  # cache keys ignore base_url, so an unreachable URL forbids any new call
            spec = {**spec, "backend": {**spec["backend"], "base_url": "http://127.0.0.1:9"}}
        (router,) = build_routers(
            [{**spec, "formulations": ["features"]}], self.ds, CallCache(".cache/calls.sqlite")
        )
        answers = router._ask_all(self.ds, np.arange(len(self.ds.examples)))
        self.jev = np.array([router._features(a[0]["answers"]) for a in answers])
        self.emb = {}
        for label, model in (("bge", "BAAI/bge-base-en-v1.5"), ("bgeL", "BAAI/bge-large-en-v1.5")):
            self.emb[label] = EmbeddingStore(model=model, max_chars=2000).matrix(
                self.ds, np.arange(len(self.ds.examples))
            )

    def labels(self, model: str) -> np.ndarray:
        """1 where *model* answered correctly, from the release's latest run files."""
        root = Path(self.cfg["dataset"]["root"])
        scores: dict[str, float] = {}
        for dset, split in {tuple(ex.id.split("/")[:2]) for ex in self.ds.examples}:
            f = max((root / dset / split / model).glob("*.json"))
            for r in json.loads(f.read_text())["records"]:
                scores[f"{dset}/{split}/{r['index']}"] = float(r["score"])
        return np.array([scores[ex.id] >= 0.5 for ex in self.ds.examples], dtype=float)

    def mean_cost(self, model: str) -> float:
        root = Path(self.cfg["dataset"]["root"])
        ids = {ex.id for ex in self.ds.examples}
        costs = []
        for dset, split in {tuple(i.split("/")[:2]) for i in ids}:
            f = max((root / dset / split / model).glob("*.json"))
            costs += [
                float(r.get("cost") or 0.0)
                for r in json.loads(f.read_text())["records"]
                if f"{dset}/{split}/{r['index']}" in ids
            ]
        return float(np.mean(costs))


def predict(method: str, s: Setup, y: np.ndarray, lab: np.ndarray, target: np.ndarray, x=None):
    """P(success) on *target* after training *method* on labeled indices *lab*."""
    q = y[:, None]
    if method == "jev_features" or method == "custom":
        x = s.jev if x is None else x
        return _PerModelLogistic().fit(x[lab], q[lab]).predict(x[target])[:, 0]
    if method in ("tfidf_logreg", "knn10"):
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
        xl = vec.fit_transform([s.texts[i] for i in lab])
        xt = vec.transform([s.texts[i] for i in target])
        if method == "tfidf_logreg":
            return _PerModelLogistic().fit(xl, q[lab]).predict(xt)[:, 0]
        return _knn(xl, xt, y[lab])
    rep, kind = method.split("_")
    e = s.emb[rep]
    if kind == "knn":
        return _knn(e[lab], e[target], y[lab])
    head = _TunedPerModelLogistic() if rep == "bgeL" else _PerModelLogistic()
    return head.fit(e[lab], q[lab]).predict(e[target])[:, 0]


def _knn(xl, xt, yl, k: int = 10) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k, xl.shape[0]), metric="cosine").fit(xl)
    _, nbrs = nn.kneighbors(xt)
    return yl[nbrs].mean(axis=1)


def curves(s: Setup, y: np.ndarray, methods: list[str], draws: np.ndarray, x_custom=None):
    """{method: {k: (seeds x B) within-category AUROC}} on the test split."""
    out = {}
    for m in methods:
        out[m] = {}
        for k in KS:
            rows = []
            for seed in SEEDS if k is not None else SEEDS[:1]:
                lab = k_shot(s.train, k, seed)
                p = predict(m, s, y, lab, s.test, x=x_custom)
                rows.append(within_rows(p, y[s.test], s.cats[s.test], draws))
            if k is None:  # all labels: one labeled set, same across seeds
                rows = rows * len(SEEDS)
            out[m][k] = np.stack(rows)
    return out


def kname(k) -> str:
    return "all" if k is None else str(k)


def reproduce(s: Setup) -> None:
    y = (s.ds.quality_matrix()[:, 0] >= 0.5).astype(float)
    draws = np.arange(len(s.test))[None, :]
    c = curves(s, y, ["jev_features", *BASELINES], draws)
    recs = json.loads(Path("results/llmrouterbench_test/metrics.json").read_text())
    names = {"jev_features": "jev:features"}
    worst = 0.0
    for m, by_k in c.items():
        for k, v in by_k.items():
            ref = [
                r["by_category"]["within"]
                for r in recs
                if r["router"] == names.get(m, m) and r.get("k") == k and r["variant"] == "raw"
            ]
            gap = abs(float(np.mean(v[:, 0])) - float(np.mean(ref)))
            worst = max(worst, gap)
            print(
                f"{m:14s} k={kname(k):>4s} ours {np.mean(v[:, 0]):.4f} pipeline {np.mean(ref):.4f}"
            )
    print(f"max |difference| = {worst:.4f}")


def seed_boot(diff: np.ndarray, seed_draws: np.ndarray) -> np.ndarray:
    """(seeds x B) differences -> B bootstrap means (column 0 = point estimate)."""
    return np.nanmean(np.take_along_axis(diff, seed_draws.T, axis=0), axis=0)


def ci(boot: np.ndarray) -> tuple[float, float, float]:
    lo, hi = np.nanpercentile(boot[1:], [2.5, 97.5])
    return float(boot[0]), float(lo), float(hi)


def primary_baseline(s: Setup) -> str:
    """Pre-registered rule: best text router at k=25 on the original target."""
    y = (s.ds.quality_matrix()[:, 0] >= 0.5).astype(float)
    draws = np.arange(len(s.test))[None, :]
    c = curves(s, y, BASELINES, draws)
    means = {m: float(np.nanmean(c[m][25][:, 0])) for m in BASELINES}
    return max(BASELINES, key=lambda m: (means[m], -BASELINES.index(m))), means


def write(name: str, lines: list[str], payload: dict) -> None:
    out = Path("results/evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"extensions_{name}.md").write_text("\n".join(lines) + "\n")
    (out / f"extensions_{name}.json").write_text(json.dumps(payload, indent=1, default=float))
    print("\n".join(lines))


def experiment_t(s: Setup) -> None:
    base, base_means = primary_baseline(s)
    rng = np.random.default_rng(0)
    draws = stratified_draws(s.cats[s.test], rng, B)
    seed_draws = rng.integers(0, len(SEEDS), size=(B, len(SEEDS)))
    seed_draws[0] = np.arange(len(SEEDS))
    methods = ["jev_features", *BASELINES]
    per_target, diffs25 = {}, []
    for t in TARGETS:
        y = s.labels(t)
        c = curves(s, y, methods, draws)
        per_target[t] = {
            "base_rate_test": float(y[s.test].mean()),
            "mean_cost": s.mean_cost(t),
            "auroc": {m: {kname(k): float(np.nanmean(c[m][k][:, 0])) for k in KS} for m in methods},
            "diff_vs_primary": {},
        }
        for k in KS:
            d = c["jev_features"][k] - c[base][k]
            per_target[t]["diff_vs_primary"][kname(k)] = ci(seed_boot(d, seed_draws))
        diffs25.append(c["jev_features"][25] - c[base][25])
        print(f"  {t}: done", flush=True)
    # Pooled primary endpoint: resample targets, seeds and test prompts together.
    target_draws = rng.integers(0, len(TARGETS), size=(B, len(TARGETS)))
    target_draws[0] = np.arange(len(TARGETS))
    per_t = np.stack([seed_boot(d, seed_draws) for d in diffs25])  # targets x B
    pooled = np.nanmean(np.take_along_axis(per_t, target_draws.T, axis=0), axis=0)
    point, lo, hi = ci(pooled)
    lines = [
        "# Experiment T: success prediction for 9 unseen models (pre-registered)",
        "",
        f"Primary baseline by the pre-registered rule: **{base}** (k=25 on the original target: "
        + ", ".join(f"{m} {v:.3f}" for m, v in base_means.items())
        + ").",
        "",
        (
            f"**Primary endpoint** (k=25, mean over targets of jev_features - {base}): "
            f"**{point:+.3f} [{lo:+.3f}, {hi:+.3f}]**; success criterion (lower bound > 0): "
            f"**{'MET' if lo > 0 else 'NOT MET'}**."
        ),
        "",
        (
            f"| target | base rate | jev k=25 | {base} k=25 | diff k=25 [95% CI] | jev all | "
            f"{base} all | labels for {base} to reach jev@25 |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for t, r in per_target.items():
        a = r["auroc"]
        target_level = a["jev_features"]["25"]
        reach = next((kname(k) for k in KS if a[base][kname(k)] >= target_level), "not reached")
        r["labels_to_reach_jev25"] = {
            m: next((kname(k) for k in KS if a[m][kname(k)] >= target_level), "not reached")
            for m in methods
        }
        d = r["diff_vs_primary"]["25"]
        lines.append(
            f"| {t} | {r['base_rate_test']:.2f} | {a['jev_features']['25']:.3f} | "
            f"{a[base]['25']:.3f} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | "
            f"{a['jev_features']['all']:.3f} | {a[base]['all']:.3f} | {reach} |"
        )
    write("T", lines, {"primary_baseline": base, "primary": [point, lo, hi], "targets": per_target})


def experiment_d(s: Setup) -> None:
    base, _ = primary_baseline(s)
    y = (s.ds.quality_matrix()[:, 0] >= 0.5).astype(float)
    methods = ["jev_features", base, "tfidf_logreg"]
    rng = np.random.default_rng(0)
    seed_draws = rng.integers(0, len(SEEDS), size=(B, len(SEEDS)))
    seed_draws[0] = np.arange(len(SEEDS))
    from dmroute.metrics.stats import _auroc_rows

    rows, pooled = [], []
    for dom in ("livecodebench", "mmlupro", "swe-bench"):
        source = s.train[s.domains[s.train] != dom]
        tgt_train = s.train[s.domains[s.train] == dom]
        tgt_test = s.test[s.domains[s.test] == dom]
        yt = y[tgt_test]
        draws = np.vstack(
            [np.arange(len(tgt_test)), rng.integers(0, len(tgt_test), (B - 1, len(tgt_test)))]
        )
        res = {}
        for m in methods:
            res[m] = {}
            for k in (0, 25, 50, 100):
                per_seed = []
                for seed in SEEDS if k else SEEDS[:1]:
                    lab = np.concatenate([source, k_shot(tgt_train, k, seed) if k else []]).astype(
                        int
                    )
                    p = predict(m, s, y, lab, tgt_test)
                    per_seed.append(_auroc_rows(p[draws], yt[draws]))
                res[m][k] = np.stack(per_seed * (1 if k else len(SEEDS)))
        d25 = seed_boot(res["jev_features"][25] - res[base][25], seed_draws)
        pooled.append(d25)
        for k in (0, 25, 50, 100):
            rows.append(
                {
                    "domain": dom,
                    "k": k,
                    "n_test": len(tgt_test),
                    **{m: float(np.nanmean(res[m][k][:, 0])) for m in methods},
                    "diff": ci(seed_boot(res["jev_features"][k] - res[base][k], seed_draws)),
                }
            )
    point, lo, hi = ci(np.nanmean(np.stack(pooled), axis=0))
    lines = [
        "# Experiment D: few-shot adaptation to a held-out domain (pre-registered)",
        "",
        (
            "Trained on other domains' train labels + k target-domain labels; AUROC on the "
            f"target domain's test split. **Primary** (k=25, mean over domains, jev_features - "
            f"{base}): **{point:+.3f} [{lo:+.3f}, {hi:+.3f}]** -> "
            f"**{'MET' if lo > 0 else 'NOT MET'}**."
        ),
        "",
        f"| domain (n test) | k | jev_features | {base} | tfidf_logreg | diff [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        d = r["diff"]
        lines.append(
            f"| {r['domain']} ({r['n_test']}) | {r['k']} | {r['jev_features']:.3f} | "
            f"{r[base]:.3f} | {r['tfidf_logreg']:.3f} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] |"
        )
    write("D", lines, {"primary_baseline": base, "primary": [point, lo, hi], "rows": rows})


def experiment_e(s: Setup) -> None:
    y = (s.ds.quality_matrix()[:, 0] >= 0.5).astype(float)
    rng = np.random.default_rng(0)
    draws = stratified_draws(s.cats[s.test], rng, B)
    seed_draws = rng.integers(0, len(SEEDS), size=(B, len(SEEDS)))
    seed_draws[0] = np.arange(len(SEEDS))
    jev = curves(s, y, ["jev_features"], draws)["jev_features"]
    lines = ["# Experiment E: why does it work? (pre-registered, original target)", ""]
    payload: dict = {"pca13": {}, "subsets": {}}
    lines += [
        "## E2. 13-dimensional embeddings (PCA fit on unlabeled train prompts), same C=1 head",
        "",
        "| representation | "
        + " | ".join(f"k={kname(k)}" for k in KS)
        + " | jev - it at k=25 [95% CI] |",
        "|---|" + "---|" * (len(KS) + 1),
    ]
    for rep in ("bge", "bgeL"):
        x = PCA(n_components=13, random_state=0).fit(s.emb[rep][s.train]).transform(s.emb[rep])
        c = curves(s, y, ["custom"], draws, x_custom=x)["custom"]
        means = [float(np.nanmean(c[k][:, 0])) for k in KS]
        d = ci(seed_boot(jev[25] - c[25], seed_draws))
        payload["pca13"][rep] = {"auroc": means, "diff25": d}
        lines.append(
            f"| {rep} PCA-13 | "
            + " | ".join(f"{m:.3f}" for m in means)
            + f" | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] |"
        )
    lines.append(
        "| jev features (13) | "
        + " | ".join(f"{np.nanmean(jev[k][:, 0]):.3f}" for k in KS)
        + " | - |"
    )
    lines += [
        "",
        "## E3. Number of Jev questions (random subsets, 20 draws each, numpy seed 0)",
        "",
        "| questions | " + " | ".join(f"k={kname(k)}" for k in KS) + " |",
        "|---|" + "---|" * len(KS),
    ]
    sub_rng = np.random.default_rng(0)
    one_draw = np.arange(len(s.test))[None, :]
    for size in (1, 3, 6, 13):
        subsets = (
            [np.arange(13)]
            if size == 13
            else [np.sort(sub_rng.choice(13, size, replace=False)) for _ in range(20)]
        )
        by_k = {kname(k): [] for k in KS}
        for cols in subsets:
            c = curves(s, y, ["custom"], one_draw, x_custom=s.jev[:, cols])["custom"]
            for k in KS:
                by_k[kname(k)].append(float(np.nanmean(c[k][:, 0])))
        payload["subsets"][size] = {
            "draws": [[list(FEATURES)[i] for i in cols] for cols in subsets],
            "auroc": by_k,
        }
        lines.append(
            f"| {size} | "
            + " | ".join(
                f"{np.mean(v):.3f} ± {np.std(v):.3f}" if size < 13 else f"{v[0]:.3f}"
                for v in by_k.values()
            )
            + " |"
        )
    write("E", lines, payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["reproduce", "T", "D", "E"])
    args = ap.parse_args()
    s = Setup()
    {"reproduce": reproduce, "T": experiment_t, "D": experiment_d, "E": experiment_e}[
        args.experiment
    ](s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
