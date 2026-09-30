"""Significance and generalization checks for the headline results (no API calls).

    uv run python scripts/evidence.py configs/llmrouterbench_test.yaml
    uv run python scripts/evidence.py configs/twinrouterbench_test.yaml --no-lodo

A. Few-shot discrimination: within-category AUROC of P(cheapest model suffices), decision-model
   features vs each text router, at every k. Hierarchical paired bootstrap: each replicate
   resamples the 5 label seeds and the shared test set (stratified by category).
B. Operating points vs always-mid (seed 42, all labels): paired bootstrap on quality and cost,
   exact McNemar on quality.
C. Leave-one-dataset-out (LLMRouterBench): heads trained on the other datasets' train split,
   scored on every non-design prompt of the held-out dataset (never seen by any head).

Reads the full run's saved predictions (results/<name>/seed*_k*/) and cached calls; run the
config with dmroute first. Writes results/evidence/<name>.{md,json}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from dmroute.backends.cache import CallCache
from dmroute.data import load
from dmroute.data.splits import grouped_split, k_shot
from dmroute.metrics import cost_quality, stats
from dmroute.metrics.calibration import MIN_CLASS
from dmroute.metrics.stats import _auroc_rows
from dmroute.policy import pick
from dmroute.routers import build_routers

B = 2000
SEEDS = [42, 999, 2024, 2025, 3407]
TEXT_ROUTERS = ["tfidf_logreg", "knn10", "bge_logreg", "bge_knn", "bgeL_logreg", "bgeL_knn"]


def splits(ds, split: dict) -> tuple[np.ndarray, np.ndarray]:
    """Train and test indices exactly as experiment.run builds them."""
    train, test = grouped_split(ds, split.get("test_frac", 0.3), split.get("test_seed", 42))
    if split.get("exclude_from_test"):
        seen = set(Path(split["exclude_from_test"]).read_text().split())
        moved = np.array([ds.examples[i].id in seen for i in test], dtype=bool)
        train, test = np.sort(np.concatenate([train, test[moved]])), test[~moved]
    return train, test


def load_p(results_dir: Path, seed: int, k, router: str, ids: list[str]) -> np.ndarray | None:
    f = results_dir / f"seed{seed}_k{k}" / f"{router}.jsonl"
    if not f.exists():
        return None
    rows = {r["id"]: r for r in map(json.loads, f.open())}
    return np.array([rows[i]["p"] for i in ids], dtype=float)


def within_rows(p: np.ndarray, y: np.ndarray, cats: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Within-category AUROC (n-weighted, categories with >= MIN_CLASS of each outcome) for
    each bootstrap row of test positions ``idx`` (shape B x n)."""
    num = np.zeros(len(idx))
    den = np.zeros(len(idx))
    for c in np.unique(cats):
        cols = cats[idx] == c  # which drawn positions fall in category c, per row
        # Stratified resampling keeps each category's count fixed, so gather per row.
        n_c = int(cols[0].sum())
        if n_c == 0:
            continue
        pos = idx[cols].reshape(len(idx), n_c)
        yy, pp = y[pos], p[pos]
        ok = (yy.sum(1) >= MIN_CLASS) & ((n_c - yy.sum(1)) >= MIN_CLASS)
        a = _auroc_rows(pp, yy)
        num += np.where(ok, a * n_c, 0.0)
        den += np.where(ok, n_c, 0.0)
    with np.errstate(invalid="ignore"):
        return num / den


def stratified_draws(cats: np.ndarray, rng: np.random.Generator, b: int) -> np.ndarray:
    """B x n test positions, resampled with replacement within each category; row 0 = identity."""
    n = len(cats)
    out = np.empty((b, n), dtype=int)
    out[0] = np.arange(n)
    for c in np.unique(cats):
        members = np.flatnonzero(cats == c)
        out[1:, members] = rng.choice(members, size=(b - 1, len(members)))
    return out


def part_a(results_dir, ds, test, feat_router, ks) -> list[dict]:
    ids = [ds.examples[i].id for i in test]
    y = (ds.quality_matrix()[test, 0] >= 0.5).astype(float)
    cats = np.array([ds.examples[i].category for i in test])
    rng = np.random.default_rng(0)
    draws = stratified_draws(cats, rng, B)
    seed_draws = rng.integers(0, len(SEEDS), size=(B, len(SEEDS)))
    seed_draws[0] = np.arange(len(SEEDS))
    rows = []
    for k in ks:
        scores = {}
        for router in [feat_router, *TEXT_ROUTERS]:
            per_seed = [load_p(results_dir, s, k, router, ids) for s in SEEDS]
            if any(p is None for p in per_seed):
                continue
            scores[router] = np.stack([within_rows(p[:, 0], y, cats, draws) for p in per_seed])
        if feat_router not in scores:
            continue
        f = scores[feat_router]  # seeds x B
        for base in TEXT_ROUTERS:
            if base not in scores:
                continue
            diff = f - scores[base]
            point = float(np.nanmean(diff[:, 0]))
            boot = np.nanmean(np.take_along_axis(diff, seed_draws.T, axis=0), axis=0)
            lo, hi = np.nanpercentile(boot[1:], [2.5, 97.5])
            rows.append(
                {
                    "k": "all" if k is None else k,
                    "router": feat_router,
                    "baseline": base,
                    "router_auroc": float(np.nanmean(f[:, 0])),
                    "baseline_auroc": float(np.nanmean(scores[base][:, 0])),
                    "diff": point,
                    "ci95": [float(lo), float(hi)],
                    "seeds_won": int((diff[:, 0] > 0).sum()),
                }
            )
    return rows


def part_b(results_dir, ds, test, routers) -> list[dict]:
    recs = json.loads((results_dir / "metrics.json").read_text())
    taus = {}
    for r in recs:
        if r.get("seed") == 42 and r.get("k") is None and r.get("variant") == "raw":
            name = "".join(ch if ch.isalnum() or ch in "-_.@" else "_" for ch in r["router"])
            taus[name] = {k: v["tau"] for k, v in r["operating_points"].items()}
    ids = [ds.examples[i].id for i in test]
    q, c = ds.quality_matrix()[test], ds.cost_matrix()[test]
    mid = len(ds.pool) // 2
    mid_q, mid_c = q[:, mid], c[:, mid]
    rows = []
    for router in routers:
        f = results_dir / "seed42_kNone" / f"{router}.jsonl"
        if not f.exists() or router not in taus:
            continue
        saved = {r["id"]: r for r in map(json.loads, f.open())}
        p = np.array([saved[i]["p"] for i in ids], dtype=float)
        rcost = np.array([saved[i]["router_cost"] for i in ids], dtype=float)
        for op in ("budget_mid", "match_frac"):
            qual, cost = cost_quality.realized(pick(p, taus[router][op]), q, c, rcost)
            dq = stats.paired_bootstrap(qual, mid_q)
            dc = stats.paired_bootstrap(cost, mid_c)
            rows.append(
                {
                    "router": router,
                    "op": op,
                    "quality": float(qual.mean()),
                    "cost": float(cost.mean()),
                    "dq": dq,
                    "dc": dc,
                    "mcnemar_p": stats.mcnemar_exact(qual >= 0.5, mid_q >= 0.5).get("p_value"),
                }
            )
    return rows, ds.model_names[mid], float(mid_q.mean()), float(mid_c.mean())


def part_c(cfg, ds, train, test_all_ok, feat_label) -> list[dict]:
    """Leave-one-dataset-out with the decision-model routers and the text routers."""
    wanted = []
    for spec in cfg["routers"]:
        if spec["type"] in ("tfidf_logreg", "knn", "embedding"):
            wanted.append(spec)
        elif spec["type"] == "decision" and spec["label"] == feat_label:
            wanted.append({**spec, "formulations": ["features", "choice", "difficulty"]})
    cache = CallCache(cfg.get("cache", ".cache/calls.sqlite"))
    routers = build_routers(wanted, ds, cache)
    q = ds.quality_matrix()
    datasets = np.array([ex.dataset for ex in ds.examples])
    rows = []
    for held in sorted(set(datasets)):
        target = np.array([i for i in test_all_ok if datasets[i] == held])
        y = (q[target, 0] >= 0.5).astype(float)
        if min(y.sum(), len(y) - y.sum()) < MIN_CLASS:
            continue
        pool = np.array([i for i in train if datasets[i] != held])
        for k in (25, None):
            seeds = SEEDS if k is not None else [42]
            for router in routers:
                vals = []
                for s in seeds:
                    lab = k_shot(pool, k, s)
                    if router.needs_labels:
                        router.fit(ds, lab)
                    p = np.array([o.p[0] for o in router.predict(ds, target)])
                    vals.append(float(_auroc_rows(p[None, :], y[None, :])[0]))
                boot = stats.bootstrap_auroc(p, y)  # CI from the last seed's predictions
                rows.append(
                    {
                        "held_out": held,
                        "n": len(target),
                        "n_neg": int(len(y) - y.sum()),
                        "k": "all" if k is None else k,
                        "router": router.name,
                        "auroc": float(np.mean(vals)),
                        "sd": float(np.std(vals)),
                        "ci95_last_seed": boot["ci95"],
                    }
                )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--features", default=None, help="decision-model features router name")
    ap.add_argument("--no-lodo", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    results_dir = Path(cfg.get("output_dir", f"results/{cfg['name']}"))
    ds = load(cfg["dataset"])
    train, test = splits(ds, cfg.get("split", {}))
    ks = [*[k for k in cfg["split"].get("k_shot", [None]) if k is not None], None]
    feat = args.features or ("jev-4k_features" if "twin" in cfg["name"] else "jev_features")

    out: dict = {"A": part_a(results_dir, ds, test, feat, ks)}
    decision = [f"{feat.split('_')[0]}_{f}" for f in ("choice", "features")]
    b_rows, mid_name, mid_q, mid_c = part_b(
        results_dir, ds, test, [*decision, "tfidf_logreg", "bge_logreg", "category_best"]
    )
    out["B"] = b_rows
    lines = [
        f"# Evidence: {cfg['name']}",
        "",
        "## A. Few-shot discrimination (within-category AUROC, P(cheapest suffices))",
        "",
        f"Hierarchical paired bootstrap ({B} replicates: 5 label seeds x stratified test set).",
        "",
        "| k | router | baseline | router | baseline | diff | 95% CI | seeds won |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in out["A"]:
        lines.append(
            f"| {r['k']} | {r['router']} | {r['baseline']} | {r['router_auroc']:.3f} | "
            f"{r['baseline_auroc']:.3f} | {r['diff']:+.3f} | "
            f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['seeds_won']}/5 |"
        )
    lines += [
        "",
        f"## B. Operating points vs always-{mid_name} ({mid_q:.3f} @ ${mid_c:.5f}), all labels",
        "",
        "| router | op | quality | cost | d quality [95% CI] | d cost [95% CI] | McNemar p |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in b_rows:
        dq, dc = r["dq"], r["dc"]
        p = r["mcnemar_p"]
        lines.append(
            f"| {r['router']} | {r['op']} | {r['quality']:.3f} | ${r['cost']:.5f} | "
            f"{dq['mean_diff']:+.3f} [{dq['ci95'][0]:+.3f}, {dq['ci95'][1]:+.3f}] | "
            f"{dc['mean_diff']:+.5f} [{dc['ci95'][0]:+.5f}, {dc['ci95'][1]:+.5f}] | "
            f"{'-' if p is None else f'{p:.3f}'} |"
        )
    if not args.no_lodo:
        seen = set()
        if cfg["split"].get("exclude_from_test"):
            seen = set(Path(cfg["split"]["exclude_from_test"]).read_text().split())
        ok = np.array([i for i in range(len(ds.examples)) if ds.examples[i].id not in seen])
        out["C"] = part_c(cfg, ds, train, ok, feat.split("_")[0])
        lines += [
            "",
            "## C. Leave-one-dataset-out (AUROC on the held-out dataset)",
            "",
            "Heads trained on the other datasets' train split (k=25: mean ± sd over 5 seeds);",
            "scored on every non-design prompt of the held-out dataset.",
            "",
            "| held out | n (neg) | k | router | AUROC | sd | 95% CI (one seed) |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in out["C"]:
            ci = r["ci95_last_seed"]
            lines.append(
                f"| {r['held_out']} | {r['n']} ({r['n_neg']}) | {r['k']} | {r['router']} | "
                f"{r['auroc']:.3f} | {r['sd']:.3f} | [{ci[0]:.2f}, {ci[1]:.2f}] |"
            )
    out_dir = Path("results/evidence")
    out_dir.mkdir(parents=True, exist_ok=True)
    md = "\n".join(lines) + "\n"
    (out_dir / f"{cfg['name']}.md").write_text(md)
    (out_dir / f"{cfg['name']}.json").write_text(json.dumps(out, indent=1, default=float))
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
