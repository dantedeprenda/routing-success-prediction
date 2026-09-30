"""Labels-to-target and the cost of collecting those labels (no API calls).

    uv run python scripts/label_budget.py

For each benchmark's original target (the cheapest model suffices), the smallest labeled-set
size at which each router's mean within-category AUROC reaches the decision-model features
router's k=25 level, and what labeling that many prompts costs: running every pool model on a
prompt (needed to train per-model heads) or only the cheap model (enough for this one label).
Writes results/evidence/label_budget.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dmroute.data import load
from evidence import splits

BENCH = {
    "llmrouterbench_test": "jev:features",
    "twinrouterbench_test": "jev-4k:features",
}
ROUTERS = ["tfidf_logreg", "knn10", "bge_logreg", "bge_knn", "bgeL_logreg", "bgeL_knn"]


def main() -> int:
    lines = ["# Labels needed to match the decision model's 25-label discrimination", ""]
    for name, feat in BENCH.items():
        cfg = yaml.safe_load(Path(f"configs/{name}.yaml").read_text())
        ds = load(cfg["dataset"])
        n_train = len(splits(ds, cfg["split"])[0])
        c = ds.cost_matrix()
        all_models, cheap_only = float(c.sum(axis=1).mean()), float(c[:, 0].mean())
        recs = json.loads(Path(f"results/{name}/metrics.json").read_text())
        ks = sorted({r["k"] for r in recs if r.get("k")}) + [None]

        def mean_within(router, k, recs=recs):
            v = [
                r["by_category"]["within"]
                for r in recs
                if r["router"] == router and r.get("k") == k and r.get("variant") == "platt"
            ]
            return float(np.mean(v)) if v else float("nan")

        level = mean_within(feat, 25)

        def reach(router, level=level, ks=ks, n_train=n_train):
            for k in ks:
                if mean_within(router, k) >= level:
                    return n_train if k is None else k
            return None

        def dollars(n, all_models=all_models, cheap_only=cheap_only):
            return f"${n * all_models:,.2f} (all models) / ${n * cheap_only:,.2f} (cheap only)"

        lines += [
            f"## {name}",
            "",
            (
                f"Target: {feat} at k=25 = {level:.3f}. Labeling one prompt costs "
                f"${all_models:.4f} with every pool model, ${cheap_only:.5f} with the cheap "
                f"model only. Train split: {n_train:,} prompts."
            ),
            "",
            "| router | best mean AUROC (any k) | labels to reach target | labeling cost |",
            "|---|---|---|---|",
            f"| {feat} | {max(mean_within(feat, k) for k in ks):.3f} | 25 | {dollars(25)} |",
        ]
        for router in ROUTERS:
            if np.isnan(mean_within(router, 25)):
                continue
            n = reach(router)
            best = max(mean_within(router, k) for k in ks)
            cell = (
                f"not reached with all {n_train:,}"
                if n is None
                else (f"all {n_train:,}" if n == n_train else str(n))
            )
            cost = f"> {dollars(n_train)}" if n is None else dollars(n)
            lines.append(f"| {router} | {best:.3f} | {cell} | {cost} |")
        lines.append("")
    out = Path("results/evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_budget.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
