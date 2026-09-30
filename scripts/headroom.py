"""Headroom scan for choosing a second model pool (preregistration/v2.md, Part 2).

Train split only: reads model outcomes and costs from the LLMRouterBench release, never router
output. For each (cheap, strong) pair with cost ratio >= 3 and train accuracy(cheap) <=
accuracy(strong), reports the oracle gain over the strong model. Run from the repo root.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from dmroute.data import load
from evidence import splits

CONFIG = "configs/llmrouterbench_test.yaml"
EXCLUDED = {"openrouter"}  # not a single named model


def outcomes(root: Path, ids: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Correctness and cost per prompt for every model that answers all ``ids``."""
    pairs = {tuple(i.split("/")[:2]) for i in ids}
    models = sorted({p.name for d, s in pairs for p in (root / d / s).iterdir() if p.is_dir()})
    correct, cost = {}, {}
    for m in models:
        score, spend = {}, {}
        for d, s in pairs:
            files = list((root / d / s / m).glob("*.json")) if (root / d / s / m).exists() else []
            for r in json.loads(max(files).read_text())["records"] if files else []:
                key = f"{d}/{s}/{r['index']}"
                score[key], spend[key] = float(r["score"]), float(r.get("cost") or 0.0)
        if all(i in score for i in ids):
            correct[m] = np.array([score[i] >= 0.5 for i in ids], dtype=float)
            cost[m] = np.array([spend[i] for i in ids])
    return correct, cost


def main() -> None:
    cfg = yaml.safe_load(Path(CONFIG).read_text())
    ds = load(cfg["dataset"])
    train, _ = splits(ds, cfg["split"])
    correct, cost = outcomes(Path(cfg["dataset"]["root"]), [ex.id for ex in ds.examples])
    acc = {m: y[train].mean() for m, y in correct.items()}
    usd = {m: c[train].mean() for m, c in cost.items()}
    print(f"{len(train)} train prompts; {len(correct)} models with full coverage")
    rows = []
    for a in correct:
        for b in correct:
            if {a, b} & EXCLUDED or usd[a] > usd[b] / 3 or acc[a] > acc[b]:
                continue
            oracle = np.maximum(correct[a][train], correct[b][train]).mean()
            rows.append(
                (oracle - acc[b], usd[b] / usd[a], a, b, acc[a], acc[b], oracle, usd[a], usd[b])
            )
    rows.sort(reverse=True)
    print("gain  ratio  cheap -> strong  acc_cheap acc_strong oracle  $cheap $strong")
    for gain, ratio, a, b, aa, ab, orc, ca, cb in rows[:12]:
        print(
            f"{gain:.3f} {ratio:5.1f}  {a} -> {b}  {aa:.3f} {ab:.3f} {orc:.3f}  {ca:.4f} {cb:.4f}"
        )


if __name__ == "__main__":
    main()
