"""Very small label budgets and deferral, on the original target (no API calls).

    uv run python scripts/small_budgets.py

Exploratory, not pre-registered. Uses the same re-implementation as scripts/extensions.py (which
reproduces the pipeline's 35/35 cells) with the network blocked, so only cached answers are read.

A. Within-category AUROC at k = 5 and 10 (25 for reference): decision-model features vs each
   text router, hierarchical paired bootstrap over the 5 label seeds and the test set. Labeled
   sets are drawn exactly as in the paper (uniform, no stratification), so a draw with one
   outcome only gives a constant predictor; the share of such draws is reported.
B. Deferral: accept the cheap model's answer on the fraction of prompts a router is most
   confident about and escalate the rest. Selective risk = cheap-model failure rate among
   accepted prompts. Reports AURC (area under the risk-coverage curve, lower is better) and risk
   at 50% coverage, over all test prompts (category knowledge counts here, so a category-prior
   baseline shows how much of it is just knowing the category).
C. Setting the threshold from the k labels: cross-fitted predictions on the labeled set pick the
   largest coverage whose estimated risk is at most half the labeled set's failure rate; the
   threshold is then applied unchanged to the test split. Reports realized coverage and risk.

Writes results/evidence/small_budgets.{md,json}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from dmroute.data.splits import k_shot
from evidence import SEEDS, stratified_draws, within_rows
from extensions import BASELINES, Setup, ci, kname, predict, seed_boot

BENCH = {  # config, decision-model label
    "llmrouterbench_test": ("configs/llmrouterbench_test.yaml", "jev"),
    "twinrouterbench_test": ("configs/twinrouterbench_test.yaml", "jev-4k"),
}
SMALL = [5, 10, 25]
DEFER_KS = [5, 10, 25, 100, None]
B = 1000
FEAT = "jev_features"


def category_prior(s: Setup, y: np.ndarray, lab: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Labeled success rate of each category, shrunk to the overall rate (1 pseudo-count)."""
    overall = float(y[lab].mean())
    out = np.full(len(target), overall)
    for c in np.unique(s.cats[target]):
        m = s.cats[lab] == c
        rate = (y[lab][m].sum() + overall) / (m.sum() + 1)
        out[s.cats[target] == c] = rate
    return out


def scores(method: str, s: Setup, y, lab, target) -> np.ndarray:
    if method == "category_prior":
        return category_prior(s, y, lab, target)
    return predict(method, s, y, lab, target)


def risk_coverage(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Selective risk after accepting the i most confident prompts, i = 1..n (ties averaged)."""
    order = np.argsort(-p, kind="stable")
    fails = 1.0 - y[order]
    risk = np.cumsum(fails) / np.arange(1, len(y) + 1)
    # Tied scores have no order; give every position in a tie block the block's mean risk.
    ps = p[order]
    out = risk.copy()
    start = 0
    for i in range(1, len(ps) + 1):
        if i == len(ps) or ps[i] != ps[start]:
            if i - start > 1:
                base = risk[start - 1] * start if start else 0.0
                blk = fails[start:i].mean()
                out[start:i] = (base + blk * np.arange(1, i - start + 1)) / np.arange(
                    start + 1, i + 1
                )
            start = i
    return out


def aurc(p, y) -> float:
    return float(risk_coverage(p, y).mean())


def risk_at(p, y, cov: float) -> float:
    return float(risk_coverage(p, y)[max(round(cov * len(y)) - 1, 0)])


def _fmt(risks: list[float]) -> str:
    """Mean realized risk over the seeds that accepted anything ('-' if none did)."""
    r = [x for x in risks if not np.isnan(x)]
    return f"{np.mean(r):.3f}" if r else "-"


def cross_fit(method, s, y, lab) -> np.ndarray | None:
    yl = y[lab]
    minority = int(min(yl.sum(), len(yl) - yl.sum()))
    if minority < 2:
        return None
    out = np.empty(len(lab))
    folds = StratifiedKFold(n_splits=min(5, minority), shuffle=True, random_state=0)
    for tr, te in folds.split(lab, yl):
        out[te] = scores(method, s, y, lab[tr], lab[te])
    return out


def pick_threshold(p_cv: np.ndarray, y_lab: np.ndarray, target: float) -> float:
    """Lowest score threshold whose accepted labeled prompts have failure rate <= target."""
    best = np.inf  # accept nothing
    for t in np.unique(p_cv):
        acc = p_cv >= t
        if (1 - y_lab[acc]).mean() <= target:
            best = min(best, t)
    return best


def run(name: str, config: str, jev: str) -> tuple[list[str], dict]:
    s = Setup(config, jev, offline=True)
    y = (s.ds.quality_matrix()[:, 0] >= 0.5).astype(float)
    yt, ct = y[s.test], s.cats[s.test]
    rng = np.random.default_rng(0)
    draws = stratified_draws(ct, rng, B)
    seed_draws = rng.integers(0, len(SEEDS), size=(B, len(SEEDS)))
    seed_draws[0] = np.arange(len(SEEDS))
    methods = [FEAT, *BASELINES]
    lines = [f"## {name}", ""]
    lines.append(
        f"Test prompts: {len(s.test):,}; cheap model fails on {1 - yt.mean():.1%} of them. "
        f"Train pool: {len(s.train):,}."
    )
    payload: dict = {"a": [], "b": [], "c": []}

    # A. within-category AUROC at small k
    one_class = {
        k: float(np.mean([len(set(y[k_shot(s.train, k, sd)])) < 2 for sd in SEEDS])) for k in SMALL
    }
    lines += [
        "",
        "### A. Within-category AUROC at small k (5 seeds; 95% CI of the difference)",
        "",
        "| k | one-outcome draws | features | baseline | baseline AUROC | diff | 95% CI | seeds won |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in SMALL:
        per = {}
        for m in methods:
            per[m] = np.stack(
                [
                    within_rows(predict(m, s, y, k_shot(s.train, k, sd), s.test), yt, ct, draws)
                    for sd in SEEDS
                ]
            )
        f = per[FEAT]
        for base in BASELINES:
            diff = f - per[base]
            d, lo, hi = ci(seed_boot(diff, seed_draws))
            won = int((diff[:, 0] > 0).sum())
            row = {
                "k": k,
                "baseline": base,
                "features": float(np.nanmean(f[:, 0])),
                "baseline_auroc": float(np.nanmean(per[base][:, 0])),
                "diff": d,
                "ci95": [lo, hi],
                "seeds_won": won,
                "one_class_share": one_class[k],
            }
            payload["a"].append(row)
            lines.append(
                f"| {k} | {one_class[k]:.0%} | {row['features']:.3f} | {base} | "
                f"{row['baseline_auroc']:.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {won}/5 |"
            )

    # B + C. deferral
    dmethods = [FEAT, *BASELINES, "category_prior"]
    lines += [
        "",
        "### B. Deferral: accept the cheap answer on the most confident prompts (mean of 5 seeds)",
        "",
        (
            f"Always accepting has risk {1 - yt.mean():.3f}; a random order has AURC "
            f"{1 - yt.mean():.3f}; the oracle AURC is {aurc(yt, yt):.3f}."
        ),
        "",
        "| k | method | AURC | risk @ 50% coverage |",
        "|---|---|---|---|",
    ]
    lines_c = [
        "",
        "### C. Threshold set from the k labels (target: half the labeled failure rate)",
        "",
        (
            "Cross-fitted on the labeled set only, applied unchanged to test. 'n/a' = fewer than 2 "
            "labels of an outcome, so no threshold can be estimated. Coverage averages the seeds "
            "with a threshold; test risk averages those that accepted anything ('-': none did)."
        ),
        "",
        "| k | method | seeds with a threshold | target risk | test coverage | test risk |",
        "|---|---|---|---|---|---|",
    ]
    for k in DEFER_KS:
        seeds = SEEDS if k is not None else SEEDS[:1]
        for m in dmethods:
            au, r50, cov, rk, tg = [], [], [], [], []
            for sd in seeds:
                lab = k_shot(s.train, k, sd)
                p = scores(m, s, y, lab, s.test)
                au.append(aurc(p, yt))
                r50.append(risk_at(p, yt, 0.5))
                if k is None or m == "category_prior":
                    continue
                p_cv = cross_fit(m, s, y, lab)
                if p_cv is None:
                    continue
                target = 0.5 * float((1 - y[lab]).mean())
                t = pick_threshold(p_cv, y[lab], target)
                acc = p >= t
                tg.append(target)
                cov.append(float(acc.mean()))
                rk.append(float((1 - yt[acc]).mean()) if acc.any() else float("nan"))
            b_row = {
                "k": kname(k),
                "method": m,
                "aurc": float(np.mean(au)),
                "risk_at_50": float(np.mean(r50)),
            }
            payload["b"].append(b_row)
            lines.append(f"| {kname(k)} | {m} | {b_row['aurc']:.3f} | {b_row['risk_at_50']:.3f} |")
            if k is not None and m != "category_prior":
                c_row = {
                    "k": k,
                    "method": m,
                    "seeds": len(cov),
                    "target": tg,
                    "coverage": cov,
                    "risk": rk,
                }
                payload["c"].append(c_row)
                if cov:
                    lines_c.append(
                        f"| {k} | {m} | {len(cov)}/5 | {np.mean(tg):.3f} | {np.mean(cov):.1%} | "
                        f"{_fmt(rk)} |"
                    )
                else:
                    lines_c.append(f"| {k} | {m} | 0/5 | n/a | n/a | n/a |")
    return lines + lines_c + [""], payload


def main() -> int:
    lines = ["# Small label budgets and deferral (exploratory, not pre-registered)", ""]
    payload = {}
    for name, (config, jev) in BENCH.items():
        part, payload[name] = run(name, config, jev)
        lines += part
    out = Path("results/evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / "small_budgets.md").write_text("\n".join(lines) + "\n")
    (out / "small_budgets.json").write_text(json.dumps(payload, indent=1, default=float))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
