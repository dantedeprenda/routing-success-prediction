"""Second model pool with more headroom (preregistration/v2.md, Part 2). Offline, $0.

    uv run python scripts/second_pool.py    # -> results/evidence/second_pool.{md,json}

Pool (frozen by the pre-registration's rule; see scripts/headroom.py): cheap = deepseek-v3-0324,
strong = gpt-5-chat. Routing sends a prompt to the cheap model when P(cheap succeeds) >= tau,
otherwise to the strong one. tau is chosen on the k labeled prompts only: the best labeled-set
quality whose labeled mean cost is <= the strong model's labeled mean cost. It is then applied
unchanged to the test split. The headroom simulation at the end is exploratory by design.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

import extensions as ext
from dmroute.data.splits import k_shot
from dmroute.metrics import cost_quality

CHEAP, STRONG = "deepseek-v3-0324", "gpt-5-chat"
ORIGINAL = ("deepseek-v3.1-terminus", "gpt-5")  # the paper's binary pair (0, 2)
KS = [25, None]
METHODS = ["jev_features", "bge_knn", "tfidf_logreg"]
JEV_CALL_USD = 0.00014  # one fan-out Jev call per prompt
B = 2000


def costs(s: ext.Setup, model: str) -> np.ndarray:
    """Per-prompt cost of *model* from the release's latest run files."""
    root = Path(s.cfg["dataset"]["root"])
    spend = {}
    for dset, split in {tuple(ex.id.split("/")[:2]) for ex in s.ds.examples}:
        f = max((root / dset / split / model).glob("*.json"))
        for r in json.loads(f.read_text())["records"]:
            spend[f"{dset}/{split}/{r['index']}"] = float(r.get("cost") or 0.0)
    return np.array([spend[ex.id] for ex in s.ds.examples])


def probs(method: str, s, q: np.ndarray, lab: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(len(idx), 2) P(success) for (cheap, strong), heads trained on labeled prompts *lab*."""
    return np.stack([ext.predict(method, s, q[:, m], lab, idx) for m in (0, 1)], axis=1)


def choose_tau(p_lab, q_lab, c_lab, r_lab) -> float:
    budget = c_lab[:, 1].mean()
    points = cost_quality.sweep(p_lab, q_lab, c_lab, r_lab)
    ok = [pt for pt in points if pt["cost"] <= budget + 1e-12] or [
        min(points, key=lambda p: p["cost"])
    ]
    return max(ok, key=lambda pt: (pt["quality"], -pt["cost"]))["tau"]


def run(s, q, c, method, k, seed):
    """Per-test-prompt (quality, cost) at the labeled-set operating point, plus P on test."""
    lab = k_shot(s.train, k, seed)
    rc = JEV_CALL_USD if method == "jev_features" else 0.0
    p_lab = probs(method, s, q, lab, lab)
    tau = choose_tau(p_lab, q[lab], c[lab], np.full(len(lab), rc))
    p_te = probs(method, s, q, lab, s.test)
    qual, cost = cost_quality.realized(
        cost_quality.pick(p_te, tau), q[s.test], c[s.test], np.full(len(s.test), rc)
    )
    return qual, cost, p_te, tau


def hier_ci(diffs: np.ndarray, rng) -> tuple[float, float, float]:
    """(seeds x prompts) differences: resample seeds and prompts together."""
    n_s, n_p = diffs.shape
    sd = rng.integers(0, n_s, size=(B, n_s))
    pd = rng.integers(0, n_p, size=(B, n_p))
    boot = np.array([diffs[sd[b]][:, pd[b]].mean() for b in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def headroom(s, cheap: str, strong: str, rng, reps: int = 200) -> list[dict]:
    """Exploratory: simulated P(cheap succeeds) of a given pooled AUROC on the test split.

    Scores are y * d + N(0, 1) over the whole test split (pooled, not within category), with
    d = sqrt(2) * Phi^-1(AUROC). The noise is independent of whether the strong model succeeds,
    so this is an upper bound for predictors that mostly read difficulty. tau is
    chosen on the test split itself (optimistic): best quality at cost <= the strong model's.
    """
    y = np.stack([s.labels(cheap), s.labels(strong)], axis=1)[s.test]
    c = np.stack([costs(s, cheap), costs(s, strong)], axis=1)[s.test]
    best = y.mean(axis=0).max()
    out = []
    for a in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        d = np.sqrt(2) * norm.ppf(a)
        gains = []
        for _ in range(reps):
            score = y[:, 0] * d + rng.standard_normal(len(y))
            p = np.stack([norm.cdf(score - d / 2), np.ones(len(y))], axis=1)
            points = cost_quality.sweep(p, y, c, np.zeros(len(y)))
            ok = [pt for pt in points if pt["cost"] <= c[:, 1].mean() + 1e-12]
            gains.append(max(pt["quality"] for pt in ok) - best)
        gains = np.array(gains)
        out.append({"auroc": a, "gain": float(gains.mean()), "p05": float(np.percentile(gains, 5))})
    return out


def main() -> int:
    ext.KS = KS
    s = ext.Setup(offline=True)
    rng = np.random.default_rng(0)
    q = np.stack([s.labels(CHEAP), s.labels(STRONG)], axis=1)
    c = np.stack([costs(s, CHEAP), costs(s, STRONG)], axis=1)
    qt, ct = q[s.test], c[s.test]
    singles = {
        CHEAP: (qt[:, 0].mean(), ct[:, 0].mean()),
        STRONG: (qt[:, 1].mean(), ct[:, 1].mean()),
    }
    best = int(np.argmax([qt[:, 0].mean(), qt[:, 1].mean()]))
    payload = {"singles": singles, "best_single": [CHEAP, STRONG][best], "results": {}}
    lines = [
        "# Second model pool (pre-registered, preregistration/v2.md Part 2)",
        "",
        f"Pool: {CHEAP} (cheap) -> {STRONG} (strong). Test split, {len(s.test)} prompts.",
        "",
        "| model | test quality | test cost |",
        "|---|---|---|",
        *(f"| {m} | {v[0]:.3f} | ${v[1]:.4f} |" for m, v in singles.items()),
        f"| oracle | {np.maximum(qt[:, 0], qt[:, 1]).mean():.3f} | - |",
        "",
        (
            "| method | k | quality | cost | d quality vs best single [95% CI] | d cost vs strong "
            "[95% CI] | random at matched cost | within AUROC (cheap / strong) |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    draws = np.arange(len(s.test))[None, :]
    for method in METHODS:
        for k in KS:
            seeds = ext.SEEDS if k is not None else ext.SEEDS[:1]
            quals, cst, aucs = [], [], []
            for seed in seeds:
                qual, cost, p_te, _ = run(s, q, c, method, k, seed)
                quals.append(qual)
                cst.append(cost)
                aucs.append(
                    [
                        float(ext.within_rows(p_te[:, m], q[s.test, m], s.cats[s.test], draws)[0])
                        for m in (0, 1)
                    ]
                )
            quals, cst = np.stack(quals), np.stack(cst)
            dq = hier_ci(quals - qt[:, best][None, :], rng)
            dc = hier_ci(cst - ct[:, 1][None, :], rng)
            frac = np.clip(
                (cst.mean() - ct[:, 0].mean()) / (ct[:, 1].mean() - ct[:, 0].mean()), 0, 1
            )
            rnd = (1 - frac) * qt[:, 0].mean() + frac * qt[:, 1].mean()
            auc = np.nanmean(np.array(aucs), axis=0)
            kn = ext.kname(k)
            payload["results"][f"{method}|{kn}"] = {
                "quality": float(quals.mean()),
                "cost": float(cst.mean()),
                "dq_vs_best_single": dq,
                "dc_vs_strong": dc,
                "random_matched_quality": float(rnd),
                "within_auroc": auc.tolist(),
            }
            lines.append(
                f"| {method} | {kn} | {quals.mean():.3f} | ${cst.mean():.4f} | "
                f"{dq[0]:+.3f} [{dq[1]:+.3f}, {dq[2]:+.3f}] | "
                f"{dc[0]:+.4f} [{dc[1]:+.4f}, {dc[2]:+.4f}] | {rnd:.3f} | "
                f"{auc[0]:.3f} / {auc[1]:.3f} |"
            )
    # Exploratory diagnostic: the same heads with tau picked on the test split (optimistic).
    # Separates "the ranking is too weak" from "25 labels cannot place the threshold".
    lines += [
        "",
        (
            "Exploratory: tau picked on the test split instead of the labeled set (optimistic), "
            "best quality at cost <= strong, minus the best single model:"
        ),
        "",
        "| method | k | gain (mean over seeds) |",
        "|---|---|---|",
    ]
    payload["test_tau_gain"] = {}
    for method in METHODS:
        for k in KS:
            gains = []
            for seed in ext.SEEDS if k is not None else ext.SEEDS[:1]:
                lab = k_shot(s.train, k, seed)
                rc = JEV_CALL_USD if method == "jev_features" else 0.0
                p_te = probs(method, s, q, lab, s.test)
                points = cost_quality.sweep(p_te, qt, ct, np.full(len(s.test), rc))
                ok = [pt for pt in points if pt["cost"] <= ct[:, 1].mean() + 1e-12]
                gains.append(max(pt["quality"] for pt in ok) - qt[:, best].mean())
            payload["test_tau_gain"][f"{method}|{ext.kname(k)}"] = float(np.mean(gains))
            lines.append(f"| {method} | {ext.kname(k)} | {np.mean(gains):+.3f} |")
    # Exploratory diagnostic: routing gains come only from discordant prompts (exactly one model
    # right). AUROC of p_cheap - p_strong for "only cheap right" vs "only strong right".
    disc = qt[:, 0] != qt[:, 1]
    only_cheap = (qt[:, 0] > qt[:, 1])[disc].astype(float)
    lines += [
        "",
        (
            f"Exploratory: discordant test prompts (exactly one model right): {int(disc.sum())} "
            f"({int(only_cheap.sum())} only cheap right). AUROC of P(cheap) - P(strong), and of "
            "P(cheap) alone, for telling them apart (0.5 = no routing signal):"
        ),
        "",
        "| method | k | AUROC of P(cheap) - P(strong) | AUROC of P(cheap) |",
        "|---|---|---|---|",
    ]
    payload["discordant_auroc"] = {}
    for method in METHODS:
        for k in KS:
            a_diff, a_cheap = [], []
            for seed in ext.SEEDS if k is not None else ext.SEEDS[:1]:
                p_te = probs(method, s, q, k_shot(s.train, k, seed), s.test)[disc]
                a_diff.append(roc_auc_score(only_cheap, p_te[:, 0] - p_te[:, 1]))
                a_cheap.append(roc_auc_score(only_cheap, p_te[:, 0]))
            payload["discordant_auroc"][f"{method}|{ext.kname(k)}"] = [
                float(np.mean(a_diff)),
                float(np.mean(a_cheap)),
            ]
            lines.append(
                f"| {method} | {ext.kname(k)} | {np.mean(a_diff):.3f} | {np.mean(a_cheap):.3f} |"
            )
    prim = payload["results"]["jev_features|25"]["dq_vs_best_single"]
    lines[4:4] = [
        (
            f"**Primary endpoint** (k = 25, jev:features quality minus best single): "
            f"**{prim[0]:+.3f} [{prim[1]:+.3f}, {prim[2]:+.3f}]**; success criterion (lower bound > 0): "
            f"**{'MET' if prim[1] > 0 else 'NOT MET'}**."
        ),
        "",
    ]
    lines += [
        "",
        "## Headroom simulation (exploratory by design)",
        "",
        (
            "Simulated P(cheap succeeds) with a given pooled AUROC; tau picked on test "
            "(optimistic). Gain = best quality at cost <= strong, minus the best single model."
        ),
        "",
        "| AUROC | new pool gain (mean / 5th pct) | original pair gain (mean / 5th pct) |",
        "|---|---|---|",
    ]
    new, orig = headroom(s, CHEAP, STRONG, rng), headroom(s, *ORIGINAL, rng)
    payload["headroom"] = {"new": new, "original": orig}
    for a, b in zip(new, orig, strict=True):
        lines.append(
            f"| {a['auroc']:.2f} | {a['gain']:+.3f} / {a['p05']:+.3f} | "
            f"{b['gain']:+.3f} / {b['p05']:+.3f} |"
        )
    out = Path("results/evidence")
    (out / "second_pool.md").write_text("\n".join(lines) + "\n")
    (out / "second_pool.json").write_text(json.dumps(payload, indent=1, default=float))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
