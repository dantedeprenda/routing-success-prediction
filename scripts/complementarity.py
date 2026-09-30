"""Difficulty vs complementarity in routing (preregistration/v3.md).

    uv run python scripts/complementarity.py answer    # Jev answers the pair-aware sets; cap $3
    uv run python scripts/complementarity.py analyze   # offline -> results/evidence/complementarity.*

H1: do success heads rank discordant prompts (exactly one model right) backwards? H2: does routing on
P(cheap) - P(strong) beat routing on P(cheap)? H3: do pair-aware questions help? H4: how many labeled
prompts are discordant, and how does the difference signal grow with k?
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

import extensions as ext
from blind_questions import jev_backend
from dmroute.cli import _load_dotenv
from dmroute.data import load
from dmroute.data.splits import k_shot
from dmroute.metrics import cost_quality
from dmroute.routers.base import fit_text
from generate_pair_questions import PAIRS, final_set
from second_pool import JEV_CALL_USD, costs

MAX_USD = 3.0  # pre-registered hard cap for H3 (generator + Jev)
KS = [25, 100, 400, None]
B = 2000
REFERENCE = ("deepseek-v3-0324", "gpt-5-chat")  # analyzed in v2: exploratory reference row only
BIG = 1e9  # the strong model's "score": always acceptable, so it is the fallback


def requests(cheap: str, strong: str) -> dict[str, dict]:
    qs = final_set(cheap, strong)["questions"]
    return {f"feat_{q}": {"type": "noul", "instructions": t} for q, t in qs.items()}


def generator_usd() -> float:
    total = 0.0
    for path in Path("configs/questions/pairs").glob("*.json"):
        u = json.loads(path.read_text())["usage"]
        total += (3 * u["input_tokens"] + 15 * u["output_tokens"]) / 1e6
    return total


def answer() -> int:
    _load_dotenv()
    backend, max_chars = jev_backend()
    ds = load(yaml.safe_load(Path(ext.CONFIG).read_text())["dataset"])
    spent = generator_usd()
    print(f"generator ${spent:.4f}", flush=True)
    for cheap, strong in PAIRS:
        questions = requests(cheap, strong)

        def one(ex, questions=questions):
            try:
                resp, _, cost = backend.ask(fit_text(ex.text, max_chars), questions)
                return resp.get("model"), cost, None
            except Exception as e:  # noqa: BLE001 - counted; a rerun retries uncached calls
                return None, 0.0, f"{type(e).__name__}: {e}"

        models, cost, failed = {}, 0.0, 0
        for start in range(0, len(ds.examples), 500):
            if spent + cost > MAX_USD:
                print(f"STOP: cap ${MAX_USD} reached", flush=True)
                return 1
            with ThreadPoolExecutor(6) as pool:
                for model, c, err in pool.map(one, ds.examples[start : start + 500]):
                    cost += c
                    models[model] = models.get(model, 0) + 1
                    failed += err is not None
        spent += cost
        print(f"{cheap} -> {strong}: models {models}, failed {failed}, ${cost:.4f}", flush=True)
    print(f"total ${spent:.4f} (cap ${MAX_USD})")
    return 0


def pair_features(s: ext.Setup, cheap: str, strong: str) -> tuple[np.ndarray, int]:
    """(n x 13) Jev answers to the pair-aware set, from the cache only; failures are 0.5."""
    backend, max_chars = jev_backend()
    backend.base_url = "http://127.0.0.1:9"  # offline: an uncached call fails instead of paying
    questions = requests(cheap, strong)
    x = np.full((len(s.ds.examples), len(questions)), 0.5)
    failed = 0
    for n, ex in enumerate(s.ds.examples):
        try:
            resp, _, _ = backend.ask(fit_text(ex.text, max_chars), questions)
            x[n] = [resp["answers"][q]["noul"] for q in questions]
        except Exception:  # noqa: BLE001
            failed += 1
    return x, failed


def heads(method: str, s, q: np.ndarray, lab: np.ndarray, idx: np.ndarray, x=None) -> np.ndarray:
    """(len(idx), 2) P(success) of (cheap, strong) from per-model heads trained on *lab*."""
    return np.stack([ext.predict(method, s, q[:, m], lab, idx, x=x) for m in (0, 1)], axis=1)


def route(score_lab, q_lab, c_lab, score_te, q_te, c_te, rc: float) -> np.ndarray:
    """Per-test-prompt quality: cheap model when score >= tau, tau picked on the labeled prompts
    (best labeled quality with labeled mean cost <= the strong model's)."""
    as_p = lambda sc: np.stack([sc, np.full(len(sc), BIG)], axis=1)
    r_lab, r_te = np.full(len(q_lab), rc), np.full(len(q_te), rc)
    points = cost_quality.sweep(as_p(score_lab), q_lab, c_lab, r_lab)
    ok = [p for p in points if p["cost"] <= c_lab[:, 1].mean() + 1e-12] or [
        min(points, key=lambda p: p["cost"])
    ]
    tau = max(ok, key=lambda p: (p["quality"], -p["cost"]))["tau"]
    qual, _ = cost_quality.realized(cost_quality.pick(as_p(score_te), tau), q_te, c_te, r_te)
    return qual


def pair_run(s, q, c, method: str, k, x=None) -> dict:
    """Seed-averaged per-test-prompt results for one pair, method and budget."""
    seeds = ext.SEEDS if k is not None else ext.SEEDS[:1]
    te = s.test
    disc = q[te, 0] != q[te, 1]
    d = (q[te, 0] > q[te, 1])[disc].astype(float)
    rc = JEV_CALL_USD if method != "bge_knn" and method != "tfidf_logreg" else 0.0
    out = {"q_pcheap": [], "q_diff": [], "auc_pcheap": [], "auc_diff": [], "lab_disc": []}
    for seed in seeds:
        lab = k_shot(s.train, k, seed)
        p_lab, p_te = heads(method, s, q, lab, lab, x), heads(method, s, q, lab, te, x)
        for name, sl, st in (
            ("pcheap", p_lab[:, 0], p_te[:, 0]),
            ("diff", p_lab[:, 0] - p_lab[:, 1], p_te[:, 0] - p_te[:, 1]),
        ):
            out[f"q_{name}"].append(route(sl, q[lab], c[lab], st, q[te], c[te], rc))
            out[f"auc_{name}"].append(roc_auc_score(d, st[disc]))
        out["lab_disc"].append(float((q[lab, 0] != q[lab, 1]).mean()))
    res = {key: np.mean(v, axis=0) for key, v in out.items()}
    res["disc"], res["d"] = disc, d
    return res


def boot_auc(p_by_pair: list[tuple[np.ndarray, np.ndarray]], rng) -> tuple[float, float, float]:
    """Mean over pairs of discordant AUROC; bootstrap resamples pairs and prompts within a pair."""
    per = []
    for d, score in p_by_pair:
        idx = rng.integers(0, len(d), size=(B, len(d)))
        idx[0] = np.arange(len(d))
        vals = []
        for b in range(B):
            db = d[idx[b]]
            vals.append(roc_auc_score(db, score[idx[b]]) if 0 < db.mean() < 1 else np.nan)
        per.append(np.array(vals))
    per = np.stack(per)
    pd = rng.integers(0, len(per), size=(B, len(per)))
    pd[0] = np.arange(len(per))
    boot = np.nanmean(np.take_along_axis(per, pd.T, axis=0), axis=0)
    return ext.ci(boot)


def boot_mean(diffs: list[np.ndarray], rng) -> tuple[float, float, float]:
    """Mean over pairs of mean per-prompt differences; resample pairs and prompts."""
    per = []
    for v in diffs:
        idx = rng.integers(0, len(v), size=(B, len(v)))
        idx[0] = np.arange(len(v))
        per.append(v[idx].mean(axis=1))
    per = np.stack(per)
    pd = rng.integers(0, len(per), size=(B, len(per)))
    pd[0] = np.arange(len(per))
    return ext.ci(np.nanmean(np.take_along_axis(per, pd.T, axis=0), axis=0))


def fmt(t) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def analyze() -> int:
    s = ext.Setup(offline=True)
    rng = np.random.default_rng(0)
    rows, payload, per_pair = [], {"pairs": {}}, {}
    for cheap, strong in [*PAIRS, REFERENCE]:
        q = np.stack([s.labels(cheap), s.labels(strong)], axis=1)
        c = np.stack([costs(s, cheap), costs(s, strong)], axis=1)
        xp, failed = (None, 0) if (cheap, strong) == REFERENCE else pair_features(s, cheap, strong)
        r = {}
        for k in KS:
            for method in ["jev_features", "bge_knn", "tfidf_logreg"]:
                r[(method, k)] = pair_run(s, q, c, method, k)
            if xp is not None and k in (400, None):
                r[("pair_aware", k)] = pair_run(s, q, c, "custom", k, x=xp)
        best = q[s.test].mean(axis=0).max()
        per_pair[(cheap, strong)] = (r, q[s.test], best)
        j = r[("jev_features", None)]
        payload["pairs"][f"{cheap} -> {strong}"] = {
            "reference_only": (cheap, strong) == REFERENCE,
            "failed_pair_aware_calls": failed,
            "n_discordant": int(j["disc"].sum()),
            "only_cheap": int(j["d"].sum()),
            "test_quality": q[s.test].mean(axis=0).tolist(),
            **{
                f"{m}|{ext.kname(k)}": {
                    "auc_pcheap": float(v["auc_pcheap"]),
                    "auc_diff": float(v["auc_diff"]),
                    "q_pcheap": float(v["q_pcheap"].mean()),
                    "q_diff": float(v["q_diff"].mean()),
                    "labeled_discordant_share": float(v["lab_disc"]),
                }
                for (m, k), v in r.items()
            },
        }
        print(f"  {cheap} -> {strong}: done", flush=True)

    confirm = [p for p in per_pair if p != REFERENCE]

    # H1 primary: AUROC of P(cheap) on discordant prompts, Jev features, all labels.
    h1 = boot_auc(
        [
            (per_pair[p][0][("jev_features", None)]["d"], _score(s, per_pair, p, "pcheap"))
            for p in confirm
        ],
        rng,
    )
    # H2 primary: quality(Diff) - quality(P-cheap), Jev features, all labels.
    h2 = boot_mean(
        [
            per_pair[p][0][("jev_features", None)]["q_diff"]
            - per_pair[p][0][("jev_features", None)]["q_pcheap"]
            for p in confirm
        ],
        rng,
    )
    h2_best = boot_mean(
        [per_pair[p][0][("jev_features", None)]["q_diff"] - per_pair[p][2] for p in confirm], rng
    )
    h3 = {
        ext.kname(k): boot_mean(
            [
                per_pair[p][0][("pair_aware", k)]["q_diff"]
                - per_pair[p][0][("jev_features", k)]["q_diff"]
                for p in confirm
            ],
            rng,
        )
        for k in (400, None)
    }
    payload.update({"H1": h1, "H2": h2, "H2_diff_vs_best_single": h2_best, "H3a": h3})

    lines = [
        "# Difficulty vs complementarity (pre-registered, preregistration/v3.md)",
        "",
        (
            f"**H1** (discordant AUROC of P(cheap), Jev features, all labels, mean over 6 pairs): "
            f"**{fmt(h1)}**; supported if the upper bound < 0.5: "
            f"**{'SUPPORTED' if h1[2] < 0.5 else 'NOT SUPPORTED'}**."
        ),
        "",
        (
            f"**H2** (routing quality, Diff minus P-cheap, Jev features, all labels, mean over pairs): "
            f"**{fmt(h2)}**; supported if the lower bound > 0: "
            f"**{'SUPPORTED' if h2[1] > 0 else 'NOT SUPPORTED'}**. Diff minus best single model: "
            f"{fmt(h2_best)}."
        ),
        "",
        "**H3a** (Diff policy quality, pair-aware minus v1 questions): "
        + "; ".join(f"k={k}: {fmt(v)}" for k, v in h3.items())
        + ".",
        "",
        "## Per pair (Jev v1 features unless noted; point estimates, mean over label seeds)",
        "",
        (
            "| pair | discordant (only cheap) | k | AUC P(cheap) | AUC diff | quality P-cheap | "
            "quality Diff | best single | labeled discordant share |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (cheap, strong), (r, qt, best) in per_pair.items():
        tag = " (reference, exploratory)" if (cheap, strong) == REFERENCE else ""
        for (m, k), v in r.items():
            if m not in ("jev_features", "pair_aware"):
                continue
            label = f"{cheap} -> {strong}{tag}" + (" [pair-aware]" if m == "pair_aware" else "")
            rows.append(
                f"| {label} | {int(v['disc'].sum())} ({int(v['d'].sum())}) | {ext.kname(k)} | "
                f"{v['auc_pcheap']:.3f} | {v['auc_diff']:.3f} | {v['q_pcheap'].mean():.3f} | "
                f"{v['q_diff'].mean():.3f} | {best:.3f} | {v['lab_disc']:.2f} |"
            )
    lines += rows
    lines += [
        "",
        "## Text heads (all labels): discordant AUROC of P(cheap) and of the difference",
        "",
        "| pair | bge_knn P(cheap) | bge_knn diff | tfidf P(cheap) | tfidf diff |",
        "|---|---|---|---|---|",
    ]
    for (cheap, strong), (r, _, _) in per_pair.items():
        b, t = r[("bge_knn", None)], r[("tfidf_logreg", None)]
        lines.append(
            f"| {cheap} -> {strong} | {b['auc_pcheap']:.3f} | {b['auc_diff']:.3f} | "
            f"{t['auc_pcheap']:.3f} | {t['auc_diff']:.3f} |"
        )
    out = Path("results/evidence")
    (out / "complementarity.md").write_text("\n".join(lines) + "\n")
    (out / "complementarity.json").write_text(json.dumps(payload, indent=1, default=float))
    print("\n".join(lines))
    return 0


def _score(s, per_pair, pair, kind: str) -> np.ndarray:
    """Discordant-prompt scores at k = all for the H1 bootstrap (recomputed; heads are
    deterministic at k = all)."""
    cheap, strong = pair
    q = np.stack([s.labels(cheap), s.labels(strong)], axis=1)
    lab = k_shot(s.train, None, ext.SEEDS[0])
    p = heads("jev_features", s, q, lab, s.test)
    disc = q[s.test, 0] != q[s.test, 1]
    return (p[:, 0] if kind == "pcheap" else p[:, 0] - p[:, 1])[disc]


if __name__ == "__main__":
    sys.exit({"answer": answer, "analyze": analyze}[sys.argv[1]]())
