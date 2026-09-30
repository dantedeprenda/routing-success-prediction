"""Head-to-head on a fixed test sample: LLM-router judges vs decision-model routers.

    uv run python scripts/judge_sample.py configs/llmrouterbench_test.yaml --n 500 --max-usd 4
    uv run python scripts/judge_sample.py configs/twinrouterbench_test.yaml --max-usd 2.5

A frontier judge on every prompt (train + test, as the full pipeline asks it) costs ~$30; this
runs it only on a stratified sample of the seed-42 test split and compares every router on
exactly those prompts. The judge needs no labels, so nothing is lost by skipping train; the
fallback thresholds are fixed values, not tuned. Decision-model and baseline predictions come
from the full run's saved test outputs (results/<name>/seed42_kNone/*.jsonl, raw variant):
run the config with dmroute first. Writes results/judge_sample/<name>.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from dmroute.backends.cache import CallCache
from dmroute.backends.chat import AnthropicDirect, ChatBackend
from dmroute.cli import _load_dotenv
from dmroute.data import load
from dmroute.data.splits import grouped_split
from dmroute.metrics import cost_quality, stats
from dmroute.policy import pick
from dmroute.routers.base import fit_text
from dmroute.routers.llm_judge import (
    MAX_MESSAGE_CHARS,
    RESPONSE_FORMAT,
    ROUTING_TIMEOUT_S,
    LLMJudge,
    system_prompt,
)

# Prices from OpenRouter, 2026-09-26 ($ per M input / output tokens).
JUDGES = {
    "judge-sonnet": ("anthropic/claude-sonnet-5", 2.0, 10.0),
    "judge-luna": ("openai/gpt-6-luna", 0.10, 0.50),
}
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
CHUNK = 40


def test_split(ds, split: dict) -> np.ndarray:
    """Seed 42's held-out test indices, exactly as experiment.run builds them."""
    _, test = grouped_split(ds, split.get("test_frac", 0.3), split.get("test_seed", 42))
    if split.get("exclude_from_test"):
        seen = set(Path(split["exclude_from_test"]).read_text().split())
        test = test[~np.array([ds.examples[i].id in seen for i in test], dtype=bool)]
    return test


def stratified_sample(ds, test: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """``n`` test indices with each dataset's share preserved (largest remainder)."""
    if n is None or n >= len(test):
        return test
    cats = np.array([ds.examples[i].dataset for i in test])
    names, counts = np.unique(cats, return_counts=True)
    exact = counts / counts.sum() * n
    quota = np.floor(exact).astype(int)
    quota[np.argsort(exact - quota)[::-1][: n - quota.sum()]] += 1
    rng = np.random.default_rng(seed)
    picked = [rng.choice(test[cats == c], q, replace=False) for c, q in zip(names, quota)]
    return np.sort(np.concatenate(picked))


def run_judge(
    label, model, pin, pout, ds, idx, cache, budget, rpm, model_ids=None, via="openrouter"
) -> tuple[list, float]:
    """Judge outputs for ``idx`` in chunks; stops before a chunk once new spend passes budget.

    Calls that still fail after the backend's retries (errors are never cached) are retried in
    up to two more passes, so a transient 429 is not scored as a routing decision.
    """
    if via == "anthropic":  # same cache key; Messages API instead of OpenRouter
        backend = AnthropicDirect(
            model=model,
            anthropic_model=model.split("/", 1)[1],
            price_in=pin,
            price_out=pout,
            rpm=rpm,
            cache=cache,
        )
    else:
        backend = ChatBackend(model=model, price_in=pin, price_out=pout, rpm=rpm, cache=cache)
    judge = LLMJudge(backend, label=label, workers=8, model_ids=model_ids)
    system = system_prompt(model_ids or ds.model_names)

    def cached(i):
        user = fit_text(ds.examples[i].text, MAX_MESSAGE_CHARS)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        request = {"model": model, "temperature": 0, "messages": msgs}
        request["response_format"] = RESPONSE_FORMAT  # LLMJudge is structured by default
        return cache.get(backend.backend_id, request) is not None

    outputs, spent = [], 0.0
    for start in range(0, len(idx), CHUNK):
        if spent > budget:
            raise SystemExit(f"{label}: stopped at ${spent:.2f} new spend (budget ${budget})")
        part = idx[start : start + CHUNK]
        fresh = np.array([not cached(i) for i in part])
        out = judge.predict(ds, part)
        for _ in range(2):
            failed = [j for j, o in enumerate(out) if "error" in o.raw]
            if not failed:
                break
            for j, o in zip(failed, judge.predict(ds, part[failed])):
                out[j] = o
        spent += sum(o.router_cost for o, f in zip(out, fresh) if f)
        outputs.extend(out)
        print(f"  {label}: {start + len(part)}/{len(idx)} new spend ${spent:.2f}", flush=True)
    return outputs, spent


def saved_outputs(results_dir: Path, ids: list[str]) -> dict[str, dict]:
    """Per-router arrays for ``ids`` from the full run's seed-42, all-label test predictions."""
    out = {}
    for f in sorted((results_dir / "seed42_kNone").glob("*.jsonl")):
        rows = {r["id"]: r for r in map(json.loads, f.open())}
        if not all(i in rows for i in ids):
            continue
        rs = [rows[i] for i in ids]
        out[f.stem] = {
            "p": np.array([r["p"] for r in rs], float),
            "conf": np.array([r["confidence"] if r["confidence"] is not None else 1.0 for r in rs]),
            "rcost": np.array([r["router_cost"] for r in rs], float),
            "lat": np.array([r["router_latency_s"] for r in rs], float),
        }
    return out


def taus(results_dir: Path) -> dict[str, dict[str, float]]:
    """Operating-point thresholds each router chose on its labeled train set (raw variant)."""
    recs = json.loads((results_dir / "metrics.json").read_text())
    out = {}
    for r in recs:
        if r.get("seed") == 42 and r.get("k") is None and r.get("variant") == "raw":
            name = "".join(ch if ch.isalnum() or ch in "-_.@" else "_" for ch in r["router"])
            out[name] = {k: v["tau"] for k, v in r["operating_points"].items()}
    return out


def point(choice, q, c, rcost, lat, extra=None) -> dict:
    qual, cost = cost_quality.realized(choice, q, c, rcost)
    return {
        "quality": float(qual.mean()),
        "cost": float(cost.mean()),
        "router_usd_per_call": float(rcost.mean()),
        "latency_p50": float(np.percentile(lat, 50)),
        "latency_p95": float(np.percentile(lat, 95)),
        "share": np.bincount(choice, minlength=q.shape[1]).tolist(),
        "_qual": qual,
        "_cost": cost,
        **(extra or {}),
    }


def main() -> int:
    _load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=None, help="sample size (default: whole test split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-usd", type=float, required=True, help="new spend cap per judge")
    ap.add_argument("--primary", default=None, help="cascade primary, e.g. jev_choice")
    ap.add_argument("--rpm", type=float, default=18, help="requests/min per judge model")
    ap.add_argument(
        "--sonnet-via",
        choices=["openrouter", "anthropic"],
        default="openrouter",
        help="provider for judge-sonnet (anthropic: ANTHROPIC_API_KEY, Messages API)",
    )
    ap.add_argument("--anthropic-rpm", type=float, default=120, help="rpm cap on the direct API")
    ap.add_argument(
        "--timeout-s",
        type=float,
        default=ROUTING_TIMEOUT_S,
        help="Omnigent's judge timeout: slower calls count as no routing (0 disables)",
    )
    ap.add_argument(
        "--model-ids",
        default=None,
        help="comma-separated names the judges see, in pool order (default: pool names)",
    )
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_name = cfg["name"] + args.tag
    results_dir = Path(cfg.get("output_dir", f"results/{cfg['name']}"))
    out_dir = Path("results/judge_sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = CallCache(cfg.get("cache", ".cache/calls.sqlite"))
    ds = load(cfg["dataset"])
    model_ids = args.model_ids.split(",") if args.model_ids else None
    idx = stratified_sample(ds, test_split(ds, cfg.get("split", {})), args.n, args.seed)
    ids = [ds.examples[i].id for i in idx]
    (out_dir / f"{cfg['name']}_ids.txt").write_text("\n".join(ids) + "\n")
    q, c = ds.quality_matrix()[idx], ds.cost_matrix()[idx]
    k = q.shape[1]
    tiered = "target_tier_id" in ds.examples[idx[0]].meta
    if tiered:
        target = np.array([ds.examples[i].meta["target_tier_id"] for i in idx])
        groups = [ds.examples[i].group for i in idx]
    print(f"{cfg['name']}: {len(idx)} sampled test examples")

    rows: dict[str, dict] = {}
    for m in range(k):
        rows[f"always:{ds.model_names[m]}"] = point(
            np.full(len(idx), m), q, c, np.zeros(len(idx)), np.zeros(len(idx))
        )

    def direct(label: str) -> bool:
        return label == "judge-sonnet" and args.sonnet_via == "anthropic"

    judges = {}
    with ThreadPoolExecutor(len(JUDGES)) as pool:  # separate per-model rate limits
        runs = {
            label: pool.submit(
                run_judge,
                label,
                *spec,
                ds,
                idx,
                cache,
                args.max_usd,
                args.anthropic_rpm if direct(label) else args.rpm,
                model_ids,
                "anthropic" if direct(label) else "openrouter",
            )
            for label, spec in JUDGES.items()
        }
    for label, fut in runs.items():
        outs, spent = fut.result()
        errors = sum("error" in o.raw for o in outs)
        choice = np.array([o.choice for o in outs])
        parsed = np.array([o.raw.get("parsed", False) for o in outs])
        lat = np.array([o.router_latency_s for o in outs])
        # Omnigent's judge fails open after ROUTING_REQUEST_TIMEOUT_S: no routing, so the
        # session keeps its default model, taken as the strongest (as for judge errors).
        timed_out = (lat > args.timeout_s) if args.timeout_s else np.zeros(len(outs), bool)
        choice = np.where(timed_out, k - 1, choice)
        judges[label] = choice
        rows[label] = point(
            choice,
            q,
            c,
            np.array([o.router_cost for o in outs]),
            np.minimum(lat, args.timeout_s) if args.timeout_s else lat,
            {
                "unparsed": int((~parsed).sum()) - errors,
                "errors": errors,
                "timeouts": int(timed_out.sum()),
                "new_spend": spent,
            },
        )

    saved, tau = saved_outputs(results_dir, ids), taus(results_dir)
    curves = {}
    for name, s in saved.items():
        if name.startswith(("always", "oracle", "random")):
            continue
        cheap_ok = (q[:, 0] >= 0.5).astype(float)
        auroc = stats.bootstrap_auroc(s["p"][:, 0], cheap_ok)
        pts = cost_quality.sweep(s["p"], q, c, s["rcost"])
        curves[name] = cost_quality.hull([(p["cost"], p["quality"]) for p in pts])
        for op in ("budget_mid", "match_frac"):
            if name in tau and op in tau[name]:
                rows[f"{name}@{op}"] = point(
                    pick(s["p"], tau[name][op]),
                    q,
                    c,
                    s["rcost"],
                    s["lat"],
                    {"auroc_cheap_ok": auroc["auroc"], "auroc_ci95": auroc["ci95"]},
                )

    primary = args.primary or next((n for n in saved if n.endswith("choice") and "jev" in n), None)
    if primary and primary in tau:
        s = saved[primary]
        base = pick(s["p"], tau[primary]["match_frac"])
        for label, jchoice in judges.items():
            j = rows[label]
            for t in THRESHOLDS:
                defer = s["conf"] < t
                choice = np.where(defer, jchoice, base)
                rcost = s["rcost"] + np.where(defer, j["router_usd_per_call"], 0.0)
                lat = s["lat"] + np.where(defer, j["latency_p50"], 0.0)
                rows[f"{primary}>{label}@{t}"] = point(
                    choice, q, c, rcost, lat, {"deferred": float(defer.mean())}
                )

    lines = [
        f"# Judge sample: {out_name}",
        "",
        (
            f"{len(idx)} stratified test examples (seed {args.seed}; ids in "
            f"`{cfg['name']}_ids.txt`). Decision routers use the thresholds they picked on "
            "labeled train data (raw variant); judges and cascades need none. "
            "Cost = model + router."
        ),
        f"Judges saw model ids: {', '.join(model_ids or ds.model_names)}.",
        "",
        (
            "| router | quality | cost | router $/call | latency p50 / p95 s | AUROC cheap-ok | "
            "Jev-curve quality at this cost | notes |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    jev_curve = curves.get(primary)
    for name, r in rows.items():
        at = (
            f"{float(cost_quality.hull_value(jev_curve, r['cost'])):.3f}"
            if jev_curve and not name.startswith(primary or "\0")
            else ""
        )
        au = (
            f"{r['auroc_cheap_ok']:.2f} [{r['auroc_ci95'][0]:.2f}, {r['auroc_ci95'][1]:.2f}]"
            if "auroc_cheap_ok" in r
            else ""
        )
        notes = []
        if "unparsed" in r:
            notes.append(
                f"unparsed {r['unparsed']}, errors {r['errors']}, "
                f"timeouts >{args.timeout_s:g}s {r['timeouts']}, new spend ${r['new_spend']:.2f}"
            )
        if "deferred" in r:
            notes.append(f"deferred {r['deferred']:.0%}")
        lines.append(
            f"| {name} | {r['quality']:.3f} | ${r['cost']:.5f} | ${r['router_usd_per_call']:.5f} | "
            f"{r['latency_p50']:.2f} / {r['latency_p95']:.2f} | {au} | {at} | {'; '.join(notes)} |"
        )

    ref = f"{primary}@match_frac"
    if ref in rows:
        lines += ["", f"Paired bootstrap vs {ref} (mean diff, 95% CI):", ""]
        for label in judges:
            dq = stats.paired_bootstrap(rows[label]["_qual"], rows[ref]["_qual"])
            dc = stats.paired_bootstrap(rows[label]["_cost"], rows[ref]["_cost"])
            lines.append(
                f"- {label}: quality {dq['mean_diff']:+.3f} [{dq['ci95'][0]:+.3f}, "
                f"{dq['ci95'][1]:+.3f}], cost {dc['mean_diff']:+.5f} "
                f"[{dc['ci95'][0]:+.5f}, {dc['ci95'][1]:+.5f}]"
            )
    if primary in saved:
        lines += [
            "",
            f"Fallback vs {primary} alone at the same cost: quality minus {primary}'s cost-quality",
            "hull at the fallback's cost, with Jev's threshold chosen on the test set itself",
            "(optimistic for Jev, so conservative for the fallback). Bootstrap over examples.",
            "",
        ]
        s = saved[primary]

        def hull_at(sel: np.ndarray, cost: float) -> float:
            pts = cost_quality.sweep(s["p"][sel], q[sel], c[sel], s["rcost"][sel])
            h = cost_quality.hull([(pt["cost"], pt["quality"]) for pt in pts])
            return float(cost_quality.hull_value(h, cost))

        rng = np.random.default_rng(0)
        draws = [np.arange(len(idx))] + [rng.integers(0, len(idx), len(idx)) for _ in range(300)]
        for name in [n for n in rows if n.startswith(f"{primary}>")]:
            fq, fc = rows[name]["_qual"], rows[name]["_cost"]
            diffs = [fq[d].mean() - hull_at(d, fc[d].mean()) for d in draws]
            lo, hi = np.percentile(diffs[1:], [2.5, 97.5])
            rows[name]["vs_primary_hull"] = {"diff": diffs[0], "ci95": [lo, hi]}
            lines.append(f"- {name}: {diffs[0]:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    if tiered:
        lines += ["", "Tier metrics (RowPass / TrajPass):", ""]
        for label, jchoice in judges.items():
            t = cost_quality.tier_metrics(jchoice, target, groups)
            lines.append(f"- {label}: {t['row_pass']:.3f} / {t['traj_pass']:.3f}")
        if primary in tau:
            base = pick(saved[primary]["p"], tau[primary]["match_frac"])
            t = cost_quality.tier_metrics(base, target, groups)
            lines.append(f"- {primary}@match_frac: {t['row_pass']:.3f} / {t['traj_pass']:.3f}")

    md = "\n".join(lines) + "\n"
    (out_dir / f"{out_name}.md").write_text(md)
    clean = {n: {k2: v for k2, v in r.items() if not k2.startswith("_")} for n, r in rows.items()}
    (out_dir / f"{out_name}.json").write_text(json.dumps(clean, indent=1))
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
