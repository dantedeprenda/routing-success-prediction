"""Run one experiment config end to end: load, split, predict (cached), evaluate, save.

Protocol: grouped train/test split per seed; routers learn only
from the k-shot labeled subset of train; thresholds, recalibration and operating
points are chosen on that labeled subset and applied unchanged to test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dmroute.backends.cache import CallCache
from dmroute.data import load
from dmroute.data.splits import grouped_split, k_shot
from dmroute.metrics import calibration, cost_quality, stats
from dmroute.policy import Recalibrator, cascade, p_matrix, pick, router_costs
from dmroute.questions import VERSION, fingerprint
from dmroute.routers import build_routers
from dmroute.schema import Dataset, RouterOutput


def _is_discrete(outputs: list[RouterOutput]) -> bool:
    return bool(np.isin(p_matrix(outputs), [0.0, 1.0]).all())


def _select_taus(p, q, c, rcost, taus=None, match_frac: float = 0.95) -> dict[str, float]:
    """Operating points chosen on labeled data only.

    match_best: cheapest tau whose quality >= the best single model's.
    match_frac: cheapest tau whose quality >= match_frac x the best single model's.
    budget_mid: best quality whose cost <= always-middle-model's cost.
    """
    singles = cost_quality.single_model_points(q, c)
    best_q = max(s[1] for s in singles)
    mid_cost = singles[len(singles) // 2][0]
    points = cost_quality.sweep(p, q, c, rcost, taus)
    best_point = max(points, key=lambda pt: (pt["quality"], -pt["cost"]))

    def cheapest_reaching(target: float) -> float:
        # Unreachable targets fall back to the router's own best point, never to a model
        # the router did not pick.
        ok = [pt for pt in points if pt["quality"] >= target - 1e-12] or [best_point]
        return min(ok, key=lambda pt: pt["cost"])["tau"]

    budget = [pt for pt in points if pt["cost"] <= mid_cost]
    if not budget:
        budget = [min(points, key=lambda pt: pt["cost"])]
    return {
        "match_best": cheapest_reaching(best_q),
        "match_frac": cheapest_reaching(match_frac * best_q),
        "budget_mid": max(budget, key=lambda pt: (pt["quality"], -pt["cost"]))["tau"],
    }


def evaluate(
    outputs_lab: list[RouterOutput],
    outputs_test: list[RouterOutput],
    ds: Dataset,
    lab: np.ndarray,
    test: np.ndarray,
    binary_pairs: list[list[int]],
    match_frac: float = 0.95,
) -> dict[str, Any]:
    q, c = ds.quality_matrix(), ds.cost_matrix()
    q_lab, c_lab, q_te, c_te = q[lab], c[lab], q[test], c[test]
    p_lab, p_te = p_matrix(outputs_lab), p_matrix(outputs_test)
    r_lab, r_te = router_costs(outputs_lab), router_costs(outputs_test)

    # A router that only names a model is one operating point, not a curve.
    taus = [0.5] if _is_discrete(outputs_test) else None
    if outputs_test[0].raw.get("oracle"):
        points = cost_quality.oracle_sweep(q_te, c_te)
    else:
        points = cost_quality.sweep(p_te, q_te, c_te, r_te, taus)
    h = cost_quality.hull([(pt["cost"], pt["quality"]) for pt in points])
    singles = cost_quality.single_model_points(q_te, c_te)
    c_min, c_max = min(s[0] for s in singles), max(s[0] for s in singles)
    best_m = max(range(len(singles)), key=lambda m: (singles[m][1], -singles[m][0]))

    res: dict[str, Any] = {
        "aiq": cost_quality.aiq_linear(h, c_min, c_max),
        "aiq_log": cost_quality.aiq(h, c_min, c_max),
        **cost_quality.perf_gain_cost_save(points, q_te, c_te),
        "curve": [{k: pt[k] for k in ("tau", "quality", "cost")} for pt in points],
        "operating_points": {},
    }
    tiered = "target_tier_id" in ds.examples[test[0]].meta
    if tiered:
        target = np.array([ds.examples[i].meta["target_tier_id"] for i in test])
        groups = [ds.examples[i].group for i in test]
    for label, tau in _select_taus(p_lab, q_lab, c_lab, r_lab, taus, match_frac).items():
        choice = pick(p_te, tau)
        qual, cost = cost_quality.realized(choice, q_te, c_te, r_te)
        best_qual, best_cost = q_te[:, best_m], c_te[:, best_m]
        res["operating_points"][label] = {
            "tau": float(tau),
            "quality": float(qual.mean()),
            "cost": float(cost.mean()),
            "share": np.bincount(choice, minlength=len(ds.pool)).tolist(),
            "vs_best_single_quality": stats.paired_bootstrap(qual, best_qual),
            "vs_best_single_cost": stats.paired_bootstrap(cost, best_cost),
            "mcnemar_vs_best_single": stats.mcnemar_exact(qual, best_qual),
        }
        if tiered:
            res["operating_points"][label]["tier"] = cost_quality.tier_metrics(
                choice, target, groups
            )
    if not _is_discrete(outputs_test):
        res["calibration"] = calibration.calibration_report(p_te, q_te)
        res["binary"] = {f"{w}-{s}": cost_quality.apgr(p_te, q_te, w, s) for w, s in binary_pairs}
        res["by_category"] = calibration.by_category(
            p_te, q_te, [ds.examples[i].category for i in test]
        )
    if tiered:
        res["tier"] = cost_quality.tier_metrics(pick(p_te, 0.5), target, groups)

    lat = np.array([o.router_latency_s for o in outputs_test])
    res["overhead"] = {
        "cost_mean": float(r_te.mean()),
        "latency_p50": float(np.percentile(lat, 50)),
        "latency_p95": float(np.percentile(lat, 95)),
        "errors": sum("error" in o.raw for o in outputs_test),
    }
    return res


def reference_rows(ds: Dataset, test: np.ndarray) -> dict[str, Any]:
    """Zero router (hull of single models = random mixing at matched cost)."""
    q, c = ds.quality_matrix()[test], ds.cost_matrix()[test]
    singles = cost_quality.single_model_points(q, c)
    h = cost_quality.hull(singles)
    c_min, c_max = min(s[0] for s in singles), max(s[0] for s in singles)
    return {
        "zero_router_aiq": cost_quality.aiq_linear(h, c_min, c_max),
        "zero_router_aiq_log": cost_quality.aiq(h, c_min, c_max),
        "single_models": {
            m.name: {"cost": s[0], "quality": s[1]} for m, s in zip(ds.pool, singles)
        },
    }


def run(config_path: str) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg.get("output_dir", f"results/{cfg['name']}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = CallCache(cfg.get("cache", ".cache/calls.sqlite"))
    ds = load(cfg["dataset"])
    routers = build_routers(cfg["routers"], ds, cache)
    split = cfg.get("split", {})
    recal_methods = cfg.get("recalibrate", ["platt"])
    binary_pairs = cfg.get("binary_pairs", [[0, len(ds.pool) - 1]])
    match_frac = cfg.get("match_frac", 0.95)

    records = []
    for seed in split.get("seeds", [42]):
        # A fixed test_seed keeps one held-out test split across seeds; seeds then vary only
        # the labeled subset. Needed because questions were designed on seed 42's train split.
        test_seed = split.get("test_seed", seed)
        train, test = grouped_split(ds, split.get("test_frac", 0.3), test_seed)
        if split.get("exclude_from_test"):
            # Examples seen while designing questions may never be scored as test.
            seen = set(Path(split["exclude_from_test"]).read_text().split())
            moved = np.array([ds.examples[i].id in seen for i in test], dtype=bool)
            train, test = np.sort(np.concatenate([train, test[moved]])), test[~moved]
        if split.get("dev_only"):
            # Question design phase: evaluate on a held-out half of train; test is never called.
            dev_ds = ds.subset(train)
            fit_local, eval_local = grouped_split(dev_ds, 0.5, seed + 1)
            train, test = train[fit_local], train[eval_local]
        refs = reference_rows(ds, test)
        for k in split.get("k_shot", [None]):
            lab = k_shot(train, k, seed)
            outputs: dict[str, tuple[list, list]] = {}
            for router in routers:
                if router.needs_labels:
                    router.fit(ds, lab)
                outputs[router.name] = (router.predict(ds, lab), router.predict(ds, test))
            for spec in cfg.get("cascades", []):
                prim, fb = outputs[spec["primary"]], outputs[spec["fallback"]]
                for t in spec["thresholds"]:
                    name = f"{spec['primary']}>{spec['fallback']}@{t}"
                    outputs[name] = (cascade(prim[0], fb[0], t), cascade(prim[1], fb[1], t))

            for name, (o_lab, o_te) in outputs.items():
                variants = {"raw": (o_lab, o_te)}
                if not _is_discrete(o_te):
                    q_lab = ds.quality_matrix()[lab]
                    for method in recal_methods:
                        rc = Recalibrator(method).fit(p_matrix(o_lab), q_lab)
                        variants[method] = (_recalibrated(rc, o_lab), _recalibrated(rc, o_te))
                for variant, (vl, vt) in variants.items():
                    res = evaluate(vl, vt, ds, lab, test, binary_pairs, match_frac)
                    records.append(
                        {"seed": seed, "k": k, "router": name, "variant": variant, **res}
                    )
                _save_predictions(
                    out_dir / f"seed{seed}_k{k}" / f"{_safe(name)}.jsonl", ds, test, o_te
                )
        records.append({"seed": seed, "k": None, "router": "_reference", **refs})

    (out_dir / "metrics.json").write_text(json.dumps(records, indent=1, default=_json_default))
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {**cfg, "questions": {"version": VERSION, "fingerprint": fingerprint(ds.pool)}}
        )
    )
    return out_dir


def _recalibrated(rc: Recalibrator, outputs: list[RouterOutput]) -> list[RouterOutput]:
    p = rc.transform(p_matrix(outputs))
    return [RouterOutput(p=row.tolist(), **_carry(o)) for row, o in zip(p, outputs)]


def _carry(o: RouterOutput) -> dict:
    return {
        "choice": o.choice,
        "confidence": o.confidence,
        "router_cost": o.router_cost,
        "router_latency_s": o.router_latency_s,
        "raw": o.raw,
    }


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_.@" else "_" for ch in name)


def _save_predictions(path: Path, ds: Dataset, idx: np.ndarray, outputs: list[RouterOutput]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, o in zip(idx, outputs):
            row = {"id": ds.examples[i].id, "p": o.p, **_carry(o)}
            f.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))
