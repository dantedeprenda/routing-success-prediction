"""Cost-quality metrics from the routing literature.

- Non-decreasing convex hull and AIQ (RouterBench, arXiv 2403.12031)
- PGR / APGR / CPT(x%) for a weak/strong pair (RouteLLM, arXiv 2406.18665)
- PerfGain / CostSave vs the best single model (LLMRouterBench, arXiv 2601.07206)
- RowPass / RowExact / TrajPass for tier labels (TwinRouterBench, arXiv 2605.18859)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from dmroute.policy import pick, tau_grid


def realized(choice: np.ndarray, q: np.ndarray, c: np.ndarray, rcost: np.ndarray | None = None):
    """Per-example quality and total cost (model + router) for a vector of picks."""
    rows = np.arange(len(choice))
    cost = c[rows, choice] + (rcost if rcost is not None else 0.0)
    return q[rows, choice], cost


def _point(tau: float, choice: np.ndarray, q, c, rcost) -> dict:
    qual, cost = realized(choice, q, c, rcost)
    return {
        "tau": float(tau),
        "quality": float(qual.mean()),
        "cost": float(cost.mean()),
        "share": np.bincount(choice, minlength=q.shape[1]).tolist(),
    }


def sweep(p: np.ndarray, q: np.ndarray, c: np.ndarray, rcost: np.ndarray, taus=None) -> list[dict]:
    """Operating points of the threshold policy across the tau grid (or given ``taus``)."""
    taus = tau_grid(p) if taus is None else taus
    return [_point(tau, pick(p, tau), q, c, rcost) for tau in taus]


def oracle_sweep(q: np.ndarray, c: np.ndarray, n: int = 200) -> list[dict]:
    """RouterBench oracle curve: per example, argmax(quality - lam * cost) over a lam grid.

    Ties go to the cheapest model. ``tau`` holds lam. Upper bound only.
    """
    scale = float(np.median(c)) or 1.0
    lams = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, n) / scale])
    points = []
    for lam in lams:
        util = q - lam * c
        best = util.max(axis=1, keepdims=True)
        choice = (util >= best - 1e-12).argmax(axis=1)
        points.append(_point(lam, choice, q, c, None))
    return points


def hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-left non-decreasing convex hull of (cost, quality) points.

    Mixing two routers at random reaches any point on the segment between them, so the
    hull is what a router can achieve. Points dominated by a cheaper, better one drop out.
    """
    pts = sorted(set(points))
    frontier: list[tuple[float, float]] = []
    for pt in pts:  # drop points not strictly better than something cheaper
        if not frontier or pt[1] > frontier[-1][1]:
            frontier.append(pt)
    upper: list[tuple[float, float]] = []
    for pt in frontier:
        while len(upper) >= 2:
            (x1, y1), (x2, y2) = upper[-2], upper[-1]
            if (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1) >= 0:
                upper.pop()
            else:
                break
        upper.append(pt)
    return upper


def hull_value(h: list[tuple[float, float]], cost):
    """Quality the hull reaches at ``cost`` (0 below its cheapest point, flat past its end)."""
    xs, ys = map(np.asarray, zip(*h))
    cost = np.asarray(cost, dtype=float)
    return np.where(cost < xs[0], 0.0, np.interp(cost, xs, ys))


def aiq(h: list[tuple[float, float]], c_min: float, c_max: float, n: int = 2000) -> float:
    """Average hull quality over [c_min, c_max] (RouterBench AIQ).

    Integrated on a log-cost grid because model prices span orders of magnitude; the
    linear-cost variant is ``aiq_linear``. Report which one a table uses.
    """
    grid = np.geomspace(max(c_min, 1e-12), c_max, n)
    return float(hull_value(h, grid).mean())


def aiq_linear(h, c_min, c_max, n: int = 2000) -> float:
    grid = np.linspace(c_min, c_max, n)
    return float(hull_value(h, grid).mean())


def single_model_points(q: np.ndarray, c: np.ndarray) -> list[tuple[float, float]]:
    return [(float(c[:, m].mean()), float(q[:, m].mean())) for m in range(q.shape[1])]


def perf_gain_cost_save(points: list[dict], q: np.ndarray, c: np.ndarray) -> dict:
    """LLMRouterBench PerfGain (best quality vs best single) and CostSave (cheapest
    point matching best-single quality)."""
    singles = single_model_points(q, c)
    best_c, best_q = max(singles, key=lambda s: (s[1], -s[0]))
    top = max(pt["quality"] for pt in points)
    matching = [pt["cost"] for pt in points if pt["quality"] >= best_q - 1e-12]
    return {
        "best_single_quality": best_q,
        "best_single_cost": best_c,
        "perf_gain": top / best_q - 1 if best_q > 0 else float("nan"),
        "cost_save": 1 - min(matching) / best_c if matching and best_c > 0 else float("nan"),
    }


def apgr(p: np.ndarray, q: np.ndarray, weak: int, strong: int, n: int = 101) -> dict:
    """RouteLLM APGR / CPT on a binary pair using P(weak suffices) as the routing score.

    Routes to ``strong`` for the fraction of examples with the lowest P(weak); PGR at
    each strong-call rate, averaged over an even grid of rates.
    """
    r_weak, r_strong = q[:, weak].mean(), q[:, strong].mean()
    gap = r_strong - r_weak
    order = np.argsort(p[:, weak], kind="stable")  # least confident in weak first
    rates = np.linspace(0, 1, n)
    pgr = []
    for rate in rates:
        to_strong = np.zeros(len(q), dtype=bool)
        to_strong[order[: round(rate * len(q))]] = True
        qual = np.where(to_strong, q[:, strong], q[:, weak]).mean()
        pgr.append((qual - r_weak) / gap if gap > 0 else float("nan"))
    pgr = np.array(pgr)

    def cpt(x):
        hit = np.flatnonzero(pgr >= x)
        return float(rates[hit[0]]) if len(hit) else float("nan")

    return {"apgr": float(np.trapezoid(pgr, rates)), "cpt50": cpt(0.5), "cpt80": cpt(0.8)}


def tier_metrics(choice: np.ndarray, target: np.ndarray, groups: list[str]) -> dict:
    """TwinRouterBench static metrics. TrajPass is the share of rows whose whole trajectory passes."""
    passed = choice >= target
    by_group: dict[str, list[bool]] = defaultdict(list)
    for g, ok in zip(groups, passed):
        by_group[g].append(bool(ok))
    traj_ok = {g: all(v) for g, v in by_group.items()}
    return {
        "row_pass": float(passed.mean()),
        "row_exact": float((choice == target).mean()),
        "traj_pass": float(np.mean([traj_ok[g] for g in groups])),
        "traj_pass_unweighted": float(np.mean(list(traj_ok.values()))),
    }
