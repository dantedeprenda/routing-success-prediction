"""Summaries and figures from a results directory's metrics.json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _fmt(values: list[float], pct: bool = False, digits: int = 3) -> str:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return "-"
    scale = 100 if pct else 1
    mean = np.mean(vals) * scale
    if len(vals) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{np.std(vals) * scale:.{digits}f}"


def summary_table(records: list[dict], k=None) -> str:
    rows = [r for r in records if r["router"] != "_reference" and r["k"] == k]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["router"], r["variant"])].append(r)
    refs = [r for r in records if r["router"] == "_reference"]

    header = (
        "| router | variant | AIQ | AIQ(log) | CostSave | Q@match | $@match | Q@frac | $@frac | "
        "APGR | ECE(mass) | Brier | AUROC(cheap) | AUROC within | router p50 s | errors |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for (router, variant), rs in sorted(grouped.items()):
        op = [r["operating_points"]["match_best"] for r in rs]
        opf = [r["operating_points"]["match_frac"] for r in rs]
        cal = [r.get("calibration") for r in rs]
        binary = [next(iter(r["binary"].values()))["apgr"] for r in rs if r.get("binary")]
        lines.append(
            "| "
            + " | ".join(
                [
                    router,
                    variant,
                    _fmt([r["aiq"] for r in rs]),
                    _fmt([r["aiq_log"] for r in rs]),
                    _fmt([r["cost_save"] for r in rs], pct=True, digits=1),
                    _fmt([o["quality"] for o in op]),
                    _fmt([o["cost"] for o in op], digits=6),
                    _fmt([o["quality"] for o in opf]),
                    _fmt([o["cost"] for o in opf], digits=6),
                    _fmt(binary),
                    _fmt([c["pooled"]["ece_mass15"] for c in cal if c]),
                    _fmt([c["pooled"]["brier"] for c in cal if c]),
                    _fmt([c["per_model"][0]["auroc"] for c in cal if c]),
                    _fmt([r["by_category"]["within"] for r in rs if r.get("by_category")]),
                    _fmt([r["overhead"]["latency_p50"] for r in rs]),
                    str(sum(r["overhead"]["errors"] for r in rs)),
                ]
            )
            + " |"
        )
    ref_line = (
        f"\nZero router (single-model hull): AIQ {_fmt([r['zero_router_aiq'] for r in refs])}, "
        f"AIQ(log) {_fmt([r['zero_router_aiq_log'] for r in refs])}\n"
    )
    return header + "\n".join(lines) + "\n" + ref_line


def category_table(records: list[dict], k=None) -> str:
    """Per-category AUROC for P(cheapest suffices) with 95% CIs (first seed, raw variant)."""
    rows = [
        r for r in records if r["k"] == k and r.get("variant") == "raw" and r.get("by_category")
    ]
    if not rows:
        return ""
    seed = rows[0]["seed"]
    rows = [r for r in rows if r["seed"] == seed]
    cats = sorted(rows[0]["by_category"]["per"])
    header = "| router | pooled | within | " + " | ".join(cats) + " |\n"
    header += "|---" * (len(cats) + 3) + "|\n"

    def cell(v):
        minority = min(v["n_pos"], v["n_neg"])
        if not np.isfinite(v["auroc"]) or minority < 10:
            return f"n/a ({v['n_pos']}+/{v['n_neg']}-)"
        return f"{v['auroc']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]"

    lines = []
    for r in rows:
        bc = r["by_category"]
        per = " | ".join(cell(bc["per"][c]) for c in cats)
        lines.append(f"| {r['router']} | {cell(bc['pooled'])} | {bc['within']:.2f} | {per} |")
    ns = ", ".join(
        f"{c} {rows[0]['by_category']['per'][c]['n_pos']}+/{rows[0]['by_category']['per'][c]['n_neg']}-"
        for c in cats
    )
    return (
        f"\nPer-category AUROC, P(cheapest suffices), seed {seed} ({ns}):\n\n{header}"
        + "\n".join(lines)
        + "\n"
    )


def plot_curves(records: list[dict], out: Path, routers: list[str] | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed = records[0]["seed"]
    rows = [
        r
        for r in records
        if r["seed"] == seed and r["router"] != "_reference" and r.get("variant") == "raw"
    ]
    k = rows[0]["k"]
    rows = [r for r in rows if r["k"] == k and (routers is None or r["router"] in routers)]
    ref = next(r for r in records if r["router"] == "_reference" and r["seed"] == seed)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, s in ref["single_models"].items():
        ax.scatter(s["cost"], s["quality"], marker="s", color="black", zorder=3)
        ax.annotate(
            name, (s["cost"], s["quality"]), fontsize=7, xytext=(3, -8), textcoords="offset points"
        )
    for r in rows:
        if r["router"].startswith("always:") or r["router"] == "oracle":
            continue
        pts = sorted((pt["cost"], pt["quality"]) for pt in r["curve"])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, lw=1.2, label=r["router"], drawstyle="steps-post" if len(pts) > 3 else None)
    ax.set_xscale("log")
    ax.set_xlabel("mean cost per request (USD, log)")
    ax.set_ylabel("mean quality")
    ax.set_title(f"cost-quality (seed {seed}, k={k})")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = out / "cost_quality.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def report(results_dir: str) -> str:
    out = Path(results_dir)
    records = json.loads((out / "metrics.json").read_text())
    ks = sorted(
        {r["k"] for r in records if r["router"] != "_reference"}, key=lambda v: (v is None, v)
    )
    parts = [f"# Results: {out.name}\n"]
    for k in ks:
        parts.append(
            f"\n## k = {'all' if k is None else k}\n\n"
            + summary_table(records, k)
            + category_table(records, k)
        )
    text = "\n".join(parts)
    (out / "summary.md").write_text(text)
    plot_curves(records, out)
    return text
