"""Paper figures (PDF + PNG) from saved results; no API calls.

    uv run python scripts/figures.py

Reads results/<name>/metrics.json, results/judge_sample/*.json and results/evidence/*.json
(run the test configs, scripts/judge_sample.py and scripts/evidence.py first) and writes
paper/figures/{fewshot,cost_quality,lodo}.{pdf,png}. Sized for an ACL two-column page
(column 3.03 in, text width 6.3 in); identity never rests on color alone (markers, line
styles and direct labels carry it in grayscale print).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dmroute.data import load
from dmroute.metrics import cost_quality
from evidence import splits

OUT = Path("paper/figures")

# Validated categorical slots (light surface): Jev, LLM judges, fallback.
JEV = "#2a78d6"
JEV_LIGHT = "#86b6ef"  # same ramp, for the zero-shot Jev variant
JUDGE = "#eb6834"
FALLBACK = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TEXT_GRAYS = ["#52514e", "#6f6e69", "#898781", "#a3a29c"]

BENCH = {
    "llmrouterbench_test": ("LLMRouterBench", "jev:features", "jev:choice", "jev_choice"),
    "twinrouterbench_test": (
        "TwinRouterBench",
        "jev-4k:features",
        "jev-4k:choice",
        "jev-4k_choice",
    ),
}
MODEL_NAMES = {  # pool ids -> printed names (TwinRouterBench tiers are named by level)
    "deepseek-v3.1-terminus": "DeepSeek-V3.1",
    "deepseek-r1-0528": "DeepSeek-R1",
    "gpt-5": "GPT-5",
    "low": "DeepSeek-V3.2",
    "mid": "MiniMax-M2.7",
    "mid_high": "Gemini-3-Flash",
    "high": "Opus 4.6",
}
TEXT_ROUTERS = [  # (metrics name, label, marker, linestyle)
    ("tfidf_logreg", "TF-IDF + logistic", "s", "-"),
    ("knn10", "TF-IDF kNN", "v", (0, (4, 2))),
    ("bge_logreg", "Embedding + logistic", "D", "-"),
    ("bge_knn", "Embedding kNN", "^", (0, (4, 2))),
]


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def metrics(name: str) -> list[dict]:
    return json.loads(Path(f"results/{name}/metrics.json").read_text())


def train_size(name: str) -> int:
    cfg = yaml.safe_load(Path(f"configs/{name}.yaml").read_text())
    ds = load(cfg["dataset"])
    return len(splits(ds, cfg["split"])[0])


def fewshot() -> None:
    """Within-category AUROC vs labeled examples: original targets and 9 unseen models.

    Bands: sd over 5 label seeds (benchmark panels) or over the 9 target models (middle panel,
    each target first averaged over seeds; pre-registered experiment T). The benchmark panels
    start at k = 5 and 10 (exploratory, scripts/small_budgets.py: means only, no band).
    """
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3), sharey=True)
    lrb_recs = metrics("llmrouterbench_test")
    lrb_ks = sorted({r["k"] for r in lrb_recs if r.get("k")})
    t_json = json.loads(Path("results/evidence/extensions_T.json").read_text())["targets"]

    small = json.loads(Path("results/evidence/small_budgets.json").read_text())

    def small_series(name, router, ks):
        """Means at k = 5 and 10 (scripts/small_budgets.py; no per-seed sd is stored)."""
        rows = small[name]["a"]
        out = []
        for k in ks:
            feat = router.endswith("features")
            hit = [r for r in rows if r["k"] == k and (feat or r["baseline"] == router)]
            key = "features" if feat else "baseline_auroc"
            out.append(hit[0][key] if hit else np.nan)
        return np.array(out), np.full(len(ks), np.nan)

    def seed_series(recs, router, ks):
        mean, sd = [], []
        for k in [*ks, None]:
            v = [
                r["by_category"]["within"]
                for r in recs
                if r["router"] == router and r.get("k") == k and r.get("variant") == "platt"
            ]
            mean.append(np.mean(v))
            sd.append(np.std(v))
        return np.array(mean), np.array(sd)

    def target_series(router, ks):
        keys = [*map(str, ks), "all"]
        per_t = np.array([[t["auroc"][router][k] for k in keys] for t in t_json.values()])
        return per_t.mean(axis=0), per_t.std(axis=0)

    panels = [
        ("LLMRouterBench", "llmrouterbench_test", "jev:features", "jev_features"),
        ("9 unseen models (pre-registered)", None, None, "jev_features"),
        ("TwinRouterBench", "twinrouterbench_test", "jev-4k:features", None),
    ]
    for ax, (title, name, feat, t_feat) in zip(axes, panels):
        if name:
            recs = metrics(name)
            ks = sorted({r["k"] for r in recs if r.get("k")})
            n_all = train_size(name)

            def series(router, recs=recs, ks=ks, name=name):
                m0, s0 = small_series(name, router, [5, 10])
                m1, s1 = seed_series(recs, router, ks)
                return np.concatenate([m0, m1]), np.concatenate([s0, s1])

            ks = [5, 10, *ks]

            jev_name = feat
        else:
            ks, n_all = lrb_ks, train_size("llmrouterbench_test")

            def series(router, ks=ks):
                return target_series(router, ks)

            jev_name = t_feat
        xs = [*ks, n_all]
        for (router, label, marker, ls), gray in zip(TEXT_ROUTERS, TEXT_GRAYS):
            m, _ = series(router)
            ax.plot(xs, m, color=gray, lw=1.0, ls=ls, marker=marker, ms=3.2, label=label)
        m, sd = series(jev_name)
        ax.fill_between(xs, m - sd, m + sd, color=JEV, alpha=0.15, lw=0)
        ax.plot(xs, m, color=JEV, lw=2.0, marker="o", ms=4.2, label="Jev features (ours)")
        ax.axhline(0.5, color=AXIS, lw=0.8)
        ax.text(xs[-1], 0.495, "chance", color=MUTED, fontsize=6.3, va="top", ha="right")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([*map(str, ks), f"all\n({n_all:,})"], fontsize=6.3)
        ax.minorticks_off()
        ax.set_title(title, loc="left", color=INK, fontsize=7.5)
        ax.set_xlabel("Labeled training prompts")
        ax.set_ylim(0.45, 0.8)
    axes[0].set_ylabel("Within-category AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [4, 0, 1, 2, 3]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.08),
        handlelength=2.2,
        columnspacing=1.2,
    )
    fig.tight_layout()
    save(fig, "fewshot")


def cost_quality_fig() -> None:
    """Cost vs quality on the full test splits: single models, Jev's curve, LLM judges."""
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    for ax, (name, (title, _, choice, choice_file)) in zip(axes, BENCH.items()):
        rows = json.loads(Path(f"results/judge_sample/{name}.json").read_text())
        singles = [
            (r["cost"], r["quality"], n.split(":", 1)[1])
            for n, r in rows.items()
            if n.startswith("always:")
        ]
        hull = cost_quality.hull([(c, q) for c, q, _ in singles])
        ax.plot(*zip(*hull), color=MUTED, lw=1.0, zorder=1, label="Single models (hull)")
        cheapest = min(c for c, _, _ in singles)
        for c, q, label in singles:
            ax.scatter(c, q, s=18, color=MUTED, zorder=2)
            offset = (5, -9) if c == cheapest else (3, -8)
            ax.annotate(
                MODEL_NAMES.get(label, label),
                (c, q),
                textcoords="offset points",
                xytext=offset,
                fontsize=6,
                color=INK_2,
            )
        curve = next(
            r["curve"]
            for r in metrics(name)
            if r["router"] == choice
            and r.get("k") is None
            and r.get("seed") == 42
            and r.get("variant") == "raw"
        )
        jh = cost_quality.hull([(p["cost"], p["quality"]) for p in curve])
        ax.plot(*zip(*jh), color=JEV, lw=1.6, zorder=3, label="Jev (threshold sweep)")
        jev = rows[f"{choice_file}@match_frac"]
        ax.scatter(
            jev["cost"],
            jev["quality"],
            s=34,
            marker="o",
            color=JEV,
            zorder=5,
            edgecolor="white",
            linewidth=0.8,
            label="Jev operating point",
        )
        for key, marker, label in [
            ("judge-sonnet", "^", "LLM judge: Sonnet 5"),
            ("judge-luna", "v", "LLM judge: GPT-6 Luna"),
        ]:
            r = rows[key]
            ax.scatter(
                r["cost"],
                r["quality"],
                s=34,
                marker=marker,
                color=JUDGE,
                zorder=5,
                edgecolor="white",
                linewidth=0.8,
                label=label,
            )
        fb = rows[f"{choice_file}>judge-sonnet@0.5"]
        ax.scatter(
            fb["cost"],
            fb["quality"],
            s=38,
            marker="D",
            color=FALLBACK,
            zorder=6,
            edgecolor="white",
            linewidth=0.8,
            label=r"Jev $\rightarrow$ Sonnet fallback",
        )
        ax.annotate(
            "fallback",
            (fb["cost"], fb["quality"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
            color=INK_2,
        )
        ax.margins(y=0.09)
        ax.set_xscale("log")
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Mean cost per task, incl. router (USD, log)")
    axes[0].set_ylabel("Quality")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), columnspacing=1.2
    )
    fig.tight_layout()
    save(fig, "cost_quality")


def lodo() -> None:
    """Held-out-dataset AUROC: Jev features vs zero-shot Jev vs the best text router."""
    rows = json.loads(Path("results/evidence/llmrouterbench_test.json").read_text())["C"]
    held = sorted(
        {r["held_out"] for r in rows},
        key=lambda d: -next(r["n"] for r in rows if r["held_out"] == d),
    )

    def get(d, router):
        return next(
            r for r in rows if r["held_out"] == d and r["k"] == "all" and r["router"] == router
        )

    fig, ax = plt.subplots(figsize=(3.03, 2.1))
    x = np.arange(len(held))
    w = 0.26
    bars = [
        ("Jev features", JEV, None, lambda d: get(d, "jev:features")),
        ("Jev zero-shot (no labels)", JEV_LIGHT, None, lambda d: get(d, "jev:choice")),
        (
            "Best text router",
            "#bdbcb5",
            "////",
            lambda d: max(
                (get(d, n) for n in ("tfidf_logreg", "knn10", "bge_logreg", "bge_knn")),
                key=lambda r: r["auroc"],
            ),
        ),
    ]
    for j, (label, color, hatch, fn) in enumerate(bars):
        vals = [fn(d) for d in held]
        heights = [v["auroc"] - 0.5 for v in vals]
        lo = [v["auroc"] - v["ci95_last_seed"][0] for v in vals]
        hi = [v["ci95_last_seed"][1] - v["auroc"] for v in vals]
        ax.bar(
            x + (j - 1) * w,
            heights,
            w * 0.92,
            bottom=0.5,
            color=color,
            hatch=hatch,
            edgecolor="white" if hatch is None else "#8f8e88",
            linewidth=0.0 if hatch is None else 0.4,
            label=label,
            zorder=2,
        )
        ax.errorbar(
            x + (j - 1) * w,
            [v["auroc"] for v in vals],
            yerr=[lo, hi],
            fmt="none",
            ecolor=INK_2,
            elinewidth=0.6,
            capsize=1.2,
            zorder=3,
        )
    ax.axhline(0.5, color=INK_2, lw=0.8, zorder=1)
    ax.set_xticks(x)
    names = {
        "livecodebench": "LiveCode",
        "mmlupro": "MMLU-Pro",
        "swe-bench": "SWE-bench",
        "aime": "AIME",
        "gpqa": "GPQA",
    }
    ax.set_xticklabels(
        [f"{names.get(d, d)}\n(n={get(d, 'jev:features')['n']})" for d in held], fontsize=6.3
    )
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUROC on held-out dataset")
    ax.grid(axis="x", visible=False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=6.2,
        handlelength=1.4,
        columnspacing=1.0,
    )
    fig.tight_layout()
    save(fig, "lodo")


def discordant() -> None:
    """Pre-registered v3: discordant-prompt AUROC of P(cheap) vs P(cheap) - P(strong), per pair.

    Reads results/evidence/complementarity.json (scripts/complementarity.py), Jev features with all
    labels. Discordant prompts are those exactly one model answers correctly; below 0.5 a score
    ranks them backwards.
    """
    res = json.loads(Path("results/evidence/complementarity.json").read_text())["pairs"]
    short = {
        "deepseek-v3-0324": "DS-V3",
        "deepseek-v3.1-terminus": "DS-V3.1",
        "kimi-k2-0905": "Kimi-K2",
        "qwen3-235b-a22b-2507": "Qwen3",
        "gemini-2.5-flash": "Gemini",
        "glm-4.6": "GLM",
        "claude-sonnet-4": "Sonnet",
        "gpt-5": "GPT-5",
    }
    pairs = [(k, v) for k, v in res.items() if not v["reference_only"]]
    labels = ["{}\nvs {}".format(*(short[m] for m in k.split(" -> "))) for k, _ in pairs]
    fig, ax = plt.subplots(figsize=(3.03, 2.0))
    x = np.arange(len(pairs))
    w = 0.38
    pc = [v["jev_features|all"]["auc_pcheap"] for _, v in pairs]
    df = [v["jev_features|all"]["auc_diff"] for _, v in pairs]
    ax.bar(x - w / 2, pc, w, color="white", edgecolor=INK_2, hatch="////", lw=0.8)
    ax.bar(x + w / 2, df, w, color=JEV)
    ax.axhline(0.5, color=INK_2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.8)
    ax.set_ylim(0.3, 0.8)
    ax.set_ylabel("AUROC on discordant prompts")
    ax.grid(axis="x", visible=False)
    handles = [
        mpl.patches.Patch(facecolor="white", edgecolor=INK_2, hatch="////", label="P(cheap)"),
        mpl.patches.Patch(color=JEV, label="P(cheap) − P(strong)"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    save(fig, "discordant")


if __name__ == "__main__":
    style()
    fewshot()
    cost_quality_fig()
    lodo()
    discordant()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
