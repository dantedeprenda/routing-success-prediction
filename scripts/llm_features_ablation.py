"""Ablation: is the few-shot gain the decision model, or the questions?

    uv run python scripts/llm_features_ablation.py configs/llmrouterbench_test.yaml --max-usd 3

A generative LLM answers the same 13 feature questions (verbalized probabilities, one
structured-output call per prompt) on the same 2,000-char router-visible text, and the same
per-model logistic head is trained on k labels. Only the prompts a k <= 200 head or the test
split needs are asked (the full train split would be 3.6k extra calls at a 20 req/min limit).
Compared with the decision-model features router's saved test predictions by the same
hierarchical paired bootstrap as scripts/evidence.py. Writes results/evidence/<name>_llm_features.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dmroute.backends.cache import CallCache
from dmroute.backends.chat import ChatBackend
from dmroute.cli import _load_dotenv
from dmroute.data import load
from dmroute.data.splits import k_shot
from dmroute.questions import FEATURES
from dmroute.routers.base import fit_text
from dmroute.routers.baselines import _PerModelLogistic
from evidence import SEEDS, load_p, splits, stratified_draws, within_rows

MODEL, PRICE_IN, PRICE_OUT = "openai/gpt-6-luna", 0.10, 0.50
MAX_CHARS = 2000
KS = [25, 50, 100, 200]
B = 2000

SYSTEM = (
    "Below is a conversation. For each question, estimate the probability (0 to 1) that the "
    "answer is yes, where 'the next assistant response' means the reply that should follow "
    "the conversation. Answer every question; do not answer the conversation itself.\n\n"
    + "\n".join(f"- {name}: {text}" for name, text in FEATURES.items())
    + "\n\nReturn JSON only, one number per question id."
)
FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "feature_probabilities",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {name: {"type": "number"} for name in FEATURES},
            "required": list(FEATURES),
            "additionalProperties": False,
        },
    },
}


def features(backend: ChatBackend, text: str) -> np.ndarray | None:
    try:
        reply, _, _ = backend.complete(SYSTEM, fit_text(text, MAX_CHARS), response_format=FORMAT)
        answer = json.loads(reply)
        return np.clip([float(answer[name]) for name in FEATURES], 0.0, 1.0)
    except Exception:  # noqa: BLE001 - counted and reported, never silently scored
        return None


def main() -> int:
    _load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--rpm", type=float, default=18)
    ap.add_argument("--features", default="jev_features", help="decision-model router to compare")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    results_dir = Path(cfg.get("output_dir", f"results/{cfg['name']}"))
    ds = load(cfg["dataset"])
    train, test = splits(ds, cfg["split"])
    labs = {(s, k): k_shot(train, k, s) for s in SEEDS for k in KS}
    needed = sorted({int(i) for idx in labs.values() for i in idx} | {int(i) for i in test})
    backend = ChatBackend(
        model=MODEL, price_in=PRICE_IN, price_out=PRICE_OUT, rpm=args.rpm, cache=CallCache()
    )

    feats: dict[int, np.ndarray | None] = {}
    spent = 0.0
    chunk = 60
    for start in range(0, len(needed), chunk):
        if spent > args.max_usd:
            raise SystemExit(f"stopped at ${spent:.2f} (budget ${args.max_usd})")
        part = needed[start : start + chunk]
        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(lambda i: features(backend, ds.examples[i].text), part))
        feats.update(zip(part, got))
        # Cost bound from token prices (cached calls cost nothing but are counted here too).
        spent = len(feats) * (1500 * PRICE_IN + 250 * PRICE_OUT) / 1e6
        print(f"  {len(feats)}/{len(needed)} prompts, <= ${spent:.2f}", flush=True)
    failed = [i for i, f in feats.items() if f is None]

    q = ds.quality_matrix()
    ids = [ds.examples[i].id for i in test]
    y = (q[test, 0] >= 0.5).astype(float)
    cats = np.array([ds.examples[i].category for i in test])
    rng = np.random.default_rng(0)
    draws = stratified_draws(cats, rng, B)
    seed_draws = rng.integers(0, len(SEEDS), size=(B, len(SEEDS)))
    seed_draws[0] = np.arange(len(SEEDS))
    x_test = np.array(
        [feats[i] if feats[i] is not None else np.full(len(FEATURES), 0.5) for i in test]
    )
    lines = [
        f"# Ablation: feature questions answered by {MODEL} ({cfg['name']})",
        "",
        (
            f"{len(needed)} prompts asked, {len(failed)} failed (imputed 0.5 at test, dropped "
            "in training). Within-category AUROC of P(cheapest suffices); diff = decision model "
            f"minus LLM; hierarchical paired bootstrap ({B} replicates)."
        ),
        "",
        "| k | decision model | LLM features | diff | 95% CI | seeds won |",
        "|---|---|---|---|---|---|",
    ]
    rows = []
    for k in KS:
        dm, llm = [], []
        for s in SEEDS:
            lab = np.array([i for i in labs[(s, k)] if feats[int(i)] is not None])
            head = _PerModelLogistic().fit(np.array([feats[int(i)] for i in lab]), q[lab])
            p_llm = head.predict(x_test)[:, 0]
            p_dm = load_p(results_dir, s, k, args.features, ids)[:, 0]
            dm.append(within_rows(p_dm, y, cats, draws))
            llm.append(within_rows(p_llm, y, cats, draws))
        dm, llm = np.stack(dm), np.stack(llm)
        diff = dm - llm
        boot = np.nanmean(np.take_along_axis(diff, seed_draws.T, axis=0), axis=0)
        lo, hi = np.nanpercentile(boot[1:], [2.5, 97.5])
        row = {
            "k": k,
            "decision_model": float(np.nanmean(dm[:, 0])),
            "llm_features": float(np.nanmean(llm[:, 0])),
            "diff": float(np.nanmean(diff[:, 0])),
            "ci95": [float(lo), float(hi)],
            "seeds_won": int((diff[:, 0] > 0).sum()),
        }
        rows.append(row)
        lines.append(
            f"| {k} | {row['decision_model']:.3f} | {row['llm_features']:.3f} | "
            f"{row['diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {row['seeds_won']}/5 |"
        )
    out = Path("results/evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg['name']}_llm_features.md").write_text("\n".join(lines) + "\n")
    (out / f"{cfg['name']}_llm_features.json").write_text(
        json.dumps(
            {"model": MODEL, "asked": len(needed), "failed": len(failed), "rows": rows}, indent=1
        )
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
