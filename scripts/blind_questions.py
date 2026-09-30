"""X2: label-blind question sets answered by Jev (preregistration/v2.md, Part 1).

    uv run python scripts/blind_questions.py answer    # Jev calls, cached; hard cap $6 in total
    uv run python scripts/blind_questions.py analyze   # offline -> results/evidence/blind_questions.*

Each set in configs/questions/qblind_{1,2,3}.json is sent exactly as v1's feature questions are:
one Noul per question with the question text as ``instructions``, one fan-out request per prompt,
the same 16,000-character router-visible text. Everything after the calls reuses the pre-registered
extensions code (splits, heads, within-category AUROC, bootstraps).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

import extensions as ext
from dmroute.backends.cache import CallCache
from dmroute.backends.systemone import SystemOneBackend
from dmroute.cli import _load_dotenv
from dmroute.data import load
from dmroute.questions import FEATURES
from dmroute.routers.base import fit_text
from evidence import stratified_draws

SETS = [1, 2, 3]
KS = [10, 25, 50, 100, 200, None]
B = 2000
MAX_USD = 6.0  # pre-registered hard cap, generator included
PRIMARY = "bge_knn"  # fixed by the rule in preregistration/v1.md


def question_set(i: int) -> dict[str, str]:
    path = Path(f"configs/questions/qblind_{i}.json")
    retry = path.with_name(f"qblind_{i}_regenerated.json")
    return json.loads((retry if retry.exists() else path).read_text())["questions"]


def generator_usd() -> float:
    """Sonnet 5 list price, $3 / $15 per M tokens (input / output)."""
    total = 0.0
    for path in Path("configs/questions").glob("qblind_*.json"):
        u = json.loads(path.read_text())["usage"]
        total += (3 * u["input_tokens"] + 15 * u["output_tokens"]) / 1e6
    return total


def jev_backend() -> tuple[SystemOneBackend, int]:
    cfg = yaml.safe_load(Path(ext.CONFIG).read_text())
    spec = next(r for r in cfg["routers"] if r.get("label") == "jev")
    return SystemOneBackend(**spec["backend"], cache=CallCache(".cache/calls.sqlite")), spec[
        "max_chars"
    ]


def requests(i: int) -> dict[str, dict]:
    return {f"feat_{q}": {"type": "noul", "instructions": t} for q, t in question_set(i).items()}


def answer() -> int:
    _load_dotenv()
    backend, max_chars = jev_backend()
    ds = load(yaml.safe_load(Path(ext.CONFIG).read_text())["dataset"])
    spent = generator_usd()
    print(f"generator ${spent:.4f}")
    for i in SETS:
        questions = requests(i)

        def one(ex, questions=questions):
            try:
                resp, _, cost = backend.ask(fit_text(ex.text, max_chars), questions)
                return ex.id, resp.get("model"), cost, None
            except Exception as e:  # noqa: BLE001 - counted; a rerun retries uncached calls
                return ex.id, None, 0.0, f"{type(e).__name__}: {e}"

        failed, models, cost = [], {}, 0.0
        chunk = 500
        for start in range(0, len(ds.examples), chunk):
            if spent + cost > MAX_USD:
                print(f"STOP: cap ${MAX_USD} reached", flush=True)
                return 1
            with ThreadPoolExecutor(6) as pool:
                for ex_id, model, c, err in pool.map(one, ds.examples[start : start + chunk]):
                    cost += c
                    models[model] = models.get(model, 0) + 1
                    if err:
                        failed.append((ex_id, err))
            done = min(start + chunk, len(ds.examples))
            print(f"Q-blind-{i}: {done}/{len(ds.examples)}, ${cost:.4f}", flush=True)
        spent += cost
        print(f"Q-blind-{i}: models {models}, failed {len(failed)}, ${cost:.4f}", flush=True)
        for ex_id, err in failed[:5]:
            print(f"  {ex_id}: {err}")
    print(f"total ${spent:.4f} (cap ${MAX_USD})")
    return 0


def features(s: ext.Setup, i: int) -> tuple[np.ndarray, int]:
    """(n x 13) P(yes) matrix for set *i* from the cache; failed prompts are NaN."""
    backend, max_chars = jev_backend()
    backend.base_url = "http://127.0.0.1:9"  # offline: an uncached call fails instead of paying
    questions = requests(i)
    x = np.full((len(s.ds.examples), len(questions)), np.nan)
    failed = 0
    for n, ex in enumerate(s.ds.examples):
        try:
            resp, _, _ = backend.ask(fit_text(ex.text, max_chars), questions)
            x[n] = [resp["answers"][q]["noul"] for q in questions]
        except Exception:  # noqa: BLE001
            failed += 1
    return x, failed


def impute(x: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Pre-registered failure rule, applied per method call: 0.5 at test (training rows with a
    failure are rare and also set to 0.5, reported in the failure count)."""
    return np.where(np.isnan(x), 0.5, x)


def contrast(c: dict, a: str, b: str, seed_draws: np.ndarray, k) -> np.ndarray:
    return ext.seed_boot(c[a][k] - c[b][k], seed_draws)


def analyze() -> int:
    ext.KS = KS
    s = ext.Setup(offline=True)
    rng = np.random.default_rng(0)
    draws = stratified_draws(s.cats[s.test], rng, B)
    seed_draws = rng.integers(0, len(ext.SEEDS), size=(B, len(ext.SEEDS)))
    seed_draws[0] = np.arange(len(ext.SEEDS))
    set_draws = rng.integers(0, len(SETS), size=(B, len(SETS)))
    set_draws[0] = np.arange(len(SETS))

    xs, fails = {}, {}
    for i in SETS:
        x, fails[i] = features(s, i)
        xs[i] = impute(x, s.train)
    y = (s.ds.quality_matrix()[:, 0] >= 0.5).astype(float)
    c = ext.curves(s, y, ["jev_features", *ext.BASELINES], draws)
    for i in SETS:
        c[f"qblind_{i}"] = ext.curves(s, y, ["custom"], draws, x_custom=xs[i])["custom"]

    def pooled(per_set: list[np.ndarray]) -> tuple[float, float, float]:
        m = np.stack(per_set)  # sets x B
        return ext.ci(np.nanmean(np.take_along_axis(m, set_draws.T, axis=0), axis=0))

    rows, payload = [], {"failed_calls": fails, "auroc": {}, "vs_primary": {}, "vs_author": {}}
    for k in KS:
        kn = ext.kname(k)
        payload["auroc"][kn] = {
            m: float(np.nanmean(c[m][k][:, 0]))
            for m in ["jev_features", *ext.BASELINES, *(f"qblind_{i}" for i in SETS)]
        }
        vp = pooled([contrast(c, f"qblind_{i}", PRIMARY, seed_draws, k) for i in SETS])
        va = pooled([contrast(c, f"qblind_{i}", "jev_features", seed_draws, k) for i in SETS])
        per_set = {i: ext.ci(contrast(c, f"qblind_{i}", PRIMARY, seed_draws, k)) for i in SETS}
        payload["vs_primary"][kn], payload["vs_author"][kn] = vp, va
        payload.setdefault("vs_primary_per_set", {})[kn] = per_set
        a = payload["auroc"][kn]
        rows.append(
            f"| {kn} | {a['jev_features']:.3f} | "
            + " / ".join(f"{a[f'qblind_{i}']:.3f}" for i in SETS)
            + f" | {a[PRIMARY]:.3f} | {vp[0]:+.3f} [{vp[1]:+.3f}, {vp[2]:+.3f}] | "
            f"{va[0]:+.3f} [{va[1]:+.3f}, {va[2]:+.3f}] |"
        )
    point, lo, hi = payload["vs_primary"]["25"]

    # Secondary 3: the 9 unseen targets at k = 25 (mean over targets and sets).
    ext.KS = [25]
    per_target = []
    for t in ext.TARGETS:
        yt = s.labels(t)
        base = ext.curves(s, yt, [PRIMARY], draws)[PRIMARY][25]
        diffs = [
            ext.seed_boot(
                ext.curves(s, yt, ["custom"], draws, x_custom=xs[i])["custom"][25] - base,
                seed_draws,
            )
            for i in SETS
        ]
        per_target.append(np.nanmean(np.stack(diffs), axis=0))
        print(f"  {t}: done", flush=True)
    target_draws = rng.integers(0, len(ext.TARGETS), size=(B, len(ext.TARGETS)))
    target_draws[0] = np.arange(len(ext.TARGETS))
    tt = ext.ci(
        np.nanmean(np.take_along_axis(np.stack(per_target), target_draws.T, axis=0), axis=0)
    )
    payload["targets_k25_vs_primary"] = tt

    # Secondary 4: closest v1 question for each label-blind question (bge-large cosine).
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer("BAAI/bge-large-en-v1.5")
    v1 = list(FEATURES.items())
    ev1 = enc.encode([t for _, t in v1], normalize_embeddings=True)
    overlap = []
    for i in SETS:
        qs = list(question_set(i).items())
        sims = enc.encode([t for _, t in qs], normalize_embeddings=True) @ ev1.T
        for (qid, _), row in zip(qs, sims, strict=True):
            j = int(row.argmax())
            overlap.append((i, qid, v1[j][0], float(row[j])))
    payload["overlap"] = overlap

    lines = [
        "# X2 routing arm: label-blind questions (pre-registered, preregistration/v2.md)",
        "",
        (
            f"Failed Jev calls (imputed 0.5): {fails}. Bootstrap: {B} replicates over question "
            "sets, label seeds and test prompts (stratified by category)."
        ),
        "",
        (
            f"**Primary endpoint** (k = 25, mean over Q-blind-1..3 of Q-blind - {PRIMARY}): "
            f"**{point:+.3f} [{lo:+.3f}, {hi:+.3f}]**; success criterion (lower bound > 0): "
            f"**{'MET' if lo > 0 else 'NOT MET'}**."
        ),
        "",
        (
            f"| k | Q-author (v1) | Q-blind 1 / 2 / 3 | {PRIMARY} | Q-blind - {PRIMARY} | "
            "Q-blind - Q-author |"
        ),
        "|---|---|---|---|---|---|",
        *rows,
        "",
        (
            f"9 unseen targets, k = 25, Q-blind - {PRIMARY} (mean over targets and sets): "
            f"{tt[0]:+.3f} [{tt[1]:+.3f}, {tt[2]:+.3f}]."
        ),
        "",
        "Closest v1 question per label-blind question (bge-large cosine):",
        "",
        "| set | question | closest v1 | cosine |",
        "|---|---|---|---|",
        *(f"| {i} | {q} | {v} | {sim:.2f} |" for i, q, v, sim in overlap),
    ]
    out = Path("results/evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / "blind_questions.md").write_text("\n".join(lines) + "\n")
    (out / "blind_questions.json").write_text(json.dumps(payload, indent=1, default=float))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit({"answer": answer, "analyze": analyze}[sys.argv[1]]())
