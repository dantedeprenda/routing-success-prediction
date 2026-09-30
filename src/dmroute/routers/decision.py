"""Decision-model routers (Jev, Laya, stub).

Every formulation of the same backend sends the identical fan-out request, so the call
cache turns four routers into one API call per example. Each formulation maps the
answers to ``p[m]`` = P(model m suffices), monotone in m by construction.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dmroute.questions import FEATURES, FORMULATIONS, build_questions, ladder, tier_label
from dmroute.routers.base import Router, fit_text
from dmroute.routers.baselines import _PerModelLogistic
from dmroute.schema import Dataset, RouterOutput, step_p


def _level_probs(answer: dict, k: int) -> np.ndarray:
    """Score answer -> probability per level, in level order."""
    probs = answer.get("probabilities")
    if isinstance(probs, list):
        out = np.array(probs, dtype=float)
    elif isinstance(probs, dict) and all(str(key).isdigit() for key in probs):
        out = np.array([probs.get(str(i), 0.0) for i in range(k)], dtype=float)
    elif isinstance(probs, dict):
        levels = ladder(k)
        out = np.array([probs.get(level, 0.0) for level in levels], dtype=float)
    else:  # only the expected score: put all mass on the nearest level
        out = np.zeros(k)
        out[int(np.clip(round(answer["score"]), 0, k - 1))] = 1.0
    return out / out.sum() if out.sum() > 0 else np.full(k, 1 / k)


def _noul_confidence(p: np.ndarray) -> float:
    return float(np.mean(np.abs(2 * p - 1)))


class DecisionModelRouter(Router):
    def __init__(
        self,
        backend,
        formulation: str,
        label: str,
        max_chars: int | None = None,
        workers: int = 8,
    ):
        if formulation not in FORMULATIONS:
            raise ValueError(f"unknown formulation {formulation!r}")
        self.backend = backend
        self.formulation = formulation
        self.label = label
        self.max_chars = max_chars
        self.workers = workers
        self.needs_labels = formulation == "features"
        self._memo: dict[str, tuple[dict | None, float, float, str | None]] = {}

    @property
    def name(self):
        return f"{self.label}:{self.formulation}"

    def _ask_all(
        self, ds: Dataset, idx: np.ndarray
    ) -> list[tuple[dict | None, float, float, str | None]]:
        questions = build_questions(ds.pool)
        todo = [i for i in idx if ds.examples[i].id not in self._memo]

        def one(i):
            ex = ds.examples[i]
            try:
                resp, lat, cost = self.backend.ask(fit_text(ex.text, self.max_chars), questions)
                return ex.id, (resp, lat, cost, None)
            except Exception as e:  # noqa: BLE001 - recorded, counted, routed to strongest
                return ex.id, (None, 0.0, 0.0, f"{type(e).__name__}: {e}")

        if getattr(self.backend, "batch_size", 1) > 1:  # local GPU model: batch forward passes
            chunk = 128
            for start in range(0, len(todo), chunk):
                part = todo[start : start + chunk]
                states = [fit_text(ds.examples[i].text, self.max_chars) for i in part]
                for i, (resp, lat, cost) in zip(part, self.backend.ask_many(states, questions)):
                    self._memo[ds.examples[i].id] = (resp, lat, cost, None)
        else:
            with ThreadPoolExecutor(self.workers) as pool:
                for ex_id, result in pool.map(one, todo):
                    self._memo[ex_id] = result
        return [self._memo[ds.examples[i].id] for i in idx]

    def _features(self, answers: dict) -> np.ndarray:
        return np.array([answers[f"feat_{name}"]["noul"] for name in FEATURES])

    def fit(self, ds, idx):
        if self.formulation != "features":
            return
        results = self._ask_all(ds, idx)
        ok = [j for j, r in enumerate(results) if r[0] is not None]
        x = np.array([self._features(results[j][0]["answers"]) for j in ok])
        self.head_ = _PerModelLogistic().fit(x, ds.quality_matrix()[idx[ok]])

    def predict(self, ds, idx):
        k = len(ds.pool)
        outputs = []
        for resp, lat, cost, err in self._ask_all(ds, idx):
            if resp is None:
                outputs.append(RouterOutput(p=step_p(k - 1, k), choice=k - 1, raw={"error": err}))
                continue
            a = resp["answers"]
            raw = {"model_version": resp.get("model")}
            if self.formulation == "choice":
                ans = a["choice"]
                labels = [tier_label(m, ds.pool) for m in range(k)]
                probs = np.array([ans["probabilities"].get(lab, 0.0) for lab in labels])
                p, conf = np.cumsum(probs), ans.get("confidence")
                choice = int(np.argmax(probs))
            elif self.formulation == "difficulty":
                probs = _level_probs(a["difficulty"], k)
                p, conf = np.cumsum(probs), a["difficulty"].get("confidence")
                choice = int(np.argmax(probs))
                raw["score"] = a["difficulty"].get("score")
            elif self.formulation == "tier_nouls":
                raw_p = np.array([a[f"tier_{m}"]["noul"] for m in range(k)])
                p, choice = np.maximum.accumulate(raw_p), None
                conf = _noul_confidence(p)
                raw["raw_p"] = raw_p.tolist()
            else:
                feats = self._features(a)
                p = self.head_.predict(feats[None, :])[0]
                p, conf, choice = np.maximum.accumulate(p), _noul_confidence(p), None
                raw["features"] = feats.tolist()
            outputs.append(
                RouterOutput(
                    p=np.clip(p, 0, 1).tolist(),
                    choice=choice,
                    confidence=conf,
                    router_cost=cost,
                    router_latency_s=lat,
                    raw=raw,
                )
            )
        return outputs
