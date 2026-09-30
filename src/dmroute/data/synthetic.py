"""Small synthetic benchmark so the whole pipeline runs without downloads or API keys.

Each example has a latent difficulty. The text carries noisy cues of that difficulty
(hard/easy vocabulary, length), so a sensible router can beat always-cheap but not the
oracle. Nothing here should be read as evidence about real routers.
"""

from __future__ import annotations

import numpy as np

from dmroute.schema import Dataset, Example, ModelSpec

POOL = [
    ModelSpec("tiny", "Small fast model for trivial requests.", 0.05, 0.10),
    ModelSpec("small", "Capable open model for routine single-step tasks.", 0.30, 1.20),
    ModelSpec("mid", "Strong general model for moderate multi-step tasks.", 1.25, 10.0),
    ModelSpec("frontier", "Frontier reasoning model for the hardest tasks.", 5.00, 25.0),
]
CAPABILITY = np.array([0.2, 0.45, 0.68, 0.9])
CATEGORIES = ["math", "code", "knowledge", "chat"]
CATEGORY_SHIFT = {"math": 0.12, "code": 0.08, "knowledge": 0.0, "chat": -0.15}

EASY_CUES = ["what is", "define", "quick question", "list", "hello", "in one word"]
HARD_CUES = [
    "prove",
    "multi-step",
    "refactor across files",
    "derive",
    "edge cases",
    "optimize",
    "formal argument",
    "concurrency",
]
TOPICS = {
    "math": ["an integral", "a combinatorics identity", "a probability puzzle", "a sum"],
    "code": ["a parser", "a caching layer", "a sorting routine", "an API client"],
    "knowledge": ["a historical event", "a chemistry fact", "a legal concept", "a city"],
    "chat": ["a greeting", "a birthday message", "weekend plans", "a joke"],
}


def load_synthetic(n: int = 400, seed: int = 0, name: str = "synthetic") -> Dataset:
    rng = np.random.default_rng(seed)
    examples = []
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        d = float(np.clip(rng.beta(2, 2) + CATEGORY_SHIFT[cat], 0.0, 1.0))
        n_hard = rng.binomial(3, d)
        n_easy = rng.binomial(2, 1 - d)
        cues = list(rng.choice(HARD_CUES, n_hard)) + list(rng.choice(EASY_CUES, n_easy))
        rng.shuffle(cues)
        topic = rng.choice(TOPICS[cat])
        filler = " Please include details." * int(rng.poisson(1 + 6 * d))
        text = f"[{cat}] About {topic}: {', '.join(cues) or 'help me'}.{filler}"

        p_success = 1 / (1 + np.exp(-(CAPABILITY - d) * 12))
        quality = {m.name: float(rng.random() < p) for m, p in zip(POOL, p_success)}
        in_tok = len(text) / 4
        out_tok = 150 + 900 * d
        cost = {m.name: (in_tok * m.price_in + out_tok * m.price_out) / 1e6 for m in POOL}
        examples.append(
            Example(
                id=f"syn-{i}",
                dataset=f"{name}-{cat}",
                text=text,
                quality=quality,
                cost=cost,
                category=cat,
                group=f"syn-{i}",
                meta={"difficulty": d},
            )
        )
    return Dataset(name, POOL, examples)
