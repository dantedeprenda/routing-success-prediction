"""Router interface.

``fit`` sees only the labeled indices a router is allowed to learn from (the k-shot
budget); ``predict`` returns one :class:`RouterOutput` per requested index.
"""

from __future__ import annotations

import numpy as np

from dmroute.schema import Dataset, RouterOutput


class Router:
    name: str = "router"
    needs_labels: bool = False

    def fit(self, ds: Dataset, idx: np.ndarray) -> None:
        pass

    def predict(self, ds: Dataset, idx: np.ndarray) -> list[RouterOutput]:
        raise NotImplementedError


def fit_text(text: str, max_chars: int | None) -> str:
    """Truncate to ``max_chars`` keeping the head (task statement) and the tail (latest step)."""
    if not max_chars or len(text) <= max_chars:
        return text
    head = max_chars // 3
    tail = max_chars - head - 5
    return f"{text[:head]}\n...\n{text[-tail:]}"
