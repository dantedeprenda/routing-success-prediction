"""Merge Laya calls from a Kaggle run into the local cache.

    uv run python kaggle/merge_cache.py ~/Downloads/laya_lrb.sqlite ~/Downloads/laya_trb.sqlite

Only ``laya:*`` rows are copied. Rows already in the local cache (the TwinRouterBench steps run
on MPS before the Kaggle run) are replaced so every Laya prediction comes from one device;
the script prints how far the replaced answers moved first.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

LOCAL = ".cache/calls.sqlite"


def _probs(response: dict) -> list[float]:
    out = []
    for qid in sorted(response["answers"]):
        a = response["answers"][qid]
        if a["type"] == "noul":
            out.append(a["noul"])
        else:
            out.extend(a["probabilities"][k] for k in sorted(a["probabilities"]))
    return out


def main(paths: list[str]) -> int:
    backup = Path(LOCAL + ".pre-kaggle")
    if not backup.exists():
        shutil.copy2(LOCAL, backup)
        print(f"backup: {backup}")
    db = sqlite3.connect(LOCAL)
    for path in paths:
        src = sqlite3.connect(path)
        rows = src.execute(
            "SELECT key, backend, request, response, latency_s, created FROM calls "
            "WHERE backend LIKE 'laya:%'"
        ).fetchall()
        diffs = []
        for key, _, _, response, *_ in rows:
            old = db.execute("SELECT response FROM calls WHERE key = ?", (key,)).fetchone()
            if old:
                a, b = _probs(json.loads(old[0])), _probs(json.loads(response))
                diffs.append(max(abs(x - y) for x, y in zip(a, b)))
        db.executemany(
            "INSERT OR REPLACE INTO calls (key, backend, request, response, latency_s, created) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        db.commit()
        backends = sorted({r[1] for r in rows})
        print(f"{path}: {len(rows)} rows ({', '.join(backends)}); replaced {len(diffs)}")
        if diffs:
            diffs.sort()
            print(
                f"  replaced answers, max |dp| per call: median {diffs[len(diffs) // 2]:.4f}, "
                f"max {diffs[-1]:.4f}"
            )
    for backend, n in db.execute(
        "SELECT backend, count(*) FROM calls WHERE backend LIKE 'laya:%' GROUP BY backend"
    ):
        print(f"local {backend}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
