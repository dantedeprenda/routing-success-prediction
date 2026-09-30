"""LLMRouterBench (arXiv 2601.07206, MIT).

Download ``bench-release.tar.gz`` from https://huggingface.co/datasets/NPULH/LLMRouterBench
and extract it so that ``<root>/<dataset>/<split>/<model>/<timestamp>.json`` exists
(``root`` is the ``results/bench`` directory of the release). Each JSON holds a
``records`` list with ``index, origin_query, prompt, score, cost, prompt_tokens,
completion_tokens``. We align records across the configured pool by
``(dataset, split, index)`` and keep only prompts every pool model answered.

Do not route on tau2: its ``origin_query`` is the hidden user-simulator script (it states
what the test checks), not anything a router would see.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from dmroute.schema import Dataset, Example, ModelSpec

# Coarse categories from the LLMRouterBench README, used by the per-category baseline.
CATEGORY = {
    "aime": "math",
    "livemathbench": "math",
    "livecodebench": "code",
    "swebench": "code",
    "swe-bench": "code",
    "gpqa": "knowledge",
    "hle": "knowledge",
    "mmlupro": "knowledge",
    "simpleqa": "knowledge",
    "arenahard": "instruction",
    "tau2": "tool",
}


def _category(dataset: str) -> str:
    key = dataset.lower().replace("-", "").replace("_", "")
    for prefix, cat in CATEGORY.items():
        if key.startswith(prefix):
            return cat
    return dataset


def router_text(dataset: str, record: dict[str, Any]) -> str:
    """What a router would see for this prompt.

    SWE-bench queries embed a partial code base after the issue; a router sees the issue.
    """
    text = record.get("origin_query") or record.get("prompt") or ""
    if dataset.startswith("swe"):
        match = re.search(r"<issue>(.*?)</issue>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def _latest_records(model_dir: Path) -> list[dict[str, Any]]:
    files = sorted(model_dir.glob("*.json"))
    if not files:
        return []
    return json.loads(files[-1].read_text())["records"]


def _pick_split(ds_dir: Path, names: list[str], override: str | None) -> Path | None:
    """One split per dataset. Releases ship overlapping splits (e.g. MMLU-Pro test_1000 is
    a subset of test_3000), so loading all of them would duplicate prompts. Default: the
    split with the most records that every pool model answered."""
    splits = [p for p in ds_dir.iterdir() if p.is_dir() and all((p / m).is_dir() for m in names)]
    if override:
        return ds_dir / override
    if not splits:
        return None
    return max(splits, key=lambda sp: min(len(_latest_records(sp / m)) for m in names))


def load_llmrouterbench(
    root: str,
    pool: list[dict[str, Any]],
    datasets: list[str] | None = None,
    splits: dict[str, str] | None = None,
    max_per_dataset: int | None = None,
    seed: int = 0,
    name: str = "llmrouterbench",
) -> Dataset:
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise FileNotFoundError(
            f"{root_path} not found; see the module docstring for download instructions"
        )
    specs = [ModelSpec(**m) for m in pool]
    names = [m.name for m in specs]
    dataset_dirs = sorted(p for p in root_path.iterdir() if p.is_dir())
    if datasets:
        wanted = set(datasets)
        dataset_dirs = [p for p in dataset_dirs if p.name in wanted]

    rng = np.random.default_rng(seed)
    examples: list[Example] = []
    for ds_dir in dataset_dirs:
        split_dir = _pick_split(ds_dir, names, (splits or {}).get(ds_dir.name))
        if split_dir is None:
            continue
        per_model = {m: {str(r["index"]): r for r in _latest_records(split_dir / m)} for m in names}
        common = set.intersection(*(set(v) for v in per_model.values()))
        ds_examples = []
        for idx in sorted(common, key=lambda i: (len(i), i)):
            first = per_model[names[0]][idx]
            ds_examples.append(
                Example(
                    id=f"{ds_dir.name}/{split_dir.name}/{idx}",
                    dataset=ds_dir.name,
                    text=router_text(ds_dir.name, first),
                    quality={
                        m: float(np.clip(float(per_model[m][idx]["score"]), 0, 1)) for m in names
                    },
                    cost={m: float(per_model[m][idx].get("cost") or 0.0) for m in names},
                    category=_category(ds_dir.name),
                    group=f"{ds_dir.name}/{idx}",
                    meta={"split": split_dir.name},
                )
            )
        if max_per_dataset and len(ds_examples) > max_per_dataset:
            keep = rng.choice(len(ds_examples), max_per_dataset, replace=False)
            ds_examples = [ds_examples[i] for i in sorted(keep)]
        examples.extend(ds_examples)
    if not examples:
        raise ValueError(f"no prompts answered by every model in {names} under {root_path}")
    return Dataset(name, specs, examples)
