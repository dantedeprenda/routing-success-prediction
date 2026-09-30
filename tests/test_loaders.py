import json
from pathlib import Path

import numpy as np

from dmroute.data.llmrouterbench import load_llmrouterbench
from dmroute.data.splits import grouped_split
from dmroute.data.synthetic import load_synthetic
from dmroute.data.twinrouterbench import load_twinrouterbench

FIXTURE = Path(__file__).parent / "fixtures" / "twinrouterbench_sample.jsonl"


def test_twinrouterbench_outcomes_follow_target_tier():
    ds = load_twinrouterbench(str(FIXTURE))
    assert ds.model_names == ["low", "mid", "mid_high", "high"]
    for ex in ds.examples:
        t = ex.meta["target_tier_id"]
        assert [ex.quality[m] for m in ds.model_names] == [float(i >= t) for i in range(4)]
    assert (np.diff(ds.cost_matrix(), axis=1) > 0).all()


def test_grouped_split_keeps_trajectories_whole():
    ds = load_twinrouterbench(str(FIXTURE))
    train, test = grouped_split(ds, 0.34, seed=1)
    train_groups = {ds.examples[i].group for i in train}
    test_groups = {ds.examples[i].group for i in test}
    assert train_groups and test_groups and not train_groups & test_groups


def test_llmrouterbench_aligns_records_across_models(tmp_path):
    def write(dataset, model, records):
        d = tmp_path / dataset / "test" / model
        d.mkdir(parents=True)
        (d / "20260101.json").write_text(json.dumps({"records": records}))

    rec = lambda i, s, c: {"index": i, "origin_query": f"q{i}", "score": s, "cost": c}
    write("aime", "cheap", [rec(0, 0, 0.1), rec(1, 1, 0.1), rec(2, 1, 0.1)])
    write("aime", "strong", [rec(0, 1, 1.0), rec(1, 1, 1.0)])  # index 2 missing -> dropped
    # A smaller overlapping split must be ignored in favor of the larger one.
    small = tmp_path / "aime" / "small"
    for m in ("cheap", "strong"):
        (small / m).mkdir(parents=True)
        (small / m / "1.json").write_text(json.dumps({"records": [rec(0, 1, 0.1)]}))
    ds = load_llmrouterbench(str(tmp_path), pool=[{"name": "cheap"}, {"name": "strong"}])
    assert [ex.text for ex in ds.examples] == ["q0", "q1"]
    assert ds.quality_matrix().tolist() == [[0, 1], [1, 1]]
    assert ds.examples[0].category == "math"


def test_synthetic_is_deterministic_and_harder_items_fail_more():
    a, b = load_synthetic(n=200, seed=3), load_synthetic(n=200, seed=3)
    assert [e.text for e in a.examples] == [e.text for e in b.examples]
    q = a.quality_matrix()
    assert q[:, 0].mean() < q[:, -1].mean()


def test_swebench_router_text_is_the_issue():
    from dmroute.data.llmrouterbench import router_text

    rec = {"origin_query": "preamble <issue>\nFix the bug\n</issue> <code>huge</code>"}
    assert router_text("swe-bench", rec) == "Fix the bug"
    assert router_text("aime", {"origin_query": "x"}) == "x"
