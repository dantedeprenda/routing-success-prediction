import json
from pathlib import Path

import yaml

from dmroute.experiment import run
from dmroute.report import report

ROOT = Path(__file__).parents[1]


def test_synthetic_config_runs_and_reports(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "synthetic.yaml").read_text())
    cfg["dataset"]["n"] = 200
    cfg["split"]["seeds"] = [42]
    cfg["output_dir"] = str(tmp_path / "out")
    cfg["cache"] = str(tmp_path / "cache.sqlite")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))

    out = run(str(path))
    records = json.loads((out / "metrics.json").read_text())
    names = {r["router"] for r in records}
    assert {"oracle", "stub:choice", "stub:features", "judge-stub", "knn10"} <= names
    assert any(n.startswith("stub:choice>judge-stub@") for n in names)

    by = {(r["router"], r["variant"], r["k"]): r for r in records if r["router"] != "_reference"}
    oracle = by[("oracle", "raw", None)]
    for (name, _, k), r in by.items():
        if k is None:
            assert r["aiq"] <= oracle["aiq"] + 1e-9, name
    assert "calibration" in by[("stub:difficulty", "platt", None)]

    text = report(str(out))
    assert "stub:choice" in text and (out / "cost_quality.png").exists()


def test_twinrouterbench_fixture_gets_tier_metrics(tmp_path):
    cfg = {
        "name": "trb",
        "output_dir": str(tmp_path / "out"),
        "cache": str(tmp_path / "cache.sqlite"),
        "dataset": {
            "loader": "twinrouterbench",
            "path": str(ROOT / "tests" / "fixtures" / "twinrouterbench_sample.jsonl"),
        },
        "split": {"test_frac": 0.34, "seeds": [0]},
        "routers": [
            {"type": "always_each"},
            {
                "type": "decision",
                "label": "stub",
                "backend": {"kind": "stub"},
                "formulations": ["difficulty"],
            },
        ],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    records = json.loads((run(str(path)) / "metrics.json").read_text())
    high = next(r for r in records if r["router"] == "always:3")
    assert high["tier"]["row_pass"] == 1.0


def test_dev_only_never_touches_test_split(tmp_path):
    from dmroute.data.splits import grouped_split
    from dmroute.data.synthetic import load_synthetic

    cfg = {
        "name": "dev",
        "output_dir": str(tmp_path / "out"),
        "cache": str(tmp_path / "cache.sqlite"),
        "dataset": {"loader": "synthetic", "n": 100},
        "split": {"test_frac": 0.3, "seeds": [7], "dev_only": True},
        "routers": [{"type": "always", "model": 0}],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = run(str(path))
    _, test = grouped_split(load_synthetic(n=100), 0.3, 7)
    test_ids = {f"syn-{i}" for i in test}
    seen = {json.loads(line)["id"] for line in (out / "seed7_kNone" / "always_0.jsonl").open()}
    assert seen and not seen & test_ids
