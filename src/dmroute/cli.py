"""dmroute command line.

dmroute run configs/synthetic.yaml       # predict (cached) + evaluate
dmroute report results/synthetic          # summary.md + cost_quality.png
dmroute fetch twinrouterbench             # download benchmark data into data/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SOURCES = {
    "twinrouterbench": (
        "https://huggingface.co/datasets/Amorph/TwinRouterBench/resolve/main/question_bank.jsonl",
        "data/twinrouterbench/question_bank.jsonl",
    ),
    "llmrouterbench": (
        "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz",
        "data/llmrouterbench/bench-release.tar.gz",
    ),
}


def _fetch(name: str) -> None:
    url, dest = SOURCES[name]
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(["curl", "-L", "--fail", "-o", str(path), url], check=True)
    if path.suffixes[-2:] == [".tar", ".gz"]:
        subprocess.run(["tar", "xzf", path.name], cwd=path.parent, check=True)
    print(f"{name}: {path.parent}")


def _load_dotenv(path: str = ".env") -> None:
    """Set KEY=VALUE lines from .env without overriding the real environment."""
    env = Path(path)
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key and not key.startswith("#") and value.strip():
            os.environ.setdefault(key, value.strip().strip("'\""))


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="dmroute")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run an experiment config")
    p_run.add_argument("config")
    p_run.add_argument("--no-report", action="store_true")
    p_rep = sub.add_parser("report", help="summarize a results directory")
    p_rep.add_argument("results_dir")
    p_fetch = sub.add_parser("fetch", help="download a benchmark")
    p_fetch.add_argument("name", choices=sorted(SOURCES))
    args = parser.parse_args(argv)

    if args.cmd == "run":
        from dmroute.experiment import run
        from dmroute.report import report

        out = run(args.config)
        print(f"wrote {out}/metrics.json")
        if not args.no_report:
            print(report(str(out)))
    elif args.cmd == "report":
        from dmroute.report import report

        print(report(args.results_dir))
    elif args.cmd == "fetch":
        _fetch(args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
