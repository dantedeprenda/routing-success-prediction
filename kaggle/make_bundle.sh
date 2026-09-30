#!/usr/bin/env bash
# Build kaggle/dmroute-bundle.zip: code + configs + only the data the Laya runs read.
# Upload it as a private Kaggle Dataset named "dmroute-bundle" (see kaggle/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."

out=kaggle/build/dmroute-bundle
rm -rf kaggle/build
mkdir -p "$out/data/twinrouterbench"

cp -R src configs pyproject.toml README.md "$out/"
find "$out" -name __pycache__ -prune -exec rm -rf {} +
cp data/design_seen_lrb.txt data/design_seen_trb.txt "$out/data/"
cp data/twinrouterbench/question_bank.jsonl "$out/data/twinrouterbench/"

# LLMRouterBench: the six test datasets x three pool models, latest run file only
# (what llmrouterbench._latest_records reads). ~360 MB instead of 6.5 GB.
uv run python - "$out" <<'EOF'
import sys, yaml, shutil
from pathlib import Path
out = Path(sys.argv[1])
cfg = yaml.safe_load(Path("configs/kaggle/llmrouterbench_laya.yaml").read_text())["dataset"]
root = Path(cfg["root"])
models = [m["name"] for m in cfg["pool"]]
n = 0
for ds in cfg["datasets"]:
    for split in sorted(p for p in (root / ds).iterdir() if p.is_dir()):
        for m in models:
            files = sorted((split / m).glob("*.json"))
            if files:
                dest = out / files[-1]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(files[-1], dest)
                n += 1
print(f"llmrouterbench: {n} files")
EOF

(cd kaggle/build && zip -qr ../dmroute-bundle.zip dmroute-bundle)
rm -rf kaggle/build
ls -lh kaggle/dmroute-bundle.zip
