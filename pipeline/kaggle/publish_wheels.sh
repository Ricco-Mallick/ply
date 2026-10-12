#!/usr/bin/env bash
# After a successful `wheels` workflow run: pull its wheel artifacts, publish them as the private Kaggle dataset
# riccomarcelino/ply-wheels (flat folder), and start the GPU worker with it mounted, so the install takes minutes.
# Usage: pipeline/kaggle/publish_wheels.sh <run-id> <kernel-dir-with-kernel-metadata.json>
set -euo pipefail
RUN="$1"; KDIR="$2"
HERE="$(cd "$(dirname "$0")/.." && pwd)"; K="$HERE/.venv/bin/kaggle"
export GH_TOKEN=$(gh auth token --user Ricco-Mallick)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
gh run download "$RUN" -R Ricco-Mallick/ply -D "$T/art"
mkdir -p "$T/ply-wheels"; find "$T/art" -name '*.whl' -exec cp {} "$T/ply-wheels/" \;
ls -la "$T/ply-wheels"
for p in gsplat fused_ssim fused_bilagrid ppisp; do ls "$T/ply-wheels" | grep -qi "^$p" || { echo "missing wheel: $p"; exit 1; }; done
printf '{ "title": "ply-wheels", "id": "riccomarcelino/ply-wheels", "licenses": [{ "name": "other" }] }\n' > "$T/ply-wheels/dataset-metadata.json"
if $K datasets status riccomarcelino/ply-wheels >/dev/null 2>&1; then $K datasets version -p "$T/ply-wheels" -m "run $RUN"; else $K datasets create -p "$T/ply-wheels"; fi
until $K datasets status riccomarcelino/ply-wheels 2>/dev/null | grep -q ready; do sleep 10; done
# the worker kernel mounts the wheels next to the code + config
python3 - "$KDIR/kernel-metadata.json" <<'PY'
import json, sys
p = sys.argv[1]; m = json.load(open(p))
if "riccomarcelino/ply-wheels" not in m["dataset_sources"]:
    m["dataset_sources"].append("riccomarcelino/ply-wheels")
json.dump(m, open(p, "w"), indent=1)
PY
cp "$HERE/kaggle/ply_worker.ipynb" "$KDIR/"
$K kernels push -p "$KDIR"
