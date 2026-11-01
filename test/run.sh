#!/usr/bin/env bash
# Reconstruct a capture with Ply.  Usage: ./test/run.sh <folder-of-clips> [out-dir]
# Requires the pipeline environment (see pipeline/README.md). Settings are planned from the capture itself.
set -euo pipefail
IN=${1:?usage: ./test/run.sh <folder-of-clips> [out-dir]}
OUT=${2:-work/$(basename "$IN")}
mkdir -p "$(dirname "$OUT")"
python -m plypipe run "$IN" --out "$OUT" --preset standard
echo
echo "done. scene manifest: $OUT/scene.json"
echo "open it at    https://ricco.codes/ply/walk.html?scene=<public url of that scene.json>"
