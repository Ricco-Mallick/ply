#!/usr/bin/env bash
# Start the Ply GPU worker on any Kaggle account (so a second account can take over when one's weekly GPU quota runs
# out). Jobs, captures and results live in the Ply API + B2, never on Kaggle, so switching accounts loses nothing.
#
#   pipeline/kaggle/start_worker.sh <kaggle-config-dir> <kaggle-username> <kernel-slug>
#   e.g. pipeline/kaggle/start_worker.sh ~/.kaggle2 secondaccount ply-worker-a2
#
# Creates/updates on that account: private datasets ply-pipeline (this repo's pipeline/ at HEAD) and ply-worker-config
# (API URL + worker token from cloud/.secrets), then pushes the worker notebook (GPU T4, internet on).
set -euo pipefail
CFG_DIR="$1"; OWNER="$2"; SLUG="$3"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; K="$ROOT/pipeline/.venv/bin/kaggle"
export KAGGLE_CONFIG_DIR="$CFG_DIR"
source "$ROOT/cloud/.secrets"
API="${PLY_API:-https://ply-api.ricco-ply.workers.dev}"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; umask 077

ds() {  # create or new-version a private dataset from a folder
  local name="$1" dir="$2"
  printf '{ "title": "%s", "id": "%s/%s", "licenses": [{ "name": "other" }] }\n' "$name" "$OWNER" "$name" > "$dir/dataset-metadata.json"
  if $K datasets status "$OWNER/$name" >/dev/null 2>&1; then $K datasets version -p "$dir" --dir-mode zip -m "update" >/dev/null
  else $K datasets create -p "$dir" --dir-mode zip >/dev/null; fi
  until $K datasets status "$OWNER/$name" 2>/dev/null | grep -q ready; do sleep 5; done
  echo "dataset $OWNER/$name ready"
}
mkdir -p "$T/code" "$T/cfg" "$T/kernel"
(cd "$ROOT" && git archive HEAD pipeline) | tar -x -C "$T/code"
mv "$T/code/pipeline"/* "$T/code/" && rmdir "$T/code/pipeline"
ds ply-pipeline "$T/code"
printf '{"api": "%s", "token": "%s"}\n' "$API" "$WORKER_TOKEN" > "$T/cfg/config.json"
ds ply-worker-config "$T/cfg"

cp "$ROOT/pipeline/kaggle/ply_worker.ipynb" "$T/kernel/"
cat > "$T/kernel/kernel-metadata.json" <<META
{ "id": "$OWNER/$SLUG", "title": "$SLUG", "code_file": "ply_worker.ipynb", "language": "python",
  "kernel_type": "notebook", "is_private": true, "enable_gpu": true, "enable_internet": true, "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["$OWNER/ply-pipeline", "$OWNER/ply-worker-config"], "competition_sources": [], "kernel_sources": [] }
META
$K kernels push -p "$T/kernel"
