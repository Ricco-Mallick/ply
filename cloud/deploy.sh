#!/usr/bin/env bash
# One-shot setup of the Ply API (Cloudflare) + GPU worker (Kaggle). Safe to re-run.
# Needs: `npx wrangler login` done, cloud/.b2 with a Backblaze B2 key (no card needed), ~/.kaggle token, gh with Ricco-Mallick.
set -euo pipefail
cd "$(dirname "$0")"
W="npx --yes wrangler@4.149.0"; export WRANGLER_SEND_METRICS=false
source .secrets   # APP_KEY, WORKER_TOKEN (generated locally, gitignored)

# storage: private B2 bucket with CORS for direct part uploads; the Worker gets its S3 endpoint + key
B2OUT=$(python3 b2_setup.py) || exit 1
eval "$B2OUT"
source .b2
sed -i -e "s#^S3_ENDPOINT = .*#S3_ENDPOINT = \"$S3_ENDPOINT\"#" -e "s#^S3_REGION = .*#S3_REGION = \"$S3_REGION\"#" -e "s#^S3_BUCKET = .*#S3_BUCKET = \"$S3_BUCKET\"#" wrangler.toml
printf '%s' "$B2_KEY_ID" | $W secret put S3_KEY_ID >/dev/null
printf '%s' "$B2_APP_KEY" | $W secret put S3_SECRET >/dev/null
$W d1 execute ply-jobs --remote --file schema.sql -y >/dev/null
printf '%s' "$APP_KEY" | $W secret put APP_KEY >/dev/null
printf '%s' "$WORKER_TOKEN" | $W secret put WORKER_TOKEN >/dev/null
URL=$($W deploy 2>&1 | grep -o 'https://[a-z0-9.-]*workers\.dev' | head -1)
echo "API: $URL"; curl -fsS "$URL/" && echo

# website -> API, then redeploy Pages
export GH_TOKEN=$(gh auth token --user Ricco-Mallick)
gh variable set PLY_API -R Ricco-Mallick/ply -b "$URL"
gh workflow run pages.yml -R Ricco-Mallick/ply

# Kaggle: private config dataset + worker kernel (GPU T4, internet on)
K=../pipeline/.venv/bin/kaggle; T=$(mktemp -d)
mkdir -p "$T/cfg" "$T/kernel"
printf '{"api": "%s", "token": "%s"}\n' "$URL" "$WORKER_TOKEN" > "$T/cfg/config.json"
printf '{"title": "ply-worker-config", "id": "riccomarcelino/ply-worker-config", "licenses": [{"name": "other"}]}\n' > "$T/cfg/dataset-metadata.json"
$K datasets create -p "$T/cfg" 2>/dev/null || $K datasets version -p "$T/cfg" -m "config"
cp ../pipeline/kaggle/ply_worker.ipynb "$T/kernel/"
cat > "$T/kernel/kernel-metadata.json" <<META
{ "id": "riccomarcelino/ply-worker", "title": "ply-worker", "code_file": "ply_worker.ipynb", "language": "python",
  "kernel_type": "notebook", "is_private": true, "enable_gpu": true, "enable_internet": true, "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["riccomarcelino/ply-pipeline", "riccomarcelino/ply-worker-config"], "competition_sources": [], "kernel_sources": [] }
META
$K kernels push -p "$T/kernel"
rm -rf "$T"
echo "Team key for the capture app: $APP_KEY"
