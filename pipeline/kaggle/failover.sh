#!/usr/bin/env bash
# Follow one job across Kaggle accounts. Account 1 = ~/.kaggle (riccomarcelino), account 2 = ~/.kaggle2 (owner in
# ~/.kaggle2/owner). When the active run ends before the job is done (quota out, machine lost, late boot), requeue the
# job and start the worker on the next account. Stops Kaggle as soon as the job is done or fails, so nothing idles.
J=${1:?usage: failover.sh <job-id> <account-1 kernel owner/slug>}
ACTIVE=1; SLUG1=${2:?usage: failover.sh <job-id> <account-1 kernel owner/slug>}; SLUG2=""; started=$(date +%s); booted=0
ROOT=~/ply; K=$ROOT/pipeline/.venv/bin/kaggle
source $ROOT/cloud/.secrets
api() { curl -s -m 15 -A ply-admin "https://ply-api.ricco-ply.workers.dev/v1/$1"; }
kag() { if [ "$ACTIVE" = 1 ]; then KAGGLE_CONFIG_DIR=~/.kaggle $K "$@"; else KAGGLE_CONFIG_DIR=~/.kaggle2 $K "$@"; fi; }
slug() { [ "$ACTIVE" = 1 ] && echo "$SLUG1" || echo "$SLUG2"; }
stop_active() { kag kernels delete "$(slug)" -y 2>&1 | tail -1; }
requeue() { curl -s -X POST "https://ply-api.ricco-ply.workers.dev/v1/worker/jobs/$J" -H "authorization: Bearer $WORKER_TOKEN" -H 'content-type: application/json' -A ply-admin -d "{\"status\":\"queued\",\"stage\":null,\"progress\":0,\"message\":\"$1\"}" >/dev/null; }
failover() {
  echo "$(date +%H:%M:%S) FAILOVER: $1"
  if [ "$ACTIVE" = 1 ] && [ -f ~/.kaggle2/owner ]; then
    OWNER2=$(cat ~/.kaggle2/owner); SLUG2="$OWNER2/ply-worker-a2"
    requeue "moving to the second GPU account"
    bash $ROOT/pipeline/kaggle/start_worker.sh ~/.kaggle2 "$OWNER2" ply-worker-a2 2>&1 | tail -3
    ACTIVE=2; started=$(date +%s); booted=0; return 0
  fi
  echo "no further account to fail over to"; exit 0
}
last=""; lastchange=$(date +%s)
while true; do
  line=$(api "jobs/$J" | python3 -c 'import sys,json;j=json.load(sys.stdin);print(j["status"],j["stage"],round(j["progress"],3),(j["message"] or "")[-300:].replace(chr(10)," | "))' 2>/dev/null)
  if [ "$line" != "$last" ]; then echo "$(date +%H:%M:%S) [acct $ACTIVE] $line"; last="$line"; lastchange=$(date +%s); fi
  case "$line" in done*) echo "JOB DONE"; stop_active; exit 0;; failed*) echo "JOB FAILED"; stop_active; exit 0;; esac
  h=$(api health); echo "$h" | grep -q '"workers":\[{' && booted=1
  st=$(kag kernels status "$(slug)" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*')
  case "$st" in *ERROR*|*COMPLETE*|*CANCEL*) failover "account $ACTIVE run ended ($st) before the job finished";; esac
  if [ "$booted" = 0 ] && [ $(( $(date +%s) - started )) -gt 480 ]; then stop_active; failover "account $ACTIVE run never checked in within 8 min"; fi
  if [ $(( $(date +%s) - lastchange )) -gt 1200 ]; then echo "NO CHANGE FOR 20 MIN"; stop_active; failover "stalled on account $ACTIVE"; fi
  sleep 20
done
