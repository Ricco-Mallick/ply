#!/usr/bin/env bash
set -u
API=https://ply-api.ricco-ply.workers.dev
M="env MODAL_PROFILE=ricco-mallick /home/ricco/ply/pipeline/.venv/bin/modal"
J=hq-chapel-1; L=/home/ricco/ply/scripts/night.log; tries=0
say(){ echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$L"; }
while :; do
  st=$(curl -s -m 20 -A ply-admin "$API/v1/jobs/$J" | python3 -c 'import json,sys;j=json.load(sys.stdin);print(j["status"], j.get("stage") or "-", round(j.get("progress") or 0,2))' 2>/dev/null)
  case "$st" in
    done*) say "chapel done: $st"; break;;
    failed*)
      tries=$((tries+1))
      if [ $tries -le 1 ]; then
        say "chapel failed ($st); requeueing once"
        cd /home/ricco/ply/cloud; printf "UPDATE jobs SET status='queued', stage=NULL, progress=0, attempts=0, message='auto-retry after failure' WHERE id='%s';\n" "$J" > ./rt.sql
        WRANGLER_SEND_METRICS=false timeout 60 npx wrangler@4.149.0 d1 execute ply-jobs --remote --file ./rt.sql -y >/dev/null 2>&1; rm -f ./rt.sql
        MODAL_PROFILE=ricco-mallick timeout 120 /home/ricco/ply/pipeline/.venv/bin/python /home/ricco/ply/scripts/spawn_worker.py >/dev/null 2>&1
      else say "chapel failed twice; leaving it"; break; fi;;
  esac
  sleep 120
done
m=$($M billing summary --json 2>/dev/null | python3 -c 'import json,sys;print(round(float(json.load(sys.stdin)["metered_cost"]),2))' 2>/dev/null)
say "stopping Modal (spend \$$m)"; $M app stop ply -y >/dev/null 2>&1
