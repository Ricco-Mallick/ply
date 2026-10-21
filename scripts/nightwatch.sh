#!/usr/bin/env bash
# overnight supervisor: keep the two high-quality twins moving, fix what fails, stop Modal at the end.
# state lives in JSON rows; every action is logged with a timestamp.
set -u
API=https://ply-api.ricco-ply.workers.dev
M="env MODAL_PROFILE=ricco-mallick /home/ricco/ply/pipeline/.venv/bin/modal"
LOG=/home/ricco/ply/scripts/night.log
JOBS="hq-classroom-1 hq-chapel-1"
ATT="hq-classroom-1=0 hq-chapel-1=0"
say(){ echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
spend(){ $M billing summary --json 2>/dev/null | python3 -c 'import json,sys;print(round(float(json.load(sys.stdin)["metered_cost"]),2))' 2>/dev/null || echo 0; }
field(){ curl -s -m 20 -A ply-admin "$API/v1/jobs/$1" | python3 -c "import json,sys;j=json.load(sys.stdin);print(j.get('$2') or '')" 2>/dev/null; }
requeue(){ curl -s -m 20 -A ply-admin >/dev/null 2>&1; cd /home/ricco/ply/cloud; printf "UPDATE jobs SET status='queued', stage=NULL, progress=0, attempts=0, message='%s' WHERE id='%s';\n" "$2" "$1" > ./rw.sql; WRANGLER_SEND_METRICS=false timeout 60 npx wrangler@4.149.0 d1 execute ply-jobs --remote --file ./rw.sql -y >/dev/null 2>&1; rm -f ./rw.sql; }
say "supervisor up; spend $(spend)"
while :; do
  tot=$(spend); case "$tot" in ''|*[!0-9.]*) tot=0;; esac
  if python3 -c "import sys;sys.exit(0 if float('$tot') > 24.0 else 1)"; then say "SPEND CEILING \$$tot: stopping Modal"; $M app stop ply -y >/dev/null 2>&1; break; fi
  done_n=0; live=0
  for j in $JOBS; do
    st=$(field "$j" status); stage=$(field "$j" stage); msg=$(field "$j" message)
    printf -v "st_${j//-/_}" '%s' "$st"
    case "$st" in
      done) done_n=$((done_n+1));;
      failed)
        n=$(echo "$ATT" | tr ' ' '\n' | sed -n "s/^$j=//p"); n=${n:-0}
        if [ "$n" -lt 3 ]; then
          n=$((n+1)); ATT=$(echo "$ATT" | sed "s/$j=[0-9]*/$j=$n/")
          say "$j failed (attempt $n): ${msg: -160}; requeueing"
          requeue "$j" "auto-retry $n after: ${msg: -100}"
          live=1
        else say "$j failed $n times; leaving it: ${msg: -200}"; fi;;
      running)
        beat=$(curl -s -m 20 -A ply-admin "$API/v1/jobs/$j" | python3 -c 'import json,sys,time;j=json.load(sys.stdin);print(int(time.time()-j["updated_at"]/1000))' 2>/dev/null || echo 0)
        if [ "${beat:-0}" -gt 600 ]; then say "$j stale (${beat}s); requeueing"; requeue "$j" "auto-retry: stale heartbeat"; fi
        live=1;;
      queued) live=1;;
    esac
  done
  [ "$done_n" -eq 2 ] && { say "both twins done; stopping Modal"; $M app stop ply -y >/dev/null 2>&1; break; }
  sleep 90
done
say "supervisor exit; spend $(spend)"
