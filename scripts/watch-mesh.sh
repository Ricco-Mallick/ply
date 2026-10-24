#!/usr/bin/env bash
J=mesh-classroom-1; API=https://ply-api.ricco-ply.workers.dev; M="env MODAL_PROFILE=ricco-mallick /home/ricco/ply/pipeline/.venv/bin/modal"
last=""; t0=$(date +%s)
while :; do
  s=$(curl -s -m 20 -A ply-admin "$API/v1/jobs/$J" | python3 -c 'import json,sys;j=json.load(sys.stdin);print(j["status"],j.get("stage"),round(j.get("progress") or 0,2),(j.get("message") or "")[-90:])' 2>/dev/null)
  k=$(echo "$s" | cut -c1-70); [ "$k" != "$last" ] && echo "$(date +%H:%M:%S) $s" >> /home/ricco/ply/scripts/mesh.log && last="$k"
  case "$s" in done*|failed*) echo "ENDED: $s" >> /home/ricco/ply/scripts/mesh.log; break;; esac
  m=$($M billing summary --json 2>/dev/null | python3 -c 'import json,sys;print(round(float(json.load(sys.stdin)["metered_cost"]),2))' 2>/dev/null)
  python3 -c "import sys;sys.exit(0 if float('${m:-0}') > 26.0 else 1)" && { echo "SPEND STOP \$$m" >> /home/ricco/ply/scripts/mesh.log; $M app stop ply -y >/dev/null 2>&1; break; }
  [ $(( $(date +%s) - t0 )) -gt 14400 ] && { echo "4h cap" >> /home/ricco/ply/scripts/mesh.log; $M app stop ply -y >/dev/null 2>&1; break; }
  sleep 90
done
