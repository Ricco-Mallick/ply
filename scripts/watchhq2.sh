#!/usr/bin/env bash
J=hq-classroom-1; M="env MODAL_PROFILE=ricco-mallick /home/ricco/ply/pipeline/.venv/bin/modal"; last=""; t0=$(date +%s)
while :; do
  s=$(curl -s -A ply-admin https://ply-api.ricco-ply.workers.dev/v1/jobs/$J | python3 -c 'import json,sys;j=json.load(sys.stdin);print(j["status"],j.get("stage"),round(j.get("progress") or 0,2),(j.get("worker") or "-"),(j.get("message") or "")[-65:])' 2>/dev/null)
  k=$(echo "$s" | cut -c1-70); [ "$k" != "$last" ] && echo "$(date +%H:%M:%S) $s" && last="$k"
  case "$s" in done*|failed*) echo "ENDED: $s"; timeout 30 $M app stop ply -y >/dev/null 2>&1; break;; esac
  m=$($M billing summary --json 2>/dev/null | python3 -c 'import json,sys;print(round(float(json.load(sys.stdin)["metered_cost"]),2))' 2>/dev/null)
  python3 -c "import sys;sys.exit(0 if float('${m:-0}') > 10.0 else 1)" && { echo "SPEND STOP at \$$m"; timeout 30 $M app stop ply -y; break; }
  [ $(( $(date +%s) - t0 )) -gt 4800 ] && { echo "80 min cap"; timeout 30 $M app stop ply -y; break; }
  sleep 60
done
