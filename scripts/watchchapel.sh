#!/usr/bin/env bash
# the chapel is the last demo: stop Modal when it ends, or after a 4-hour cap
J=b0f7efc9-de2f-445c-8a00-0b6fd8a33ea1; M=~/ply/pipeline/.venv/bin/modal; t0=$(date +%s); last=""
while :; do
  s=$(curl -s -A ply-admin https://ply-api.ricco-ply.workers.dev/v1/jobs/$J | python3 -c 'import json,sys;j=json.load(sys.stdin);print(j["status"],j.get("stage"),round(j.get("progress") or 0,2),(j.get("message") or "")[-70:])' 2>/dev/null)
  k=$(echo "$s" | cut -c1-60); [ "$k" != "$last" ] && echo "$(date +%H:%M:%S) $s" && last="$k"
  case "$s" in done*|failed*) echo "CHAPEL ENDED: $s"; timeout 30 $M app stop ply -y >/dev/null 2>&1; break;; esac
  [ $(( $(date +%s) - t0 )) -gt 14400 ] && { echo "4h cap"; timeout 30 $M app stop ply -y; break; }
  sleep 60
done
