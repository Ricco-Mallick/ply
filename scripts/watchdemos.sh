#!/usr/bin/env bash
# follow both demo jobs; stop Modal on failure, past a spend ceiling, or after 4 hours
A=9b20d642-0979-4abc-99e0-2a1cf94e19f5; B=b0f7efc9-de2f-445c-8a00-0b6fd8a33ea1; M=~/ply/pipeline/.venv/bin/modal
t0=$(date +%s); last=""; base=5.44
while :; do
  a=$(curl -s -A ply-admin https://ply-api.ricco-ply.workers.dev/v1/jobs/$A | python3 -c 'import json,sys;j=json.load(sys.stdin);print(j["status"],j.get("stage"),round(j.get("progress") or 0,2),(j.get("message") or "")[-90:])' 2>/dev/null)
  b=$(curl -s -A ply-admin https://ply-api.ricco-ply.workers.dev/v1/jobs/$B | python3 -c 'import json,sys;j=json.load(sys.stdin);print(j["status"],j.get("stage"),round(j.get("progress") or 0,2),(j.get("message") or "")[-90:])' 2>/dev/null)
  cur="$a | $b"
  [ "$cur" != "$last" ] && echo "$(date +%H:%M:%S) $cur" && last="$cur"
  case "$a" in failed*) echo "CLASSROOM FAILED: stopping Modal"; $M app stop ply -y >/dev/null 2>&1; break;; esac
  case "$b" in failed*) echo "CHAPEL FAILED: stopping Modal"; $M app stop ply -y >/dev/null 2>&1; break;; esac
  case "$a" in done*) case "$b" in done*) echo "BOTH DONE"; break;; esac;; esac
  m=$($M billing summary --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["metered_cost"])' 2>/dev/null)
  if python3 -c "import sys;sys.exit(0 if float('${m:-0}') > ${base} + 9 else 1)"; then echo "SPEND CEILING at \$$m: stopping"; $M app stop ply -y >/dev/null 2>&1; break; fi
  [ $(( $(date +%s) - t0 )) -gt 14400 ] && { echo "4h cap: stopping"; $M app stop ply -y >/dev/null 2>&1; break; }
  sleep 60
done
