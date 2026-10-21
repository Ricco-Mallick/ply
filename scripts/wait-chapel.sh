#!/usr/bin/env bash
API=https://ply-api.ricco-ply.workers.dev
while :; do
  s=$(curl -s -m 20 -A ply-admin "$API/v1/jobs/hq-chapel-1" | python3 -c 'import json,sys;j=json.load(sys.stdin);print(j["status"])' 2>/dev/null)
  case "$s" in done|failed) break;; esac
  sleep 60
done
echo "chapel ended: $s"
