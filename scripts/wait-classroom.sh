#!/usr/bin/env bash
API=https://ply-api.ricco-ply.workers.dev
while :; do
  s=$(curl -s -m 20 -A ply-admin "$API/v1/jobs/hq-classroom-1" | python3 -c 'import json,sys;j=json.load(sys.stdin);print(j["status"])' 2>/dev/null)
  case "$s" in done|failed) break;; esac
  sleep 60
done
echo "classroom ended: $s"
curl -s -m 20 -A ply-admin "$API/v1/scenes/hq-classroom-1/pipeline.log" | grep -E '"_probe|"stage": "export"|"stage": "train", "event": "done"' | tail -6
