#!/usr/bin/env bash
# append every stage/progress change of the two HQ jobs to scripts/progress.log
API=https://ply-api.ricco-ply.workers.dev; L=/home/ricco/ply/scripts/progress.log; last=""
while :; do
  line=""
  for j in hq-classroom-1 hq-chapel-1; do
    s=$(curl -s -m 20 -A ply-admin "$API/v1/jobs/$j" | python3 -c 'import json,sys;j=json.load(sys.stdin);print(j["status"],j["stage"],round(j["progress"] or 0,2),(j["message"] or "")[-60:])' 2>/dev/null)
    line="$line | $j: $s"
  done
  k=$(echo "$line" | cut -c1-90)
  [ "$k" != "$last" ] && echo "$(date '+%m-%d %H:%M:%S')$line" >> "$L" && last="$k"
  case "$line" in *"hq-classroom-1: done"*) case "$line" in *"hq-chapel-1: done"*) break;; esac;; esac
  sleep 120
done
