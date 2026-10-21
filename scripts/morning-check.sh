#!/usr/bin/env bash
# verify both high-quality twins the way a viewer would: files served, place count, tour, spawn inside bounds
API=https://ply-api.ricco-ply.workers.dev
for J in hq-classroom-1 hq-chapel-1; do
  echo "== $J"
  curl -s -m 20 -A ply-admin "$API/v1/jobs/$J" | python3 -c '
import json,sys
j=json.load(sys.stdin); r=j.get("result") or {}; rep=r.get("report") or {}
print(" status:", j["status"], j.get("stage"), "| message:", (j.get("message") or "")[:120])
print(" files:", list((r.get("files") or {}).keys()))
if rep.get("export"): e=rep["export"]; print(" export:", {k:e.get(k) for k in ("spzMB","liteMB","gaussiansIn","gaussiansOut","cropDropped","floaterFilter")})
if rep.get("sfm"): s=rep["sfm"]; print(" sfm:", {k:s.get(k) for k in ("images","registered","trajectoryJumps","meanReprojErrorPx")})'
  for f in scene.spz scene_lite.spz scene.json places.json; do
    printf " %-14s " "$f"; curl -s -m 30 -o /dev/null -w "%{http_code} %{size_download}B\n" -A ply-admin "$API/v1/scenes/$J/$f"
  done
  curl -s -m 20 -A ply-admin "$API/v1/scenes/$J/scene.json" | python3 -c '
import json,sys
d=json.load(sys.stdin); b=d["bounds"]; sp=d["spawn"]["position"]
inside = b["min"][0]<=sp[0]<=b["max"][0] and b["min"][2]<=sp[2]<=b["max"][2]
print(" spawn", [round(v,1) for v in sp], "inside room:", inside, "| tour shots:", len(d.get("tour") or []))'
  curl -s -m 20 -A ply-admin "$API/v1/scenes/$J/places.json" | python3 -c 'import json,sys;d=json.load(sys.stdin);p=d.get("places") or [];print(" places:",len(p))'
done
