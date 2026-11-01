# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Assemble a room bundle for a modelling agent: the evidence, the measurements, and a folder to model in.

  python scripts/agent_bundle.py <job-id>            # pull results from the API/storage
  python scripts/agent_bundle.py --out <job-out-dir> # use a local pipeline output directory

The bundle is deliberately small and readable: shell.json, places.json, scene.json, a spread of posed frames,
and report.md — the measurement sheet an agent reads before it models anything. Frames are optional: a job without
evidence frames still produces the measurement sheet.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


API = os.environ.get("PLY_API", "https://ply-api.ricco-ply.workers.dev")
UA = {"user-agent": "ply-agent-bundle/1.0"}


def fetch(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["curl", "-sSL", "-A", UA["user-agent"], "-o", str(dest), "-w", "%{http_code}", url], capture_output=True, text=True)
    if r.stdout.strip() != "200":
        dest.unlink(missing_ok=True)
        return False
    return True


def api_json(path: str):
    out = subprocess.run(["curl", "-sS", "-A", UA["user-agent"], f"{API}{path}"], capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except Exception:
        return None


def measurement_sheet(shell: dict, places: dict, scene: dict) -> str:
    lines = ["# Measurement sheet", ""]
    walls = shell.get("walls") or []
    floor = shell.get("floorY", 0)
    ceil = shell.get("ceilingY")
    lines += [f"- floor at y = {floor:.2f} m", f"- ceiling estimate {ceil:.2f} m (splat percentile: verify against frames)" if ceil else "- ceiling: not measured", ""]
    lines += [f"## Walls ({len(walls)})", "", "| # | from | to | length (m) | angle (deg) |", "|---|---|---|---|---|"]
    for i, w in enumerate(walls):
        a, b = w["a"], w["b"]
        lines.append(f"| {i} | ({a[0]:.2f}, {a[1]:.2f}) | ({b[0]:.2f}, {b[1]:.2f}) | {w['length']:.2f} | {w.get('angle', 0):.1f} |")
    lines += ["", f"## Openings ({len(shell.get('openings') or [])})", "", "| kind | wall | at (m along) | width | sill | height |", "|---|---|---|---|---|---|"]
    for o in shell.get("openings") or []:
        lines.append(f"| {o['kind']} | {o['wall']} | {o['u']:.2f} | {o['width']:.2f} | {o['sill']:.2f} | {o['height']:.2f} |")
    objects = shell.get("objects") or (places.get("places") or [])
    counts: dict[str, int] = {}
    for o in objects:
        counts[o.get("kind") or (o.get("ai") or {}).get("cls") or "?"] = counts.get(o.get("kind") or (o.get("ai") or {}).get("cls") or "?", 0) + 1
    lines += ["", f"## Objects ({len(objects)})", "", "| kind | count |", "|---|---|"]
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    biggest = sorted(objects, key=lambda o: -((o.get("half") or [0, 0, 0])[0] * (o.get("half") or [0, 0, 0])[2]))[:30]
    lines += ["", "### The 30 largest (model these properly)", "", "| id | kind | centre (x, z) | footprint (m) | height (m) |", "|---|---|---|---|---|"]
    for o in biggest:
        h = o.get("half") or [0.1, 0.1, 0.1]
        c = o.get("center") or o.get("p") or [0, 0, 0]
        lines.append(f"| {o.get('id')} | {o.get('kind') or (o.get('ai') or {}).get('cls')} | ({c[0]:.2f}, {c[2]:.2f}) | {2*h[0]:.2f} × {2*h[2]:.2f} | {2*h[1]:.2f} |")
    if scene.get("spawn"):
        p = scene["spawn"]["position"]
        lines += ["", f"- spawn point: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"]
    lines += ["", "## Rules", "", "- metres, +Y up, floor at 0, scene frame",
              "- measurements are ±5 cm from the photogrammetric solve; say so where it matters",
              "- model the hero objects, proxy the rest", "- verify with scripts/render_pose.mjs + scripts/compare.py"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", nargs="?", help="job id (or omit with --out)")
    ap.add_argument("--out", type=Path, help="local pipeline output directory")
    ap.add_argument("--dest", type=Path, default=None)
    a = ap.parse_args()

    dest = a.dest or Path(f"room-{a.job or 'local'}")
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "evidence" / "frames").mkdir(parents=True)
    (dest / "model").mkdir()

    if a.out:
        out = a.out
        for name in ("shell.json", "places.json", "scene.json"):
            if (out / name).exists():
                shutil.copyfile(out / name, dest / name)
        # evidence frames, if the pipeline kept a spread
        for src in sorted((out / "evidence" / "frames").glob("*.jpg"))[:48]:
            shutil.copyfile(src, dest / "evidence" / "frames" / src.name)
        if (out / "evidence" / "poses.json").exists():
            shutil.copyfile(out / "evidence" / "poses.json", dest / "evidence" / "poses.json")
    else:
        base = f"{API}/v1/scenes/{a.job}"
        for name in ("shell.json", "places.json", "scene.json"):
            if not fetch(f"{base}/{name}", dest / name):
                print(f"missing {name}", file=sys.stderr)
        fetch(f"{base}/evidence/poses.json", dest / "evidence" / "poses.json")
        poses = json.loads((dest / "evidence" / "poses.json").read_text()) if (dest / "evidence" / "poses.json").exists() else []
        for p in poses[:48]:
            fetch(f"{base}/{p['file']}", dest / "evidence" / "frames" / Path(p["file"]).name)

    shell = json.loads((dest / "shell.json").read_text()) if (dest / "shell.json").exists() else {}
    places = json.loads((dest / "places.json").read_text()) if (dest / "places.json").exists() else {}
    scene = json.loads((dest / "scene.json").read_text()) if (dest / "scene.json").exists() else {}
    (dest / "report.md").write_text(measurement_sheet(shell, places, scene))
    print(f"bundle at {dest.resolve()}")
    print(f"  walls {len(shell.get('walls') or [])} · openings {len(shell.get('openings') or [])} · objects {len(shell.get('objects') or [])} · frames {len(list((dest / 'evidence' / 'frames').glob('*.jpg')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
