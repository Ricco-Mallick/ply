"""Phase 6, route 8a on the real scan: step 0's points-1cm.npz -> the twin folder, on a Modal CPU container.

    modal run --detach scripts/lidar/lidar_twin.py --name office-1

Starts one CPU function with .spawn() and returns: read the result from `modal app logs ply-lidar-twin` (the LIDAR_TWIN line)
or `modal volume ls ply-bench lidar/office-1/twin`. Needs the ply-bench Volume only (no secret, no GPU): the input is
lidar/<name>/points-1cm.npz from scripts/lidar/inspect_e57.py --step export, the output lidar/<name>/twin/ holds scene.spz,
scene_lite.spz, scene.json, occupancy.json/.png, shell.json, topview.webp, report.json and log.jsonl (the 3DGS .ply files
stay on the container's disk: 250 MB per million splats at SH degree 3, 62 MB at degree 0, which is what runs here).

It runs `python -m plypipe lidar` from the repository's own plypipe folder, so the twin is what the tests checked. The phone
file (scene_lite.spz, 800 k splats) is merged on the CPU by lidar_splat: splat-transform's own `-d` cut needs a WebGPU adapter
and fails on a container with no GPU (tests/test_export.py's phone-file test fails the same way on a CPU machine). The work
is capped at the High preset's 2 M splats: an office thinned to 1 cm has 5 to 30 M points, coarsened to about 1.5 to 3 cm.
Measured on 2 contended CPUs here, per million splats: 20 s for local planes, 165 s for the scanner-visibility test of the
normals (the cost, and it grows with the number of stations), 5 s for the rest of the stage, 0.55 GB of memory over a 0.6 GB
base; the 30 M input points add about 3 GB while they are thinned. Export (three splat-transform runs) is 1 to 2 minutes
at 2 M splats. So 2 M splats: 12 minutes and 2 to 5 GB. Estimate 20 minutes with the read of the npz and a slower machine;
1.5 x that is the 1800 s timeout. 4 CPUs, 12 GB: $0.000079 a second, about $0.09 expected, $0.14 at the timeout.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App("ply-lidar-twin")
vol = modal.Volume.from_name("ply-bench", create_if_missing=True)
PLYPIPE = Path(__file__).resolve().parents[2] / "pipeline" / "plypipe"
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("curl", "ca-certificates")
         .run_commands("curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs",
                       "npm i -g @playcanvas/splat-transform@3.10.1 && npx -y @playcanvas/splat-transform@3.10.1 --version")
         .pip_install("numpy>=1.26", "opencv-python-headless>=4.9", "scipy>=1.11")
         .add_local_dir(PLYPIPE, "/root/pipeline/plypipe"))
VOL, ROOT = "/vol", "/root/pipeline"  # where the container mounts the Volume and the plypipe folder
KEEP = ("scene.spz", "scene_lite.spz", "scene.json", "occupancy.json", "occupancy.png", "shell.json", "topview.webp", "report.json", "log.jsonl")


@app.function(image=image, cpu=4, memory=12288, timeout=1800, volumes={VOL: vol})
def twin(name: str = "office-1", points: str = "points-1cm.npz", args: str = "--sh-degree 0"):
    """lidar/<name>/<points> -> lidar/<name>/twin/. `args` are more plypipe lidar options, e.g. "--cap 1500000 --up z"."""
    src = Path(VOL) / "lidar" / name / points
    out = Path(VOL) / "lidar" / name / "twin"
    work = Path("/tmp/twin")
    shutil.rmtree(work, ignore_errors=True)
    t0 = time.time()
    cmd = [sys.executable, "-m", "plypipe", "lidar", str(src), "--out", str(work), *args.split()]
    print("running", " ".join(cmd), flush=True)
    run = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT})
    out.mkdir(parents=True, exist_ok=True)
    for f in KEEP:
        if (work / f).exists():
            shutil.copy2(work / f, out / f)
    vol.commit()
    report = {"exit": run.returncode, "seconds": round(time.time() - t0, 1), "twin": str(out),
              "bytes": {f.name: f.stat().st_size for f in sorted(out.iterdir())}}
    rep = work / "report.json"
    if rep.exists():
        lid = json.loads(rep.read_text()).get("lidar", {})
        report["lidar"] = {k: lid.get(k) for k in ("points", "splats", "voxelM", "fScore", "up", "yawDeg", "phaseSeconds", "peakMB")}
    print("LIDAR_TWIN " + json.dumps(report), flush=True)
    if run.returncode:
        raise RuntimeError(f"plypipe lidar exited {run.returncode}")
    return report


@app.local_entrypoint()
def main(name: str = "office-1", points: str = "points-1cm.npz", args: str = "--sh-degree 0"):
    call = twin.spawn(name, points, args)
    print(f"[lidar] twin started, call {call.object_id}; read it with `modal app logs ply-lidar-twin` (LIDAR_TWIN line) "
          f"or `modal volume ls ply-bench lidar/{name}/twin`", flush=True)
