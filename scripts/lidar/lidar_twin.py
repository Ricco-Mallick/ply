"""Phase 6, route 8a on the real scan: step 0's points-1cm.npz -> the twin folder, on a Modal CPU container.

    modal run --detach scripts/lidar/lidar_twin.py --name office-1
    modal run --detach scripts/lidar/lidar_twin.py --name office-1 --dst twin-8b --publish lidar-office-8b

Starts one CPU function with .spawn() and returns: read the result from `modal app logs ply-lidar-twin` (the LIDAR_TWIN line)
or `modal volume ls ply-bench lidar/office-1/twin`. Needs the ply-bench Volume only (no secret, no GPU): the input is
lidar/<name>/points-1cm.npz from scripts/lidar/inspect_e57.py --step export, the output lidar/<name>/twin/ holds scene.spz,
scene_lite.spz, scene.json, occupancy.json/.png, shell.json, topview.webp, report.json and log.jsonl (the 3DGS .ply files
stay on the container's disk: 250 MB per million splats at SH degree 3, 62 MB at degree 0, which is what runs here).

`--dst` names the volume folder (default `twin`, so a rerun replaces it; give a new name to keep the old one), and
`--publish <scene id>` uploads the files the viewer needs to the results dataset, scenes/<scene id>/ on Hugging Face,
through plypipe.hf (the ply-hf secret's HF token; no other secret and no GPU). It never writes the same scene id twice
by accident: the name is the caller's, and `lidar-office-8a` and `-8b` are separate prefixes.

It runs `python -m plypipe lidar` from the repository's own plypipe folder, so the twin is what the tests checked. The phone
file (scene_lite.spz, 800 k splats) is merged on the CPU by lidar_splat: splat-transform's own `-d` cut needs a WebGPU adapter
and fails on a container with no GPU (tests/test_export.py's phone-file test fails the same way on a CPU machine). The work
is capped at the High preset's 2 M splats: an office thinned to 1 cm has 5 to 30 M points, coarsened to about 1.5 to 3 cm.
Measured on this 2-CPU machine, shared with other jobs, on 2.7 M synthetic points with 18 stations at the 2 M cap: 47 s to
read, align and thin and to find local planes, 238 s for the scanner-visibility test of the normals (the cost; it grows with
the splat count and the number of stations), 11 s for the rest of the stage, then export (three splat-transform runs) about
35 s: 5.5 minutes in all, 1.5 GB peak. A real 30 M-point npz adds about 3 GB and 2 minutes while it is thinned and held
against the splats for the F-score. Estimate 20 minutes for 2 M splats and an unknown number of stations; 1.5 x that is the
1800 s timeout. 4 CPUs, 12 GB: $0.000079 a second, about $0.05 expected, $0.14 at the timeout.
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
HERE = Path(__file__).resolve()  # in the container this file is /root/lidar_twin.py, which has no repository above it
PLYPIPE = (HERE.parents[2] if len(HERE.parents) > 2 else Path("/root")) / "pipeline" / "plypipe"
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("curl", "ca-certificates")
         .run_commands("curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs",
                       "npm i -g @playcanvas/splat-transform@3.10.1 && npx -y @playcanvas/splat-transform@3.10.1 --version")
         .pip_install("numpy>=1.26", "opencv-python-headless>=4.9", "scipy>=1.11")
         .add_local_dir(PLYPIPE, "/root/pipeline/plypipe"))
VOL, ROOT = "/vol", "/root/pipeline"  # where the container mounts the Volume and the plypipe folder
KEEP = ("scene.spz", "scene_lite.spz", "scene.json", "occupancy.json", "occupancy.png", "shell.json", "topview.webp", "places.json",
        "report.json", "log.jsonl")
# what the viewer fetches: the same set lidar-office-8a publishes, plus the places
PUBLISH = ("scene.spz", "scene_lite.spz", "scene.json", "occupancy.json", "occupancy.png", "shell.json", "topview.webp", "places.json")
REPO = "Ricco-Marcelino/ply-data"


@app.function(image=image, cpu=4, memory=12288, timeout=1800, volumes={VOL: vol}, secrets=[modal.Secret.from_name("ply-hf")])
def twin(name: str = "office-1", points: str = "points-1cm.npz", args: str = "--sh-degree 0", dst: str = "twin",
         publish: str = ""):
    """lidar/<name>/<points> -> lidar/<name>/<dst>/, and with `publish`, the viewer's files to scenes/<publish>/ on the
    results dataset. `args` are more plypipe lidar options, e.g. "--cap 1500000 --up z"."""
    src = Path(VOL) / "lidar" / name / points
    out = Path(VOL) / "lidar" / name / dst
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
        report["lidar"] = {k: lid.get(k) for k in ("points", "splats", "voxelM", "fScore", "up", "yawDeg", "phaseSeconds", "peakMB",
                                                  "grid", "walk", "route", "stations", "pointcloud", "places", "ceilingY")}
    if publish:
        from plypipe import hf
        os.environ.setdefault("PLY_HF_REPO", REPO)
        files = [(f"scenes/{publish}/{f}", out / f) for f in PUBLISH if (out / f).exists()]
        hf.upload_many(files, message=f"lidar: {publish}, from lidar/{name}/{dst}")
        report["published"] = {"repo": os.environ["PLY_HF_REPO"], "prefix": f"scenes/{publish}", "files": [f for f, _ in files],
                               "url": hf.url(f"scenes/{publish}/scene.json")}
    print("LIDAR_TWIN " + json.dumps(report), flush=True)
    if run.returncode:
        raise RuntimeError(f"plypipe lidar exited {run.returncode}")
    return report


@app.local_entrypoint()
def main(name: str = "office-1", points: str = "points-1cm.npz", args: str = "--sh-degree 0", dst: str = "twin",
         publish: str = ""):
    call = twin.spawn(name, points, args, dst, publish)
    print(f"[lidar] twin started, call {call.object_id}; read it with `modal app logs ply-lidar-twin` (LIDAR_TWIN line) "
          f"or `modal volume ls ply-bench lidar/{name}/twin`", flush=True)
