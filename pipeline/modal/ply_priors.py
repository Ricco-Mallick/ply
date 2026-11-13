"""Measure a saved twin's metric scale with MoGe-2 depth (plan section 9: 2b with ViT-L, 2c with ViT-B). One short GPU run.

    MODAL_PROFILE=<account> modal run pipeline/modal/ply_priors.py --src chapel/high/job --tag chapel-l --wait
    MODAL_PROFILE=<account> modal run pipeline/modal/ply_priors.py --src chapel/high/job --tag chapel-b --model moge2-b --wait

The saved job is read from the `ply-bench` Volume exactly as ply_eval.py restores it (data/sparse/0, dense/images, scene.json).
`plypipe priors` then predicts metric depth on about 150 frames and compares it with the solve's track points. On a saved
twin the solve is already scaled by the camera-height prior (2a), so the ratio is a correction to that scale; the printed
`twinScale` is the scale 2b gives the twin, `scene.scale` the one it has (2a, or AR poses). Compare each with the laser
reference, and apply the winner with `plypipe rescale <folder> --factor <ratio>` (priors/priors.json holds `scale`).

Writes /vol/priors/<tag>/ (priors.json, readout.json, depth_half.npz). MoGe is installed when the container starts, because
the worker image's last layer is the mounted pipeline code and a layer cannot follow it; once it is in the image
(pipeline/modal/ply_modal.py) the install step is a no-op. The function times out at 20 minutes (L4, about $0.27 at most).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for root in (HERE.parent, Path("/root/pipeline")):
    if (root / "plypipe").is_dir():
        sys.path.append(str(root))
from ply_modal import image as worker_image  # noqa: E402
from plypipe import evalrun  # noqa: E402

GPU = os.environ.get("PLY_PRIORS_GPU", "L4")
TIMEOUT_MIN = 20
MOGE = "moge @ git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8"  # the last commit before MoGe-3 changed the package
UTILS3D = "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"  # what that commit pins
image = worker_image.add_local_file(HERE / "ply_modal.py", "/root/ply_modal.py")
app = modal.App("ply-priors", image=image)
vol = modal.Volume.from_name("ply-bench")  # not created here: a wrong account fails at once
JOB = Path("/tmp/work/job")


@app.function(gpu=GPU, cpu=2, memory=16 * 1024, timeout=TIMEOUT_MIN * 60, volumes={"/vol": vol})
def measure(src: str, tag: str, model: str, frames: int) -> dict:
    started = time.time()
    source = Path("/vol") / src
    if not (source / "data" / "sparse" / "0" / "images.bin").exists():
        raise FileNotFoundError(f"{source} lacks data/sparse/0: seed the Volume with the saved job (modal volume ls ply-bench {src})")
    try:
        import moge  # noqa: F401
    except ImportError:
        t = time.time()
        subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--system", "--no-deps", MOGE, UTILS3D, "scipy"], check=True)
        print(f"[priors] installed MoGe in {time.time() - t:.0f} s", flush=True)
    shutil.rmtree(JOB, ignore_errors=True)
    got = evalrun.restore(source, JOB)
    print(f"[priors] restored {got['files']} files ({got['bytes'] / 2**20:.0f} MB) in {time.time() - started:.0f} s", flush=True)
    env = {**os.environ, "PYTHONPATH": "/root/pipeline"}
    proc = subprocess.run([sys.executable, "-m", "plypipe", "priors", str(JOB), "--priors-model", model, "--priors-frames", str(frames),
                           "--priors-max-seconds", "600"], env=env, capture_output=True, text=True)
    print(proc.stderr[-3000:], flush=True)
    if proc.returncode:
        raise RuntimeError(f"plypipe priors exited with {proc.returncode}: {proc.stdout[-1500:]}")
    readout = json.loads((JOB / "priors" / "readout.json").read_text())
    saved = Path("/vol/priors") / tag
    shutil.rmtree(saved, ignore_errors=True)
    shutil.copytree(JOB / "priors", saved)
    (saved / "readout.json").write_text(json.dumps({"src": src, "model": model, **readout}, indent=2))
    vol.commit()
    readout["wallSeconds"] = round(time.time() - started, 1)
    print(json.dumps(readout, indent=2), flush=True)
    return readout


@app.local_entrypoint()
def main(src: str = "", tag: str = "", model: str = "moge2-l", frames: int = 150, wait: bool = False):
    if not src or not tag:
        raise SystemExit("give --src <saved job folder on the Volume> and --tag <name for the result>")
    if model not in ("moge2-l", "moge2-b"):
        raise SystemExit("--model is moge2-l (2b) or moge2-b (2c)")
    if wait:
        measure.remote(src, tag, model, frames)
        return
    call = measure.spawn(src, tag, model, frames)
    print(f"[priors] started {call.object_id}: not waiting. `modal app logs ply-priors` follows it; the result lands in /vol/priors/{tag}/", flush=True)
