"""Ply GPU worker on Modal: an L4 that wakes when a job is queued and stops when the queue is empty.

    modal secret create ply-worker PLY_API=https://ply-api.ricco-ply.workers.dev PLY_WORKER_TOKEN=... PLY_TRIGGER_TOKEN=...
    modal deploy pipeline/modal/ply_modal.py          # prints the trigger URL for the Cloudflare API (MODAL_TRIGGER)

Image (built once, cached): CUDA 12.8 devel, Python 3.12, torch 2.11 cu128, gsplat main + PPISP + fused-ssim /
fused-bilagrid compiled for A100 / A10 / L4 / H100, pycolmap-cuda12 4.2.1, ffmpeg, Node 22 + splat-transform.
The pipeline code is mounted at run time, so code changes redeploy in seconds without rebuilding.
Budget: one container at a time, 3-minute idle exit, 3-hour hard cap per wake-up.
"""
import os
import subprocess
import sys
from pathlib import Path

import modal

PIPE = Path(__file__).resolve().parents[1]
GPU = os.environ.get("PLY_MODAL_GPU", "L4")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "build-essential", "curl", "ca-certificates", "libgl1", "libglib2.0-0")
    .run_commands("curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs",
                  "npm i -g @playcanvas/splat-transform@3.10.1 && npx -y @playcanvas/splat-transform@3.10.1 --version || true")
    # gcc, not the clang Modal's Python reports from its own build (torch's extension builder would pick that up)
    .env({"TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0", "MAX_JOBS": "8", "PLY_GSPLAT_DIR": "/opt/gsplat", "PLY_SKIP_PLYPIPE": "1",
          "CC": "gcc", "CXX": "g++", "LDSHARED": "gcc -pthread -shared"})
    .add_local_file(PIPE / "scripts" / "install_gpu.sh", "/opt/ply-install/install_gpu.sh", copy=True)
    .run_commands("bash -o pipefail -c 'bash /opt/ply-install/install_gpu.sh 2>&1 | tail -40'")
    .add_local_dir(PIPE / "plypipe", "/root/pipeline/plypipe")
)
app = modal.App("ply", image=image)
secret = modal.Secret.from_name("ply-worker")
hf = modal.Secret.from_name("ply-hf")  # HF_TOKEN: gated facebook/sam3 weights for the label stage


@app.function(gpu=GPU, cpu=4, memory=16384, timeout=3 * 3600, max_containers=1, secrets=[secret, hf])  # L4 + 4 cores + 16 GB = ~$1.12/h
def work():
    """Claim and run jobs until the queue has been empty for 3 minutes."""
    env = dict(os.environ, PYTHONPATH="/root/pipeline", PLY_WORKER_NAME=f"modal-{GPU.lower()}", PLY_WORK_DIR="/tmp/work",
               PLY_IDLE_EXIT_MIN="3", MPLBACKEND="Agg")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    return subprocess.run([sys.executable, "-m", "plypipe.worker"], env=env, check=False).returncode


@app.function(image=modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]"), secrets=[secret])
@modal.fastapi_endpoint(method="POST")
def trigger(body: dict):
    """Called by the Ply API when a job is queued ({"token": ...}). Starts the GPU worker unless one is up or starting."""
    import hmac
    if not hmac.compare_digest(str(body.get("token", "")), os.environ["PLY_TRIGGER_TOKEN"]):
        return {"ok": False, "error": "unauthorised"}
    stats = modal.Function.from_name("ply", "work").get_current_stats()
    if stats.num_total_runners or stats.backlog:
        return {"ok": True, "started": False, "running": stats.num_total_runners, "backlog": stats.backlog}
    modal.Function.from_name("ply", "work").spawn()
    return {"ok": True, "started": True}


@app.local_entrypoint()
def smoke():
    """modal run pipeline/modal/ply_modal.py: check the image (GPU, torch, gsplat, pycolmap) without touching the queue."""
    print(check.remote())


@app.function(gpu=GPU, cpu=1, memory=4096, timeout=300)
def check():
    import torch, gsplat, pycolmap  # noqa: E401
    import ppisp  # noqa: F401  (PPISP present = the full training stack)
    out = subprocess.run(["npx", "-y", "@playcanvas/splat-transform@3.10.1", "--version"], capture_output=True, text=True)
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "gsplat": gsplat.__version__,
            "pycolmap": pycolmap.__version__, "pycolmap_cuda": pycolmap.has_cuda, "splat-transform": out.stdout.strip() or out.stderr[-200:]}
