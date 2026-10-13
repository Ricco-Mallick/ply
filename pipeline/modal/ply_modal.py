"""Ply GPU worker on Modal: an L4 that wakes when a job is queued and stops when the queue is empty.

    modal secret create ply-worker PLY_API=https://ply-api.ricco-ply.workers.dev PLY_WORKER_TOKEN=... PLY_TRIGGER_TOKEN=...
    modal deploy pipeline/modal/ply_modal.py          # prints the trigger URL for the Cloudflare API (MODAL_TRIGGER)

Image (built once, cached): CUDA 12.8 runtime, Python 3.12, torch 2.11 cu128, gsplat main + PPISP + fused-ssim /
fused-bilagrid from CI wheels (sm_89: L4 / L40S), pycolmap-cuda12 4.2.1, transformers 5 (SAM 3), ffmpeg, Node 22 +
splat-transform.
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

# CUDA extensions come prebuilt from CI (.github/workflows/wheels.yml, python 3.12, sm_89 for L4 / L40S):
#   gh run download <run> -R Ricco-Mallick/ply -D pipeline/wheels   (flattened: one folder of .whl files)
# Building them here took >90 min of Modal CPU (and $1.4 of credit); installing wheels takes a minute.
WHEELS = PIPE / "wheels"
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "curl", "ca-certificates", "libgl1", "libglib2.0-0")
    .run_commands("curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs",
                  "npm i -g @playcanvas/splat-transform@3.10.1 && npx -y @playcanvas/splat-transform@3.10.1 --version || true")
    .env({"PLY_GSPLAT_DIR": "/opt/gsplat", "PLY_SKIP_PLYPIPE": "1", "PLY_WHEELS": "/opt/ply-wheels"})
    .add_local_dir(WHEELS, "/opt/ply-wheels", copy=True)
    .add_local_file(PIPE / "scripts" / "install_gpu.sh", "/opt/ply-install/install_gpu.sh", copy=True)
    .run_commands("bash -o pipefail -c 'bash /opt/ply-install/install_gpu.sh 2>&1 | tail -60'")
    # SAM 3 weights baked in (3.4 GB, fetched once at build with the HF token): no download on every wake-up, and no
    # Modal Volume (Volumes bill storage even under a $0 spend limit)
    .env({"HF_HOME": "/opt/hf"})
    .run_commands("python -c \"from huggingface_hub import snapshot_download; snapshot_download('facebook/sam3', allow_patterns=['*.json', '*.txt', 'model.safetensors'])\"",
                  secrets=[modal.Secret.from_name("ply-hf")])
    .add_local_dir(PIPE / "plypipe", "/root/pipeline/plypipe")
)
app = modal.App("ply", image=image)
secret = modal.Secret.from_name("ply-worker")
hf = modal.Secret.from_name("ply-hf")  # HF_TOKEN: gated facebook/sam3 weights for the label stage


TIMEOUT_S = int(4.5 * 3600)  # longer than the longest job (all stage limits, ~3.7 h) so a started job always finishes


@app.function(gpu=GPU, cpu=4, memory=16384, timeout=TIMEOUT_S, max_containers=1, secrets=[secret, hf])  # L4 + 4 cores + 16 GB = ~$1.12/h
def work():
    """Claim and run jobs until the queue has been empty for 3 minutes. The worker only starts a job it can finish
    before this container's timeout (PLY_DEADLINE); later work waits for the next wake-up (API trigger or cron)."""
    import time
    env = dict(os.environ, PYTHONPATH="/root/pipeline", PLY_WORKER_NAME=f"modal-{GPU.lower()}", PLY_WORK_DIR="/tmp/work",
               PLY_IDLE_EXIT_MIN="3", MPLBACKEND="Agg", PLY_DEADLINE=str(time.time() + TIMEOUT_S - 300))
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    return subprocess.run([sys.executable, "-m", "plypipe.worker"], env=env, check=False).returncode


@app.function(image=modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]"), secrets=[secret])
@modal.fastapi_endpoint(method="POST")
def trigger(body: dict):
    """Called by the Ply API when a job is queued ({"token": ...}). Starts the GPU worker unless one is up or starting.
    Two calls at the same instant can both spawn; max_containers=1 queues the second, which then finds the queue empty
    and exits after its 3 idle minutes (~$0.06 worst case)."""
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
    import fused_ssim  # noqa: F401
    from transformers import Sam3Model  # noqa: F401  (the label stage)
    out = subprocess.run(["npx", "-y", "@playcanvas/splat-transform@3.10.1", "--version"], capture_output=True, text=True)
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "gsplat": gsplat.__version__,
            "pycolmap": pycolmap.__version__, "pycolmap_cuda": pycolmap.has_cuda, "splat-transform": out.stdout.strip() or out.stderr[-200:]}
