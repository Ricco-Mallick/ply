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
HF_REPO = os.environ.get("PLY_HF_REPO", "Ricco-Marcelino/ply-data")  # results storage (private dataset repo)
# read HERE, at deploy time on the developer's machine: values read inside work() see the container's env,
# which never has these, so deploy-time flags like PLY_STAGE_LIMITS were silently ignored before
STAGE_LIMITS = os.environ.get("PLY_STAGE_LIMITS", "sfm=100")
EXPORT_PROBES = os.environ.get("PLY_EXPORT_PROBES", "1")
WORKER_NAME = os.environ.get("PLY_WORKER_NAME", f"modal-{GPU.lower()}")  # deploy-time: a second account needs its own row

# CUDA extensions come prebuilt from CI (.github/workflows/wheels.yml, python 3.12, sm_89 for L4 / L40S):
#   gh run download <run> -R Ricco-Mallick/ply -D pipeline/wheels   (flattened: one folder of .whl files)
# Building them here took >90 min of Modal CPU (and $1.4 of credit); installing wheels takes a minute.
WHEELS = PIPE / "wheels"
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "curl", "ca-certificates", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .run_commands("curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs",
                  "npm i -g @playcanvas/splat-transform@3.10.1 && npx -y @playcanvas/splat-transform@3.10.1 --version || true")
    .env({"PLY_GSPLAT_DIR": "/opt/gsplat", "UV_LINK_MODE": "copy"})
    # one cached step per layer, uv instead of pip (pip unpacking the 900 MB torch wheel ran the builder out of memory),
    # same pins as scripts/install_gpu.sh (the Kaggle / RunPod path)
    .run_commands("python -m pip install -q uv")
    .run_commands("python -m uv pip install --system torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128")
    .run_commands("git init -q /opt/gsplat && cd /opt/gsplat && git remote add origin https://github.com/nerfstudio-project/gsplat.git"
                  " && git fetch -q --depth 1 origin 512d366b67073d77ca099ede742683c165dfc23b && git checkout -q FETCH_HEAD")
    .add_local_dir(WHEELS, "/opt/ply-wheels", copy=True)
    .run_commands("python -m uv pip install --system --no-deps /opt/ply-wheels/*.whl")
    .run_commands("grep -v -E '^(torch|torchvision)==|^nvidia-ncore|^pycolmap|fused-ssim|fused-bilagrid|^ppisp|nerfview|^#|^$' /opt/gsplat/examples/requirements.txt > /tmp/reqs.txt"
                  " && python -m uv pip install --system -r /tmp/reqs.txt 'git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787'"
                  " jaxtyping nvtx 'rich>=12' ninja 'torch==2.11.0' 'torchvision==0.26.0' --index-strategy unsafe-best-match"
                  " --extra-index-url https://download.pytorch.org/whl/cu128")
    .run_commands("python -m uv pip install --system pycolmap-cuda12==4.2.1 'opencv-python-headless>=4.9' 'transformers==5.19.0' 'accelerate>=1.1.0' pillow trimesh")
    .run_commands("python -c \"import torch, gsplat, pycolmap, ppisp, fused_ssim, transformers; print('versions', torch.__version__, gsplat.__version__, pycolmap.__version__, transformers.__version__)\"")
    # SAM 3 weights baked in (3.4 GB, fetched once at build with the HF token): no download on every wake-up, and no
    # Modal Volume (Volumes bill storage even under a $0 spend limit)
    .env({"HF_HOME": "/opt/hf"})
    .run_commands("python -c \"from huggingface_hub import snapshot_download; snapshot_download('facebook/sam3', allow_patterns=['*.json', '*.txt', 'model.safetensors'])\"",
                  secrets=[modal.Secret.from_name("ply-hf")])
    # ALIKED + LightGlue in PyTorch (the pycolmap wheels have no ONNX runtime for COLMAP's own), with their weights
    # and DINOv2 (pair retrieval) baked in, so a job never downloads them
    .env({"TORCH_HOME": "/opt/torch"})
    .run_commands("python -m uv pip install --system 'kornia>=0.7' && python -m uv pip install --system --no-deps 'lightglue @ git+https://github.com/cvg/LightGlue@5a650c444718caffc2669dc0094cde908f42dd73'",
                  "python -c \"from lightglue import ALIKED, LightGlue; ALIKED(max_num_keypoints=4096); LightGlue(features='aliked')\"",
                  "python -c \"from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov2-small')\"")
    # the worker runs the pipeline as a subprocess of the image's python, which can't see Modal's injected client:
    # the CPU-solve hand-off (plypipe.handoff) needs it there to spawn `solve` and commit the Volume
    .run_commands(f"python -m uv pip install --system modal=={modal.__version__}")
    .env({"PLY_IMAGE_REV": "16"})  # bump to force the code layer to rebuild when Modal reuses a cached image
    .add_local_dir(PIPE / "plypipe", "/root/pipeline/plypipe")
)
app = modal.App("ply", image=image)
secret = modal.Secret.from_name("ply-worker")
# long captures solve their cameras on a CPU box (plypipe.handoff): COLMAP's mapper never uses the GPU, and on the
# chapel it was 93 of 119 min of a billed L40S. The job folder travels through this Volume between the two.
CPU_SFM = os.environ.get("PLY_CPU_SFM", "0")  # off until a live job has been watched end to end; deploy with PLY_CPU_SFM=1
CPU_SFM_MIN = os.environ.get("PLY_CPU_SFM_MIN", "400")
handoff_vol = modal.Volume.from_name("ply-handoff", create_if_missing=True)
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("pycolmap==4.2.1", "numpy>=1.26", "opencv-python-headless>=4.9", f"modal=={modal.__version__}")
             .env({"PLY_IMAGE_REV": "1"})
             .add_local_dir(PIPE / "plypipe", "/root/pipeline/plypipe"))
HANDOFF_ENV = {"PLY_HANDOFF_DIR": "/handoff", "PLY_HANDOFF_VOLUME": "ply-handoff", "PLY_HANDOFF_APP": "ply"}
hf = modal.Secret.from_name("ply-hf")  # HF_TOKEN: gated facebook/sam3 weights for the label stage


# longer than the longest possible job: the sum of every stage limit (JOB_MAX_MIN, 290 min once the dense mesh
# stage was added). The worker refuses any job it cannot finish inside the remaining deadline, so a timeout
# shorter than that sum makes it refuse everything.
TIMEOUT_S = int(6.0 * 3600)


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=TIMEOUT_S, max_containers=1, secrets=[secret, hf],
              volumes={"/handoff": handoff_vol},
              # the module is re-imported in the container without the deploy shell's variables: bake the switch in
              env={"PLY_CPU_SFM": CPU_SFM, "PLY_CPU_SFM_MIN": CPU_SFM_MIN})  # 8 cores + 32 GB: 2560 px frames and ~800 of them  # L4 + 4 cores + 16 GB = ~$1.12/h
def work(worker_name: str = WORKER_NAME):
    """Claim and run jobs until the queue has been empty for 3 minutes. The worker only starts a job it can finish
    before this container's timeout (PLY_DEADLINE); later work waits for the next wake-up (API trigger or cron)."""
    import time
    env = dict(os.environ, PYTHONPATH="/root/pipeline", PLY_WORKER_NAME=worker_name, PLY_WORK_DIR="/tmp/work", PLY_HF_REPO=HF_REPO, PLY_MODAL_GPU=GPU, PLY_STAGE_LIMITS="sfm=300,export=60",
               PLY_IDLE_EXIT_MIN="3", MPLBACKEND="Agg", PLY_DEADLINE=str(time.time() + TIMEOUT_S - 300),
               PLY_CPU_SFM=CPU_SFM, PLY_CPU_SFM_MIN=CPU_SFM_MIN, **HANDOFF_ENV)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    return subprocess.run([sys.executable, "-m", "plypipe.worker"], env=env, check=False).returncode


@app.function(image=cpu_image, cpu=8, memory=32768, timeout=3 * 3600, secrets=[secret], volumes={"/handoff": handoff_vol})
def solve(jid: str):
    """Map a handed-over job's matches here (8 cores + 32 GB: $0.63/h against $1.43/h for the L4 box), then re-queue
    the job and start a GPU worker for the rest unless one is already up."""
    env = dict(os.environ, PYTHONPATH="/root/pipeline", MPLBACKEND="Agg", **HANDOFF_ENV)
    code = subprocess.run([sys.executable, "-m", "plypipe.handoff", jid], env=env, check=False).returncode
    if code == 0:
        stats = modal.Function.from_name("ply", "work").get_current_stats()
        if not (stats.num_total_runners or stats.backlog):
            modal.Function.from_name("ply", "work").spawn()
    return code


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
    import json
    import torch, gsplat, pycolmap  # noqa: E401
    import ppisp  # noqa: F401  (PPISP present = the full training stack)
    import fused_ssim  # noqa: F401
    from transformers import Sam3Model  # noqa: F401  (the label stage)
    out = subprocess.run(["npx", "-y", "@playcanvas/splat-transform@3.10.1", "--version"], capture_output=True, text=True)
    info = {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "cuda": str(torch.version.cuda), "gsplat": str(gsplat.__version__),
            "pycolmap": str(pycolmap.__version__), "pycolmap_cuda": bool(pycolmap.has_cuda), "splat-transform": (out.stdout.strip() or out.stderr[-200:])}
    # a real CUDA kernel from the compiled gsplat wheel: rasterise 1000 random Gaussians
    from gsplat import rasterization
    n, dev = 1000, "cuda"
    img, alpha, _ = rasterization(torch.randn(n, 3, device=dev) + torch.tensor([0, 0, 4.0], device=dev), torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1),
                                  torch.rand(n, 3, device=dev) * .1, torch.rand(n, device=dev), torch.rand(n, 3, device=dev),
                                  torch.eye(4, device=dev)[None], torch.tensor([[[300., 0, 160], [0, 300., 120], [0, 0, 1]]], device=dev), 320, 240)
    info["gsplat_render"] = f"{tuple(img.shape)} mean alpha {float(alpha.mean()):.3f}"
    return json.dumps(info)
