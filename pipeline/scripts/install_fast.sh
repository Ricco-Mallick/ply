#!/usr/bin/env bash
# Fast GPU install (minutes, not an hour): prebuilt gsplat instead of compiling it.
#
# gsplat only publishes prebuilt wheels up to 1.5.3 (torch 2.4, CUDA 12.4, Python 3.10), and its training example
# wants a different package also called "pycolmap" (rmbrualla's parser). So training gets its own Python 3.10 venv
# (made by uv in seconds), while the rest of the pipeline stays on the machine's Python with the official
# pycolmap-cuda12 for SfM. plypipe's train stage runs the venv's python via PLY_TRAIN_PYTHON.
# Trade-off vs install_gpu.sh: no PPISP (gsplat 1.5.3 predates it); the trainer uses its per-image appearance model.
# Only fused-ssim / fused-bilagrid compile (small, a few minutes).
#
# Env: PLY_VENV (default /kaggle/working/venv310), PLY_GSPLAT_DIR (examples checkout), TORCH_CUDA_ARCH_LIST, PYTHON.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PLY_VENV:-/kaggle/working/venv310}"
GSPLAT_DIR="${PLY_GSPLAT_DIR:-/opt/gsplat153}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export MAX_JOBS="${MAX_JOBS:-4}"
PY="${PYTHON:-python3}"
GSPLAT_WHL="https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"
log() { printf '{"stage":"install","event":"progress","msg":"%s"}\n' "$*"; }

log "pipeline env: pycolmap-cuda12 4.2.1 + plypipe"
$PY -m pip install -q uv "pycolmap-cuda12==4.2.1"
$PY -m pip install -q -e "$HERE"
UV="$PY -m uv"

log "training venv: python 3.10"
[ -x "$VENV/bin/python" ] || $UV venv -q --python 3.10 "$VENV"
VPY="$VENV/bin/python"
$UV pip install -q --python "$VPY" setuptools wheel ninja

log "training venv: torch 2.4.1+cu124"
$UV pip install -q --python "$VPY" torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

log "training venv: prebuilt gsplat 1.5.3"
$UV pip install -q --python "$VPY" "$GSPLAT_WHL"
if [ ! -d "$GSPLAT_DIR/examples" ]; then git clone -q --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat "$GSPLAT_DIR"; fi

log "training venv: example requirements (fused-ssim, fused-bilagrid compile here)"
grep -v -E '^#|^$|fused-ssim|fused-bilagrid' "$GSPLAT_DIR/examples/requirements.txt" > /tmp/ply-reqs153.txt
$UV pip install -q --python "$VPY" -r /tmp/ply-reqs153.txt
FSSIM=$(grep -o 'git+https://github.com/rahul-goel/fused-ssim@[0-9a-f]*' "$GSPLAT_DIR/examples/requirements.txt")
FBILA=$(grep -o 'git+https://github.com/harry7557558/fused-bilagrid@[0-9a-f]*' "$GSPLAT_DIR/examples/requirements.txt")
$UV pip install -q --python "$VPY" --no-build-isolation "$FSSIM" "$FBILA"

log "check"
"$VPY" - <<'PY'
import json, torch, gsplat, fused_ssim
print(json.dumps({"stage": "install", "event": "done", "trainPython": "3.10", "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "gsplat": gsplat.__version__}))
PY
$PY -c "import pycolmap, json; print(json.dumps({'stage': 'install', 'event': 'done', 'pycolmap': pycolmap.__version__}))"
