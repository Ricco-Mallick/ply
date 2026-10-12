#!/usr/bin/env bash
# GPU environment for plypipe: the single source of truth used by the Dockerfile (RunPod) and the
# Kaggle notebook. Pinned to the compatibility-checked stack:
#   torch 2.11.0+cu128 / torchvision 0.26.0, gsplat main @ GSPLAT_SHA (v1.6.0, MCMC + PPISP),
#   pycolmap-cuda12 4.2.1 (COLMAP 4.2.1 with global SfM, ALIKED + LightGlue), splat-transform 3.10.1.
#
# Env:
#   TORCH_CUDA_ARCH_LIST  GPUs to compile CUDA extensions for. Kaggle T4 = "7.5", RTX 4090 = "8.9",
#                         L40S = "8.9", A100 = "8.0". Default covers T4/A100/3090/4090/L40S/H100.
#   PLY_GSPLAT_DIR        where to clone gsplat (default /opt/gsplat; the examples are used for training)
#   PLY_WHEELS            folder of prebuilt wheels (e.g. a Kaggle Dataset) to skip ~15 min of CUDA builds
#   PLY_SAVE_WHEELS       folder to write freshly built wheels into, for reuse as PLY_WHEELS
#   MAX_JOBS              parallel nvcc jobs (default 4; lower it if the build runs out of RAM)
set -euo pipefail

GSPLAT_SHA=512d366b67073d77ca099ede742683c165dfc23b
FUSED_SSIM=git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38
FUSED_BILAGRID=git+https://github.com/harry7557558/fused-bilagrid@49f0ef06c9f81810fb9b5dd9027cf1844950cc16
PPISP=git+https://github.com/nv-tlabs/ppisp@v1.2.1
NERFVIEW=git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GSPLAT_DIR="${PLY_GSPLAT_DIR:-/opt/gsplat}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
PY="${PYTHON:-python3}"
PIP="$PY -m pip"
log() { printf '{"stage":"install","event":"progress","msg":"%s"}\n' "$*"; }

log "pip tooling"
$PIP install -q -U pip wheel setuptools ninja packaging

log "torch 2.11.0+cu128"
$PIP install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

log "gsplat @ ${GSPLAT_SHA:0:7} -> ${GSPLAT_DIR}"
if [ ! -d "$GSPLAT_DIR/.git" ]; then
  mkdir -p "$GSPLAT_DIR"
  git -C "$GSPLAT_DIR" init -q
  git -C "$GSPLAT_DIR" remote add origin https://github.com/nerfstudio-project/gsplat.git
fi
git -C "$GSPLAT_DIR" fetch -q --depth 1 origin "$GSPLAT_SHA"
git -C "$GSPLAT_DIR" checkout -q FETCH_HEAD
git -C "$GSPLAT_DIR" submodule update -q --init --recursive --depth 1

CUDA_PKGS=("$GSPLAT_DIR" "$FUSED_SSIM" "$FUSED_BILAGRID" "$PPISP")
# CI builds each package in its own job (PLY_ONLY_PKG=gsplat|fused-ssim|fused-bilagrid|ppisp) so they compile in parallel
case "${PLY_ONLY_PKG:-}" in
  gsplat) CUDA_PKGS=("$GSPLAT_DIR");; fused-ssim) CUDA_PKGS=("$FUSED_SSIM");; fused-bilagrid) CUDA_PKGS=("$FUSED_BILAGRID");; ppisp) CUDA_PKGS=("$PPISP");;
esac
if [ -n "${PLY_WHEELS:-}" ] && ls "$PLY_WHEELS"/gsplat-*.whl >/dev/null 2>&1; then
  log "installing prebuilt CUDA wheels from $PLY_WHEELS"
  $PIP install -q --no-deps "$PLY_WHEELS"/*.whl
else
  log "building CUDA extensions for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (10-20 min)"
  if [ -n "${PLY_SAVE_WHEELS:-}" ]; then
    mkdir -p "$PLY_SAVE_WHEELS"
    $PIP wheel -q --no-build-isolation --no-deps -w "$PLY_SAVE_WHEELS" "${CUDA_PKGS[@]}"
    $PIP install -q --no-deps "$PLY_SAVE_WHEELS"/*.whl
  else
    $PIP install -q --no-build-isolation --no-deps "${CUDA_PKGS[@]}"
  fi
fi

# CI (.github/workflows/wheels.yml): only the CUDA wheels are wanted; the rest needs the target machine
if [ -n "${PLY_WHEELS_ONLY:-}" ]; then log "wheels only: built $(ls "$PLY_SAVE_WHEELS" | tr '\n' ' ')"; exit 0; fi

log "gsplat runtime + example requirements"
# examples/requirements.txt pins torch 2.9.1 and pulls nvidia-ncore (only for the NCore dataset
# backend); both are filtered out so pip cannot replace the cu128 torch installed above.
grep -v -E '^(torch|torchvision)==|^nvidia-ncore|fused-ssim|fused-bilagrid|^ppisp|nerfview|^#|^$' \
  "$GSPLAT_DIR/examples/requirements.txt" > /tmp/ply-gsplat-reqs.txt
$PIP install -q -r /tmp/ply-gsplat-reqs.txt "$NERFVIEW" jaxtyping nvtx "rich>=12"

log "pycolmap-cuda12 4.2.1 + plypipe"
$PIP install -q "pycolmap-cuda12==4.2.1"
$PIP install -q -e "$HERE"

log "check"
$PY - <<'PY'
import torch, gsplat, pycolmap
print({"stage": "install", "event": "done", "torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       "gsplat": gsplat.__version__, "pycolmap": pycolmap.__version__})
PY
