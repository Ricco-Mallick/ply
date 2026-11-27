#!/usr/bin/env bash
# Local GPU box (a college PC, a friend's laptop): the pipeline inside WSL2 Ubuntu on Windows 11, or on plain Ubuntu.
#
#   sudo bash pipeline/local/install_local.sh system   apt packages, CUDA toolkit 12.4 (nvcc only, never a driver), Node 22
#   bash pipeline/local/install_local.sh                 venvs, torch, gsplat, pycolmap, splat-transform, then a GPU smoke test
#
# start.bat (start.ps1) runs both for you on Windows. The Windows NVIDIA driver is the only driver: WSL must not get one of its own.
# Training uses the same route as the Kaggle T4 (scripts/install_fast.sh): prebuilt gsplat 1.5.3 in a Python 3.10 venv,
# pycolmap-cuda12 4.2.1 for the camera solve. A GTX 1650 is Turing (sm 7.5), the same architecture as a T4.
# fused-ssim compiles against torch cu124, which is why the toolkit here is 12.4 exactly (not the newest).
#
# Everything lands in ~/.ply (venvs, gsplat examples, node packages) and ~/.ply/env.sh; jobs go to ~/ply-work.
# Remove it all with: rm -rf ~/.ply ~/ply-work ~/.cache/huggingface ~/.cache/uv ~/.cache/pip
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # pipeline/
CUDA_VER=12-4
CUDA_DIR=/usr/local/cuda-12.4
NODE_MAJOR=22
log() { printf '\033[1m[ply-local]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[ply-local] %s\033[0m\n' "$*" >&2; exit 1; }
is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }

system() {
  [ "$(id -u)" = 0 ] || die "the system part needs root: sudo bash $0 system"
  export DEBIAN_FRONTEND=noninteractive
  log "apt packages"
  apt-get update -q
  apt-get install -y -q build-essential git curl ca-certificates xz-utils ffmpeg python3-venv python3-dev python3-pip \
    pkg-config libgl1 libglib2.0-0 unzip

  if [ ! -x "$CUDA_DIR/bin/nvcc" ]; then
    # wsl-ubuntu: NVIDIA's repo for WSL, which has the toolkit and no driver. On plain Ubuntu the distro's own repo
    . /etc/os-release
    repo=wsl-ubuntu; is_wsl || repo="ubuntu${VERSION_ID//./}"
    log "CUDA toolkit ${CUDA_VER/-/.} from developer.download.nvidia.com ($repo)"
    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/keyring.deb" "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i "$tmp/keyring.deb"
    rm -rf "$tmp"
    apt-get update -q
    apt-get install -y -q "cuda-toolkit-$CUDA_VER"
  fi
  "$CUDA_DIR/bin/nvcc" --version | tail -1

  if ! command -v node >/dev/null || [ "$(node -p 'process.versions.node.split(".")[0]')" -lt 20 ]; then
    log "Node $NODE_MAJOR (official tarball, checksum verified) -> /opt/node"
    base="https://nodejs.org/dist/latest-v${NODE_MAJOR}.x"
    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/SHASUMS256.txt" "$base/SHASUMS256.txt"
    file=$(grep -o "node-v[0-9.]*-linux-x64.tar.xz" "$tmp/SHASUMS256.txt" | head -1)
    curl -fsSL -o "$tmp/$file" "$base/$file"
    (cd "$tmp" && grep " $file\$" SHASUMS256.txt | sha256sum -c -)
    rm -rf /opt/node && mkdir -p /opt/node && tar -xJf "$tmp/$file" -C /opt/node --strip-components 1
    rm -rf "$tmp"
    ln -sf /opt/node/bin/node /usr/local/bin/node; ln -sf /opt/node/bin/npm /usr/local/bin/npm; ln -sf /opt/node/bin/npx /usr/local/bin/npx
  fi
  node --version
  log "system part done"
}

user() {
  [ "$(id -u)" != 0 ] || die "run the user part as yourself, not root: bash $0"
  local smi; smi=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
  [ -x "$smi" ] || die "nvidia-smi not found: is the NVIDIA driver installed on Windows (and is this WSL 2, not WSL 1)?"
  [ -x "$CUDA_DIR/bin/nvcc" ] || die "CUDA toolkit missing: run 'sudo bash $0 system' first"
  local sm; sm=$("$smi" --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  log "GPU: $("$smi" --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1) (sm $sm)"
  case "$sm" in 5.*|6.*) die "compute capability $sm is too old for torch's CUDA 12 builds: this machine can't train";; esac

  ROOT="$HOME/.ply"; mkdir -p "$ROOT"
  local ram_gb; ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
  # nvcc jobs: about 4 GB of RAM each while compiling fused-ssim
  local jobs=$(( ram_gb / 4 )); [ "$jobs" -lt 1 ] && jobs=1; [ "$jobs" -gt 4 ] && jobs=4

  [ -x "$ROOT/venv/bin/python" ] || python3 -m venv "$ROOT/venv"
  "$ROOT/venv/bin/python" -m pip install -q -U pip wheel setuptools

  log "pipeline + training environments (scripts/install_fast.sh; 10-25 min, mostly downloads)"
  rm -f "$ROOT/venv310.failed"
  CUDA_HOME="$CUDA_DIR" PATH="$CUDA_DIR/bin:$PATH" TORCH_CUDA_ARCH_LIST="$sm" MAX_JOBS="$jobs" \
    PLY_VENV="$ROOT/venv310" PLY_GSPLAT_DIR="$ROOT/gsplat153" PYTHON="$ROOT/venv/bin/python" \
    bash "$HERE/scripts/install_fast.sh"
  [ ! -f "$ROOT/venv310.failed" ] || die "the training environment failed to build (see the output above); frames/sfm/occupancy still work"
  "$ROOT/venv/bin/python" -m pip install -q trimesh

  log "splat-transform 3.10.1 -> $ROOT/node"
  npm install -g -s --prefix "$ROOT/node" @playcanvas/splat-transform@3.10.1

  cat > "$ROOT/env.sh" <<EOF
# written by pipeline/local/install_local.sh
export CUDA_HOME=$CUDA_DIR
export PATH="$ROOT/node/bin:$CUDA_DIR/bin:/usr/lib/wsl/lib:\$PATH"
export PLY_PY="$ROOT/venv/bin/python"
export PLY_TRAIN_PYTHON="$ROOT/venv310/bin/python"
export PLY_GSPLAT_DIR="$ROOT/gsplat153"
export PLY_REPO="$(cd "$HERE/.." && pwd)"
export PLY_WORK="\${PLY_WORK:-$HOME/ply-work}"
alias plylocal="bash \$PLY_REPO/pipeline/local/plylocal.sh"
EOF
  grep -q '.ply/env.sh' "$HOME/.bashrc" 2>/dev/null || echo '[ -f "$HOME/.ply/env.sh" ] && . "$HOME/.ply/env.sh"' >> "$HOME/.bashrc"

  log "smoke test: gsplat renders on this GPU"
  "$ROOT/venv310/bin/python" - <<'PY'
import torch, gsplat
from gsplat import rasterization
d = "cuda"
n = 2000
means = torch.randn(n, 3, device=d) * 0.5 + torch.tensor([0, 0, 3.0], device=d)
quats = torch.nn.functional.normalize(torch.randn(n, 4, device=d), dim=-1)
scales = torch.rand(n, 3, device=d) * 0.02
opac = torch.rand(n, device=d)
cols = torch.rand(n, 3, device=d)
K = torch.tensor([[300, 0, 160], [0, 300, 120], [0, 0, 1.0]], device=d)[None]
img, alpha, _ = rasterization(means, quats, scales, opac, cols, torch.eye(4, device=d)[None], K, 320, 240, packed=True)
torch.cuda.synchronize()
from fused_ssim import fused_ssim  # noqa: F401  (simple_trainer imports it at the top)
print(f"[ply-local] gsplat {gsplat.__version__} on {torch.cuda.get_device_name(0)}: render ok, "
      f"alpha mean {float(alpha.mean()):.3f}, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
PY
  "$ROOT/venv/bin/python" -c "import pycolmap; print('[ply-local] pycolmap', pycolmap.__version__, 'cuda', getattr(pycolmap, 'has_cuda', '?'))"
  (cd "$HERE" && "$ROOT/venv/bin/python" -m plypipe.device --dir "$HOME")
  log "done. Open a new WSL shell (or: . ~/.ply/env.sh), then: plylocal help"
}

case "${1:-user}" in
  system) system;;
  user) user;;
  *) die "usage: $0 [system|user]";;
esac
