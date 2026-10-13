#!/usr/bin/env bash
# Local, no-GPU rehearsal of the training stage exactly as Kaggle runs it with install_fast.sh: a Python 3.10 venv with
# torch 2.4.1 (CPU build here), prebuilt gsplat 1.5.3 + its examples, then plypipe's train stage with --train-dry-run on
# a synthetic aligned scene. Catches import/config/parser problems (the class of bugs that cost GPU runs) in seconds.
#   pipeline/scripts/rehearse_train_cpu.sh [workdir]      (needs ~1.7 GB free)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; W="${1:-/tmp/ply-rehearse}"; UV="$HERE/.venv/bin/uv"; export UV_NO_CACHE=1
mkdir -p "$W"
[ -x "$W/venv310/bin/python" ] || "$UV" venv -q --python 3.10 "$W/venv310"
VPY="$W/venv310/bin/python"
"$UV" pip install -q --python "$VPY" torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
printf 'torch==2.4.1\ntorchvision==0.19.1\nnumpy<2\n' > "$W/con.txt"
"$UV" pip install -q --python "$VPY" -c "$W/con.txt" "https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"
[ -d "$W/gsplat153" ] || git clone -q --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat "$W/gsplat153"
grep -v -E '^#|^$|fused-ssim|fused-bilagrid' "$W/gsplat153/examples/requirements.txt" > "$W/reqs.txt"
"$UV" pip install -q --python "$VPY" -c "$W/con.txt" -r "$W/reqs.txt"
rm -rf "$W/scene"
"$HERE/.venv/bin/python" -c "
import sys, pathlib, cv2, numpy as np
sys.path[:0] = ['$HERE', '$HERE/tests']
from synth import write_synthetic_model
from plypipe import colmap_txt
d = pathlib.Path('$W/scene'); write_synthetic_model(d/'dense'/'sparse_txt')
imgs = d/'dense'/'images'; imgs.mkdir(parents=True)
for im in colmap_txt.read_model(d/'dense'/'sparse_txt').images:
    cv2.imwrite(str(imgs/im.name), (np.random.default_rng(im.image_id).random((1200, 1600, 3))*255).astype('uint8'))
"
# Kaggle exports an inline matplotlib backend; plant it to prove the train stage overrides it
MPLBACKEND=module://matplotlib_inline.backend_inline PLY_TRAIN_PYTHON="$VPY" PLY_GSPLAT_DIR="$W/gsplat153" \
  "$HERE/.venv/bin/python" -m plypipe run "$W/x.mp4" --out "$W/scene" --from align --until train --train-dry-run --preset standard --quiet
MPLBACKEND=Agg PYTHONPATH="$HERE" "$VPY" -m plypipe.train_launcher --examples "$W/gsplat153/examples" --data "$W/scene/data" \
  --result "$W/scene/train" --preset standard --dry-run | grep plyTrain
echo "rehearsal OK (delete $W to free ~1.7 GB)"
