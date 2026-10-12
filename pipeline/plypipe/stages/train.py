"""train: aligned data_dir -> 3D Gaussian splat PLY (gsplat, MCMC strategy, PPISP appearance).

Needs the GPU environment (torch 2.11+cu128, gsplat built from the pinned SHA, the example
requirements). The gsplat checkout is found via --gsplat-dir, $PLY_GSPLAT_DIR or /opt/gsplat.
Output: <out>/scene.ply (scene frame: +Y up, floor y=0, metres).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from plypipe.util import Ctx, run

GSPLAT_SHA = "512d366b67073d77ca099ede742683c165dfc23b"  # nerfstudio-project/gsplat main, 2026-09-19 (v1.6.0)


def gsplat_examples(ctx: Ctx) -> Path:
    root = Path(ctx.opt("gsplat_dir") or os.environ.get("PLY_GSPLAT_DIR") or "/opt/gsplat")
    ex = root / "examples"
    if not (ex / "simple_trainer.py").exists():
        raise RuntimeError(f"gsplat examples not found at {ex}. Clone gsplat at {GSPLAT_SHA} and pass --gsplat-dir.")
    return ex


def run_stage(ctx: Ctx) -> dict:
    ex = gsplat_examples(ctx)
    preset = ctx.opt("preset", "standard")
    result = ctx.train_dir
    # PLY_TRAIN_PYTHON: training may live in its own venv (install_fast.sh: prebuilt gsplat 1.5.3 on Python 3.10)
    cmd = [os.environ.get("PLY_TRAIN_PYTHON") or sys.executable, "-m", "plypipe.train_launcher", "--examples", str(ex), "--data", str(ctx.data_dir),
           "--result", str(result), "--preset", preset]
    if ctx.opt("ppisp") is False:
        cmd.append("--no-ppisp")
    for k, flag in (("iterations", "--iterations"), ("sh_degree", "--sh-degree"), ("cap_max", "--cap-max")):
        if ctx.opt(k) is not None:
            cmd += [flag, str(ctx.opt(k))]
    if ctx.opt("antialiased"):
        cmd.append("--antialiased")
    pkg_root = Path(__file__).resolve().parents[2]
    env = {"PYTHONPATH": os.pathsep.join([str(pkg_root), os.environ.get("PYTHONPATH", "")])}
    run(cmd, ctx, "train", env=env, progress_every=50)
    plys = sorted((result / "ply").glob("point_cloud_*.ply"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not plys:
        raise RuntimeError(f"training finished but no PLY in {result / 'ply'}")
    shutil.copy2(plys[-1], ctx.out / "scene.ply")
    preset_info = json.loads((result / "ply_preset.json").read_text()) if (result / "ply_preset.json").exists() else {}
    return {"ply": "scene.ply", "sizeMB": round((ctx.out / "scene.ply").stat().st_size / 2**20, 1),
            "fromStep": int(plys[-1].stem.rsplit("_", 1)[1]), **preset_info}
