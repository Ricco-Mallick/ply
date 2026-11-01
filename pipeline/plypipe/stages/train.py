# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
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

from plypipe import plan
from plypipe.util import Ctx, read_scene, run

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
    if ctx.opt("train_dry_run"):
        cmd.append("--dry-run")
    # the training budget comes from this model, not from a house number: surface area sets the gaussian
    # cap, the solved views set the steps, and the preset only scales the two
    scene = read_scene(ctx) or {}
    # scene.json names the shell ("shell.json") rather than inlining it: resolve the name, and treat a
    # missing or malformed file as "no shell" instead of failing the whole training stage
    shell = scene.get("shell")
    if isinstance(shell, str) and shell:
        try:
            shell = json.loads((ctx.out / shell).read_text())
        except (OSError, ValueError):
            shell = {}
    shell = shell if isinstance(shell, dict) else {}
    bounds = scene.get("bounds")
    bounds = bounds if isinstance(bounds, dict) else {}
    views = len(list(ctx.frames_dir.glob("*.jpg"))) if ctx.frames_dir.exists() else 0
    if not views:  # a resumed job may have the solved frames only
        views = len(list((ctx.data_dir / "images").glob("*.jpg"))) if (ctx.data_dir / "images").exists() else 0
    height = min(shell.get("ceilingY") or 0, (bounds.get("max") or [0, 0, 0])[1] or 0) or None
    budget = plan.model_plan(views, bounds or None, preset, height,
                             iterations=ctx.opt("iterations"), cap_max=ctx.opt("cap_max"))
    plan.write(ctx.out, "model", budget)
    ctx.log.emit("train", "progress", f"plan: {budget['why']}")
    cmd += ["--iterations", str(budget["iterations"]), "--cap-max", str(budget["capMax"])]
    if ctx.opt("sh_degree") is not None:
        cmd += ["--sh-degree", str(ctx.opt("sh_degree"))]
    if not ctx.opt("no_antialias"):
        cmd.append("--antialiased")
    pkg_root = Path(__file__).resolve().parents[2]
    # MPLBACKEND: notebooks (Kaggle) export an inline backend that a separate training venv can't load
    env = {"PYTHONPATH": os.pathsep.join([str(pkg_root), os.environ.get("PYTHONPATH", "")]), "MPLBACKEND": "Agg",
           # gsplat's cli() spawns one process per visible GPU (Kaggle gives two T4s); train on one
           "CUDA_VISIBLE_DEVICES": os.environ.get("PLY_TRAIN_GPU", "0")}
    run(cmd, ctx, "train", env=env, progress_every=50)
    if ctx.opt("train_dry_run"):
        return {"dryRun": True}
    plys = sorted((result / "ply").glob("point_cloud_*.ply"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not plys:
        raise RuntimeError(f"training finished but no PLY in {result / 'ply'}")
    shutil.copy2(plys[-1], ctx.out / "scene.ply")
    preset_info = json.loads((result / "ply_preset.json").read_text()) if (result / "ply_preset.json").exists() else {}
    return {"ply": "scene.ply", "sizeMB": round((ctx.out / "scene.ply").stat().st_size / 2**20, 1),
            "fromStep": int(plys[-1].stem.rsplit("_", 1)[1]), **preset_info,
            "plan": {k: budget[k] for k in ("iterations", "capMax", "why")}}
