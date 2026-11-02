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
    elif ctx.opt("ppisp") is True:
        cmd.append("--ppisp")
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
    budget = plan.model_plan(views, bounds or None, preset, height, iterations=ctx.opt("iterations"),
                             cap_max=ctx.opt("cap_max"), megapixels=view_megapixels(ctx.data_dir / "images"))
    plan.write(ctx.out, "model", budget)
    ctx.log.emit("train", "progress", f"plan: {budget['why']}")
    cmd += ["--iterations", str(budget["iterations"]), "--cap-max", str(budget["capMax"])]
    spread = camera_spread(ctx.data_dir / "sparse" / "0")
    if spread:
        if spread["maxOverMedian"] > 60:  # sfm drops such views; a model from elsewhere may still carry one
            raise RuntimeError(f"a camera sits {spread['maxOverMedian']:.0f}x the median camera distance from the others: "
                               "gsplat would size its learning rates from it and diverge. Re-run sfm (--force sfm).")
        if ctx.opt("scale_mcmc"):  # opt-in: 29.02 vs 28.94 dB unscaled/scaled on Dr Johnson (quick, 9 held-out views)
            cmd += ["--world-scale", f"{spread['medianM']:.4f}"]
    if ctx.opt("sh_degree") is not None:
        cmd += ["--sh-degree", str(ctx.opt("sh_degree"))]
    if not ctx.opt("no_antialias"):
        cmd.append("--antialiased")
    holdout = int(ctx.opt("holdout", 30) or 0)
    if holdout > 1:
        cmd += ["--holdout", str(holdout)]
    for flag in ("pose_opt", "depth_loss", "bilateral_grid", "random_bkgd"):
        if ctx.opt(flag):
            cmd.append("--" + flag.replace("_", "-"))
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
    scores = held_out_scores(result, ctx.out / "eval")
    if scores:
        ctx.log.emit("train", "progress", f"held-out views: PSNR {scores['psnr']:.2f}, SSIM {scores['ssim']:.3f}, "
                     f"LPIPS {scores['lpips']:.3f} over {scores['views']} views")
    return {"ply": "scene.ply", "sizeMB": round((ctx.out / "scene.ply").stat().st_size / 2**20, 1),
            "fromStep": int(plys[-1].stem.rsplit("_", 1)[1]), **preset_info,
            "plan": {k: budget[k] for k in ("iterations", "capMax", "why")},
            **({"eval": scores} if scores else {})}


def camera_spread(sparse: Path) -> dict | None:
    """Median and farthest camera distance from the cameras' median centre, in scene units (metres)."""
    import numpy as np
    from plypipe import colmap_txt
    try:
        centres, _, _ = colmap_txt.read_model(sparse).cameras()
    except (OSError, ValueError, StopIteration):
        return None
    if len(centres) < 3:
        return None
    dist = np.linalg.norm(centres - np.median(centres, axis=0), axis=1)
    med = float(np.median(dist))
    return {"medianM": med, "maxOverMedian": float(dist.max() / med) if med > 0 else float("inf")} if med > 0 else None


def view_megapixels(images: Path) -> float | None:
    """Size of one training view, from the first undistorted image."""
    try:
        import cv2
        first = next(iter(sorted(images.glob("*.jpg"))))
        h, w = cv2.imread(str(first), cv2.IMREAD_REDUCED_GRAYSCALE_2).shape[:2]
        return round(4 * h * w / 1e6, 3)
    except (StopIteration, AttributeError, ImportError, OSError):
        return None


def held_out_scores(result: Path, keep: Path, renders: int = 4) -> dict | None:
    """gsplat's scores on the held-out views (stats/val_step*.json) plus a few render|photo pairs as evidence."""
    stats = sorted((result / "stats").glob("val_step*.json"), key=lambda p: int(p.stem.rsplit("step", 1)[1]))
    if not stats:
        return None
    raw = json.loads(stats[-1].read_text())
    names = {"psnr": "psnr", "ssim": "ssim", "lpips": "lpips", "cc_psnr": "ccPsnr", "cc_ssim": "ccSsim",
             "cc_lpips": "ccLpips", "num_GS": "gaussians", "ellipse_time": "secondsPerView"}
    out = {v: round(float(raw[k]), 4) for k, v in names.items() if k in raw}
    pngs = sorted((result / "renders").glob(f"val_step{stats[-1].stem.rsplit('step', 1)[1].lstrip('0') or '0'}_*.png"))
    out["views"] = len(pngs) or None
    keep.mkdir(parents=True, exist_ok=True)
    for png in pngs[:: max(1, len(pngs) // renders)][:renders]:  # left: photo, right: render (gsplat's canvas)
        shutil.copy2(png, keep / png.name)
    shutil.copy2(stats[-1], keep / "val_stats.json")
    return out
