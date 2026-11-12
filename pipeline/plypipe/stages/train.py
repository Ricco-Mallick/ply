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
import time
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


def launcher(ctx: Ctx, result: Path) -> list[str]:
    # PLY_TRAIN_PYTHON: training may live in its own venv (install_fast.sh: prebuilt gsplat 1.5.3 on Python 3.10)
    return [os.environ.get("PLY_TRAIN_PYTHON") or sys.executable, "-m", "plypipe.train_launcher", "--examples", str(gsplat_examples(ctx)),
            "--data", str(ctx.data_dir), "--result", str(result), "--preset", ctx.opt("preset", "standard")]


def split_flags(ctx: Ctx) -> tuple[list[str], int]:
    """What every run on a job must share: how the splats are drawn and which views are held out. Returns the
    launcher flags and the block's size."""
    flags = [] if ctx.opt("no_antialias") else ["--antialiased"]
    holdout = int(ctx.opt("holdout", 30) or 0)
    if holdout > 1:
        flags += ["--holdout", str(holdout)]
    block = int(ctx.opt("holdout_block", 0) or 0)
    if block:
        if holdout <= 1:
            raise RuntimeError("--holdout-block is scored next to the every-Nth views: it needs --holdout N (N > 1)")
        flags += ["--holdout-block", str(block), "--frames-index", str(ctx.out / "frames_index.json")]
    if ctx.opt("opacity_reg") is not None:
        flags += ["--opacity-reg", str(ctx.opt("opacity_reg"))]
    return flags, block


def launcher_env() -> dict:
    pkg_root = Path(__file__).resolve().parents[2]
    # MPLBACKEND: notebooks (Kaggle) export an inline backend that a separate training venv can't load
    return {"PYTHONPATH": os.pathsep.join([str(pkg_root), os.environ.get("PYTHONPATH", "")]), "MPLBACKEND": "Agg",
            # gsplat's cli() spawns one process per visible GPU (Kaggle gives two T4s); train on one
            "CUDA_VISIBLE_DEVICES": os.environ.get("PLY_TRAIN_GPU", "0")}


def train_on(ctx: Ctx, ply: Path, steps: int, sh_degree: int, result: Path, stage: str) -> Path:
    """Train on from the model in `ply` for `steps`, at its own splat count (plypipe.finetune), on the views the job
    trained on and with no appearance model. `result` is this run's alone. Returns the PLY it leaves."""
    shutil.rmtree(result, ignore_errors=True)
    flags, _ = split_flags(ctx)
    cmd = launcher(ctx, result) + ["--no-ppisp", "--from-ply", str(ply), "--iterations", str(steps), "--sh-degree", str(sh_degree), *flags]
    if ctx.opt("random_bkgd"):
        cmd.append("--random-bkgd")
    # no part-way PLYs: the worker asks for them for its live view of the train stage, and nothing shows these
    run(cmd, ctx, stage, env={**launcher_env(), "PLY_LIVE_SNAPSHOTS": ""}, progress_every=50)
    plys = sorted((result / "ply").glob("point_cloud_*.ply"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not plys:
        raise RuntimeError(f"the fine-tune finished but left no PLY in {result / 'ply'}")
    return plys[-1]


def finetune(ctx: Ctx, ply: Path, steps: int, sh_degree: int, stage: str = "export") -> dict:
    """The phone file's fine-tune (train_on). Returns the fitted PLY's path and the trainer's held-out scores.
    <out>/train-lite is this run's alone."""
    result = ctx.out / "train-lite"
    fitted = train_on(ctx, ply, steps, sh_degree, result, stage)
    scores = held_out_scores(result, ctx.out / "eval" / "lite")
    held = held_out_scores(result, ctx.out / "eval" / "lite", stage="block") if split_flags(ctx)[1] else None
    return {"ply": fitted, **({"eval": scores} if scores else {}), **({"evalBlock": held} if held else {})}


def fold_exposure(ctx: Ctx, result: Path, ply: Path, steps: int, block: int) -> tuple[Path, Path, dict]:
    """Variant Q1 (plypipe.exposure) on the model the trainer left in `ply`: fit the colour map from its raw render to
    the training photos, fold the map into the splats' colours (<result>/folded.ply), and train on for `steps` with
    no appearance model (<result>/settle), so the model settles in the colours the viewer shows.

    Returns the model to ship, the run whose held-out scores are that model's, and the record for the report. A fit
    that cannot be made, is refused or fails ships the model as trained; a settling run that fails ships the folded
    model. Both say why."""
    from plypipe import exposure
    out: dict = {"steps": steps, "folded": False, "settled": False}
    if not (ctx.opt("bilateral_grid") or ctx.opt("ppisp")):
        ctx.log.emit("train", "warn", "--exposure-fold without an appearance model: the colours are the photos' already, so the map is "
                     "near the identity and the steps after it are plain extra training")
    t0 = time.time()
    try:
        held = []
        if block:  # the trainer's own list: a block it did not name is a block the fit could read by mistake
            try:
                held = json.loads((result / "ply_block.json").read_text())["frames"]
            except (OSError, ValueError, KeyError):
                raise RuntimeError("the trainer left no list of the held-out block's views, which the fit must not read") from None
        fit = exposure.fit_views(ply, ctx.data_dir, int(ctx.opt("holdout", 30) or 0), held, antialiased=not ctx.opt("no_antialias"))
        if fit is None:
            fit = {"why": "not fitted: needs a GPU with gsplat, and PINHOLE cameras"}
        out.update(fit)
        if "why" not in fit:
            model = exposure.fold_ply(ply, result / "folded.ply", fit["matrix"], fit["offset"])
    except Exception as e:
        out["why"] = f"the fit failed: {str(e)[-160:]}"
    out["fitSeconds"] = round(time.time() - t0, 1)
    if "why" in out:
        ctx.log.emit("train", "warn", f"exposure fold skipped, the model ships as trained: {out['why']}")
        return ply, result, out
    out["folded"] = True
    ctx.log.emit("train", "progress", f"exposure fold: gains {', '.join(f'{g:.3f}' for g in fit['gains'])} from {fit['views']} training views; "
                 f"brightness there x{fit.get('brightnessBefore')} -> x{fit.get('brightnessAfter')} of the photos")
    if steps <= 0:
        return result / "folded.ply", result, out
    try:  # the fit leaves this process holding cached GPU memory, which the trainer would go without
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    t0 = time.time()
    try:
        settled = train_on(ctx, result / "folded.ply", steps, model["shDegree"], result / "settle", "train")
    except Exception as e:
        ctx.log.emit("train", "warn", f"exposure fold: the {steps} steps after it failed, the folded model ships without them: {str(e)[-160:]}")
        return result / "folded.ply", result, {**out, "why": f"the {steps} steps after the fold failed: {str(e)[-160:]}"}
    return settled, result / "settle", {**out, "settled": True, "settleSeconds": round(time.time() - t0, 1)}


def run_stage(ctx: Ctx) -> dict:
    preset = ctx.opt("preset", "standard")
    result = ctx.train_dir
    cmd = launcher(ctx, result)
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
    flags, block = split_flags(ctx)
    cmd += flags
    for flag in ("pose_opt", "depth_loss", "bilateral_grid", "random_bkgd"):
        if ctx.opt(flag):
            cmd.append("--" + flag.replace("_", "-"))
    run(cmd, ctx, "train", env=launcher_env(), progress_every=50)
    if ctx.opt("train_dry_run"):
        return {"dryRun": True}
    plys = sorted((result / "ply").glob("point_cloud_*.ply"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not plys:
        raise RuntimeError(f"training finished but no PLY in {result / 'ply'}")
    ship, scored, fold = plys[-1], result, None
    if ctx.opt("exposure_fold") is not None:
        ship, scored, fold = fold_exposure(ctx, result, plys[-1], int(ctx.opt("exposure_fold")), block)
    shutil.copy2(ship, ctx.out / "scene.ply")
    preset_info = json.loads((result / "ply_preset.json").read_text()) if (result / "ply_preset.json").exists() else {}
    keep = ctx.out / "eval"
    if scored != result:  # the scores below are the settled model's: the trainer's scores of the model before the fold go beside the fit
        before = {"eval": held_out_scores(result, keep / "trained"), "evalBlock": held_out_scores(result, keep / "trained", stage="block") if block else None}
        fold["trained"] = {k: v for k, v in before.items() if v}
    scores = held_out_scores(scored, keep)
    if scores:
        ctx.log.emit("train", "progress", f"held-out views: PSNR {scores['psnr']:.2f}, SSIM {scores['ssim']:.3f}, "
                     f"LPIPS {scores['lpips']:.3f} over {scores['views']} views")
    extra = block_report(scored, keep) if block else {}
    if block and not extra.get("evalBlock"):
        ctx.log.emit("train", "warn", "the held-out block was requested but the trainer left no scores for it")
    return {"ply": "scene.ply", "sizeMB": round((ctx.out / "scene.ply").stat().st_size / 2**20, 1),
            "fromStep": int(plys[-1].stem.rsplit("_", 1)[1]), **preset_info,
            "plan": {k: budget[k] for k in ("iterations", "capMax", "why")},
            **({"eval": scores} if scores else {}), **extra, **({"exposureFold": fold} if fold else {})}


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


def held_out_scores(result: Path, keep: Path, renders: int = 4, stage: str = "val") -> dict | None:
    """gsplat's scores on the held-out views (stats/<stage>_step*.json) plus a few render|photo pairs as evidence."""
    stats = sorted((result / "stats").glob(f"{stage}_step*.json"), key=lambda p: int(p.stem.rsplit("step", 1)[1]))
    if not stats:
        return None
    raw = json.loads(stats[-1].read_text())
    names = {"psnr": "psnr", "ssim": "ssim", "lpips": "lpips", "cc_psnr": "ccPsnr", "cc_ssim": "ccSsim",
             "cc_lpips": "ccLpips", "num_GS": "gaussians", "ellipse_time": "secondsPerView"}
    out = {v: round(float(raw[k]), 4) for k, v in names.items() if k in raw}
    pngs = sorted((result / "renders").glob(f"{stage}_step{stats[-1].stem.rsplit('step', 1)[1].lstrip('0') or '0'}_*.png"))
    out["views"] = len(pngs) or None
    keep.mkdir(parents=True, exist_ok=True)
    for png in pngs[:: max(1, len(pngs) // renders)][:renders]:  # left: photo, right: render (gsplat's canvas)
        shutil.copy2(png, keep / png.name)
    shutil.copy2(stats[-1], keep / f"{stage}_stats.json")
    return out


def block_report(result: Path, keep: Path) -> dict:
    """The held-out block's scores and image names, and the per-view scores of both hold-outs (eval/per_view.json)."""
    out: dict = {}
    scores = held_out_scores(result, keep, stage="block")
    if scores:
        out["evalBlock"] = scores
    try:
        block = json.loads((result / "ply_block.json").read_text())
        out.update({"blockImages": block["frames"], "blockClip": block["clip"], "blockSeconds": block["seconds"]})
    except (OSError, ValueError, KeyError):
        pass
    views = sorted((result / "stats").glob("per_view_step*.json"), key=lambda p: int(p.stem.rsplit("step", 1)[1]))
    if views:
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(views[-1], keep / "per_view.json")
        out["perView"] = "eval/per_view.json"
    return out
