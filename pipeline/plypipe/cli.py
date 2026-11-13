# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""plypipe CLI.

    python -m plypipe run <video|frames_dir> --out work/<scene> [options]

Stages run in order: frames, sfm, align, occupancy, train, export. A finished stage leaves
<out>/.done/<stage>.json and is skipped on the next run, so a crashed or pre-empted job
(Kaggle, RunPod spot) resumes where it stopped. --force <stage> re-runs that stage and every
stage after it. --only / --until restrict the range (e.g. --until occupancy on a CPU box).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from plypipe import STAGES, __version__
from plypipe.log import Log
from plypipe.util import Ctx, clear_from, is_done, mark_done


def _stage_fn(name: str):
    if name == "frames":
        from plypipe.stages import frames
        return frames.run_stage
    if name == "sfm":
        from plypipe.stages import sfm
        return sfm.run_stage
    if name == "align":
        from plypipe.stages import align
        return align.run_stage
    if name == "occupancy":
        from plypipe.stages import occupancy
        return occupancy.run_stage
    if name == "label":
        from plypipe.stages import label
        return label.run_stage
    if name == "mesh":
        from plypipe.stages import mesh
        return mesh.run_stage
    if name == "train":
        from plypipe.stages import train
        return train.run_stage
    if name == "export":
        from plypipe.stages import export
        return export.run_stage
    raise KeyError(name)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="plypipe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline (resumable)")
    r.add_argument("input", type=Path, help="a video, a folder of videos (clips of one space, in name order), or a folder of images")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--only", choices=STAGES, help="run just this stage")
    r.add_argument("--from", dest="from_stage", choices=STAGES, help="start at this stage")
    r.add_argument("--until", choices=STAGES, help="stop after this stage")
    r.add_argument("--force", choices=STAGES, help="re-run this stage and all later ones")
    r.add_argument("--quiet", action="store_true", help="log only to <out>/log.jsonl")
    g = r.add_argument_group("frames")
    # no argparse defaults here: a value present in opts beats plan.input_plan(), which measures the clips
    g.add_argument("--fps", type=float, help="kept frames per second of video (default: planned from the walk, ~2.2)")
    g.add_argument("--oversample", type=int, default=4, help="candidates per kept frame for the sharpness pick")
    g.add_argument("--max-frames", type=int, help="cap on kept frames (lowers fps for long videos)")
    g.add_argument("--long-side", type=int, help="working resolution (default: planned from the clips; 1600 for photos)")
    g = r.add_argument_group("label")
    g.add_argument("--no-labels", action="store_true", help="skip the SAM 3 object labelling")
    g.add_argument("--label-frames", type=int, default=120, help="frames SAM 3 looks at (spread over the capture)")
    g.add_argument("--label-threshold", type=float, default=0.5)
    g = r.add_argument_group("sfm")
    g.add_argument("--features", choices=["aliked", "aliked-lg", "sift"], default="aliked",
                   help="aliked: COLMAP's own (needs a pycolmap with ONNX; SIFT otherwise). aliked-lg: ALIKED + LightGlue "
                        "in PyTorch (GPU). sift: classic")
    g.add_argument("--matcher", choices=["auto", "sequential", "exhaustive", "vocabtree"], default="auto")
    g.add_argument("--no-loop-detection", action="store_true", help="sequential matching without retrieval loop closure")
    g.add_argument("--sfm-part", choices=["all", "match"], default="all",
                   help="match: stop the sfm stage after features and matching (the GPU half of a split solve); "
                        "map the database later with --reuse-matches")
    g.add_argument("--split-min", type=int, default=400,
                   help="--sfm-part match only splits captures with at least this many frames (smaller ones solve here)")
    g.add_argument("--reuse-matches", action="store_true",
                   help="re-solve from the features and matches already in colmap/database.db (skips extraction and matching)")
    g.add_argument("--no-mapper-fallback", action="store_true",
                   help="keep the global solve even when its camera path jumps or it drops views")
    g.add_argument("--camera-model", choices=["SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE"], default="SIMPLE_RADIAL",
                   help="lens model for SfM: OPENCV for wide action cams (DJI, GoPro), OPENCV_FISHEYE beyond ~150 deg")
    g.add_argument("--exhaustive-max", type=int, default=250, help="auto: exhaustive up to this many frames")
    g.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    # no argparse default: an explicit choice beats the automatic one (ordered video of --incremental-min frames or
    # more goes straight to the incremental mapper), and a default would look explicit
    g.add_argument("--mapper", choices=["global", "incremental"],
                   help="global SfM (fast) or incremental (slower, more robust on rotation-heavy captures). "
                        "Default: global, except ordered video of --incremental-min frames or more")
    g.add_argument("--incremental-min", type=int, default=400,
                   help="ordered video with at least this many frames skips the global mapper when --mapper is not given")
    g.add_argument("--focal-35mm", type=float,
                   help="35 mm-equivalent focal of the lens used (phone spec, e.g. 24 for the S24 FE main camera; "
                        "~26 with video stabilisation crop). Starts SfM from the right intrinsics.")
    g = r.add_argument_group("align / occupancy")
    g.add_argument("--height-cam", type=float, default=1.45,
                   help="median phone height above the floor in metres (sets scale); 0 = keep SfM units")
    g.add_argument("--cell", type=float, default=0.1, help="occupancy cell size in metres")
    g = r.add_argument_group("train / export")
    g.add_argument("--preset", choices=["quick", "standard", "high"], default="standard")
    g.add_argument("--no-ppisp", dest="ppisp", action="store_false", default=None, help="no PPISP (the default)")
    g.add_argument("--ppisp", dest="ppisp", action="store_true",
                   help="PPISP exposure model in training (the exported twin renders brighter than the photos)")
    g.add_argument("--iterations", type=int, help="override the preset's training steps")
    g.add_argument("--sh-degree", type=int, choices=[0, 1, 2, 3], help="override the preset's SH degree")
    g.add_argument("--cap-max", type=int, help="override the preset's MCMC splat cap")
    g.add_argument("--min-opacity", type=float, default=0.03,
                   help="export drops splats below this activated opacity (0.03 default; lower keeps fine detail)")
    g.add_argument("--train-dry-run", action="store_true", help="rehearsal: load the scene in the trainer but skip GPU training")
    g.add_argument("--holdout", type=int, default=30,
                   help="hold every Nth view out of training and score the twin on it (PSNR/SSIM/LPIPS); 0 disables")
    g.add_argument("--holdout-block", type=int, default=0,
                   help="also hold out this many consecutive frames from the middle of the longest clip and score them "
                        "apart from the every-Nth views (needs --holdout); 0 disables")
    g.add_argument("--opacity-reg", type=float,
                   help="MCMC opacity regulariser (A/B option; the trainer's default is 0.01)")
    g.add_argument("--pose-opt", action="store_true", help="refine camera poses during training (A/B option)")
    g.add_argument("--depth-loss", action="store_true", help="supervise depth at the SfM track points (A/B option)")
    g.add_argument("--bilateral-grid", action="store_true",
                   help="exposure/white balance per photo with bilateral grids instead of PPISP (the twin keeps the photos' brightness)")
    g.add_argument("--exposure-affine", action="store_true",
                   help="exposure/white balance as one affine colour transform per photo (the bilateral grid at 1x1x1), no more "
                        "than --exposure-fold can fold into the splats (A/B option)")
    g.add_argument("--exposure-fold", type=int, metavar="STEPS",
                   help="after training (meant for --bilateral-grid): fit the colour map from the twin's own colours to the photos on "
                        "training views, fold it into the splats, and train on for STEPS with no appearance model, so the viewer "
                        "shows the photos' brightness (A/B option; 0 folds without the steps)")
    g.add_argument("--random-bkgd", action="store_true", help="random training backgrounds: no see-through splats")
    g.add_argument("--scale-mcmc", action="store_true",
                   help="scale gsplat's MCMC noise and scale_reg to the room size (A/B: no gain measured on Dr Johnson)")
    g.add_argument("--antialiased", dest="antialiased", action="store_true", default=True,
                   help="antialias the splats while training (on by default; --no-antialias turns it off)")
    g.add_argument("--no-antialias", dest="no_antialias", action="store_true",
                   help="skip antialiasing (only for a speed test: distant splats sparkle without it)")
    g.add_argument("--gsplat-dir", type=Path, help="gsplat checkout (default $PLY_GSPLAT_DIR or /opt/gsplat)")
    g.add_argument("--ply", type=Path, help="export this PLY instead of <out>/scene.ply")
    g.add_argument("--apply-align", action="store_true", help="export: transform an unaligned PLY into the scene frame")
    g.add_argument("--no-crop", action="store_true")
    g.add_argument("--lite-max", type=int, help="splats in the phone file (default 800000); a larger twin is cut down to this many")
    g.add_argument("--lite-finetune", type=int, metavar="STEPS",
                   help="export: train the phone file on for this many steps at its own splat count and SH degree, and ship it "
                        "when the viewer shows it better than the one it started from. Default 3000, when the phone file has "
                        "fewer splats or SH bands than the twin and a GPU can score it; 0 disables")
    g.add_argument("--dense-occupancy", dest="dense_occupancy", action="store_true", default=True,
                   help="export: rebuild the walking grid from the twin's own depth (5 cm cells; walls and furniture the camera "
                        "solve has no points on), with the sparse grid where the depth saw nothing, and use it unless fewer "
                        "capture positions are walkable on it. The default where a GPU can render")
    g.add_argument("--no-dense-occupancy", dest="dense_occupancy", action="store_false",
                   help="export: keep the walking grid from the camera solve")
    g.add_argument("--dense-layout", dest="dense_layout", action="store_true", default=True,
                   help="export: measure the walls, doors and ceiling (shell.json) from dense_points.npz and the camera path, "
                        "and use it when it passes its own checks, else keep the sparse shell. The default where dense points exist")
    g.add_argument("--no-dense-layout", dest="dense_layout", action="store_false",
                   help="export: keep the layout measured from the camera solve's sparse points")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out, quiet=a.quiet)
    opts = {k: v for k, v in vars(a).items() if k not in ("cmd", "input", "out")}
    if opts.get("gsplat_dir"):
        opts["gsplat_dir"] = str(opts["gsplat_dir"])
    ctx = Ctx(out=out, log=log, opts=opts)

    order = list(STAGES)
    lo = order.index(a.only or a.from_stage or "frames")
    hi = order.index(a.only or a.until or "export")  # the splat is the product: the three.js twin/mesh is opt-in
    if a.force:
        clear_from(ctx, order[order.index(a.force):])
    (out / "job.json").write_text(json.dumps({"input": str(a.input), "opts": opts, "version": __version__}, indent=2, default=str))

    for name in order[lo:hi + 1]:
        if is_done(ctx, name):
            log.emit(name, "skip", "already done")
            continue
        log.emit(name, "start")
        t0 = time.time()
        try:
            fn = _stage_fn(name)
            report = fn(ctx, a.input) if name == "frames" else fn(ctx)
        except Exception as e:  # report and stop; the next run resumes here
            log.emit(name, "error", str(e), trace=traceback.format_exc()[-2000:])
            return 1
        report["seconds"] = round(time.time() - t0, 1)
        if report.get("partial"):  # a split stage paused part-way: not done, and nothing after it can run yet
            log.emit(name, "paused", report["partial"], **report)
            return 0
        mark_done(ctx, name, report)
        log.emit(name, "done", **report)
    # the same gate the worker applies: say what is wrong with this run, in words that name the cause
    if hi == len(order) - 1:
        from plypipe import plan, verify
        rep = out / "report.json"
        quality = verify.check(json.loads(rep.read_text()) if rep.exists() else {}, plan.read(out))
        for issue in quality["issues"]:
            log.emit("verify", "warn" if issue["level"] == "warn" else "error", issue["message"])
        if not quality["ok"]:
            log.emit("verify", "error", "quality gate failed: this twin should not be published")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
