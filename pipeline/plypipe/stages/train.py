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

from plypipe import device, plan, restore
from plypipe.util import Ctx, Paused, read_scene, run

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
    from plypipe import people
    masks = ctx.dense_dir / people.MASKS
    if not ctx.opt("no_people_mask") and masks.is_dir() and any(masks.glob("*.png")):
        flags += ["--masks", str(masks)]  # people (plypipe.people): no colour or depth loss on them, in every run on this job
    return flags, block


def launcher_env() -> dict:
    pkg_root = Path(__file__).resolve().parents[2]
    # MPLBACKEND: notebooks (Kaggle) export an inline backend that a separate training venv can't load
    return {"PYTHONPATH": os.pathsep.join([str(pkg_root), os.environ.get("PYTHONPATH", "")]), "MPLBACKEND": "Agg",
            # gsplat's cli() spawns one process per visible GPU (Kaggle gives two T4s); train on one
            "CUDA_VISIBLE_DEVICES": os.environ.get("PLY_TRAIN_GPU", "0")}


def train_on(ctx: Ctx, ply: Path, steps: int, sh_degree: int, result: Path, stage: str, grow_cap: int | None = None,
             round_no: int | None = None) -> Path:
    """Train on from the model in `ply` for `steps`, on the views the job trained on and with no appearance model.
    Without `grow_cap` at its own splat count (plypipe.finetune: MCMC moves splats and adds none); with it, MCMC adds
    splats up to `grow_cap` (a growth round, plypipe.growth). `result` is this run's alone, with its own restore
    points: a round that was stopped continues where it was. Returns the PLY it leaves."""
    flags, _ = split_flags(ctx)
    cmd = launcher(ctx, result) + ["--no-ppisp", "--from-ply", str(ply), "--iterations", str(steps), "--sh-degree", str(sh_degree), *flags]
    if grow_cap:
        cmd += ["--grow-cap", str(int(grow_cap))]
    if ctx.opt("random_bkgd"):
        cmd.append("--random-bkgd")
    rdir = (result / "restore").resolve()
    cmd += ["--restore", str(rdir)]
    key = restore.restore_key(cmd)
    if restore.usable(rdir, key):
        cmd.append("--resume")
    else:
        shutil.rmtree(result, ignore_errors=True)
    # no part-way PLYs: the worker asks for them for its live view of the train stage, and nothing shows these
    env = {**launcher_env(), "PLY_LIVE_SNAPSHOTS": "", "PLY_RESTORE_KEY": key, **({"PLY_TRAIN_ROUND": str(round_no)} if round_no else {})}
    run(cmd, ctx, stage, env=env, progress_every=50, own_session=True)
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
    if not (ctx.opt("bilateral_grid") or ctx.opt("exposure_affine") or ctx.opt("ppisp")):
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
    dev = plan.read(ctx.out).get("device") or {}  # plypipe.device: what this machine's GPU and RAM can hold
    budget = plan.model_plan(views, bounds or None, preset, height, iterations=ctx.opt("iterations"),
                             cap_max=ctx.opt("cap_max"), megapixels=view_megapixels(ctx.data_dir / "images"),
                             max_gaussians=dev.get("maxGaussians"))
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
    if dev.get("packed"):
        cmd.append("--packed")
    if dev.get("loaderWorkers") is not None:
        cmd += ["--loader-workers", str(dev["loaderWorkers"])]
    for flag in ("pose_opt", "depth_loss", "bilateral_grid", "exposure_affine", "random_bkgd"):
        if ctx.opt(flag):
            cmd.append("--" + flag.replace("_", "-"))
    # an iPhone LiDAR capture trains against its measured depth (Echo-3-style: clean walls, no fog); --no-lidar-depth off
    lidar = ctx.out / "lidar.json"
    if lidar.is_file() and ctx.opt("lidar_depth", True) is not False:
        cmd += ["--lidar-depth", str(lidar)]
        ctx.log.emit("train", "progress", "depth loss on the LiDAR maps of the kept frames")
    elif (ctx.out / "scandepth.json").is_file() and ctx.opt("lidar_depth", True) is not False:
        # route 8c (plypipe.scan): the laser scan, registered to the video, is each view's depth and the training start
        cmd += ["--lidar-depth", str(ctx.out / "scandepth.json")]
        if (ctx.out / "scan" / "init.npz").is_file():
            cmd += ["--init-points", str(ctx.out / "scan" / "init.npz")]
        ctx.log.emit("train", "progress", "depth loss on the laser scan, training starts from its points")
    oom_retry = None
    # restore points (plypipe.restore): autosave, save-and-exit on a signal, and pick up at the saved step when the
    # same settings run again (a closed window, Ctrl+C, a cloud container that was stopped)
    rdir = (result / "restore").resolve()
    cmd += ["--restore", str(rdir)]
    rkey = restore.restore_key(cmd)
    found = restore.usable(rdir, rkey)
    if found:
        cmd.append("--resume")
        ctx.log.emit("train", "progress", f"restore point found: continuing at step {found['step'] + 1} of {found.get('of')} "
                                          f"({found.get('splats')} splats, saved {found.get('why')})")
    elif restore.read_meta(rdir):
        ctx.log.emit("train", "warn", "a restore point from other settings was left alone: training starts over")
    env = {**launcher_env(), "PLY_RESTORE_KEY": rkey}
    base_done = result / "base_done.json"
    try:
        if _done_key(base_done) == rkey and list((result / "ply").glob("point_cloud_*.ply")):
            ctx.log.emit("train", "progress", "base training already finished: continuing with the growth rounds")
        else:
            run(cmd, ctx, "train", env=env, progress_every=50, own_session=True)
    except RuntimeError as e:
        # a small card (plypipe.device) whose per-splat estimate was optimistic: one retry at a lower cap, from scratch,
        # rather than a failed job. An explicit --cap-max is the operator's number and is not second-guessed
        if not device.oom(str(e)) or ctx.opt("cap_max") or not dev.get("maxGaussians"):
            raise
        lower = max(100_000, int(budget["capMax"] * device.OOM_RETRY_FACTOR))
        ctx.log.emit("train", "warn", f"CUDA out of memory at {budget['capMax']:,} splats: retrying once at {lower:,}")
        oom_retry = {"from": budget["capMax"], "to": lower}
        i = cmd.index("--cap-max")
        cmd[i + 1] = str(lower)
        budget = {**budget, "capMax": lower, "why": budget["why"] + f"; out of memory, retried at {lower:,}"}
        plan.write(ctx.out, "model", budget)
        shutil.rmtree(result, ignore_errors=True)
        cmd = [c for c in cmd if c != "--resume"]
        rkey = env["PLY_RESTORE_KEY"] = restore.restore_key(cmd)
        run(cmd, ctx, "train", env=env, progress_every=50, own_session=True)
    if ctx.opt("train_dry_run"):
        return {"dryRun": True}
    base_done.write_text(json.dumps({"key": rkey}))
    plys = sorted((result / "ply").glob("point_cloud_*.ply"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not plys:
        raise RuntimeError(f"training finished but no PLY in {result / 'ply'}")
    ship, scored, fold = plys[-1], result, None
    grown = None
    why_not = growth_blocked(ctx)
    if why_not:
        ctx.log.emit("train", "progress", f"growth rounds off: {why_not}")
    else:
        preset0 = json.loads((result / "ply_preset.json").read_text()) if (result / "ply_preset.json").exists() else {}
        ship, scored, grown = grow(ctx, result, plys[-1], int(budget["iterations"]), int(preset0.get("shDegree", 3)), dev, budget)
    if ctx.opt("exposure_fold") is not None:
        ship, scored, fold = fold_exposure(ctx, result, plys[-1], int(ctx.opt("exposure_fold")), block)
    shutil.copy2(ship, ctx.out / "scene.ply")
    dens = None
    try:  # soft patches: too few splats where the cameras looked (grow), or too few cameras (film again)
        from plypipe import growth
        dens = growth.density(growth.ply_means(ctx.out / "scene.ply"), growth.cameras_from_scene(ctx.out),
                              float((read_scene(ctx) or {}).get("floorY", 0.0)))
        if dens.get("underFilmed"):
            ctx.log.emit("train", "warn", f"{dens['underFilmed']} floor cells ({dens['cellM']} m) have splats but were barely filmed: "
                         f"film again near {dens['underFilmedAt'][:5]} (scene x, z) for an even twin")
        if dens.get("underDense"):
            ctx.log.emit("train", "progress", f"{dens['underDense']} well-filmed cells are still thin ({dens['underDenseShare']:.1%})")
    except Exception as e:
        ctx.log.emit("train", "warn", f"density check skipped: {e}")
    preset_info = json.loads((result / "ply_preset.json").read_text()) if (result / "ply_preset.json").exists() else {}
    # Windows drivers can spill CUDA memory into system RAM instead of failing (the "sysmem fallback"): no OOM, so no
    # retry, just a run many times slower. Reserved memory above the card's own size is that spill
    if preset_info.get("peakReservedGB") and preset_info.get("vramTotalGB") and preset_info["peakReservedGB"] > preset_info["vramTotalGB"]:
        ctx.log.emit("train", "warn", f"training used {preset_info['peakReservedGB']} GB on a {preset_info['vramTotalGB']} GB card: it spilled into "
                     "system RAM (slow). Set 'CUDA - Sysmem Fallback Policy' to 'Prefer No Sysmem Fallback' in the NVIDIA Control Panel, "
                     "or lower BYTES_PER_GAUSSIAN's estimate in plypipe/device.py")
    keep = ctx.out / "eval"
    if scored != result and fold is not None:  # the scores below are the settled model's: the trainer's scores of the model before the fold go beside the fit
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
            **({"oomRetry": oom_retry} if oom_retry else {}),
            **({"growth": grown} if grown else {}), **({"density": dens} if dens else {}),
            **({"eval": scores} if scores else {}), **extra, **({"exposureFold": fold} if fold else {})}


def _done_key(path: Path) -> str | None:
    try:
        return json.loads(path.read_text()).get("key")
    except (OSError, ValueError):
        return None


def growth_blocked(ctx: Ctx) -> str | None:
    """Why growth rounds cannot be judged fairly on this run, or None. A round must train like the base run (no
    appearance model: train_on trains plain colours) and be scored on held-out views."""
    if ctx.opt("no_growth"):
        return "--no-growth"
    if ctx.opt("train_dry_run"):
        return "dry run"
    if int(ctx.opt("holdout", 30) or 0) <= 1:
        return "no held-out views (--holdout 0): nothing to measure a round on"
    for flag in ("ppisp", "bilateral_grid", "exposure_affine", "exposure_fold"):
        if ctx.opt(flag) not in (None, False):
            return f"--{flag.replace('_', '-')}: rounds train without an appearance model, so they would not compare"
    return None


def _view_masks(ctx: Ctx) -> list[Path | None]:
    """People keep-masks of the held-out views, in gsplat's val order (sorted names, every Nth)."""
    from plypipe import people
    n = int(ctx.opt("holdout", 30) or 0)
    names = sorted(p.name for p in (ctx.data_dir / "images").glob("*.jpg")) if (ctx.data_dir / "images").is_dir() else []
    mdir = ctx.dense_dir / people.MASKS
    return [(mdir / people.mask_name(nm)) if (mdir / people.mask_name(nm)).is_file() else None for i, nm in enumerate(names) if n > 1 and i % n == 0]


def grow(ctx: Ctx, base_result: Path, base_ply: Path, base_steps: int, sh_degree: int, dev: dict, budget: dict) -> tuple[Path, Path, dict]:
    """More splats while the held-out views say it helps (plypipe.growth). Returns the PLY and result folder of the best
    round and the record of every round. Resumable: finished rounds are kept in growth.json, a stopped round has its
    own restore point."""
    from plypipe import growth
    state_p = base_result / "growth.json"
    masks = _view_masks(ctx)
    try:
        hist = json.loads(state_p.read_text())
    except (OSError, ValueError):
        ev = held_out_scores(base_result, ctx.out / "eval" / "grow-0")
        hist = {"rules": growth.rules(), "best": 0, "rounds": [{
            "round": 0, "result": str(base_result), "ply": str(base_ply), "splats": _splats(base_ply), "steps": base_steps,
            "eval": ev, "patches": growth.patches(base_result, masks), "kept": True, "why": "base run"}]}
        state_p.write_text(json.dumps(hist, indent=1))
    plan.write(ctx.out, "growth", {"rules": growth.rules()})
    stop = None
    while True:
        best = hist["rounds"][hist["best"]]
        cap, why = growth.next_cap(best["splats"], dev.get("maxGaussians"))
        if cap is None:
            stop = {"reason": "gpu-full", "why": why}
            break
        i = len(hist["rounds"])
        steps = growth.round_steps(base_steps)
        ctx.log.emit("train", "progress", f"growth round {i}: {why} splats, {steps:,} steps from round {best['round']}")
        rres = base_result / f"grow-{i}"
        try:
            ply = train_on(ctx, Path(best["ply"]), steps, sh_degree, rres, "train", grow_cap=cap, round_no=i)
        except Paused:
            raise
        except RuntimeError as e:
            oom = device.oom(str(e))
            hist["rounds"].append({"round": i, "cap": cap, "kept": False, "why": "out of GPU memory" if oom else f"failed: {str(e)[-200:]}"})
            state_p.write_text(json.dumps(hist, indent=1))
            stop = {"reason": "gpu-full" if oom else "round-failed", "why": hist["rounds"][-1]["why"]}
            break
        cand = {"round": i, "result": str(rres), "ply": str(ply), "splats": _splats(ply), "steps": steps, "cap": cap,
                "eval": held_out_scores(rres, ctx.out / "eval" / f"grow-{i}"), "patches": growth.patches(rres, masks)}
        ok, verdict = growth.judge(best, cand)
        cand.update(kept=ok, why=verdict)
        hist["rounds"].append(cand)
        if ok:
            hist["best"] = i
        state_p.write_text(json.dumps(hist, indent=1))
        db, bdb = growth.score_db(cand["eval"]), growth.score_db(best["eval"])
        ctx.log.emit("train", "progress", f"round {i}: {cand['splats']:,} splats, {db if db is None else round(db, 2)} dB "
                     f"(best {bdb if bdb is None else round(bdb, 2)}), patches {(cand['patches'] or {}).get('share')}: "
                     f"{'kept' if ok else 'not kept'} ({verdict})")
        if not ok:
            stop = {"reason": "converged", "why": verdict}
            break
    improving = any(r.get("kept") and r["round"] > 0 for r in hist["rounds"]) or bool(budget.get("deviceCapMax"))
    hist["stop"] = stop
    hist["needsBiggerGpu"] = bool(stop and stop["reason"] == "gpu-full" and improving)
    if hist["needsBiggerGpu"]:
        ctx.log.emit("train", "warn", "this twin was still improving when the GPU ran out of room: a bigger GPU would make it sharper "
                     f"({stop['why']})")
    state_p.write_text(json.dumps(hist, indent=1))
    best = hist["rounds"][hist["best"]]
    summary = {"rounds": [{k: r.get(k) for k in ("round", "splats", "steps", "kept", "why")} | {"db": growth.score_db(r.get("eval")),
               "lpips": (r.get("eval") or {}).get("lpips"), "patchShare": (r.get("patches") or {}).get("share")} for r in hist["rounds"]],
               "best": hist["best"], "stop": stop, "needsBiggerGpu": hist["needsBiggerGpu"], "rules": hist["rules"]}
    return Path(best["ply"]), Path(best["result"]), summary


def _splats(ply: Path) -> int:
    """Splat count from the PLY header alone (a 2M-splat file is ~500 MB)."""
    with open(ply, "rb") as f:
        for _ in range(64):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"{ply}: no vertex count in the header")


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
