# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""export: scene.ply -> scene.spz (full) + scene_lite.spz (SH2, for phones).

Uses @playcanvas/splat-transform (pinned) through npx. The PLY is already in the scene frame
(training ran on the aligned model), so no axis change is applied. Gaussians with NaN/Inf are
removed and everything outside the room box (scene bounds + 1 m, y in [-0.5, top + 0.5]) is
cropped, which drops most floaters for free.

Cleanup (on unless --no-clean): near-invisible Gaussians (opacity < 3 %) and oversized blobs (any axis > 1 m) are
dropped with value filters, then splat-transform's floater filter removes Gaussians that don't contribute to any solid
voxel. The floater filter needs WebGPU; without it the stage logs a warning and exports the value-filtered model.
The phone file keeps SH bands up to LITE_SH and, above LITE_MAX Gaussians, is decimated (phones crash on
multi-million-splat scenes). Both files get their SH back at SH_BITS (see plypipe.spz): splat-transform rounds it to
4-5 bits, which cost Dr Johnson 1 dB on desktop; dropping SH entirely cost the phone file another 2.6 dB.

If a PLY from elsewhere (trained on the unaligned model) is given via --ply, pass --apply-align
and the stage applies scene.json's transform with splat-transform's translate/rotate/scale.
"""
from __future__ import annotations

import os

import math
import shutil
from pathlib import Path

import numpy as np

from plypipe import clean as clean_mod
from plypipe import spz as spz_mod
from plypipe import topview as topview_mod
from plypipe.util import Ctx, need, read_scene, run, splat_transform_cmd, update_scene

SPLAT_TRANSFORM = "@playcanvas/splat-transform@3.10.1"
LITE_MAX = 800_000
LITE_SH = 2
# 6 bits keeps all but ~0.07 dB of the full byte on Dr Johnson, at 29 MB against 38 MB (22 MB as splat-transform writes it)
SH_BITS = 6
# splat-transform compares activated values: opacity 0..1, scales in metres
CLEAN = ["-V", "opacity,gt,0.03", "-V", "scale_0,lt,1", "-V", "scale_1,lt,1", "-V", "scale_2,lt,1"]


def ply_count(ply: Path) -> int:
    """Vertex count from a PLY header."""
    with ply.open("rb") as f:
        for _ in range(200):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return 0


# splat-transform runs its actions (-r/-s/-t/-B) in PlayCanvas's engine frame, which is the PLY
# frame turned 180 deg about Z: engine = D @ ply with D = diag(-1, -1, 1). Measured against the
# real tool in tests/test_export.py. Its SPZ writer stores positions unconverted (spz
# CoordinateSystem.UNSPECIFIED), so the SPZ keeps the PLY's scene frame (+Y up, three.js-style).
D = np.diag([-1.0, -1.0, 1.0])


def _euler_for_splat_transform(R: np.ndarray) -> tuple[float, float, float]:
    """Degrees (a, b, c) such that `-r a,b,c` applies R = Rz(c) @ Ry(b) @ Rx(a) in the engine frame."""
    b = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    if abs(R[2, 0]) < 0.99999:
        a = math.atan2(R[2, 1], R[2, 2])
        c = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: fold everything into a
        a = math.atan2(-R[1, 2], R[1, 1])
        c = 0.0
    return math.degrees(a), math.degrees(b), math.degrees(c)


def align_actions(T: np.ndarray) -> list[str]:
    """splat-transform actions applying p' = s R p + t to PLY-frame data (rotate, scale, translate)."""
    sR = T[:3, :3]
    s = float(np.cbrt(np.linalg.det(sR)))
    R_e = D @ (sR / s) @ D
    t_e = D @ T[:3, 3]
    ax, ay, az = _euler_for_splat_transform(R_e)
    return ["-r", f"{ax:.6f},{ay:.6f},{az:.6f}", "-s", f"{s:.8f}", "-t", f"{t_e[0]:.6f},{t_e[1]:.6f},{t_e[2]:.6f}"]


def value_filter_flags(opacity: float, scale: float | None) -> list[str]:
    """splat-transform `-V` filters in its user-facing units: opacity as a probability, scale in metres.

    splat-transform (3.10.1) converts these to the PLY's raw logit / log-scale itself and rejects an
    opacity outside (0, 1), so passing a logit here crashes the export instead of filtering.
    """
    p = min(max(float(opacity), 1e-4), 1 - 1e-4)
    f = ["-V", f"opacity,gt,{p:.4f}"]
    if scale:
        f += ["-V", f"scale_0,lt,{scale:.2f}", "-V", f"scale_1,lt,{scale:.2f}", "-V", f"scale_2,lt,{scale:.2f}"]
    return f


def room_bounds(scene: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Room box in the scene (PLY) frame: scene bounds + 1 m sideways, y from -0.5 to top + 0.5."""
    b = scene.get("bounds")
    if not b:
        return None
    mn, mx = b["min"], b["max"]
    return (np.array([mn[0] - 1.0, -0.5, mn[2] - 1.0]),
            np.array([mx[0] + 1.0, max(mx[1], 2.5) + 0.5, mx[2] + 1.0]))


def crop_box(scene: dict) -> str | None:
    """The room box in splat-transform's frame."""
    rb = room_bounds(scene)
    if rb is None:
        return None
    lo, hi = rb
    e_lo, e_hi = np.minimum(D @ lo, D @ hi), np.maximum(D @ lo, D @ hi)
    return ",".join(f"{v:.3f}" for v in (*e_lo, *e_hi))


# a splat outside the room box is only cropped if the training views barely see it: under this many full-resolution
# pixels of blend weight summed over the sampled views. The box alone cut the chapel's vault (its sparse points sit
# below the vault's splats), the views through its windows and its floor reflections: 3.4 dB. 91 % of the splats
# outside the box drew nothing at all; dropping those under 1 px removed 96 % of them at no measurable cost.
NEEDED_PX = 1.0


def needed_crop(src: Path, dest: Path, lo: np.ndarray, hi: np.ndarray, weights: np.ndarray,
                min_px: float = NEEDED_PX) -> dict:
    """Write src without the splats that are outside [lo, hi] and that no training view needs."""
    header, c = clean_mod.read_ply(src)
    xyz = np.stack([c["x"], c["y"], c["z"]], 1)
    outside = ((xyz < lo) | (xyz > hi)).any(1)
    if len(weights) != len(xyz):
        raise ValueError(f"{len(weights)} weights for {len(xyz)} splats")
    drop = outside & (weights < min_px)
    names = list(c.keys())
    keep = np.stack([c[k] for k in names], 1)[~drop].astype("<f4")
    lines = [ln for ln in header.splitlines() if ln and not ln.startswith(("element vertex", "end_header"))]
    at = next(i for i, ln in enumerate(lines) if ln.startswith("property"))
    out = "\n".join([*lines[:at], f"element vertex {len(keep)}", *lines[at:], "end_header"]) + "\n"
    with Path(dest).open("wb") as f:
        f.write(out.encode("latin-1"))
        f.write(keep.tobytes())
    return {"method": "needed", "outside": int(outside.sum()), "dropped": int(drop.sum()),
            "keptOutside": int((outside & ~drop).sum()), "minPx": min_px}


def run_stage(ctx: Ctx) -> dict:
    st = splat_transform_cmd()
    if st[0] == "npx":
        need("npx", "Install Node.js >= 20 (splat-transform runs through npx).")
    ply = Path(ctx.opt("ply") or ctx.out / "scene.ply")
    if not ply.exists():
        raise FileNotFoundError(f"{ply} missing: run the train stage or pass --ply")
    scene = read_scene(ctx)
    # SPZ v3 (gzip): the web viewer's Spark 2.3.1 rejects v4, splat-transform's default ("Unsupported SPZ version: 4")
    base = [*st, "-w", "--spz-version", "3", str(ply)]
    if ctx.opt("apply_align"):
        base += align_actions(np.array(scene["transform"]))
    base += ["-N"]
    box = crop_box(scene) if not ctx.opt("no_crop") else None
    if box:
        base += ["-B", box]
    clean = not ctx.opt("no_clean")
    total = ply_count(ply)
    # each probe is a full pass over the model; a 4M-splat export can spend more time probing than exporting
    probes = os.environ.get("PLY_EXPORT_PROBES", "1") != "0"
    # the value filters cut splats larger than 1 m; in a big room (a chapel, a hall) big splats are the walls, not
    # floaters, so scale the cap to the scene and fall back to opacity only if it would still gut the model
    diag = 10.0
    b0 = scene.get("bounds") or {}
    if b0:
        diag = float(np.linalg.norm(np.array(b0["max"]) - np.array(b0["min"]))) or 10.0
    min_op = float(ctx.opt("min_opacity", 0.03))
    scale_cap = max(1.0, 0.12 * diag)
    # the value filters cut thin, translucent gaussians; how many that is depends on how the model was
    # trained, so the thresholds are tried against the model itself instead of assumed. The floor is
    # "keep at least 85% of what training fitted": below that the filter is deleting detail, not noise.
    candidates = [
        {"opacity": min_op, "scale": scale_cap, "why": f"opacity>{min_op:.3f} and scale<{scale_cap:.2f}"},
        {"opacity": 0.02, "scale": None, "why": "opacity>0.020 (scale caps dropped)"},
        {"opacity": 0.01, "scale": None, "why": "opacity>0.010 (measured: the stricter filters ate detail)"},
    ]
    def flags_for(c: dict) -> list[str]:
        return value_filter_flags(c["opacity"], c["scale"])
    clean_flags = flags_for(candidates[0])
    chosen = candidates[0]
    if clean and probes:  # probe the value filters before using them on the real pass
        probe_v = ctx.out / "_clean_probe.ply"
        try:
            for c in candidates:
                run([*st, "-w", str(ply), "-N", *flags_for(c), str(probe_v)], ctx, "export")
                kept_v = ply_count(probe_v)
                chosen = c
                if not total or kept_v >= 0.85 * total:
                    clean_flags = flags_for(c)
                    break
                ctx.log.emit("export", "warn", f"{c['why']} kept only {kept_v}/{total} splats: trying a looser filter")
            else:
                clean_flags = flags_for(chosen)
        except Exception as e:
            ctx.log.emit("export", "warn", f"value-filter probe failed ({str(e)[-120:]})")
        finally:
            probe_v.unlink(missing_ok=True)
    # a crop box taken from the solve's bounds can be wrong when the floor fit was poor (a drone walk with little
    # floor): it then deletes most of the twin. Measure, and drop the crop if it takes more than half.
    drop_crop = False
    if box and probes:
        probe = ctx.out / "_crop_probe.ply"
        try:
            run([*st, "-w", str(ply), "-N", "-B", box, str(probe)], ctx, "export")
            kept = ply_count(probe)
            if total and kept < 0.7 * total:
                drop_crop = True
                ctx.log.emit("export", "warn", f"crop box kept only {kept}/{total} splats: exporting without it")
                base = [b for i, b in enumerate(base) if not (b == "-B" or (i and base[i - 1] == "-B"))]
                box = None
        except Exception as e:
            ctx.log.emit("export", "warn", f"crop probe failed ({str(e)[-120:]}); using the box as it is")
        finally:
            probe.unlink(missing_ok=True)
    full = ctx.out / "scene.spz"
    lite = ctx.out / "scene_lite.spz"
    cleaned = ctx.out / "scene_clean.ply"
    no_box = [b for i, b in enumerate(base) if not (b == "-B" or (i and base[i - 1] == "-B"))]
    with_box_cmd = base

    def build(with_box: bool, with_floaters: bool) -> dict | None:
        """One cleaned PLY. The floater pass is Python, not WebGPU: a container without an adapter used to skip
        it silently, and the blobs shipped with the twin. Returns the floater stats (None when not run)."""
        run((with_box_cmd if with_box else no_box) + (clean_flags if clean else []) + [str(cleaned)], ctx, "export")
        if not (clean and with_floaters):
            return None
        scanned = cleaned.with_name("scene_filtered.ply")
        # the opacity floor is the one the probe chose: a stricter second cut here would undo the probe
        stats = clean_mod.floater_filter(cleaned, scanned, min_opacity=chosen["opacity"])
        if stats["keptFraction"] and stats["keptFraction"] >= 0.5:
            scanned.replace(cleaned)
            return stats
        scanned.unlink(missing_ok=True)
        return {**stats, "skipped": True, "why": f"kept only {stats['keptFraction']}"}

    # with a GPU, every cleanup step is measured the way the viewer shows the twin (viewer_eval: the held-out photos,
    # no appearance model) against the trained model itself, and a step that opens holes or costs detail is dropped.
    # Splat counts were the wrong measure: a crop that removed 6 % of Dr Johnson's splats hid 43 % of every view.
    holdout = int(ctx.opt("holdout", 30) or 0)
    aa = not ctx.opt("no_antialias")
    choice = None
    viewer = None
    try:
        from plypipe import viewer_eval
        reference = viewer_eval.evaluate(ply, ctx.data_dir, holdout, aa) if clean else None
    except Exception as e:
        reference = None
        ctx.log.emit("export", "warn", f"viewer evaluation unavailable: {str(e)[-160:]}")
    # with a GPU the crop keeps whatever the training views draw outside the box (a vault, a window's view, floor
    # reflections) and drops only what they never see; an aligned-on-export PLY is in another frame, so it keeps the box
    room_crop = {"method": "box"} if box else None
    needed_src = ctx.out / "_room_crop.ply"
    if box and reference and not ctx.opt("apply_align"):
        try:
            weights = viewer_eval.training_weights(ply, ctx.data_dir, holdout, antialiased=aa)
            if weights is not None:
                room_crop = needed_crop(ply, needed_src, *room_bounds(scene), weights)
                with_box_cmd = [str(needed_src) if a == str(ply) else a for a in no_box]
                ctx.log.emit("export", "progress", f"room crop: {room_crop['dropped']} of {room_crop['outside']} splats "
                             f"outside the room are unseen by the training views; {room_crop['keptOutside']} kept")
        except Exception as e:
            room_crop = {"method": "box", "why": f"weights failed: {str(e)[-160:]}"}
            ctx.log.emit("export", "warn", f"room crop falls back to the plain box: {str(e)[-160:]}")
    clean_stats = None
    if reference:
        tried = []
        steps = [(True, True), (False, True), (False, False)] if box else [(False, True), (False, False)]
        for with_box, with_floaters in steps:
            stats = build(with_box, with_floaters)
            score = viewer_eval.evaluate(cleaned, ctx.data_dir, holdout, aa)
            name = "+".join(n for n, on in (("crop", with_box), ("floaters", with_floaters)) if on) or "values only"
            tried.append({"cleanup": name, **(score or {})})
            ok = score and score["psnr"] >= reference["psnr"] - 0.3 and score["holes"] <= reference["holes"] + 0.01
            if ok or (with_box, with_floaters) == steps[-1]:
                choice = {"cleanup": name, "reference": reference, "tried": tried}
                viewer, clean_stats = score, stats
                drop_crop = drop_crop or (box is not None and not with_box)
                box = box if with_box else None
                break
            ctx.log.emit("export", "warn", f"cleanup '{name}' cost the viewer {reference['psnr'] - score['psnr']:.2f} dB and "
                         f"{(score['holes'] - reference['holes']):.1%} more see-through pixels: trying a lighter one")
        if clean_stats is None:  # the floater pass was measured and left out: record why, so the gate can tell
            clean_stats = {"method": "cpu-grid-27", "skipped": True, "why": "measured: it opened holes in the twin",
                           "kept": ply_count(cleaned)}
    else:
        clean_stats = build(box is not None, True) if clean else build(box is not None, False)
    floaters = bool(clean_stats and not clean_stats.get("skipped"))
    if floaters:
        ctx.log.emit("export", "progress", f"floaters: removed {clean_stats['removedIsolated']} isolated and "
                     f"{clean_stats['removedFaint']} faint splats, kept {clean_stats['kept']}")
    clipped = clean_mod.clipped_fraction(cleaned)
    if viewer is None:
        try:  # what the viewer will show, scored on the held-out photos (GPU only; None elsewhere)
            from plypipe import viewer_eval
            viewer = viewer_eval.evaluate(cleaned, ctx.data_dir, holdout, aa)
        except Exception as e:
            ctx.log.emit("export", "warn", f"viewer evaluation skipped: {str(e)[-160:]}")
    if viewer:
        ctx.log.emit("export", "progress", f"as the viewer shows it: PSNR {viewer['psnr']:.2f}, brightness ×{viewer['brightness']:.2f}, "
                     f"holes {viewer['holes']:.1%} over {viewer['views']} held-out views")
    src = [*st, "-w", "--spz-version", "3", str(cleaned)]
    run(src + [str(full)], ctx, "export")
    sh = {"full": _restore_sh(ctx, full, cleaned)}
    n = ply_count(cleaned)
    if n > LITE_MAX:  # decimation must end in a .ply: decimate, then convert
        lite_ply = ctx.out / "scene_lite.ply"
        run([*st, "-w", str(cleaned), "-H", str(LITE_SH), "-d", str(LITE_MAX), str(lite_ply)], ctx, "export")
        run([*st, "-w", "--spz-version", "3", str(lite_ply), str(lite)], ctx, "export")
        sh["lite"] = _restore_sh(ctx, lite, lite_ply)
        lite_ply.unlink(missing_ok=True)
    else:
        run(src + ["-H", str(LITE_SH), str(lite)], ctx, "export")
        sh["lite"] = _restore_sh(ctx, lite, cleaned)
    shipped = _score_shipped(ctx, st, {"full": full, "lite": lite}, holdout, aa) if viewer else None
    needed_src.unlink(missing_ok=True)
    update_scene(ctx, {"splat": {"file": "scene.spz", "lite": "scene_lite.spz", "format": "spz"}})
    top = _topview(ctx, cleaned if cleaned.exists() else full)
    return {"spz": full.name, "spzMB": round(full.stat().st_size / 2**20, 2), "lite": lite.name, "liteMB": round(lite.stat().st_size / 2**20, 2),
            "cropBox": box, "cropDropped": drop_crop, "gaussiansIn": total, "gaussiansOut": n, "floaterFilter": floaters, "liteGaussians": min(n, LITE_MAX),
            "keptFraction": round(n / total, 3) if total else None,
            "exportFilters": {"opacity": chosen["opacity"], "scaleCap": chosen["scale"], "why": chosen["why"]},
            "floaterClean": clean_stats, "clippedFraction": clipped, **({"viewerEval": viewer} if viewer else {}),
            **({"roomCrop": room_crop} if room_crop else {}),
            "shRestore": sh, **({"shippedEval": shipped} if shipped else {}),
            **({"cleanupChoice": choice} if choice else {}), **({"topview": top} if top else {})}


def _topview(ctx: Ctx, source: Path) -> dict | None:
    """The twin seen from above, next to scene.spz, for the viewer's minimap. The export is good without it."""
    try:
        top = topview_mod.build(source, ctx.out, read_scene(ctx))
        update_scene(ctx, {"topview": top})
        ctx.log.emit("export", "progress", f"top view: {top['width']}x{top['height']} px at {top['metersPerPixel'] * 100:g} cm, "
                     f"cut at {top['cutY']} m")
        return top
    except Exception as e:
        ctx.log.emit("export", "warn", f"top view skipped: {str(e)[-160:]}")
        return None


def _restore_sh(ctx: Ctx, spz: Path, ply: Path) -> dict:
    try:
        return spz_mod.restore_sh(spz, ply, bits=SH_BITS)
    except Exception as e:  # the file splat-transform wrote is still a valid twin
        ctx.log.emit("export", "warn", f"SH restore skipped for {spz.name}: {str(e)[-160:]}")
        return {"restored": False, "why": str(e)[-160:]}


def _score_shipped(ctx: Ctx, st: list[str], files: dict[str, Path], holdout: int, aa: bool) -> dict | None:
    """viewer_eval of the SPZ files themselves (decoded back to PLY): what compression costs, measured."""
    from plypipe import viewer_eval
    out = {}
    for name, spz in files.items():
        dec = ctx.out / f"_{name}_decoded.ply"
        try:
            run([*st, "-w", str(spz), str(dec)], ctx, "export")
            out[name] = viewer_eval.evaluate(dec, ctx.data_dir, holdout, aa)
        except Exception as e:
            ctx.log.emit("export", "warn", f"shipped-file evaluation skipped for {spz.name}: {str(e)[-160:]}")
        finally:
            dec.unlink(missing_ok=True)
    if out.get("full"):
        ctx.log.emit("export", "progress", f"as shipped: PSNR {out['full']['psnr']:.2f} (scene.spz)"
                     + (f", {out['lite']['psnr']:.2f} (phone file)" if out.get("lite") else ""))
    return out or None
