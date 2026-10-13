"""align: put the reconstruction in the scene frame the viewer expects.

    right-handed, +Y up, floor plane at y = 0, metres, walls along X / Z,
    origin on the floor under the median camera position.

1. gravity guess = mean of the cameras' up vectors (a phone is held roughly upright);
2. floor = lowest well-supported RANSAC plane within 25 deg of that guess, below the cameras;
3. yaw = dominant (Manhattan) direction of wall/furniture points -> +X;
4. scale = camera-height prior (median camera height above the floor = --height-cam metres).

AR scans (frames.py wrote ar_poses.json) replace 1, 2 and 4: a similarity fit of the SfM camera centres onto the phone's
tracked positions gives metric scale and gravity directly (ARCore's 'local-floor' space is +y up with the floor at 0);
the floor height is then refined on the points, and the yaw / origin steps run as usual.

The aligned model is written as text to data/sparse/0 next to data/images (the undistorted
frames), which is exactly gsplat's data_dir layout, so training happens in the scene frame
and the trained PLY needs no further transform.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from plypipe import colmap_txt, geometry, tour
from plypipe.util import Ctx, update_scene


def run_stage(ctx: Ctx) -> dict:
    src = ctx.dense_dir / "sparse_txt"
    model = colmap_txt.read_model(src)
    if model.num_points < 100:
        raise RuntimeError(f"only {model.num_points} sparse points; cannot find a floor")
    centers, ups, fwds = model.cameras()
    good = (model.track_len >= 3) & (model.error < float(ctx.opt("max_reproj_px", 2.0)))
    pts = model.xyz[good] if good.sum() >= 200 else model.xyz
    up_guess = ups.mean(axis=0)
    if np.linalg.norm(up_guess) < 0.3:
        ctx.log.emit("align", "warn", "camera up vectors disagree (rotated video?); floor guess may be poor")
    height_cam = ctx.opt("height_cam", 1.45)
    ar = _ar_transform(ctx, model, centers, pts)
    if ar:
        T, info, plane = ar
    else:
        plane = geometry.ransac_floor(pts, centers, up_guess, max_tilt_deg=float(ctx.opt("max_tilt_deg", 25.0)))
        T, info = geometry.similarity_to_scene(plane, pts, centers, cam_height_m=float(height_cam) if height_cam else None)

    aligned = colmap_txt.transformed(model, T)
    out_sparse = ctx.data_dir / "sparse" / "0"
    if out_sparse.exists():
        shutil.rmtree(out_sparse)
    colmap_txt.write_model(aligned, out_sparse)
    # binary too: gsplat <= 1.5.x reads models via rmbrualla's pycolmap, whose text reader is broken on Python 3
    colmap_txt.write_model_bin(aligned, out_sparse)
    img_link = ctx.data_dir / "images"
    if img_link.is_symlink() or img_link.exists():
        if img_link.is_symlink() or img_link.is_file():
            img_link.unlink()
        else:
            shutil.rmtree(img_link)
    try:
        os.symlink(os.path.relpath(ctx.dense_dir / "images", ctx.data_dir), img_link)
    except OSError:  # filesystems without symlinks (some Windows / FUSE mounts)
        shutil.copytree(ctx.dense_dir / "images", img_link)

    c2, _, f2 = aligned.cameras()
    p2 = aligned.xyz[good] if good.sum() >= 200 else aligned.xyz
    floor_res = np.abs(p2[np.abs(p2[:, 1]) < 0.15][:, 1])
    step = max(1, len(c2) // 200)
    path = [[round(float(v), 3) for v in c] for c in c2[::step]]
    # spawn: first camera, eye height 1.6 m, looking along its horizontal forward direction
    fwd = f2[0].copy()
    fwd[1] = 0
    fwd = fwd / (np.linalg.norm(fwd) or 1)
    spawn_pos = [float(c2[0][0]), 1.6, float(c2[0][2])]
    spawn = {"position": spawn_pos, "lookAt": [spawn_pos[0] + float(fwd[0]), 1.6, spawn_pos[2] + float(fwd[2])]}
    scene = update_scene(ctx, {
        "version": 1,
        "units": "meters",
        "up": [0, 1, 0],
        "handedness": "right",
        "floorY": 0.0,
        "scale": info["scale"],
        "scaleSource": info["scaleSource"],
        "cameraHeight": float(height_cam) if height_cam else None,
        "transform": T.tolist(),
        "bounds": geometry.robust_bounds(p2),
        "cameraPath": path,
        "spawn": spawn,
        "tour": tour.build_tour(c2, geometry.robust_bounds(p2)),
    })
    return {"floorInliers": int(plane.inliers.sum()), "floorInlierRatio": float(plane.inliers.mean()),
            "floorTiltDeg": info["floorTiltDeg"], "yawDeg": info["yawDeg"], "scale": info["scale"],
            "scaleSource": info["scaleSource"], "cameraHeightModelUnits": info["cameraHeightModel"],
            "floorResidualMedianM": float(np.median(floor_res)) if floor_res.size else None,
            "bounds": scene["bounds"], **({"ar": info["ar"]} if "ar" in info else {})}


def _ar_transform(ctx: Ctx, model, centers: np.ndarray, pts: np.ndarray):
    """World -> scene from the phone's tracked poses, or None (no AR scan, too few matches, or a poor fit)."""
    f = ctx.out / "ar_poses.json"
    if not f.exists():
        return None
    names = [im.name for im in sorted(model.images, key=lambda im: im.name)]
    pos = {n: i for i, n in enumerate(names)}
    best = None
    for seg in json.loads(f.read_text())["segments"]:  # separate AR sessions have separate origins: use the best one
        pairs = [(pos[fr["frame"]], fr["m"][12:15]) for fr in seg["frames"] if fr["frame"] in pos]
        if len(pairs) >= 6 and (best is None or len(pairs) > len(best)):
            best = pairs
    if not best:
        ctx.log.emit("align", "warn", "AR poses present but too few frames registered; using the camera-height prior")
        return None
    src = centers[[i for i, _ in best]]
    dst = np.array([p for _, p in best], float)
    s, R, t = geometry.umeyama(src, dst)
    res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    if s <= 0 or float(np.median(res)) > float(ctx.opt("ar_max_residual_m", 0.25)):
        ctx.log.emit("align", "warn", f"AR fit too loose (median {np.median(res):.2f} m over {len(best)} frames); using the camera-height prior")
        return None
    T1 = np.eye(4)
    T1[:3, :3] = s * R
    T1[:3, 3] = t
    p1 = (pts @ T1[:3, :3].T) + T1[:3, 3]
    c1 = (centers @ T1[:3, :3].T) + T1[:3, 3]
    # gravity is exact from ARCore; refine the floor height on the points (ARCore's floor guess can be a few cm off)
    plane = geometry.ransac_floor(p1, c1, np.array([0.0, 1.0, 0.0]), max_tilt_deg=6.0)
    T2, info = geometry.similarity_to_scene(plane, p1, c1, cam_height_m=None)
    T = T2 @ T1
    tilt = float(np.degrees(np.arccos(np.clip(plane.normal @ np.array([0.0, 1.0, 0.0]), -1, 1))))
    info.update({"scale": s * info["scale"], "scaleSource": "ar-poses", "floorTiltDeg": tilt,
                 "ar": {"frames": len(best), "residualMedianM": float(np.median(res)), "residualP90M": float(np.percentile(res, 90)), "scale": s}})
    ctx.log.emit("align", "progress", f"AR poses: {len(best)} frames, median fit {np.median(res) * 100:.1f} cm")
    return T, info, plane
