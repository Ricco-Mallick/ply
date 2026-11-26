# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Phone video plus a laser scan of the same space (IMPROVEMENT-PLAN phase 6, route 8c): the phone's pictures on the
scanner's geometry.

    python -m plypipe run <videos> --out work/<job> --scan hf:lidar/office-1/points-1cm.npz

The align stage levels the video's camera solve as always (floor at y = 0, +Y up, scale from the camera height). The
scan goes into the same kind of frame through plypipe.lidar_splat.align, so it lands exactly where the scan-only twin
(lidar-office-8b) is. Then, in `align`:

1. register: what is left between the two is a turn about the vertical, a scale and a floor offset. The turn and the
   scale are searched (4 deg, 3 %), the offset comes from correlating the two top-down density images (points between
   knee and head height: walls and furniture, not the floor), and the best candidates are refined by trimmed ICP
   against the scan, video points outside the scanned area ignored (the videos may cover rooms the scanner never saw).
2. gate: the fit is used only if the video points inside the scanned area sit on the scan (median distance and share
   within 5 cm) and the scale stayed sensible. Otherwise the run says why and goes on video-only: a misregistered depth
   loss is worse than none.
3. depth: the scan, z-buffered into every registered view at a quarter of its size, is that view's depth for the depth
   loss (scandepth/ + scandepth.json, read by plypipe.lidar_depth). Pixels well behind their neighbourhood are scan
   points seen through gaps and are dropped; a view whose own camera-solve points disagree with the scan by more than
   5 % (furniture moved since the scan, a person) gets no scan depth at all.
4. start: the scan's points (scan/init.npz, scene frame) join the camera-solve points as the gaussians training starts
   from (train_launcher --init-points), so walls the phone saw only as blank paint start on the measured surface.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

VOXEL = 0.02            # scan points kept for registration, depth and the training start
BAND = (0.25, 2.2)      # heights (m) whose top-down density is correlated: furniture and walls, not floor or ceiling
CELL = 0.2              # top-down raster cell (m): a coarse search, ICP does the fine work
SCALES = (0.7, 1.4, 0.03)
YAW_STEP = 4.0
GATE_MEDIAN_M = 0.05
GATE_SHARE = 0.5        # inside-the-scan video points within 5 cm of it
DEPTH_DOWN = 6          # depth maps at a sixth of the view (they ride in the checkpoint: ~1300 views)
AGREE = 0.05            # a view's own track points vs the scan depth, median relative difference
RANGE_M = (0.2, 15.0)


# ---- the scan in the scene frame ----------------------------------------------------------------------------------

def resolve(spec: str, out: Path) -> Path:
    """A local path, or 'hf:<key>' fetched from the results dataset (PLY_HF_REPO, default Ricco-Marcelino/ply-data; a
    read-only HF_TOKEN is enough) into <out>/scan/."""
    if spec.startswith("hf:"):
        import os
        from huggingface_hub import hf_hub_download
        key = spec[3:]
        dest = out / "scan" / "source" / key
        if not dest.is_file():
            hf_hub_download(repo_id=os.environ.get("PLY_HF_REPO") or "Ricco-Marcelino/ply-data", repo_type="dataset", filename=key,
                            token=os.environ.get("HF_TOKEN"), local_dir=str(out / "scan" / "source"))
        return dest
    return Path(spec)


def scene_scan(spec: str, out: Path, log=None) -> dict:
    """The scan in its scene frame (the frame lidar_splat gives the scan-only twin), thinned to VOXEL; cached in
    <out>/scan/scene.npz."""
    cache = out / "scan" / "scene.npz"
    if cache.is_file():
        z = np.load(cache)
        return {"points": z["points"], "colors": z["colors"], "info": json.loads(str(z["info"]))}
    from plypipe import geometry, lidar_splat
    src = lidar_splat.load_points(resolve(spec, out))
    T, info = lidar_splat.align(src["points"], src["colors"], src["stations"])
    pts = geometry.apply(T, src["points"].astype(np.float64)).astype(np.float32)
    pts, cols = lidar_splat.voxel_mean(pts, src["colors"], VOXEL)
    cache.parent.mkdir(parents=True, exist_ok=True)
    info = {"source": spec, "points": int(len(pts)), "voxelM": VOXEL, "transform": T.tolist(),
            "floorInlierShare": info.get("floorInlierShare")}
    np.savez(cache, points=pts, colors=cols, info=json.dumps(info))
    import shutil
    shutil.rmtree(out / "scan" / "source", ignore_errors=True)  # the download is not needed again and would ride in every checkpoint
    if log:
        log(f"scan: {len(pts):,} points at {VOXEL * 100:.0f} cm in the scene frame")
    return {"points": pts, "colors": cols, "info": info}


# ---- registration ------------------------------------------------------------------------------------------------

def _yaw(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _similarity(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def _raster(xz: np.ndarray, lo: np.ndarray, n: int) -> np.ndarray:
    ij = np.floor((xz - lo) / CELL).astype(np.int64)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    img = np.bincount(ij[ok, 1] * n + ij[ok, 0], minlength=n * n).reshape(n, n).astype(np.float64)
    return np.log1p(img)


def coarse(video: np.ndarray, scan: np.ndarray, scales=SCALES, yaw_step: float = YAW_STEP, keep: int = 4) -> list[dict]:
    """Candidates {s, yaw, t, score}: the best few turn/scale/offset triples by top-down correlation."""
    sb = scan[(scan[:, 1] > BAND[0]) & (scan[:, 1] < BAND[1])]
    span = max(np.ptp(video[:, [0, 2]], axis=0).max() * scales[1], np.ptp(sb[:, [0, 2]], axis=0).max()) + 2.0
    n = int(2 ** math.ceil(math.log2(2 * span / CELL)))
    lo_s = sb[:, [0, 2]].mean(0) - n * CELL / 2
    A = np.fft.rfft2(_raster(sb[:, [0, 2]], lo_s, n))
    out = []
    for s in np.arange(scales[0], scales[1] + 1e-9, scales[2]):
        vb = video[(video[:, 1] * s > BAND[0]) & (video[:, 1] * s < BAND[1])]
        if len(vb) < 50:
            continue
        for yaw in np.arange(0.0, 360.0, yaw_step):
            p = (vb @ _yaw(yaw).T) * s
            c0 = p[:, [0, 2]].mean(0)
            lo_v = c0 - n * CELL / 2
            B = np.fft.rfft2(_raster(p[:, [0, 2]], lo_v, n))
            corr = np.fft.irfft2(A * np.conj(B), s=(n, n))
            k = int(np.argmax(corr))
            dj, di = divmod(k, n)
            dj, di = (dj + n // 2) % n - n // 2, (di + n // 2) % n - n // 2
            # scan pixel (i, j) matches video pixel (i - di, j - dj): shift the video by lo_s - lo_v + (di, dj) cells
            shift = lo_s - lo_v + np.array([di, dj]) * CELL
            out.append({"s": float(s), "yaw": float(yaw), "t": np.array([shift[0], 0.0, shift[1]]), "score": float(corr.flat[k])})
    out.sort(key=lambda c: -c["score"])
    picked: list[dict] = []
    for c in out:  # distinct candidates: not the same turn and scale nudged by one step
        if all(abs((c["yaw"] - p["yaw"] + 180) % 360 - 180) > 10 or abs(c["s"] / p["s"] - 1) > 0.06 for p in picked):
            picked.append(c)
        if len(picked) == keep:
            break
    return picked


def icp(video: np.ndarray, scan: np.ndarray, T0: np.ndarray, iters: int = 40, tree=None) -> tuple[np.ndarray, dict]:
    """Trimmed similarity ICP of the video points onto the scan, from T0. Only video points inside the scan's box
    (plus 30 cm) take part. Returns T and the fit's numbers."""
    from scipy.spatial import cKDTree
    from plypipe import geometry
    tree = tree or cKDTree(scan)
    lo, hi = scan.min(0) - 0.3, scan.max(0) + 0.3
    T = T0.copy()
    for it in range(iters):
        p = geometry.apply(T, video)
        inside = (p >= lo).all(1) & (p <= hi).all(1)
        if inside.sum() < 50:
            break
        d, j = tree.query(p[inside], k=1)
        cut = max(0.04, 0.4 * (1 - it / iters) ** 2)
        m = d < cut
        if m.sum() < 30:
            break
        s, R, t = geometry.umeyama(video[inside][m], scan[j[m]])
        T = _similarity(s, R, t)
    p = geometry.apply(T, video)
    inside = (p >= lo).all(1) & (p <= hi).all(1)
    d = tree.query(p[inside], k=1)[0] if inside.any() else np.array([np.inf])
    s = float(np.cbrt(np.linalg.det(T[:3, :3])))
    R = T[:3, :3] / s
    tilt = float(np.degrees(np.arccos(np.clip(R[1, 1], -1, 1))))
    return T, {"inside": int(inside.sum()), "insideShare": round(float(inside.mean()), 3),
               "medianM": round(float(np.median(d)), 4), "within5cm": round(float((d < 0.05).mean()), 3),
               "scale": round(s, 4), "tiltDeg": round(tilt, 2)}


def register(video: np.ndarray, scan: np.ndarray, log=None) -> tuple[np.ndarray | None, dict]:
    """Video points (scene frame of the video's own align) -> T onto the scan's scene frame, or None and why."""
    from scipy.spatial import cKDTree
    t0 = time.time()
    if len(video) > 200_000:
        video = video[np.random.default_rng(0).choice(len(video), 200_000, replace=False)]
    cands = coarse(video, scan)
    tree = cKDTree(scan)
    tried = []
    for c in cands:
        T0 = _similarity(c["s"], _yaw(c["yaw"]), c["t"])
        T, fit = icp(video, scan, T0, tree=tree)
        tried.append((T, {**fit, "start": {"scale": round(c["s"], 3), "yawDeg": c["yaw"], "score": round(c["score"], 1)}}))
        if log:
            log(f"scan registration: start yaw {c['yaw']:.0f} deg scale {c['s']:.3f} -> median {fit['medianM'] * 100:.1f} cm, "
                f"{fit['within5cm']:.0%} within 5 cm")
    tried.sort(key=lambda r: (-r[1]["within5cm"], r[1]["medianM"]))
    T, fit = tried[0]
    rep = {**fit, "candidates": [r[1] for r in tried], "seconds": round(time.time() - t0, 1)}
    why = []
    if fit["inside"] < 500:
        why.append(f"only {fit['inside']} video points inside the scanned area")
    if fit["medianM"] > GATE_MEDIAN_M:
        why.append(f"median distance {fit['medianM'] * 100:.1f} cm > {GATE_MEDIAN_M * 100:.0f} cm")
    if fit["within5cm"] < GATE_SHARE:
        why.append(f"{fit['within5cm']:.0%} within 5 cm < {GATE_SHARE:.0%}")
    if not (SCALES[0] <= fit["scale"] <= SCALES[1]):
        why.append(f"scale {fit['scale']:.3f} outside {SCALES[0]}-{SCALES[1]}")
    if fit["tiltDeg"] > 3:
        why.append(f"tilted {fit['tiltDeg']:.1f} deg")
    if len(tried) > 1 and tried[1][1]["within5cm"] > 0.9 * fit["within5cm"] and abs(
            (tried[1][1]["start"]["yawDeg"] - fit["start"]["yawDeg"] + 180) % 360 - 180) > 30:
        rep["ambiguous"] = True  # a symmetric room: logged, not refused, when the best fit is clearly tight
        if fit["medianM"] > 0.03:
            why.append("two different turns fit almost equally well")
    if why:
        rep["refused"] = "; ".join(why)
        return None, rep
    return T, rep


# ---- depth maps from the scan ------------------------------------------------------------------------------------

def _qvec_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def pinhole(cameras_lines: list[str]) -> dict[int, tuple[int, int, float, float, float, float]]:
    """camera id -> (width, height, fx, fy, cx, cy) for PINHOLE / SIMPLE_PINHOLE cameras."""
    out = {}
    for ln in cameras_lines:
        if not ln or ln.startswith("#"):
            continue
        p = ln.split()
        w, h, v = int(p[2]), int(p[3]), [float(x) for x in p[4:]]
        if p[1] == "PINHOLE":
            out[int(p[0])] = (w, h, v[0], v[1], v[2], v[3])
        elif p[1] == "SIMPLE_PINHOLE":
            out[int(p[0])] = (w, h, v[0], v[0], v[1], v[2])
    return out


def render_depth(pts: np.ndarray, R: np.ndarray, t: np.ndarray, cam: tuple, down: int = DEPTH_DOWN) -> np.ndarray:
    """Scan depth (metres along the optical axis, 0 = none) for one view, at 1/down of its size."""
    from scipy.ndimage import minimum_filter
    W, H, fx, fy, cx, cy = cam
    w, h = max(1, round(W / down)), max(1, round(H / down))
    kx, ky = w / W, h / H
    pc = pts @ R.T + t
    z = pc[:, 2]
    ok = z > RANGE_M[0]
    pc, z = pc[ok], z[ok]
    u = (fx * pc[:, 0] / z + cx) * kx  # COLMAP pixel coordinates (centre +0.5) scaled; floor -> index
    v = (fy * pc[:, 1] / z + cy) * ky
    ok = (u >= 0) & (v >= 0) & (u < w) & (v < h) & (z < RANGE_M[1])
    idx = np.floor(v[ok]).astype(np.int64) * w + np.floor(u[ok]).astype(np.int64)
    z = z[ok]
    depth = np.full(w * h, np.inf, np.float32)
    if len(idx):
        order = np.lexsort((z, idx))
        i_s = idx[order]
        first = np.r_[True, i_s[1:] != i_s[:-1]]
        depth[i_s[first]] = z[order][first]
    depth = depth.reshape(h, w)
    near = minimum_filter(depth, size=5, mode="nearest")
    depth[~(depth <= near * 1.06 + 0.03)] = np.inf  # seen through a gap between nearer points
    depth[~np.isfinite(depth)] = 0
    return depth


def agreement(depth: np.ndarray, uv: np.ndarray, z: np.ndarray, down_x: float, down_y: float) -> tuple[float | None, int]:
    """Median |scan / track - 1| at the view's own camera-solve points, and how many were compared."""
    h, w = depth.shape
    i = np.floor(uv[:, 0] * down_x).astype(int)
    j = np.floor(uv[:, 1] * down_y).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < w) & (j < h) & (z > RANGE_M[0])
    d = depth[j[ok], i[ok]]
    zz = z[ok]
    m = d > 0
    if m.sum() < 20:
        return None, int(m.sum())
    return float(np.median(np.abs(d[m] / zz[m] - 1))), int(m.sum())


def write_depths(model, pts: np.ndarray, out: Path, log=None) -> dict:
    """scandepth/<frame>.npy (float16 metres) and scandepth.json for every view of the aligned model."""
    cams = pinhole(model.cameras_lines)
    xyz = {int(pid): model.xyz[k] for k, pid in enumerate(model.point_ids)}
    d = out / "scandepth"
    d.mkdir(parents=True, exist_ok=True)
    frames, refused, none_seen = {}, [], 0
    for n, im in enumerate(sorted(model.images, key=lambda im: im.name)):
        cam = cams.get(im.camera_id)
        if cam is None:
            continue
        R, t = _qvec_R(im.qvec), np.asarray(im.tvec, float)
        dep = render_depth(pts, R, t, cam)
        if not (dep > 0).any():
            none_seen += 1
            continue
        a = np.array(im.points2d_line.split(), float).reshape(-1, 3) if im.points2d_line.strip() else np.zeros((0, 3))
        a = a[a[:, 2] >= 0]
        known = [k for k in range(len(a)) if int(a[k, 2]) in xyz]
        if known:
            P = np.array([xyz[int(a[k, 2])] for k in known])
            zt = (P @ R.T + t)[:, 2]
            err, cnt = agreement(dep, a[known, :2], zt, dep.shape[1] / cam[0], dep.shape[0] / cam[1])
            if err is not None and err > AGREE:
                refused.append({"frame": im.name, "medianRel": round(err, 3), "points": cnt})
                continue
        f = f"{Path(im.name).stem}.npy"
        np.save(d / f, dep.astype(np.float16))
        frames[im.name] = {"depth": f}
        if log and n % 100 == 0:
            log(f"scan depth: view {n + 1}/{len(model.images)}", done=n + 1, total=len(model.images))
    (out / "scandepth.json").write_text(json.dumps({"source": "scan", "units": "m", "dir": "scandepth",
                                                    "rangeM": list(RANGE_M), "frames": frames}))
    return {"views": len(frames), "refusedDisagree": len(refused), "outsideScan": none_seen, "refused": refused[:40]}


def write_init(scan: dict, out: Path) -> Path:
    """The scan's points (scene frame) for the training start."""
    f = out / "scan" / "init.npz"
    np.savez(f, points=scan["points"].astype(np.float32), colors=scan["colors"].astype(np.uint8))
    return f


# ---- the align stage's step ----------------------------------------------------------------------------------------

def fuse(ctx, aligned, T: np.ndarray, info: dict, good: np.ndarray) -> tuple[np.ndarray, dict, dict | None]:
    """align's hook: returns T (composed with the registration when it passed), info, and the report (None: no --scan).
    The depth maps and the training start are written after the model is final (fuse_outputs)."""
    spec = ctx.opt("scan")
    if not spec:
        return T, info, None
    emit = lambda msg, **kw: ctx.log.emit("align", "progress", msg, **kw)
    try:
        sc = scene_scan(spec, ctx.out, emit)
    except Exception as e:
        ctx.log.emit("align", "warn", f"scan not used: could not load it ({type(e).__name__}: {str(e)[:200]})")
        return T, info, {"used": False, "refused": f"load: {str(e)[:200]}"}
    video = aligned.xyz[good] if good.sum() >= 200 else aligned.xyz
    Treg, rep = register(np.asarray(video, float), sc["points"].astype(np.float64), emit)
    rep = {"used": Treg is not None, "scan": sc["info"], **rep}
    if Treg is None:
        ctx.log.emit("align", "warn", f"scan not used, training on the video alone: {rep['refused']}")
        return T, info, rep
    s = rep["scale"]
    ctx.log.emit("align", "progress", f"scan registered: median {rep['medianM'] * 100:.1f} cm, {rep['within5cm']:.0%} within 5 cm, "
                 f"scale x{s:.3f} on the camera-height prior")
    info = {**info, "scale": info["scale"] * s, "scaleSource": "lidar-scan"}
    return Treg @ T, info, rep


def fuse_outputs(ctx, model, rep: dict) -> dict:
    """After align wrote the final model: depth maps and the training start (only when the registration passed)."""
    sc = scene_scan(ctx.opt("scan"), ctx.out)
    dep = write_depths(model, sc["points"], ctx.out, lambda msg, **kw: ctx.log.emit("align", "progress", msg, **kw))
    write_init(sc, ctx.out)
    ctx.log.emit("align", "progress", f"scan depth for {dep['views']} views ({dep['refusedDisagree']} refused: the scan disagrees)")
    try:
        overlay(model, sc["points"], ctx.out / "scan-fit.png")
    except Exception as e:
        ctx.log.emit("align", "warn", f"scan overlay skipped: {e}")
    return dep


def overlay(model, scan: np.ndarray, dest: Path, cell: float = 0.03) -> None:
    """Top view: the scan in grey, the video's camera-solve points in orange, cameras in white, for a look before training."""
    import cv2
    v = model.xyz
    allp = np.vstack([scan[:, [0, 2]], v[:, [0, 2]]])
    lo, hi = np.percentile(allp, 0.5, 0) - 0.5, np.percentile(allp, 99.5, 0) + 0.5
    w, h = [int(x) for x in np.ceil((hi - lo) / cell)]
    k = min(1.0, 2000 / max(w, h))
    w, h = max(1, int(w * k)), max(1, int(h * k))
    img = np.zeros((h, w, 3), np.uint8)

    def dots(p, col):
        ij = ((p - lo) / (hi - lo) * [w - 1, h - 1]).astype(int)
        ok = (ij >= 0).all(1) & (ij[:, 0] < w) & (ij[:, 1] < h)
        img[ij[ok, 1], ij[ok, 0]] = col
    band = (scan[:, 1] > BAND[0]) & (scan[:, 1] < BAND[1])
    dots(scan[band][:, [0, 2]], (150, 150, 150))
    dots(v[(v[:, 1] > BAND[0]) & (v[:, 1] < BAND[1])][:, [0, 2]], (0, 140, 255))
    c, _, _ = model.cameras()
    dots(c[:, [0, 2]], (255, 255, 255))
    cv2.imwrite(str(dest), img)
