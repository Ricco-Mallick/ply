# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Pure-numpy geometry used by the align and occupancy stages.

Conventions
-----------
* COLMAP: quaternions are (qw, qx, qy, qz) and describe cam_from_world; camera axes are
  x right, y down, z forward.
* Scene frame (what the viewer reads): right-handed, +Y up, floor at y = 0, metres.
* A similarity transform is stored as a 4x4 matrix T with T[:3,:3] = s * R, so
  p_scene = T @ [p_world, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- rotations

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(qw, qx, qy, qz) -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (qw, qx, qy, qz) with qw >= 0."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b (Rodrigues)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-9:  # opposite: rotate 180 deg about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ---------------------------------------------------------------- cameras

def camera_centers_and_ups(qvecs: np.ndarray, tvecs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From COLMAP cam_from_world poses return world camera centres, world 'up' (-y_cam)
    and world viewing directions (+z_cam)."""
    centers, ups, fwds = [], [], []
    for q, t in zip(qvecs, tvecs):
        R = quat_to_mat(q)
        centers.append(-R.T @ t)
        ups.append(R.T @ np.array([0.0, -1.0, 0.0]))
        fwds.append(R.T @ np.array([0.0, 0.0, 1.0]))
    return np.array(centers), np.array(ups), np.array(fwds)


def gravity_guess(ups: np.ndarray, fwds: np.ndarray) -> tuple[np.ndarray, str]:
    """World up from the cameras: the direction every camera's right axis is perpendicular to.

    A walk keeps the camera level (little roll) but pitches it freely, and pitch correlates with where the room is
    worth looking: on the chapel most upward shots faced the altar, which tilted the mean of the cameras' up vectors
    4.8 deg and sloped the floor 0.86 m over 16 m. The right axes stay horizontal whatever the pitch, so the
    least-spread direction of their scatter is gravity. That needs the camera to have turned (right axes spanning
    the horizontal plane); otherwise, or if the answer disagrees wildly with the mean up (a sideways video), the mean
    up vector is returned as before."""
    mean = ups.mean(axis=0)
    m = mean / (np.linalg.norm(mean) or 1.0)
    rights = np.cross(-ups, fwds)  # x_cam = y_cam x z_cam, with y_cam = -up
    w, v = np.linalg.eigh(rights.T @ rights / max(len(rights), 1))
    g = v[:, 0] * (1.0 if v[:, 0] @ m >= 0 else -1.0)
    if w[1] < 0.1 or w[0] > 0.05 or np.degrees(np.arccos(np.clip(g @ m, -1, 1))) > 25.0:
        return m, "mean-up"
    return g, "camera-level"


def transform_pose(q: np.ndarray, t: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-express a cam_from_world pose after the world is mapped by p' = s R p + d.

    x_cam' = s * x_cam keeps every projection identical, giving
    R_c' = R_c R^T and t_c' = s t_c - R_c' d.
    """
    sR = T[:3, :3]
    s = np.cbrt(np.linalg.det(sR))
    R = sR / s
    d = T[:3, 3]
    Rc = quat_to_mat(q)
    Rc2 = Rc @ R.T
    t2 = s * np.asarray(t, float) - Rc2 @ d
    return mat_to_quat(Rc2), t2


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    return pts @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------- floor plane

@dataclass
class Plane:
    normal: np.ndarray  # unit, oriented towards the cameras (i.e. "up")
    offset: float       # plane: normal . p = offset
    inliers: np.ndarray  # boolean mask over the input points
    tilt_deg: float     # angle between normal and the gravity guess


def fit_plane_lsq(pts: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n, float(n @ c)


def ransac_floor(points: np.ndarray, cam_centers: np.ndarray, up_guess: np.ndarray,
                 max_tilt_deg: float = 25.0, iters: int = 2000, thresh: float | None = None,
                 seed: int = 0) -> Plane:
    """Find the floor: the best-supported plane that is near-perpendicular to the gravity
    guess and lies below the cameras.

    thresh defaults to 1.5% of the camera-to-points vertical extent, so it is scale free.
    """
    rng = np.random.default_rng(seed)
    g = up_guess / np.linalg.norm(up_guess)
    h_pts = points @ g
    h_cam = cam_centers @ g
    cam_level = float(np.median(h_cam))
    below = np.flatnonzero(h_pts < cam_level)
    if below.size < 3:
        raise ValueError("no points below the cameras: cannot find a floor")
    extent = cam_level - float(np.percentile(h_pts[below], 2))
    if thresh is None:
        thresh = max(extent * 0.015, 1e-6)
    cos_max = np.cos(np.radians(max_tilt_deg))
    cand = points[below]
    if cand.shape[0] > 20000:  # score on a sample; the final fit uses every point
        cand = cand[rng.choice(cand.shape[0], 20000, replace=False)]
    # in-plane axes for measuring how much AREA a plane's inliers cover
    e1 = np.cross(g, [1.0, 0, 0] if abs(g[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(g, e1)
    uv = np.c_[cand @ e1, cand @ e2]
    span = np.percentile(uv, 98, axis=0) - np.percentile(uv, 2, axis=0)
    cell = max(float(span.max()) / 60.0, 1e-9)
    planes = []  # (coverage, count, height along g, n, d)
    for _ in range(iters):
        idx = rng.choice(cand.shape[0], 3, replace=False)
        p0, p1, p2 = cand[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if n @ g < 0:
            n = -n
        if n @ g < cos_max:
            continue
        d = float(n @ p0)
        inl = np.abs(cand @ n - d) < thresh
        count = int(inl.sum())
        if count < 3:
            continue
        cov = np.unique(np.floor(uv[inl] / cell).astype(np.int64), axis=0).shape[0]
        planes.append((cov, count, d / float(n @ g), n, d))
    if not planes:
        raise ValueError("RANSAC found no near-horizontal plane")
    # The floor is the most EXTENDED horizontal surface, not the one with the most points: a
    # textured table top can carry more sparse points than a plain floor. Among planes covering
    # at least half the best area, take the lowest.
    top = max(p[0] for p in planes)
    strong = [p for p in planes if p[0] >= 0.5 * top]
    low = min(p[2] for p in strong)
    # within that lowest band, the best-supported hypothesis (not the lowest, which is biased
    # towards planes tilted through the noise)
    _, _, _, n, d = max((p for p in strong if p[2] <= low + 3 * thresh), key=lambda p: (p[1], p[0]))
    # refine: least squares on inliers, repeated so the band re-centres on the fitted plane
    n2, d2 = n, d
    for _ in range(3):
        mask = np.abs(points @ n2 - d2) < thresh
        n2, d2 = fit_plane_lsq(points[mask])
        if n2 @ g < 0:
            n2, d2 = -n2, -d2
    mask = np.abs(points @ n2 - d2) < thresh
    # orient towards the cameras
    if np.median(cam_centers @ n2) < d2:
        n2, d2 = -n2, -d2
    tilt = float(np.degrees(np.arccos(np.clip(n2 @ g, -1, 1))))
    return Plane(normal=n2, offset=float(d2), inliers=mask, tilt_deg=tilt)


# ---------------------------------------------------------------- alignment

def dominant_yaw(xz: np.ndarray, step_deg: float = 0.25) -> float:
    """Angle (radians, in [0, pi/2)) of the room's Manhattan axes in the horizontal plane.

    Walls are lines, so when the points are rotated onto the right axes their x and z
    histograms collapse into sharp spikes. Sweep the angle and keep the one with the highest
    histogram energy (sum of squared bin counts), then refine around it.
    """
    xz = np.asarray(xz, float)
    if xz.shape[0] > 20000:
        xz = xz[np.random.default_rng(0).choice(xz.shape[0], 20000, replace=False)]
    xz = xz - np.median(xz, axis=0)
    span = float(np.percentile(np.abs(xz), 98)) * 2 or 1.0
    bins = 200
    edges = np.linspace(-span, span, bins + 1)

    def energy(theta: float) -> float:
        c, s = np.cos(theta), np.sin(theta)
        u = xz[:, 0] * c + xz[:, 1] * s
        v = -xz[:, 0] * s + xz[:, 1] * c
        hu, _ = np.histogram(u, edges)
        hv, _ = np.histogram(v, edges)
        return float((hu.astype(np.float64) ** 2).sum() + (hv.astype(np.float64) ** 2).sum())

    coarse = np.radians(np.arange(0.0, 90.0, 1.0))
    best = coarse[int(np.argmax([energy(t) for t in coarse]))]
    fine = best + np.radians(np.arange(-1.0, 1.0 + 1e-9, step_deg))
    best = fine[int(np.argmax([energy(t) for t in fine]))]
    return float(best % (np.pi / 2))


def similarity_to_scene(plane: Plane, points: np.ndarray, cam_centers: np.ndarray,
                        cam_height_m: float | None = 1.45, yaw_align: bool = True) -> tuple[np.ndarray, dict]:
    """Build T (world -> scene): floor normal -> +Y, floor at y=0, walls along X/Z,
    origin under the median camera position, metric scale from the camera-height prior.
    """
    n = plane.normal
    R1 = rotation_between(n, np.array([0.0, 1.0, 0.0]))
    # floor point: the projection of the camera median onto the plane
    c_med = np.median(cam_centers, axis=0)
    origin = c_med - (c_med @ n - plane.offset) * n
    p = (points - origin) @ R1.T
    c = (cam_centers - origin) @ R1.T
    yaw = 0.0
    if yaw_align:
        band = p[(p[:, 1] > 0.15 * max(np.median(c[:, 1]), 1e-9)) & (p[:, 1] < 1.2 * max(np.median(c[:, 1]), 1e-9))]
        if band.shape[0] >= 10:
            yaw = dominant_yaw(band[:, [0, 2]])
    # rotate about +Y so the dominant axis lands on +X (in xz: angle a -> 0)
    R2 = rot_y(yaw)  # maps the horizontal direction at angle `yaw` (atan2(z, x)) onto +X
    R = R2 @ R1
    heights = (cam_centers - origin) @ n
    med_h = float(np.median(heights))
    if cam_height_m and med_h > 0:
        s = cam_height_m / med_h
        source = "camera-height-prior"
    else:
        s = 1.0
        source = "none"
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = -s * R @ origin
    info = {"scale": s, "scaleSource": source, "cameraHeightModel": med_h, "yawDeg": float(np.degrees(yaw)),
            "floorTiltDeg": plane.tilt_deg}
    return T, info


def robust_bounds(pts: np.ndarray, lo: float = 2.0, hi: float = 98.0, margin: float = 0.25) -> dict:
    mn = np.percentile(pts, lo, axis=0) - margin
    mx = np.percentile(pts, hi, axis=0) + margin
    return {"min": [float(v) for v in mn], "max": [float(v) for v in mx]}


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising |s R src + t - dst|^2 (Umeyama 1991). src, dst: (n, 3), n >= 3."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if var > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


# ---------------------------------------------------------------- cloud against a reference (docs/IMPROVEMENT-PLAN.md section 5)

def nn_distance(points: np.ndarray, ref: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Distance from each of `points` to its nearest point of `ref`. scipy's k-d tree when the image has scipy (the worker's does not
    list it), else exact brute force through torch in chunks."""
    try:
        from scipy.spatial import cKDTree
        return cKDTree(ref).query(points, workers=-1)[0]
    except ImportError:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        r = torch.as_tensor(np.ascontiguousarray(ref, dtype=np.float32), device=dev)
        out = [torch.cdist(torch.as_tensor(np.ascontiguousarray(points[i:i + chunk], dtype=np.float32), device=dev), r).min(1).values.cpu().numpy()
               for i in range(0, len(points), chunk)]
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def cloud_scores(pred: np.ndarray, ref: np.ndarray, taus: tuple[float, ...] = (0.05, 0.10), cap: float = 1.0) -> dict:
    """Precision, recall and F-score at each tau (metres) of `pred` against `ref`, plus accuracy (pred to ref), completeness (ref to pred)
    and their mean as the chamfer distance, distances capped at `cap` so a stray splat cannot outweigh a room. Both clouds in one frame."""
    a, c = np.minimum(nn_distance(pred, ref), cap), np.minimum(nn_distance(ref, pred), cap)
    out = {"accuracyM": round(float(a.mean()), 4), "completenessM": round(float(c.mean()), 4), "chamferM": round(float((a.mean() + c.mean()) / 2), 4)}
    for tau in taus:
        p, r = float((a < tau).mean()), float((c < tau).mean())
        out[f"f{round(tau * 100)}cm"] = {"precision": round(p, 4), "recall": round(r, 4), "f": round(2 * p * r / (p + r), 4) if p + r else 0.0}
    return out
