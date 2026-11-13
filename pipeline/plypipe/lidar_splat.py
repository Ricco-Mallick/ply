# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""lidar_splat: a laser scan as a Gaussian-splat twin, route 8a of IMPROVEMENT-PLAN phase 6. No training, no GPU.

    python -m plypipe lidar <points.npz | points.ply> --out work/<scene>

The input is the scan as one point cloud: scripts/lidar/inspect_e57.py's `points-1cm.npz` (points, colours, the scanner
stations, all scans in one frame) or a coloured PLY (scripts/lidar/ingest_e57.py's, no stations). From it:

1. up: the file's own up axis is found from the surface normals (the axis most surfaces are perpendicular to, with a
   mild prior for E57's +Z) and its sign from the room (the floor side holds the furniture, the ceiling side does not;
   with a single layer, the side the stations are on). `--up +z|-z|+x|...` overrides; the evidence goes in the report.
2. the scene frame of stages/align.py: the floor plane through RANSAC (geometry.ransac_floor, below the stations) at
   y = 0 with +Y up, the walls on the axes (geometry.dominant_yaw), the origin on the floor under the median station.
   Scale stays the scan's own metres: scene.json says scaleSource "lidar".
3. one Gaussian per point after a voxel thinning to `--voxel` (1 cm), capped at plan.py's High splat count: the voxel
   is coarsened until the cloud is under the cap, and the size used is in the report.
4. the disk: a robust plane fit through the 16 nearest neighbours (points off the plane, such as the other side of an
   edge, are down-weighted twice) gives the normal, oriented towards the nearest scanner station (the cloud's centroid
   without stations). The rotation takes the disk's thin axis to the normal. Its two in-plane scales are half the mean
   distance to the 4 nearest neighbours, never more than half a voxel (so no disk is wider than two voxels across at
   2 sigma); the thin axis is a tenth of that. Opacity 0.95. Colour is the point's own, as the trainer stores it:
   SH0 = (rgb / 255 - 0.5) / 0.2821, higher bands zero.
5. the products an export needs, from the points themselves: occupancy.json/.png from stages/occupancy.build_grid with
   the scan's own sight lines (scanner station to floor point, the way the trained model's depth is used) and no depth
   renders; shell.json from shell.build; a camera path that walks the stations on that grid, a spawn at the first.

The twin is then the existing export stage: scene.spz, scene_lite.spz, topview. What a trained job has and a scan does
not (held-out photos, training views for the phone file's fine-tune, depth renders for the walking grid) is switched
off through export's own options (holdout 0, lite_finetune 0, dense_occupancy off) and listed in the report as skipped.

Accuracy: the splat centres are the voxel means of the scan's points, so the scan's 1 cm F-score is a property of the
voxel size; the report measures it (`fScore`) against the input cloud and the ship rule is >= 95 %.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from plypipe import geometry, plan as plan_mod
from plypipe import shell as shell_mod
from plypipe import tour as tour_mod
from plypipe import walkcheck
from plypipe.util import Ctx, mark_done, update_scene

C0 = 0.28209479177387814
VOXEL = 0.01
OPACITY = 0.95
THIN = 0.1            # thickness of a disk as a share of its in-plane scale
NEIGHBOURS = 16
EYE = 1.6             # spawn and path height, as stages/align.py
PRIOR_Z = 1.25        # E57's world frame is +Z up: another axis must score this much more to win
STAGE = "lidar"


def splat_cap() -> int:
    """The High preset's splat count (plan.py keeps it per preset, and a scan is a High job)."""
    return int(plan_mod._PRESET_CAP_MAX["high"])


# ------------------------------------------------------------------ input

def read_points_ply(path: Path) -> dict:
    """x, y, z and red, green, blue of a binary little-endian PLY, as scripts/lidar/ingest_e57.py writes it."""
    with Path(path).open("rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            head += line
        text = head.decode("latin-1").splitlines()
        if "format binary_little_endian 1.0" not in text:
            raise ValueError(f"{path}: only binary little-endian PLY is supported")
        n = next(int(ln.split()[-1]) for ln in text if ln.startswith("element vertex"))
        kinds = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "float32": "<f4", "float64": "<f8", "int": "<i4", "uint": "<u4"}
        props = [ln.split() for ln in text if ln.startswith("property")]
        dt = np.dtype([(p[-1], kinds[p[1]]) for p in props])
        a = np.fromfile(f, dtype=dt, count=n)
    pts = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    origin = np.round(pts.mean(0), 3)
    rgb = np.stack([a[c] for c in ("red", "green", "blue")], 1).astype(np.uint8) if "red" in a.dtype.names else np.full((n, 3), 128, np.uint8)
    return {"points": (pts - origin).astype(np.float32), "colors": rgb, "stations": None, "origin": origin}


def load_points(path: Path) -> dict:
    """points float32 [N, 3], colors uint8 [N, 3], stations float32 [S, 3] or None, origin float64 [3] (the file frame's offset)."""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path)
        st = np.asarray(z["stations"], np.float64) if "stations" in z.files and len(z["stations"]) else None
        return {"points": np.asarray(z["points"], np.float32), "colors": np.asarray(z["colors"], np.uint8), "stations": st,
                "origin": np.asarray(z["origin"], np.float64) if "origin" in z.files else np.zeros(3)}
    if path.suffix == ".ply":
        return read_points_ply(path)
    raise ValueError(f"{path}: expected a .npz (inspect_e57.py) or a .ply")


# ------------------------------------------------------------------ thinning

def _pack(cells: np.ndarray) -> np.ndarray:
    c = cells - cells.min(0)
    if c.max() >= 1 << 21:
        raise ValueError("the scan spans more than two million voxels along one axis: is the voxel size right?")
    return c[:, 0] | (c[:, 1] << 21) | (c[:, 2] << 42)


def voxel_mean(pts: np.ndarray, cols: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """One point per occupied voxel: the mean position and the mean colour of what falls in it."""
    cells = np.floor((pts - pts.min(0)) / voxel).astype(np.int64)
    _, first, inv = np.unique(_pack(cells), return_index=True, return_inverse=True)
    n = np.bincount(inv, minlength=len(first)).astype(np.float64)
    out = np.stack([np.bincount(inv, pts[:, j].astype(np.float64), minlength=len(first)) for j in range(3)], 1) / n[:, None]
    rgb = np.stack([np.bincount(inv, cols[:, j].astype(np.float64), minlength=len(first)) for j in range(3)], 1) / n[:, None]
    return out.astype(np.float32), np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def thin_to_cap(pts: np.ndarray, cols: np.ndarray, voxel: float, cap: int) -> tuple[np.ndarray, np.ndarray, float]:
    """voxel_mean at `voxel`, coarsened until at most `cap` points are left (a surface's voxel count goes with 1 / voxel
    squared, so the step is the square root of the overshoot). Returns the points, the colours and the voxel used."""
    v = voxel
    for _ in range(12):
        p, c = voxel_mean(pts, cols, v)
        if len(p) <= cap:
            return p, c, v
        v *= max(1.03, math.sqrt(len(p) / cap) * 1.02)
    raise RuntimeError(f"could not thin {len(pts)} points under {cap} (voxel {v:.3f} m)")


# ------------------------------------------------------------------ local planes

def _quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternions (w, x, y, z) of rotation matrices [m, 3, 3], the largest of the four candidates first for stability."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    t = m00 + m11 + m22
    pick = np.argmax(np.stack([t, m00, m11, m22]), axis=0)
    q = np.zeros((len(R), 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        s0 = np.sqrt(1 + t) * 2
        s1 = np.sqrt(1 + m00 - m11 - m22) * 2
        s2 = np.sqrt(1 + m11 - m00 - m22) * 2
        s3 = np.sqrt(1 + m22 - m00 - m11) * 2
        cand = [np.stack([s0 / 4, (R[:, 2, 1] - R[:, 1, 2]) / s0, (R[:, 0, 2] - R[:, 2, 0]) / s0, (R[:, 1, 0] - R[:, 0, 1]) / s0], 1),
                np.stack([(R[:, 2, 1] - R[:, 1, 2]) / s1, s1 / 4, (R[:, 0, 1] + R[:, 1, 0]) / s1, (R[:, 0, 2] + R[:, 2, 0]) / s1], 1),
                np.stack([(R[:, 0, 2] - R[:, 2, 0]) / s2, (R[:, 0, 1] + R[:, 1, 0]) / s2, s2 / 4, (R[:, 1, 2] + R[:, 2, 1]) / s2], 1),
                np.stack([(R[:, 1, 0] - R[:, 0, 1]) / s3, (R[:, 0, 2] + R[:, 2, 0]) / s3, (R[:, 1, 2] + R[:, 2, 1]) / s3, s3 / 4], 1)]
    for i in range(4):
        q[pick == i] = cand[i][pick == i]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def local_frames(pts: np.ndarray, k: int = NEIGHBOURS, sigma: float = 0.005, passes: int = 2, chunk: int = 100_000,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For every point, the plane its k nearest neighbours lie on: (normal [N, 3], in-plane axis [N, 3], mean distance to
    the 4 nearest neighbours [N], the 8 nearest neighbours' indices [N, 8]). The fit is a weighted PCA, refitted `passes` times with weights 1 / (1 + (r / sigma)^2)^2
    of each neighbour's distance r from the plane: at an edge or a corner the neighbours on the far face stop pulling
    the plane away from the face the point is on (a plain PCA tilts the normals within 3 cm of an edge by tens of degrees).
    The sign of the normal is arbitrary here."""
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, np.float32)
    k = min(k, len(pts) - 1)
    tree = cKDTree(pts)
    n_out, t_out, d_out = np.empty((len(pts), 3), np.float32), np.empty((len(pts), 3), np.float32), np.empty(len(pts), np.float32)
    nbr = np.empty((len(pts), min(8, k)), np.int32)
    for s in range(0, len(pts), chunk):
        dist, idx = tree.query(pts[s:s + chunk], k=k + 1, workers=-1)
        nb = pts[idx].astype(np.float64)
        w = np.ones(idx.shape)
        for it in range(passes + 1):
            c = (nb * w[..., None]).sum(1) / w.sum(1, keepdims=True)
            r = nb - c[:, None]
            _, V = np.linalg.eigh(np.einsum("mk,mki,mkj->mij", w, r, r))
            n = V[:, :, 0]
            if it < passes:
                off = np.einsum("mki,mi->mk", r, n)
                w = 1.0 / (1.0 + (off / sigma) ** 2) ** 2
        n_out[s:s + chunk], t_out[s:s + chunk] = n, V[:, :, 2]
        d_out[s:s + chunk] = dist[:, 1:5].mean(1)
        nbr[s:s + chunk] = idx[:, 1:nbr.shape[1] + 1]
    return n_out, t_out, d_out, nbr


def _occluded(p: np.ndarray, n: np.ndarray, s: np.ndarray, lo: np.ndarray, dims: np.ndarray, occupied: np.ndarray, voxel: float,
              step: float = 0.04, own: float = 0.06, near_station: float = 0.4, budget: int = 4_000_000) -> np.ndarray:
    """For each segment p -> s: does it pass through an occupied voxel other than the surface the point is on? Samples are
    `step` apart from 5 cm off the point to `near_station` short of the station (its own tripod is in the cloud), and a
    sample within `own` of the point's tangent plane is its own surface."""
    out = np.zeros(len(p), bool)
    d = s - p
    L = np.linalg.norm(d, axis=1)
    d /= np.maximum(L, 1e-9)[:, None]
    m = np.maximum(0, np.floor((L - near_station - 0.05) / step).astype(np.int64))
    start = 0
    while start < len(p):
        end = start + max(1, int(np.searchsorted(np.cumsum(m[start:]), budget)))
        mm = m[start:end]
        ray = np.repeat(np.arange(start, end), mm)
        k = np.arange(len(ray)) - np.repeat(np.cumsum(mm) - mm, mm)
        t = 0.05 + k * step
        q = p[ray] + d[ray] * t[:, None]
        c = np.floor((q - lo) / voxel).astype(np.int64)
        ok = (c >= 0).all(1) & (c < dims).all(1) & (np.abs(np.einsum("ij,ij->i", n[ray], q - p[ray])) >= own)
        key = (c[ok, 0] * dims[1] + c[ok, 1]) * dims[2] + c[ok, 2]
        pos = np.minimum(np.searchsorted(occupied, key), len(occupied) - 1)
        hit = ray[ok][occupied[pos] == key]
        out[np.unique(hit)] = True
        start = end
    return out


def orient(pts: np.ndarray, normals: np.ndarray, stations: np.ndarray | None, nbr: np.ndarray | None = None, voxel: float = 0.05,
           candidates: int = 6) -> np.ndarray:
    """Normals turned to face the scanner station that saw the point: the nearest one whose line to the point crosses
    nothing else in the cloud (a voxel grid of `voxel` with at least 2 points in a cell), preferring one that sees the surface
    head on to one that grazes it, then settled against the neighbours (`agree`). The plain nearest station is wrong for a wall with a station in the next room behind it,
    and for the far side of a table. Where no station is clear, the nearest. Without stations, normals face the cloud's centroid, which is inside a room."""
    out = normals.copy()
    if stations is None or not len(stations):
        flip = np.einsum("ij,ij->i", normals, pts.mean(0) - pts) < 0
        out[flip] *= -1
        return out
    from scipy.spatial import cKDTree
    K = min(candidates, len(stations))
    near = np.asarray(cKDTree(stations).query(pts, k=K, workers=-1)[1]).reshape(len(pts), K)
    lo = np.minimum(pts.min(0), stations.min(0)) - 0.1
    dims = np.ceil((np.maximum(pts.max(0), stations.max(0)) + 0.1 - lo) / voxel).astype(np.int64) + 1
    cells = np.floor((pts - lo) / voxel).astype(np.int64)
    keys, cnt = np.unique((cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2], return_counts=True)
    occupied = keys[cnt >= 2]
    chosen, best = near[:, 0].copy(), np.full(len(pts), -1.0)
    pending = np.arange(len(pts))
    for r in range(K):
        if not len(pending):
            break
        blocked = _occluded(pts[pending], normals[pending], stations[near[pending, r]], lo, dims, occupied, voxel)
        clear = pending[~blocked]
        vec = stations[near[clear, r]] - pts[clear]
        dist = np.linalg.norm(vec, axis=1)
        cos = np.abs(np.einsum("ij,ij->i", normals[clear], vec)) / np.maximum(dist, 1e-9)
        better = cos / (dist + 1.0) > best[clear]  # a clear station that sees the surface head on beats one that grazes it
        chosen[clear[better]] = near[clear[better], r]
        best[clear[better]] = (cos / (dist + 1.0))[better]
        pending = np.concatenate([pending[blocked], clear[cos < 0.7]])
    flip = np.einsum("ij,ij->i", normals, stations[chosen] - pts) < 0
    out[flip] *= -1
    return agree(out, nbr) if nbr is not None else out


def agree(normals: np.ndarray, nbr: np.ndarray, coplanar: float = 0.9) -> np.ndarray:
    """A normal that faces against most of the neighbours lying on its own plane (|dot| above `coplanar`) is turned round:
    what is left of the sight-line test after its few misses at edges, where a ray grazes a corner."""
    dots = np.einsum("ij,ikj->ik", normals, normals[nbr])
    vote = np.where(np.abs(dots) > coplanar, np.sign(dots), 0).sum(1)
    out = normals.copy()
    out[vote < 0] *= -1
    return out


# ------------------------------------------------------------------ the splats

def disks(pts: np.ndarray, cols: np.ndarray, normals: np.ndarray, axis: np.ndarray, spacing: np.ndarray, voxel: float,
          sh_degree: int = 3) -> dict[str, np.ndarray]:
    """The trainer's PLY columns for one disk per point (see the module docstring for the rules)."""
    n = len(pts)
    t1 = axis - np.einsum("ij,ij->i", axis, normals)[:, None] * normals
    bad = np.linalg.norm(t1, axis=1) < 1e-6
    if bad.any():  # an in-plane axis that is the normal itself: any perpendicular will do
        alt = np.cross(normals[bad], [1.0, 0, 0])
        alt[np.linalg.norm(alt, axis=1) < 1e-3] = np.cross(normals[bad][np.linalg.norm(alt, axis=1) < 1e-3], [0, 1.0, 0])
        t1[bad] = alt
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normals, t1)
    R = np.stack([t1, t2, normals], axis=2)  # columns: the disk's axes, its thin one last
    sig = np.clip(0.5 * spacing, 0.2 * voxel, 0.5 * voxel)
    q = _quat(R)
    rgb = cols.astype(np.float32) / 255.0
    cols_out: dict[str, np.ndarray] = {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2]}
    for i in range(3):
        cols_out[f"f_dc_{i}"] = (rgb[:, i] - 0.5) / C0
    for i in range(3 * ((sh_degree + 1) ** 2 - 1)):
        cols_out[f"f_rest_{i}"] = np.zeros(n, np.float32)
    cols_out["opacity"] = np.full(n, math.log(OPACITY / (1 - OPACITY)), np.float32)
    cols_out["scale_0"] = cols_out["scale_1"] = np.log(sig).astype(np.float32)
    cols_out["scale_2"] = np.log(sig * THIN).astype(np.float32)
    for i in range(4):
        cols_out[f"rot_{i}"] = q[:, i].astype(np.float32)
    return cols_out


def write_ply(path: Path, cols: dict[str, np.ndarray], chunk: int = 500_000) -> None:
    """A binary little-endian 3DGS PLY, float properties only, in the order of `cols` (what plypipe.clean.read_ply reads)."""
    names = list(cols)
    n = len(cols["x"])
    head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join(f"property float {k}\n" for k in names) + "end_header\n"
    with Path(path).open("wb") as f:
        f.write(head.encode())
        for s in range(0, n, chunk):
            f.write(np.stack([cols[k][s:s + chunk] for k in names], 1).astype("<f4").tobytes())


def fscore(splats: np.ndarray, scan: np.ndarray, radius: float = 0.01, sample: int = 1_000_000, seed: int = 0) -> dict:
    """Precision (splat centres within `radius` of the scan), recall (scan points within `radius` of a splat centre) and
    their F-score, the scan sampled down to `sample` points."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    scan = scan[rng.choice(len(scan), sample, replace=False)] if len(scan) > sample else scan
    p = float((cKDTree(scan).query(splats, workers=-1)[0] <= radius).mean())
    r = float((cKDTree(splats).query(scan, workers=-1)[0] <= radius).mean())
    return {"radiusM": radius, "precision": round(p, 4), "recall": round(r, 4), "f": round(2 * p * r / ((p + r) or 1), 4)}


# ------------------------------------------------------------------ the scene frame

def find_up(pts: np.ndarray, cols: np.ndarray, stations: np.ndarray | None, hint: str = "auto", seed: int = 0) -> tuple[np.ndarray, dict]:
    """The file's up direction (a unit vector along one of its own axes, floor to ceiling) and the evidence for it, with
    the floor's rough place as `floorOffset` (floor = {p: up . p = floorOffset})."""
    sub, sc = thin_to_cap(pts, cols, 0.03, 250_000)[:2]
    n = local_frames(sub, k=12, sigma=0.01)[0]
    flat = (np.abs(n) > math.cos(math.radians(10))).mean(0)  # share of points whose normal is along each axis
    evidence: dict = {"flatShareByAxis": [round(float(v), 3) for v in flat]}
    axis = "xyz".index(hint[-1]) if hint != "auto" else int(np.argmax(flat * np.array([1.0, 1.0, PRIOR_Z])))
    a = np.eye(3)[axis]
    h = sub @ a
    horiz = np.abs(n[:, axis]) > math.cos(math.radians(15))
    if not horiz.any():
        raise ValueError(f"no horizontal surface along axis {'xyz'[axis]}: pass --up")
    lo, hi = np.percentile(h[horiz], [0.2, 99.8])
    bins = np.arange(lo, hi + 0.02, 0.02)
    count = np.histogram(h[horiz], bins)[0]
    layers = np.flatnonzero(count >= 0.25 * count.max())
    low, high = float(bins[layers[0]] + 0.01), float(bins[layers[-1]] + 0.01)
    evidence.update(axis="xyz"[axis], lowLayer=round(low, 2), highLayer=round(high, 2))
    if hint != "auto":
        floor_low = hint[0] != "-"
        evidence["why"] = f"given: {hint}"
    elif high - low < 1.5:  # a floor and no ceiling in the scan: the stations stand above it
        have = stations is not None and len(stations)
        floor_low = not have or float(np.median(stations @ a)) >= low
        evidence["why"] = "one horizontal layer: stations above it" if have else "one horizontal layer"
    else:
        near_low = float(((h > low + 0.08) & (h < low + 1.2)).sum())
        near_high = float(((h < high - 0.08) & (h > high - 1.2)).sum())
        evidence.update(clutterLowSide=int(near_low), clutterHighSide=int(near_high))
        if abs(near_low - near_high) < 0.15 * max(near_low, near_high, 1.0):  # a bare room: the ceiling is the lighter layer
            lum = sc.astype(np.float32).mean(1)
            b_low, b_high = (float(lum[horiz & (np.abs(h - z) < 0.06)].mean()) for z in (low, high))
            floor_low = b_low <= b_high
            evidence["why"] = f"clutter about equal on both sides: the darker layer ({b_low:.0f} against {b_high:.0f}) is the floor"
        else:
            floor_low = near_low > near_high
            evidence["why"] = "the floor side holds the furniture"
    evidence["floorOffset"] = low if floor_low else -high
    return a * (1.0 if floor_low else -1.0), evidence


def refine_floor(pts: np.ndarray, g: np.ndarray, offset: float, bands: tuple = (0.04, 0.03, 0.02, 0.015, 0.01, 0.01)) -> geometry.Plane:
    """The floor plane from the level one at `offset`: refitted to the points within a band of it, the band narrowing to
    1 cm. (geometry.ransac_floor is built for sparse SfM points and takes the lowest of many planes: on a laser floor,
    flat to millimetres, that is one tilted by half a degree.)"""
    n, d = g, offset
    for band in bands:
        m = np.abs(pts @ n - d) < band
        if m.sum() < 100:
            break
        n, d = geometry.fit_plane_lsq(pts[m])
        if n @ g < 0:
            n, d = -n, -d
    m = np.abs(pts @ n - d) < bands[-1]
    return geometry.Plane(normal=n, offset=float(d), inliers=m, tilt_deg=float(np.degrees(np.arccos(np.clip(n @ g, -1, 1)))))


def align(pts: np.ndarray, cols: np.ndarray, stations: np.ndarray | None, up: str = "auto", seed: int = 0) -> tuple[np.ndarray, dict]:
    """T (4x4, the file frame to the scene frame, scale 1) and what it was decided from: up along +Y, the floor plane at
    y = 0, the walls on X and Z, the origin on the floor under the median station."""
    g, evidence = find_up(pts, cols, stations, up, seed)
    rng = np.random.default_rng(seed)
    sub = pts[rng.choice(len(pts), 400_000, replace=False)] if len(pts) > 400_000 else pts
    sub = sub.astype(np.float64)
    cams = stations if stations is not None and len(stations) else np.median(sub, axis=0, keepdims=True)
    plane = refine_floor(sub, g, evidence["floorOffset"])
    if plane.inliers.sum() < 100:
        raise ValueError(f"the floor at {evidence['floorOffset']:.2f} m along the up axis has under 100 points within 1 cm: pass --up")
    T, info = geometry.similarity_to_scene(plane, sub, cams, cam_height_m=None)
    info["floorInlierShare"] = round(float(plane.inliers.mean()), 3)
    info["up"] = [round(float(v), 4) for v in g]
    info["upEvidence"] = evidence
    return T, info


def ceiling_height(y: np.ndarray, above: float = 1.8) -> float:
    """The ceiling: the densest layer of points above head height (a scan sees lights, ducts and the outside through
    windows, which move a percentile); the 99th percentile when nothing is above head height."""
    top = y[y > above]
    if len(top) < 100:
        return float(np.percentile(y, 99.0))
    cnt, edges = np.histogram(top, np.arange(above, min(float(top.max()), above + 12.0) + 0.02, 0.02))
    b = int(np.argmax(cnt))
    layer = top[(top >= edges[b] - 0.03) & (top <= edges[b + 1] + 0.03)]
    return float(np.median(layer))


# ------------------------------------------------------------------ walking

def _shifted(arr: np.ndarray, dj: int, di: int) -> tuple[np.ndarray, np.ndarray]:
    """arr and arr moved by (dj rows, di columns), cut to the cells both have."""
    h, w = arr.shape
    return arr[:h - dj, max(0, -di): w - max(0, di)], arr[dj:, max(0, di): w - max(0, -di)]


def route(grid: np.ndarray, meta: dict, stations_xz: np.ndarray, reach_m: float = 1.5, clearance_m: float = 0.35) -> tuple[np.ndarray, list[int]]:
    """A walk through the scanner stations in scan order on the occupancy grid, as xz points about 0.5 m apart, and which
    stations it passes. Each leg is the cheapest path over the lattice the viewer lets the visitor stand on, with a
    penalty near blocked cells so the route keeps to the middle of passages. A station the visitor cannot stand within
    `reach_m` of (a tripod in a closet, a room the scan only looked into) is left out."""
    import cv2
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    ok, xs, zs = walkcheck.standable(grid, meta)
    step = walkcheck.STEP
    nz, nx = ok.shape
    ncomp, comp = cv2.connectedComponents(ok.astype(np.uint8), connectivity=8)
    X, Z = np.meshgrid(xs, zs)
    snap: list[tuple[int, int] | None] = []
    for s in stations_xz:
        d = np.where(ok, np.hypot(X - s[0], Z - s[1]), np.inf)
        j = int(np.argmin(d))
        snap.append(np.unravel_index(j, d.shape) if d.flat[j] <= reach_m else None)
    comps = [comp[c] for c in snap if c is not None]
    if not comps:
        return np.zeros((0, 2)), []
    main = np.bincount(comps).argmax()  # the component most stations are in
    keep = [i for i, c in enumerate(snap) if c is not None and comp[c] == main]
    clear = cv2.distanceTransform(ok.astype(np.uint8), cv2.DIST_L2, 3) * step
    cost = 1.0 + 4.0 * np.exp(-clear / clearance_m)
    ids = -np.ones(ok.shape, np.int64)
    ids[ok & (comp == main)] = np.arange(int((ok & (comp == main)).sum()))
    rows, cols_, w = [], [], []
    for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a, b = _shifted(ids, dj, di)
        ca, cb = _shifted(cost, dj, di)
        m = (a >= 0) & (b >= 0)
        rows.append(a[m])
        cols_.append(b[m])
        w.append(0.5 * (ca[m] + cb[m]) * step * math.hypot(dj, di))
    r, c, wt = np.concatenate(rows), np.concatenate(cols_), np.concatenate(w)
    n = int((ids >= 0).sum())
    graph = coo_matrix((np.r_[wt, wt], (np.r_[r, c], np.r_[c, r])), shape=(n, n)).tocsr()
    inv = np.stack(np.nonzero(ids >= 0), 1)  # node -> (row, column): the nodes were numbered in row-major order
    node = lambda rc: int(ids[rc[0], rc[1]])
    path: list[int] = []
    for a, b in zip(keep[:-1], keep[1:]):
        d, pred = dijkstra(graph, indices=node(snap[a]), return_predecessors=True)
        j, leg = node(snap[b]), []
        while j >= 0 and j != node(snap[a]):
            leg.append(j)
            j = int(pred[j])
        path += [node(snap[a])] + leg[::-1]
    path.append(node(snap[keep[-1]]))
    xz = np.stack([xs[inv[path, 1]], zs[inv[path, 0]]], 1)
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))]
    if seg[-1] > 0:
        t = np.arange(0.0, seg[-1], 0.5)
        xz = np.stack([np.interp(np.r_[t, seg[-1]], seg, xz[:, 0]), np.interp(np.r_[t, seg[-1]], seg, xz[:, 1])], 1)
    return xz, keep


def open_spot(grid: np.ndarray, meta: dict) -> np.ndarray:
    """Where a visitor starts when there are no stations: the standable point furthest from anything blocked."""
    import cv2
    ok, xs, zs = walkcheck.standable(grid, meta)
    if not ok.any():
        raise RuntimeError("no standable floor in the occupancy grid")
    clear = cv2.distanceTransform(ok.astype(np.uint8), cv2.DIST_L2, 3)
    j, i = np.unravel_index(int(np.argmax(clear)), clear.shape)
    return np.array([xs[i], zs[j]])


def products(ctx: Ctx, pts: np.ndarray, stations: np.ndarray | None, T: np.ndarray, info: dict, splat_info: dict) -> dict:
    """occupancy.json/.png, shell.json and scene.json from the scan's points in the scene frame (`pts`, float32) and the
    stations in it."""
    from plypipe.stages import occupancy
    cell = float(ctx.opt("cell", 0.05))
    cams = stations if stations is not None and len(stations) else np.zeros((0, 3))
    rng = np.random.default_rng(3)
    low = np.flatnonzero((pts[:, 1] > -0.1) & (pts[:, 1] < 0.6))
    low = low[rng.choice(len(low), min(len(low), 100_000), replace=False)] if len(low) else low
    if len(cams) and len(low):  # the scanner's own sight lines to everything it saw low down: free space, as a trained twin's depth is
        from scipy.spatial import cKDTree
        _, j = cKDTree(cams).query(pts[low], workers=-1)
        rays = (cams[j], pts[low])
    else:
        rays = None
    # build_grid frees a corridor along its camera path, and a scanner's stations are not a walk: a line between two of them
    # would cut through a table. Each station goes in as its own 20 cm stub, which path_segments never joins to the next
    stubs = np.stack([cams + d for d in ((-0.1, 0, 0), (0, 0, 0), (0.1, 0, 0))], 1).reshape(-1, 3)
    grid, meta = occupancy.build_grid(pts, stubs, cell=cell, rays=rays, min_pts=2)
    meta["source"] = "lidar"
    if len(cams):
        xz, kept = route(grid, meta, cams[:, [0, 2]])
    else:
        xz, kept = open_spot(grid, meta)[None], []
    route_note = {"stations": len(cams), "walkable": len(kept), "pathPoints": len(xz)}
    if len(xz) == 0:  # no station can be stood at: start at the roomiest spot
        xz = open_spot(grid, meta)[None]
    path3 = np.c_[xz[:, 0], np.full(len(xz), EYE), xz[:, 1]]
    ahead = path3[min(len(path3) - 1, 4)] - path3[0]
    ahead[1] = 0.0
    if np.linalg.norm(ahead) < 0.3:  # one stop: look across the room
        ahead = np.array([pts[:, 0].mean() - path3[0, 0], 0.0, pts[:, 2].mean() - path3[0, 2]])
    ahead /= np.linalg.norm(ahead) or 1.0
    spawn = {"position": [round(float(v), 3) for v in path3[0]], "lookAt": [round(float(path3[0, 0] + ahead[0]), 3), EYE, round(float(path3[0, 2] + ahead[2]), 3)]}
    meta["walk"] = walkcheck.check(grid, meta, (path3[0, 0], path3[0, 2]), cams[:, [0, 2]] if len(cams) else xz)
    occupancy._write(ctx, grid, meta)

    ceiling = ceiling_height(pts[:, 1])
    (ctx.out / "shell.json").write_text(json.dumps({**shell_mod.build(pts, 0.0, ceiling, []), "source": "lidar scan"}, indent=1))
    bounds = geometry.robust_bounds(pts, lo=0.2, hi=99.8, margin=0.1)
    bounds["min"][1] = min(bounds["min"][1], -0.1)
    update_scene(ctx, {
        "version": 1, "units": "meters", "up": [0, 1, 0], "handedness": "right", "floorY": 0.0, "scale": 1.0, "scaleSource": "lidar",
        "cameraHeight": round(float(np.median(cams[:, 1])), 3) if len(cams) else None, "transform": T.tolist(), "bounds": bounds,
        "cameraPath": [[round(float(v), 3) for v in p] for p in path3], "spawn": spawn, "tour": tour_mod.build_tour(path3, bounds),
        "shell": "shell.json", "lidar": {**splat_info, "stationsScene": [[round(float(v), 3) for v in s] for s in cams], "ceilingY": round(ceiling, 3),
                                         "align": {k: info[k] for k in ("up", "upEvidence", "yawDeg", "floorTiltDeg") if k in info}}})
    return {"walk": meta["walk"], "route": route_note, "grid": {"cell": cell, "width": meta["width"], "height": meta["height"], **meta["stats"]},
            "ceilingY": round(ceiling, 3), "walls": len(json.loads((ctx.out / "shell.json").read_text())["walls"])}


# ------------------------------------------------------------------ the stage

SKIPPED = ("viewer evaluation and the cleanup choices measured with it (held-out photos)", "the phone file's fine-tune (training views)",
           "the walking grid from depth (a renderer and photos: the grid is the scan's own, see `occupancy`)")


def run_stage(ctx: Ctx, source: Path) -> dict:
    t0 = time.time()
    d = load_points(source)
    pts, cols, stations = d["points"], d["colors"], d["stations"]
    n_in = len(pts)
    ctx.log.emit("lidar", "progress", f"{n_in} points, {0 if stations is None else len(stations)} scanner stations")
    T, info = align(pts, cols, stations, ctx.opt("up", "auto"))
    ctx.log.emit("lidar", "progress", f"up {info['up']} ({info['upEvidence'].get('why')}), floor tilt {info['floorTiltDeg']:.2f} deg, yaw {info['yawDeg']:.1f} deg")
    scene_pts = (pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    st = None if stations is None else (stations @ T[:3, :3].T + T[:3, 3])
    cap = int(ctx.opt("cap") or splat_cap())
    voxel0 = float(ctx.opt("voxel", VOXEL))
    centres, colours, voxel = thin_to_cap(scene_pts, cols, voxel0, cap)
    ctx.log.emit("lidar", "progress", f"{len(centres)} splats at a {voxel * 100:.2f} cm voxel" + ("" if voxel == voxel0 else f" (coarsened from {voxel0 * 100:.2f} cm to stay under {cap})"))
    normals, axis, spacing, nbr = local_frames(centres, k=int(ctx.opt("k", NEIGHBOURS)), sigma=0.5 * voxel)
    normals = orient(centres, normals, st, nbr)
    sh = int(ctx.opt("sh_degree", 3))
    write_ply(ctx.out / "scene.ply", disks(centres, colours, normals, axis, spacing, voxel, sh))
    f = fscore(centres, scene_pts, 0.01)
    splat_info = {"source": Path(source).name, "points": n_in, "splats": len(centres), "voxelM": round(voxel, 5), "voxelRequestedM": voxel0, "capSplats": cap,
                  "origin": np.asarray(d["origin"]).tolist(), "shDegree": sh}
    out = products(ctx, scene_pts, st, T, info, splat_info)
    return {**splat_info, "up": info["up"], "upEvidence": info["upEvidence"], "yawDeg": info["yawDeg"], "floorTiltDeg": info["floorTiltDeg"],
            "fScore": f, "coarsened": voxel > voxel0, "skipped": list(SKIPPED), **out, "seconds": round(time.time() - t0, 1)}


def run(ctx: Ctx, source: Path) -> dict:
    """The whole twin: this stage, then the export stage with the options a scan can use. Returns the export report."""
    from plypipe.stages import export
    rep = run_stage(ctx, source)
    mark_done(ctx, STAGE, rep)
    if ctx.opt("no_export"):
        return rep
    # splat-transform's value filters would be probed against a model whose opacity and scale are constants: skip the two passes
    prev = os.environ.get("PLY_EXPORT_PROBES")
    os.environ["PLY_EXPORT_PROBES"] = "0"
    try:
        ctx.opts.update({"holdout": 0, "lite_finetune": 0, "dense_occupancy": False})
        ex = export.run_stage(ctx)
    finally:
        if prev is None:
            os.environ.pop("PLY_EXPORT_PROBES", None)
        else:
            os.environ["PLY_EXPORT_PROBES"] = prev
    mark_done(ctx, "export", ex)
    return ex


def main(a) -> int:
    import traceback

    from plypipe.log import Log
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out, quiet=a.quiet)
    opts = {k: v for k, v in vars(a).items() if k not in ("cmd", "input", "out", "quiet") and v is not None}
    ctx = Ctx(out=out, log=log, opts=opts)
    (out / "job.json").write_text(json.dumps({"input": str(a.input), "kind": "lidar", "opts": opts}, indent=2, default=str))
    log.emit("lidar", "start")
    try:
        rep = run(ctx, a.input)
    except Exception as e:
        log.emit("lidar", "error", str(e), trace=traceback.format_exc()[-2000:])
        return 1
    log.emit("lidar", "done", **{k: v for k, v in rep.items() if not isinstance(v, (dict, list))})
    return 0
