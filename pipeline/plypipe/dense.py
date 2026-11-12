# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""dense: surface points from the trained splat's own depth (IMPROVEMENT-PLAN 2.2).

The camera solve leaves a few thousand points, none of them on a plain wall. The trained splat knows every surface the
photos show, so its expected depth is rendered at every 5th training view at half resolution and each pixel is kept
only when other views confirm it:

    usable      alpha >= 0.9 (the splats cover the pixel), depth finite and inside (near, far), and no jump of more than
                5 % to the next pixel: at a silhouette the expected depth is a blend of the two surfaces, a point in
                mid-air
    confirmed   the pixel's 3D point is projected into its K neighbouring views; a view agrees when the depth it
                rendered there (bilinear, all four pixels usable) is within 2 % of the point's depth in that view. A
                view that sees more than 5 % farther looks through the point and contradicts it; one that sees nearer
                is an occlusion and says nothing. A point needs two agreeing views and more agreeing than
                contradicting ones.
    neighbours  the K views with the smallest  |C_j - C_i| + d_i * |f_j - f_i|  (camera centres C, viewing directions f,
                d_i the view's median depth: the second term is how far apart the two views look), so a walk that
                turns round is not paired with itself by frame order. Views closer than 5 % of d_i are left out: a
                blob in front of the lens blends into two views from one spot the same way, and their agreement would
                confirm it.

What a semi-transparent floater does to expected depth is exactly what this removes: it drags the depth to a point
between itself and the wall behind, and every view that is not looking through the same blob sees the wall there.

Normals are the cross product of finite differences of the back-projected depth map (central where both neighbours
are usable, one-sided at the last pixel before a hole or a silhouette), turned to face the camera, in
the world frame. Points are yielded view by view (`consistent_points` is a generator), so what is held is the usable
depth of every view (N x H x W floats) and one view's points: never the points of every view. `Cloud` keeps one point
per 2 cm voxel for dense_points.npz (points float32, normals float16), which the mesh and layout stages read; it is not
shipped to the viewer.

The core is written against what numpy and torch share, so the same lines run on numpy arrays (the tests, CI) and on
CUDA tensors (export). Only `render_views` needs a GPU: it is gsplat's rasteriser in "ED" mode and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from plypipe.viewer_eval import _quat_to_R, held_out

FILE = "dense_points.npz"


@dataclass
class View:
    """A pinhole view at the rendered size: world-to-camera pose (x right, y down, z forward), and, once rendered,
    expected z-depth and alpha as (H, W) numpy arrays or torch tensors."""
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    name: str = ""
    depth: Any = None
    alpha: Any = None

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class Points:
    """What one view adds: its confirmed points and their normals in the world frame, as float32 numpy arrays."""
    view: int
    name: str
    center: np.ndarray
    points: np.ndarray
    normals: np.ndarray


def training_views(data_dir: Path, every: int = 30, step: int = 5, scale: float = 0.5) -> list[View] | None:
    """Every `step`-th training view of the aligned model at `scale` of the photo's size: the views
    viewer_eval.training_weights renders. Held-out views (every `every`-th) are left out, the splat was never fitted
    to them. None unless every camera is PINHOLE."""
    from plypipe import colmap_txt

    model = colmap_txt.read_model(Path(data_dir) / "sparse" / "0")
    cams = {}
    for ln in model.cameras_lines:
        if ln and not ln.startswith("#"):
            p = ln.split()
            if p[1] != "PINHOLE":
                return None
            cams[int(p[0])] = (int(p[2]), int(p[3]), *map(float, p[4:8]))
    ims = sorted(model.images, key=lambda im: im.name)
    held = set(held_out([im.name for im in ims], every))
    views = []
    for i in [i for i in range(len(ims)) if i not in held][::max(1, step)]:
        im = ims[i]
        w, h, fx, fy, cx, cy = cams[im.camera_id]
        K = np.array([[fx * scale, 0, cx * scale], [0, fy * scale, cy * scale], [0, 0, 1]])
        views.append(View(K, _quat_to_R(np.asarray(im.qvec, float)), np.asarray(im.tvec, float),
                          max(1, int(w * scale)), max(1, int(h * scale)), im.name))
    return views


def render_views(ply: Path, views: list[View], antialiased: bool = True) -> Iterator[View] | None:
    """The splats' expected depth and alpha at each view, one view at a time on the GPU. None without CUDA and gsplat."""
    try:
        import torch
        from gsplat import rasterization
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return _render(ply, views, antialiased, "cuda", rasterization)


def _render(ply: Path, views: list[View], antialiased: bool, dev: str, rasterization) -> Iterator[View]:
    import torch

    from plypipe import clean

    _, c = clean.read_ply(ply)
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    means = t(np.stack([c["x"], c["y"], c["z"]], 1))
    quats = t(np.stack([c[f"rot_{i}"] for i in range(4)], 1))
    scales = torch.exp(t(np.stack([c[f"scale_{i}"] for i in range(3)], 1)))
    opac = torch.sigmoid(t(c["opacity"]))
    for v in views:
        view = np.eye(4)
        view[:3, :3], view[:3, 3] = v.R, v.t
        with torch.no_grad():
            # "ED": the one channel is depth already divided by alpha. No colours are drawn; the means stand in for
            # them because gsplat before 1.6 checks their shape in every mode
            depth, alpha, _ = rasterization(means, quats, scales, opac, means, t(view)[None], t(v.K)[None], v.width, v.height,
                                            render_mode="ED", rasterize_mode="antialiased" if antialiased else "classic")
        yield replace(v, depth=depth[0, ..., -1], alpha=alpha[0, ..., 0])


# ------------------------------------------------------------------ numpy or torch

def _ns(a):
    """numpy for arrays, torch for tensors."""
    if isinstance(a, np.ndarray):
        return np
    import torch
    return torch


def _like(ref, a):
    """A numpy array as float32 of ref's kind, on ref's device."""
    a = np.asarray(a, np.float32)
    return a if isinstance(ref, np.ndarray) else ref.new_tensor(a)


def _f32(a):
    return a.astype(np.float32) if isinstance(a, np.ndarray) else a.float()


def _int(a):
    return a.astype(np.int64) if isinstance(a, np.ndarray) else a.long()


def _np(a) -> np.ndarray:
    return a if isinstance(a, np.ndarray) else a.cpu().numpy()


# ------------------------------------------------------------------ the core

def usable_depth(v: View, min_alpha: float = 0.9, near: float = 0.1, far: float = 50.0, edge: float = 0.05):
    """The view's depth as float32, 0 where it cannot be used: thin alpha, depth outside (near, far), or a jump of
    more than `edge` to a neighbouring pixel (which also takes the rim of every hole)."""
    xp = _ns(v.depth)
    d = _f32(v.depth)
    zero = xp.zeros_like(d)
    d = xp.where((v.alpha >= min_alpha) & xp.isfinite(d) & (d > near) & (d < far), d, zero)
    jx = xp.abs(d[:, 1:] - d[:, :-1]) > edge * xp.minimum(d[:, 1:], d[:, :-1])
    jy = xp.abs(d[1:] - d[:-1]) > edge * xp.minimum(d[1:], d[:-1])
    cut = xp.zeros_like(d > 0)
    cut[:, 1:] |= jx
    cut[:, :-1] |= jx
    cut[1:] |= jy
    cut[:-1] |= jy
    return xp.where(cut, zero, d)


def neighbours(centers: np.ndarray, forwards: np.ndarray, depth: np.ndarray, k: int = 8, min_parallax: float = 0.05) -> np.ndarray:
    """(N, k) indices of each view's neighbours, -1 where there are fewer: the views that stand and look most alike,
    leaving out those closer than `min_parallax` of the view's median depth (see the module docstring)."""
    C, f, d = np.asarray(centers, float), np.asarray(forwards, float), np.asarray(depth, float)
    base = np.linalg.norm(C[:, None] - C[None], axis=2)
    cost = base + d[:, None] * np.linalg.norm(f[:, None] - f[None], axis=2)
    cost[(base < min_parallax * d[:, None]) | (d[:, None] <= 0) | (d[None] <= 0)] = np.inf  # an empty view confirms nothing
    np.fill_diagonal(cost, np.inf)
    order = np.argsort(cost, axis=1, kind="stable")[:, :k]
    out = np.where(np.isfinite(np.take_along_axis(cost, order, 1)), order, -1)
    return np.pad(out, ((0, 0), (0, max(0, k - out.shape[1]))), constant_values=-1)


def _pixel_rays(v: View) -> np.ndarray:
    """(H, W, 3) camera-space direction of each pixel at depth 1; gsplat's pixel centres are at +0.5."""
    u, w = np.meshgrid(np.arange(v.width) + 0.5, np.arange(v.height) + 0.5)
    return np.stack([(u - v.K[0, 2]) / v.K[0, 0], (w - v.K[1, 2]) / v.K[1, 1], np.ones_like(u)], -1)


def _along_rows(P, ok):
    """Difference of the points P (H, W, 3) along each row, and where it exists: from the pixel before to the pixel
    after where both are usable, else to or from the one that is (the last pixel before a hole or a silhouette)."""
    xp = _ns(P)
    ahead, behind, has_ahead, has_behind = xp.zeros_like(P), xp.zeros_like(P), xp.zeros_like(ok), xp.zeros_like(ok)
    ahead[:, :-1] = P[:, 1:] - P[:, :-1]
    has_ahead[:, :-1] = ok[:, 1:] & ok[:, :-1]
    behind[:, 1:] = ahead[:, :-1]
    has_behind[:, 1:] = has_ahead[:, :-1]
    return ahead * has_ahead[..., None] + behind * has_behind[..., None], has_ahead | has_behind


def _normals(P, ok):
    """Unit normals facing the camera from finite differences of the camera-space points P (H, W, 3), and where they
    exist: the pixel is usable and has a usable neighbour in its row and in its column."""
    xp = _ns(P)
    du, has_u = _along_rows(P, ok)
    dv, has_v = _along_rows(P.swapaxes(0, 1), ok.swapaxes(0, 1))
    dv, has_v = dv.swapaxes(0, 1), has_v.swapaxes(0, 1)
    n = xp.stack([dv[..., 1] * du[..., 2] - dv[..., 2] * du[..., 1],
                  dv[..., 2] * du[..., 0] - dv[..., 0] * du[..., 2],
                  dv[..., 0] * du[..., 1] - dv[..., 1] * du[..., 0]], -1)
    length = xp.sqrt((n * n).sum(-1))
    has = has_u & has_v & (length > 0)
    away = (n * P).sum(-1) > 0
    return n / xp.where(has, xp.where(away, -length, length), xp.ones_like(length))[..., None], has


def _sample(D, px, py):
    """Bilinear depth at pixel coordinates, and where it was seen: inside the image with all four pixels usable."""
    xp = _ns(D)
    H, W = D.shape
    fu, fv = px - 0.5, py - 0.5
    u0, v0 = xp.floor(fu), xp.floor(fv)
    a, b = fu - u0, fv - v0
    inside = (u0 >= 0) & (v0 >= 0) & (u0 <= W - 2) & (v0 <= H - 2)
    u0, v0 = _int(xp.clip(u0, 0, W - 2)), _int(xp.clip(v0, 0, H - 2))
    d00, d01, d10, d11 = D[v0, u0], D[v0, u0 + 1], D[v0 + 1, u0], D[v0 + 1, u0 + 1]
    seen = inside & (d00 > 0) & (d01 > 0) & (d10 > 0) & (d11 > 0)
    return (d00 * (1 - a) + d01 * a) * (1 - b) + (d10 * (1 - a) + d11 * a) * b, seen


def consistent_points(views: Iterable[View], k: int = 8, tol: float = 0.02, min_agree: int = 2, through: float = 0.05,
                      min_alpha: float = 0.9, near: float = 0.1, far: float = 50.0, edge: float = 0.05,
                      min_parallax: float = 0.05, stats: dict | None = None) -> Iterator[Points]:
    """The points of each view that its neighbours confirm (module docstring), one view at a time.

    `views` is read once and only each view's usable depth is kept, so it can be a generator of renders. With
    `min_agree` 0 nothing is checked and every usable pixel comes back. `stats`, when given, counts pixels: usable,
    kept, unconfirmed (too few views agree) and contradicted (as many views look through the point as agree)."""
    cams, maps = [], []
    for v in views:
        maps.append(usable_depth(v, min_alpha, near, far, edge))
        cams.append(replace(v, depth=None, alpha=None))
    count = stats if stats is not None else {}
    count.update(views=len(cams), pixels=0, usable=0, kept=0, unconfirmed=0, contradicted=0)
    if not cams:
        return
    xp = _ns(maps[0])
    median = np.array([float(xp.median(d[d > 0])) if bool((d > 0).any()) else 0.0 for d in maps])
    near_by = neighbours(np.array([v.center for v in cams]), np.array([v.R[2] for v in cams]), median, k, min_parallax)
    for i, (v, d) in enumerate(zip(cams, maps)):
        P = _like(d, _pixel_rays(v)) * d[..., None]
        normal, has = _normals(P, d > 0)
        sel = (d > 0) & has
        R = _like(d, v.R)
        world = (P[sel] - _like(d, v.t)) @ R  # rows: R^T (p - t)
        normal = normal[sel] @ R
        agree = _int(world[:, 0] * 0)
        looks_through = _int(world[:, 0] * 0)
        for j in near_by[i][near_by[i] >= 0] if min_agree else ():
            o = cams[j]
            fx, fy, cx, cy = (float(x) for x in (o.K[0, 0], o.K[1, 1], o.K[0, 2], o.K[1, 2]))  # plain floats: numpy scalars and CUDA tensors do not mix
            q = world @ _like(d, o.R.T) + _like(d, o.t)
            z = q[:, 2]
            front = z > near
            zs = xp.where(front, z, xp.ones_like(z))
            dj, seen = _sample(maps[j], fx * q[:, 0] / zs + cx, fy * q[:, 1] / zs + cy)
            rel = (dj - z) / zs
            seen = seen & front
            agree = agree + (seen & (xp.abs(rel) <= tol))
            looks_through = looks_through + (seen & (rel > through))
        confirmed = agree >= min_agree
        keep = confirmed & (agree > looks_through) if min_agree else confirmed
        count["pixels"] += v.width * v.height
        count["usable"] += int(sel.sum())
        count["kept"] += int(keep.sum())
        count["unconfirmed"] += int((~confirmed).sum())
        count["contradicted"] += int((confirmed & ~keep).sum())
        yield Points(i, v.name, v.center, _np(world[keep]), _np(normal[keep]))


class Cloud:
    """The points that are saved: the first one seen in each `voxel` cube, so the cloud grows with the surface and not
    with the number of views. Over `cap` points the voxel doubles and the cloud is thinned again."""

    def __init__(self, voxel: float = 0.02, cap: int = 4_000_000):
        self.voxel, self.cap = voxel, cap
        self._keys = np.zeros(0, np.int64)  # sorted voxel keys of what is held
        self._parts: list[tuple[np.ndarray, np.ndarray]] = []

    def __len__(self) -> int:
        return len(self._keys)

    def _key(self, p: np.ndarray) -> np.ndarray:
        q = np.clip(np.floor(p / self.voxel).astype(np.int64) + (1 << 20), 0, (1 << 21) - 1)  # 21 bits an axis: +-20 km at 2 cm
        return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]

    def add(self, points: np.ndarray, normals: np.ndarray) -> None:
        key, first = np.unique(self._key(points), return_index=True)
        at = np.searchsorted(self._keys, key)
        new = np.ones(len(key), bool)
        if len(self._keys):
            new = (at == len(self._keys)) | (self._keys[np.minimum(at, len(self._keys) - 1)] != key)
        self._keys = np.insert(self._keys, at[new], key[new])
        self._parts.append((points[first[new]].astype(np.float32), normals[first[new]].astype(np.float16)))
        while len(self) > self.cap:
            self.voxel *= 2
            pts, nrm = self.arrays()
            self._keys, first = np.unique(self._key(pts), return_index=True)
            self._parts = [(pts[first], nrm[first])]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._parts:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float16)
        self._parts = [(np.concatenate([p for p, _ in self._parts]), np.concatenate([n for _, n in self._parts]))]
        return self._parts[0]

    def save(self, path: Path) -> dict:
        pts, nrm = self.arrays()
        np.savez_compressed(path, points=pts, normals=nrm, voxel=np.float32(self.voxel))
        return {"file": Path(path).name, "points": len(pts), "voxel": round(self.voxel, 4), "mb": round(Path(path).stat().st_size / 2**20, 2)}
