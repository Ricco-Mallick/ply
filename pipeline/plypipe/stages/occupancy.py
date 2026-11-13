# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""occupancy: 2D walkable grid for WASD collision in the R1 viewer (R2 replaces it with a navmesh).

From the aligned sparse model (scene frame, metres):
    blocked  = cells holding >= `min_pts` points between 0.2 m and 1.8 m above the floor
               (walls, furniture: anything a walking person would hit)
    free     = cells with floor points (|y| < 0.08 m); cells the cameras saw through low down: a sight line from a
               camera to a point it observed, where the line runs below 0.6 m, crosses nothing solid there, and a
               cell needs `min_rays` such lines; and the walked corridor (see below)
    unknown  = everything else, including free cells not connected to the walked corridor; small unknown holes
               enclosed by free cells are filled

The walked corridor, `path_radius` (0.2 m, about half a person's width) either side of the camera path, wins over
blocked: the person fitted there, and the viewer's walker is 0.44 m wide, so the old one-cell line left it stuck
behind any stray point or pew end. If the phone was held out over furniture, the corridor opens that strip of it.
The path is broken where consecutive cameras jump (a new clip starts elsewhere), so a corridor never cuts a wall.

Nothing is closed morphologically: a plain wall has no sparse points and no sight lines through it, so it stays
unknown, which the viewer treats as blocked.

Writes occupancy.png (8-bit grey: 255 free, 0 blocked, 127 unknown; image row r = +Z,
column c = +X, top-left pixel = cell at the origin corner) and occupancy.json, which also carries
the walk check of plypipe.walkcheck (`walk`) and where the grid came from (`source`: "sparse" or "depth").

The grid from depth (IMPROVEMENT-PLAN 2.3; `from_depth`, run by export where a GPU can render, --no-dense-occupancy off)
replaces the one above unless fewer capture positions are walkable on it; where the depth decided nothing it takes
the sparse grid's floor, and it keeps the sparse grid's walls. The sparse grid knows a wall only where the solve found points
on it, and floor only where it found points or the cameras walked. The trained splat's depth, kept where other views
confirm it (plypipe.dense), is a range scan from every view, in 5 cm cells over the same area:

    seen blocked  by a view with a point in the cell between 0.2 m and 1.8 m above the floor
    seen free     by a view with a ray that crosses the cell while the ray is between the floor and 1.8 m: up to the
                  cell before a point in the band, to the end for a ray that lands on the floor (its own cell is free
                  too), and as far as 1.8 m for a ray that ends higher. A ray under a table top does not exist (it ends
                  on the top), and one over it is outvoted by the views that see the top
    blocked       seen blocked by 3 views: this wins, so a table or a wall thickened by depth noise stays closed
    free          otherwise, seen free by 3 views; or within the walked corridor, which still wins over everything
    unknown       the rest, which the viewer blocks. One view's stray point blocks nothing, and two views free nothing

A view votes once per cell whatever its pixel count. Free space not connected to the walk goes back to unknown as
above. Unknown holes enclosed by free cells are filled only up to 0.25 m2 and only if no view saw any of their cells
blocked, so free space never closes over, or joins across, a cell that anything was seen in.

The grid from a scan's own returns (`grid_from_pointcloud`, run by `plypipe lidar` for IMPROVEMENT-PLAN 8a) needs no
poses at all: a laser scan of one station is the cloud, and the station's pose in the file is not a place anyone stood
(on `lidar-office-1` it is 5 m outside its own cloud). In 5 cm cells over the same area:

    blocked  by a cell holding `min_pts` returns between 0.2 m and 1.8 m above the floor: a wall, a desk, a chair
    free     by a cell with a return on the floor (|y| < 0.08 m): the scanner saw the floor there
             or by a cell with a return on the ceiling plane (1.8 m up to the measured ceiling + 15 cm) and no return
             below it: the beam went over the cell to the ceiling, so nothing stands in it, and the floor under it is
             only hidden (dark, glossy, or in a shadow of the scanner's own tripod). A ceiling that is under 0.6 m
             above head height is not a ceiling (an outdoor scan, a cathedral): then only floor returns free cells
    unknown  everything else: outside the scanned area, inside a wall, behind furniture

Nothing is closed morphologically and no free space is dropped for being disconnected: a cell is free only where a
return says so, and what the scanner never saw stays unknown, which the viewer blocks. Unknown pockets enclosed by
free cells are filled up to 0.25 m2, and only where no return was seen in them.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from plypipe import colmap_txt, geometry, walkcheck
from plypipe.util import Ctx, update_scene

FREE, BLOCKED, UNKNOWN = 255, 0, 127
DEPTH_CELL = 0.05


def path_segments(cam: np.ndarray, jump_m: float = 0.75, jump_ratio: float = 8.0) -> list[np.ndarray]:
    """The camera path (in capture order) cut wherever one step is much longer than a walking step."""
    cam = np.asarray(cam, float).reshape(-1, 3)
    if len(cam) < 2:
        return [cam] if len(cam) else []
    step = np.linalg.norm(np.diff(cam[:, [0, 2]], axis=0), axis=1)
    cut = step > max(jump_m, jump_ratio * float(np.median(step)))
    return np.split(cam, np.flatnonzero(cut) + 1)


def observations(model: colmap_txt.Model, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(camera centre, point) for every observation of the kept points."""
    ims = model.images
    centers, _, _ = geometry.camera_centers_and_ups(np.array([i.qvec for i in ims]), np.array([i.tvec for i in ims]))
    row = {im.image_id: k for k, im in enumerate(ims)}
    cam_idx, pt_idx = [], []
    for p in np.flatnonzero(keep):
        for img in model.point_tails[p].split()[4::2]:  # "R G B ERROR IMAGE_ID POINT2D_IDX ..."
            k = row.get(int(img))
            if k is not None:
                cam_idx.append(k)
                pt_idx.append(p)
    if not cam_idx:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(centers)[cam_idx], model.xyz[pt_idx]


def _samples(c: np.ndarray, d: np.ndarray, L: np.ndarray, t0: np.ndarray, t1: np.ndarray, lo: np.ndarray, cell: float,
             shape: tuple[int, int], per_cell: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(ray, flat cell) of the cells each ray c + t d crosses between t0 and t1, from `per_cell` samples per cell; L is
    the ray's length on the floor plan."""
    H, W = shape
    n = np.ceil((t1 - t0) * L / (cell / per_cell)).astype(np.int64) + 1
    ray = np.repeat(np.arange(len(n)), n)
    k = np.arange(len(ray)) - np.repeat(np.cumsum(n) - n, n)
    t = t0[ray] + (t1 - t0)[ray] * k / np.maximum(n[ray] - 1, 1)
    col = np.floor((c[ray, 0] + d[ray, 0] * t - lo[0]) / cell).astype(np.int64)
    r = np.floor((c[ray, 2] + d[ray, 2] * t - lo[1]) / cell).astype(np.int64)
    inside = (col >= 0) & (col < W) & (r >= 0) & (r < H)
    return ray[inside], r[inside] * W + col[inside]


def seen_free(cam: np.ndarray, pts: np.ndarray, lo: np.ndarray, cell: float, shape: tuple[int, int],
              below: float = 0.6, stop: float = 0.2, max_len: float = 12.0, chunk: int = 200_000) -> np.ndarray:
    """Per cell, how many sight lines pass over it lower than `below` (stopping `stop` short of the point seen)."""
    H, W = shape
    counts = np.zeros(H * W, np.int64)
    cam, pts = np.asarray(cam, float), np.asarray(pts, float)
    for s in range(0, len(cam), chunk):
        c, p = cam[s:s + chunk], pts[s:s + chunk]
        d = p - c
        L = np.hypot(d[:, 0], d[:, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = np.where(c[:, 1] > below, (c[:, 1] - below) / (c[:, 1] - p[:, 1]), 0.0)
            t1 = 1.0 - stop / L
        ok = (p[:, 1] < below) & (L > stop) & (L < max_len) & np.isfinite(t0) & (t1 > t0)
        if not ok.any():
            continue
        ray, flat = _samples(c[ok], d[ok], L[ok], t0[ok], t1[ok], lo, cell, shape)
        key = np.unique(ray * (H * W) + flat)  # each line counts once per cell
        counts += np.bincount(key % (H * W), minlength=H * W)
    return counts.reshape(H, W)


def fill_holes(grid: np.ndarray, max_cells: int, veto: np.ndarray | None = None) -> int:
    """Unknown pockets of at most `max_cells`, bordered only by free cells, become free. Returns cells filled.
    A pocket with a cell marked in `veto` is left as it is."""
    unknown = (grid == UNKNOWN).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(unknown, connectivity=4)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    H, W = grid.shape
    filled = 0
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area > max_cells or x == 0 or y == 0 or x + w == W or y + h == H:
            continue
        y0, y1, x0, x1 = y - 1, y + h + 1, x - 1, x + w + 1
        m = (labels[y0:y1, x0:x1] == k).astype(np.uint8)
        rim = cv2.dilate(m, cross).astype(bool) & ~m.astype(bool)
        if veto is not None and veto[y0:y1, x0:x1][m.astype(bool)].any():
            continue
        if (grid[y0:y1, x0:x1][rim] == FREE).all():
            grid[y0:y1, x0:x1][m.astype(bool)] = FREE
            filled += int(area)
    return filled


def _cells(p: np.ndarray, lo: np.ndarray, cell: float, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, columns) of the points that fall inside the grid, and which points those are."""
    c = np.floor((p[:, 0] - lo[0]) / cell).astype(int)
    r = np.floor((p[:, 2] - lo[1]) / cell).astype(int)
    ok = (c >= 0) & (c < shape[1]) & (r >= 0) & (r < shape[0])
    return r[ok], c[ok], ok


def walked_corridor(cam: np.ndarray, lo: np.ndarray, cell: float, shape: tuple[int, int], radius: float) -> tuple[np.ndarray, int]:
    """The cells within `radius` of the camera path, and the number of segments the path was cut into."""
    path = np.zeros(shape, np.uint8)
    segments = path_segments(cam) if cam.size else []
    for seg in segments:
        sr, sc, _ = _cells(seg, lo, cell, shape)
        p2 = np.stack([sc, sr], axis=1).astype(np.int32)
        for a, b in zip(p2[:-1], p2[1:]):
            cv2.line(path, tuple(map(int, a)), tuple(map(int, b)), 1, 1)
        if len(p2) == 1:
            path[p2[0, 1], p2[0, 0]] = 1
    pr_cells = max(1, int(round(radius / cell)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pr_cells + 1, 2 * pr_cells + 1))
    return cv2.dilate(path, disk).astype(bool), len(segments)


def _forget_unreached(grid: np.ndarray, walked: np.ndarray) -> None:
    """Free space nobody could reach from the walk (sight lines through gaps in an outer wall or through a window,
    stray floor points outside) is not evidence of floor: it goes back to unknown."""
    if walked.any():
        _, lab = cv2.connectedComponents((grid == FREE).astype(np.uint8), connectivity=4)
        keep = np.unique(lab[walked])
        grid[(grid == FREE) & ~np.isin(lab, keep[keep > 0])] = UNKNOWN


def build_grid(points: np.ndarray, cam_path: np.ndarray, cell: float = 0.1, band: tuple[float, float] = (0.2, 1.8),
               floor_band: float = 0.08, min_pts: int = 2, path_radius: float = 0.2, margin: float = 0.5,
               rays: tuple[np.ndarray, np.ndarray] | None = None, ray_below: float = 0.6, min_rays: int = 3,
               hole_m2: float = 0.5) -> tuple[np.ndarray, dict]:
    pts = np.asarray(points, float)
    cam = np.asarray(cam_path, float).reshape(-1, 3)
    xz = np.concatenate([pts[:, [0, 2]], cam[:, [0, 2]]]) if cam.size else pts[:, [0, 2]]
    lo = np.percentile(xz, 1, axis=0) - margin
    hi = np.percentile(xz, 99, axis=0) + margin
    W = int(np.ceil((hi[0] - lo[0]) / cell))
    H = int(np.ceil((hi[1] - lo[1]) / cell))
    if W * H > 4_000_000:
        raise ValueError(f"grid {W}x{H} too large; is the scale wrong?")

    def cells(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _cells(p, lo, cell, (H, W))

    occ = np.zeros((H, W), np.int32)
    obst = pts[(pts[:, 1] > band[0]) & (pts[:, 1] < band[1])]
    if obst.size:
        r, c, _ = cells(obst)
        np.add.at(occ, (r, c), 1)
    blocked = occ >= min_pts

    floor = np.zeros((H, W), bool)
    fl = pts[np.abs(pts[:, 1]) < floor_band]
    if fl.size:
        r, c, _ = cells(fl)
        floor[r, c] = True

    walked, segments = walked_corridor(cam, lo, cell, (H, W), path_radius)

    seen = np.zeros((H, W), bool)
    n_rays = 0
    if rays is not None and len(rays[0]):
        n_rays = len(rays[0])
        seen = seen_free(rays[0], rays[1], lo, cell, (H, W), below=ray_below) >= min_rays

    grid = np.full((H, W), UNKNOWN, np.uint8)
    grid[floor | seen] = FREE
    grid[blocked] = BLOCKED
    grid[walked] = FREE  # someone walked here: wins over whatever the sparse points say
    _forget_unreached(grid, walked)
    filled = fill_holes(grid, max(1, int(hole_m2 / (cell * cell))))

    meta = {"version": 1, "image": "occupancy.png", "cellSize": cell, "origin": [float(lo[0]), float(lo[1])],
            "width": W, "height": H, "colAxis": "+x", "rowAxis": "+z",
            "values": {"free": FREE, "blocked": BLOCKED, "unknown": UNKNOWN},
            "obstacleBand": list(band), "floorBand": [-floor_band, floor_band],
            "walkedRadius": path_radius, "pathSegments": segments,
            "seenFree": {"rays": n_rays, "below": ray_below, "minRays": min_rays, "cells": int(seen.sum())},
            "holesFilled": filled, "source": "sparse",
            "stats": {"free": int((grid == FREE).sum()), "blocked": int((grid == BLOCKED).sum()),
                      "unknown": int((grid == UNKNOWN).sum())}}
    return grid, meta


def depth_votes(views, lo: np.ndarray, cell: float, shape: tuple[int, int], band: tuple[float, float] = (0.2, 1.8),
                max_range: float = 12.0, max_rays: int = 20_000, chunk: int = 5_000) -> dict:
    """Per cell, how many views saw it free and how many saw it blocked (module docstring).

    `views` yields (camera centre, points) in the scene frame, the points being what that view saw (plypipe.dense).
    Every point in the band is a blocked vote for its cell. The rays are thinned at random to `max_rays` a view: a
    view votes once per cell, and at half resolution it has several rays through every cell it sees. Beyond
    `max_range` on the floor plan nothing is voted: 2 % of depth is five cells there."""
    H, W = shape
    free, blocked = np.zeros(H * W, np.int32), np.zeros(H * W, np.int32)
    n_views = n_rays = 0
    for k, (center, pts) in enumerate(views):
        c, p = np.asarray(center, float), np.asarray(pts, float).reshape(-1, 3)
        n_views += 1
        if not len(p) or c[1] < 0:
            continue
        hit = (p[:, 1] >= band[0]) & (p[:, 1] <= band[1]) & (np.hypot(p[:, 0] - c[0], p[:, 2] - c[2]) <= max_range)
        r, col, _ = _cells(p[hit], lo, cell, shape)
        solid, seen = np.zeros(H * W, bool), np.zeros(H * W, bool)
        solid[r * W + col] = True
        if len(p) > max_rays:
            keep = np.random.default_rng(k).choice(len(p), max_rays, replace=False)
            p, hit = p[keep], hit[keep]
        for s in range(0, len(p), chunk):
            q = p[s:s + chunk]
            d = q - c
            L = np.hypot(d[:, 0], d[:, 2])
            with np.errstate(divide="ignore", invalid="ignore"):
                top, floor = (band[1] - c[1]) / d[:, 1], -c[1] / d[:, 1]   # where the ray is 1.8 m up, and where it lands
                t0 = np.where(c[1] > band[1], top, 0.0)
                t1 = np.where(q[:, 1] > band[1], top, np.where(q[:, 1] < 0, floor, 1.0))
                t1 = np.where(hit[s:s + chunk], 1.0 - cell / L, t1)           # a point in the band: the ray stops a cell short
                t1 = np.minimum(t1, max_range / L)
            ok = np.isfinite(t0) & np.isfinite(t1) & (t0 >= 0) & (t0 <= 1) & (t1 >= t0)
            if not ok.any():
                continue
            # one sample a cell: a cell this ray steps over is crossed by the rays of the pixels beside it
            _, flat = _samples(np.broadcast_to(c, d.shape)[ok], d[ok], L[ok], t0[ok], t1[ok], lo, cell, shape, per_cell=1)
            seen[flat] = True
            n_rays += int(ok.sum())
        blocked += solid
        free += seen & ~solid  # a view does not call a cell free that it saw something in
    return {"free": free.reshape(H, W), "blocked": blocked.reshape(H, W), "views": n_views, "rays": n_rays}


def grid_from_depth(views, cam_path: np.ndarray, lo: np.ndarray, shape: tuple[int, int], cell: float = DEPTH_CELL,
                    band: tuple[float, float] = (0.2, 1.8), min_views: int = 3, path_radius: float = 0.2, hole_m2: float = 0.25,
                    max_range: float = 12.0, max_rays: int = 20_000, sparse: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """The grid from the views' confirmed depth points (module docstring), over the area that starts at `lo` (x, z)
    with `shape` cells. `views` yields (camera centre, points) and is read once; `cam_path` is every capture position.

    `sparse`: the camera solve's grid over the same area, any cell size. Each grid is right where it has evidence. On
    the chapel the depth saw every pew the sparse points leave walkable; on Dr Johnson it left a corridor's floor
    unknown that the sparse grid has, and looked through a gap in a wall the sparse points close (2026-11-12,
    docs/EXPERIMENTS.md). So a cell the depth left undecided takes the sparse grid's free floor, unless any view saw it
    blocked or it lies within 10 cm of a cell the depth blocked (the inside of a wall, which no view sees, lies between
    two faces it did), and the sparse grid's blocked cells stay blocked."""
    H, W = shape
    cam = np.asarray(cam_path, float).reshape(-1, 3)
    votes = depth_votes(views, lo, cell, shape, band, max_range, max_rays)
    blocked = votes["blocked"] >= min_views
    free = (votes["free"] >= min_views) & ~blocked
    grid = np.full(shape, UNKNOWN, np.uint8)
    grid[free] = FREE
    grid[blocked] = BLOCKED
    merged = {}
    if sparse is not None:
        fine = cv2.resize(sparse, (W, H), interpolation=cv2.INTER_NEAREST)
        k = 2 * max(1, int(round(0.1 / cell))) + 1
        beside = cv2.dilate(blocked.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        take = (grid == UNKNOWN) & (fine == FREE) & (votes["blocked"] == 0) & ~beside
        wall = (fine == BLOCKED) & (grid != BLOCKED)
        grid[take] = FREE
        grid[wall] = BLOCKED
        merged = {"sparseFree": int(take.sum()), "sparseBlocked": int(wall.sum())}
    walked, segments = walked_corridor(cam, lo, cell, shape, path_radius)
    grid[walked] = FREE  # someone walked here: wins over whatever the depth says
    _forget_unreached(grid, walked)
    filled = fill_holes(grid, max(1, int(hole_m2 / (cell * cell))), veto=votes["blocked"] > 0)
    meta = {"version": 1, "image": "occupancy.png", "cellSize": cell, "origin": [float(lo[0]), float(lo[1])],
            "width": W, "height": H, "colAxis": "+x", "rowAxis": "+z",
            "values": {"free": FREE, "blocked": BLOCKED, "unknown": UNKNOWN},
            "obstacleBand": list(band), "floorBand": [0.0, band[0]],
            "walkedRadius": path_radius, "pathSegments": segments,
            "seenFree": {"rays": votes["rays"], "below": band[1], "minRays": min_views, "cells": int(free.sum())},
            "holesFilled": filled, "source": "depth",
            "depth": {"views": votes["views"], "minViews": min_views, "maxRange": max_range, "holeM2": hole_m2,
                      "seenFree": int((votes["free"] >= min_views).sum()), "seenBlocked": int(blocked.sum()),
                      "seenBlockedByFewer": int(((votes["blocked"] > 0) & ~blocked).sum()), **merged},
            "stats": {"free": int((grid == FREE).sum()), "blocked": int((grid == BLOCKED).sum()),
                      "unknown": int((grid == UNKNOWN).sum())}}
    return grid, meta


def ceiling_layer(y: np.ndarray, above: float = 1.8, step: float = 0.02, band: float = 0.03,
                  fallback: float = 99.0) -> float:
    """The ceiling: the densest layer of returns above head height (a scan also sees lights, ducts and, through a window,
    the outside, which move a percentile); the `fallback` percentile when nothing is that far up."""
    top = np.asarray(y, float)
    top = top[top > above]
    if len(top) < 100:
        return float(np.percentile(y, fallback))
    cnt, edges = np.histogram(top, np.arange(above, min(float(top.max()), above + 12.0) + step, step))
    b = int(np.argmax(cnt))
    layer = top[(top >= edges[b] - band) & (top <= edges[b + 1] + band)]
    return float(np.median(layer))


def pointcloud_votes(points: np.ndarray, lo: np.ndarray, cell: float, shape: tuple[int, int], band: tuple[float, float] = (0.2, 1.8),
                     floor_band: float = 0.08, top: float | None = None, rays: tuple[np.ndarray, np.ndarray] | None = None,
                     ray_below: float = 0.6, min_rays: int = 3, max_len: float = 12.0, chunk: int = 200_000) -> dict:
    """Per cell, how many returns the scan has on the floor, in the walking band and on the ceiling plane, and whether
    the scan's own sight lines crossed it (module docstring). `points` are in the scene frame with the floor at y = 0."""
    H, W = shape
    p = np.asarray(points, float).reshape(-1, 3)
    masks = [("floor", lambda y: np.abs(y) < floor_band), ("solid", lambda y: (y >= band[0]) & (y <= band[1]))]
    if top is not None:  # a ceiling return says the beam went over the cell; without a ceiling above the band there is none
        masks.append(("ceiling", lambda y: (y > band[1]) & (y <= float(top))))
    counts = {"floor": np.zeros(H * W, np.int64), "solid": np.zeros(H * W, np.int64), "ceiling": np.zeros(H * W, np.int64)}
    for s in range(0, len(p), chunk):  # a 184 M-point cloud is never held as one boolean mask
        q = p[s:s + chunk]
        for name, rule in masks:
            m = rule(q[:, 1])
            if m.any():
                r, c, _ = _cells(q[m], lo, cell, shape)
                counts[name] += np.bincount(r * W + c, minlength=H * W)
    seen, n_rays = np.zeros(shape, bool), 0
    if rays is not None and len(rays[0]):
        n_rays = len(rays[0])
        seen = seen_free(rays[0], rays[1], lo, cell, shape, below=ray_below, max_len=max_len) >= min_rays
    return {"floor": counts["floor"].reshape(shape), "solid": counts["solid"].reshape(shape),
            "ceiling": counts["ceiling"].reshape(shape), "seen": seen, "rays": n_rays}


def grid_from_pointcloud(points: np.ndarray, lo: np.ndarray | None = None, shape: tuple[int, int] | None = None, cell: float = DEPTH_CELL,
                         band: tuple[float, float] = (0.2, 1.8), floor_band: float = 0.08, min_pts: int = 2, top: float | str | None = "auto",
                         hole_m2: float = 0.25, margin: float = 0.5, rays: tuple[np.ndarray, np.ndarray] | None = None,
                         min_rays: int = 3) -> tuple[np.ndarray, dict]:
    """The walking grid from a scan's own returns (module docstring), over the area that starts at `lo` (x, z) with
    `shape` cells, or over 1 % to 99 % of the cloud plus `margin` when neither is given.

    `rays` (station to point, on a scan whose station is known) also frees cells the scanner saw through low down; a
    station that is nowhere near its own cloud is not evidence and must be passed as None. `top` is the ceiling return
    above which a cell is read as seen through: "auto" measures it (ceiling_layer), a number fixes it, None turns that
    rule off."""
    p = np.asarray(points, float).reshape(-1, 3)
    if lo is None or shape is None:
        xz = p[:, [0, 2]]
        lo = np.percentile(xz, 1, axis=0) - margin
        hi = np.percentile(xz, 99, axis=0) + margin
        shape = (int(np.ceil((hi[1] - lo[1]) / cell)), int(np.ceil((hi[0] - lo[0]) / cell)))
    H, W = shape
    if W * H > 4_000_000:
        raise ValueError(f"grid {W}x{H} too large; is the scale wrong?")
    if top == "auto":
        top = ceiling_layer(p[:, 1])
    if top is not None and float(top) - band[1] < 0.6:  # nothing above head height to see through: no ceiling rule
        top = None
    votes = pointcloud_votes(p, np.asarray(lo, float), cell, shape, band, floor_band, top, rays, min_rays=min_rays)
    blocked = votes["solid"] >= min_pts
    seen_through = (votes["ceiling"] >= 1) & (votes["solid"] == 0)
    free = (votes["floor"] >= 1) | seen_through | votes["seen"]
    grid = np.full(shape, UNKNOWN, np.uint8)
    grid[free] = FREE
    grid[blocked] = BLOCKED
    filled = fill_holes(grid, max(1, int(hole_m2 / (cell * cell))), veto=votes["solid"] > 0)
    n_free, _ = cv2.connectedComponents((grid == FREE).astype(np.uint8), connectivity=4)  # the free areas, then the rest
    meta = {"version": 1, "image": "occupancy.png", "cellSize": cell, "origin": [float(lo[0]), float(lo[1])],
            "width": W, "height": H, "colAxis": "+x", "rowAxis": "+z",
            "values": {"free": FREE, "blocked": BLOCKED, "unknown": UNKNOWN},
            "obstacleBand": list(band), "floorBand": [-floor_band, floor_band],
            "seenFree": {"rays": votes["rays"], "below": band[1], "minRays": min_rays, "cells": int(votes["seen"].sum())},
            "holesFilled": filled, "source": "pointcloud",
            "pointcloud": {"minPts": min_pts, "ceilingY": None if top is None else round(float(top), 3),
                           "floorCells": int((votes["floor"] >= 1).sum()), "seenThroughCells": int(seen_through.sum()),
                           "seenCells": int(votes["seen"].sum()), "freeComponents": n_free - 1},
            "stats": {"free": int((grid == FREE).sum()), "blocked": int((grid == BLOCKED).sum()),
                      "unknown": int((grid == UNKNOWN).sum())}}
    return grid, meta


def world_to_cell(meta: dict, x: float, z: float) -> tuple[int, int]:
    """(row, col) for a scene-frame position; mirrors what the viewer does."""
    return (int(np.floor((z - meta["origin"][1]) / meta["cellSize"])),
            int(np.floor((x - meta["origin"][0]) / meta["cellSize"])))


def grid_from_model(model: colmap_txt.Model, cell: float = 0.1, max_reproj_px: float = 2.0) -> tuple[np.ndarray, dict]:
    good = (model.track_len >= 3) & (model.error < max_reproj_px)
    if good.sum() < 200:
        good = np.ones(model.num_points, bool)
    centers, _, _ = model.cameras()
    low = good & (model.xyz[:, 1] > -0.1) & (model.xyz[:, 1] < 0.6)  # only low points can carve (seen_free)
    return build_grid(model.xyz[good], centers, cell=cell, rays=observations(model, low))


def _sparse_grid(ctx: Ctx) -> tuple[np.ndarray, dict, np.ndarray, tuple[float, float]]:
    """The grid from the sparse model with its walk check, every capture position, and where the viewer starts (x, z)."""
    model = colmap_txt.read_model(ctx.data_dir / "sparse" / "0")
    grid, meta = grid_from_model(model, cell=float(ctx.opt("cell", 0.1)),
                                 max_reproj_px=float(ctx.opt("max_reproj_px", 2.0)))
    scene_p = ctx.out / "scene.json"
    scene = json.loads(scene_p.read_text()) if scene_p.exists() else {}
    centers, _, _ = model.cameras()
    sp = (scene.get("spawn") or {}).get("position") or centers[0]
    meta["walk"] = walkcheck.check(grid, meta, (sp[0], sp[2]), centers[:, [0, 2]])
    return grid, meta, centers, (float(sp[0]), float(sp[2]))


def _write(ctx: Ctx, grid: np.ndarray, meta: dict) -> None:
    cv2.imwrite(str(ctx.out / "occupancy.png"), grid)
    (ctx.out / "occupancy.json").write_text(json.dumps(meta, indent=2))
    update_scene(ctx, {"occupancy": {"image": "occupancy.png", "meta": "occupancy.json"}})


def run_stage(ctx: Ctx) -> dict:
    grid, meta, _, _ = _sparse_grid(ctx)
    _write(ctx, grid, meta)
    return {"width": meta["width"], "height": meta["height"], **meta["stats"], "walk": meta["walk"]}


def from_depth(ctx: Ctx, ply: Path, holdout: int = 30, antialiased: bool = True, step: int = 5, scale: float = 0.5, fusion=None) -> dict:
    """The grid from the depth of the splats in `ply` (module docstring), with the sparse grid's evidence where the
    depth has none (grid_from_depth), written in place of the sparse grid unless fewer capture positions are walkable
    on it. Less floor to stand on is not a reason to refuse it: on the chapel the floor it takes away is the pews.
    When it is refused the sparse grid is written again, with the reason, and the grid that lost is left beside it as
    occupancy_depth.png. The confirmed points are saved as dense_points.npz for the mesh and layout stages.

    `fusion` (stages.mesh.Fusion) is fed every view's kept depth and colours as the pass goes, so the TSDF mesh costs no second render.

    Returns the record for the export report. Without CUDA and gsplat, or with cameras that are not PINHOLE, nothing
    is built or written. Raises if the pass itself fails: the caller decides what that costs."""
    from plypipe import dense

    cams = dense.training_views(ctx.data_dir, holdout, step, scale)
    rendered = dense.render_views(ply, cams, antialiased, **({"colour": True} if fusion is not None else {})) if cams else None
    if rendered is None:
        return {"used": "sparse", "why": "not built: needs a GPU with gsplat, and PINHOLE cameras"}
    sparse, sparse_meta, centers, spawn = _sparse_grid(ctx)
    finer = sparse_meta["cellSize"] / DEPTH_CELL  # the same area as the sparse grid, in smaller cells
    shape = (int(np.ceil(sparse_meta["height"] * finer)), int(np.ceil(sparse_meta["width"] * finer)))
    pixels, cloud = {}, dense.Cloud()
    if fusion is not None:
        fusion.begin(cams)

    def scans():
        for p in dense.consistent_points(rendered, stats=pixels, fuse=fusion is not None):
            cloud.add(p.points, p.normals)
            if fusion is not None:
                fusion.add(p)
            yield p.center, p.points

    grid, meta = grid_from_depth(scans(), centers, np.array(sparse_meta["origin"]), shape, sparse=sparse)
    meta["walk"] = now = walkcheck.check(grid, meta, spawn, centers[:, [0, 2]])
    meta["depth"].update(pixels=pixels, points=cloud.save(ctx.out / dense.FILE))
    was = sparse_meta["walk"]
    out = {"views": meta["depth"]["views"], "walk": now, "sparseWalk": was, "points": meta["depth"]["points"]}
    if (now["cameras"] or 0) < (was["cameras"] or 0):
        why = f"fewer capture positions are walkable on it: {now['cameras']} against {was['cameras']}"
        cv2.imwrite(str(ctx.out / "occupancy_depth.png"), grid)
        sparse_meta["depth"] = {**meta["depth"], "stats": meta["stats"], "walk": now, "rejected": why}
        _write(ctx, sparse, sparse_meta)
        return {"used": "sparse", "why": why, **out}
    meta["depth"]["sparseWalk"] = was
    _write(ctx, grid, meta)
    return {"used": "depth", **out}
