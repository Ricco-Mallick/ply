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
the walk check of plypipe.walkcheck (`walk`).
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from plypipe import colmap_txt, geometry, walkcheck
from plypipe.util import Ctx, update_scene

FREE, BLOCKED, UNKNOWN = 255, 0, 127


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
        c, d, L, t0, t1 = c[ok], d[ok], L[ok], t0[ok], t1[ok]
        n = np.ceil((t1 - t0) * L / (0.5 * cell)).astype(np.int64) + 1  # two samples per cell crossed
        ray = np.repeat(np.arange(len(n)), n)
        k = np.arange(len(ray)) - np.repeat(np.cumsum(n) - n, n)
        t = t0[ray] + (t1 - t0)[ray] * k / np.maximum(n[ray] - 1, 1)
        col = np.floor((c[ray, 0] + d[ray, 0] * t - lo[0]) / cell).astype(np.int64)
        r = np.floor((c[ray, 2] + d[ray, 2] * t - lo[1]) / cell).astype(np.int64)
        inside = (col >= 0) & (col < W) & (r >= 0) & (r < H)
        key = np.unique(ray[inside] * (H * W) + r[inside] * W + col[inside])  # each line counts once per cell
        counts += np.bincount(key % (H * W), minlength=H * W)
    return counts.reshape(H, W)


def fill_holes(grid: np.ndarray, max_cells: int) -> int:
    """Unknown pockets of at most `max_cells`, bordered only by free cells, become free. Returns cells filled."""
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
        if (grid[y0:y1, x0:x1][rim] == FREE).all():
            grid[y0:y1, x0:x1][m.astype(bool)] = FREE
            filled += int(area)
    return filled


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
        c = np.floor((p[:, 0] - lo[0]) / cell).astype(int)
        r = np.floor((p[:, 2] - lo[1]) / cell).astype(int)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        return r[ok], c[ok], ok

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

    path = np.zeros((H, W), np.uint8)
    segments = path_segments(cam) if cam.size else []
    for seg in segments:
        sr, sc, _ = cells(seg)
        p2 = np.stack([sc, sr], axis=1).astype(np.int32)
        for a, b in zip(p2[:-1], p2[1:]):
            cv2.line(path, tuple(map(int, a)), tuple(map(int, b)), 1, 1)
        if len(p2) == 1:
            path[p2[0, 1], p2[0, 0]] = 1
    pr_cells = max(1, int(round(path_radius / cell)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pr_cells + 1, 2 * pr_cells + 1))
    walked = cv2.dilate(path, disk).astype(bool)

    seen = np.zeros((H, W), bool)
    n_rays = 0
    if rays is not None and len(rays[0]):
        n_rays = len(rays[0])
        seen = seen_free(rays[0], rays[1], lo, cell, (H, W), below=ray_below) >= min_rays

    grid = np.full((H, W), UNKNOWN, np.uint8)
    grid[floor | seen] = FREE
    grid[blocked] = BLOCKED
    grid[walked] = FREE  # someone walked here: wins over whatever the sparse points say
    if walked.any():
        # free space nobody could reach from the walk (sight lines through gaps in an outer wall, stray floor
        # points outside) is not evidence of floor
        _, lab = cv2.connectedComponents((grid == FREE).astype(np.uint8), connectivity=4)
        keep = np.unique(lab[walked])
        grid[(grid == FREE) & ~np.isin(lab, keep[keep > 0])] = UNKNOWN
    filled = fill_holes(grid, max(1, int(hole_m2 / (cell * cell))))

    meta = {"version": 1, "image": "occupancy.png", "cellSize": cell, "origin": [float(lo[0]), float(lo[1])],
            "width": W, "height": H, "colAxis": "+x", "rowAxis": "+z",
            "values": {"free": FREE, "blocked": BLOCKED, "unknown": UNKNOWN},
            "obstacleBand": list(band), "floorBand": [-floor_band, floor_band],
            "walkedRadius": path_radius, "pathSegments": len(segments),
            "seenFree": {"rays": n_rays, "below": ray_below, "minRays": min_rays, "cells": int(seen.sum())},
            "holesFilled": filled,
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


def run_stage(ctx: Ctx) -> dict:
    model = colmap_txt.read_model(ctx.data_dir / "sparse" / "0")
    grid, meta = grid_from_model(model, cell=float(ctx.opt("cell", 0.1)),
                                 max_reproj_px=float(ctx.opt("max_reproj_px", 2.0)))
    scene_p = ctx.out / "scene.json"
    scene = json.loads(scene_p.read_text()) if scene_p.exists() else {}
    centers, _, _ = model.cameras()
    sp = (scene.get("spawn") or {}).get("position") or centers[0]
    meta["walk"] = walkcheck.check(grid, meta, (sp[0], sp[2]), centers[:, [0, 2]])
    cv2.imwrite(str(ctx.out / "occupancy.png"), grid)
    (ctx.out / "occupancy.json").write_text(json.dumps(meta, indent=2))
    update_scene(ctx, {"occupancy": {"image": "occupancy.png", "meta": "occupancy.json"}})
    return {"width": meta["width"], "height": meta["height"], **meta["stats"], "walk": meta["walk"]}
