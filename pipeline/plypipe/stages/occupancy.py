# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""occupancy: 2D walkable grid for WASD collision in the R1 viewer (R2 replaces it with a navmesh).

From the aligned sparse points (scene frame, metres):
    blocked  = cells holding >= `min_pts` points between 0.2 m and 1.8 m above the floor
               (walls, furniture: anything a walking person would hit)
    free     = cells with floor points (|y| < 0.08 m) or that the capturing phone passed over
               (the person physically walked there), closed to bridge sparse-floor gaps
    unknown  = everything else

Camera-path cells always win over blocked: stray sparse points must not wall off the route
the capture actually walked.

Writes occupancy.png (8-bit grey: 255 free, 0 blocked, 127 unknown; image row r = +Z,
column c = +X, top-left pixel = cell at the origin corner) and occupancy.json.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from plypipe import colmap_txt
from plypipe.util import Ctx, update_scene

FREE, BLOCKED, UNKNOWN = 255, 0, 127


def build_grid(points: np.ndarray, cam_path: np.ndarray, cell: float = 0.1, band: tuple[float, float] = (0.2, 1.8),
               floor_band: float = 0.08, min_pts: int = 2, path_radius: float = 0.3, close_m: float = 0.3,
               margin: float = 0.5) -> tuple[np.ndarray, dict]:
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

    floor = np.zeros((H, W), np.uint8)
    fl = pts[np.abs(pts[:, 1]) < floor_band]
    if fl.size:
        r, c, _ = cells(fl)
        floor[r, c] = 1
    path = np.zeros((H, W), np.uint8)
    if cam.size:
        # rasterise the path as connected segments so gaps between subsampled cameras are walked too
        pr, pc, ok = cells(cam)
        pts2 = np.stack([pc, pr], axis=1).astype(np.int32)
        for a, b in zip(pts2[:-1], pts2[1:]):
            cv2.line(path, tuple(map(int, a)), tuple(map(int, b)), 1, 1)
        if len(pts2) == 1:
            path[pr[0], pc[0]] = 1
    pr_cells = max(1, int(round(path_radius / cell)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pr_cells + 1, 2 * pr_cells + 1))
    path_wide = cv2.dilate(path, disk)

    k = max(1, int(round(close_m / cell)))
    close_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    free = cv2.morphologyEx(np.maximum(floor, path_wide), cv2.MORPH_CLOSE, close_el).astype(bool)

    grid = np.full((H, W), UNKNOWN, np.uint8)
    grid[free] = FREE
    grid[blocked] = BLOCKED
    grid[path.astype(bool)] = FREE  # the walked line itself is never blocked

    meta = {"version": 1, "image": "occupancy.png", "cellSize": cell, "origin": [float(lo[0]), float(lo[1])],
            "width": W, "height": H, "colAxis": "+x", "rowAxis": "+z",
            "values": {"free": FREE, "blocked": BLOCKED, "unknown": UNKNOWN},
            "obstacleBand": list(band), "floorBand": [-floor_band, floor_band],
            "stats": {"free": int((grid == FREE).sum()), "blocked": int((grid == BLOCKED).sum()),
                      "unknown": int((grid == UNKNOWN).sum())}}
    return grid, meta


def world_to_cell(meta: dict, x: float, z: float) -> tuple[int, int]:
    """(row, col) for a scene-frame position; mirrors what the viewer does."""
    return (int(np.floor((z - meta["origin"][1]) / meta["cellSize"])),
            int(np.floor((x - meta["origin"][0]) / meta["cellSize"])))


def run_stage(ctx: Ctx) -> dict:
    model = colmap_txt.read_model(ctx.data_dir / "sparse" / "0")
    good = (model.track_len >= 3) & (model.error < float(ctx.opt("max_reproj_px", 2.0)))
    pts = model.xyz[good] if good.sum() >= 200 else model.xyz
    centers, _, _ = model.cameras()
    grid, meta = build_grid(pts, centers, cell=float(ctx.opt("cell", 0.1)))
    cv2.imwrite(str(ctx.out / "occupancy.png"), grid)
    (ctx.out / "occupancy.json").write_text(json.dumps(meta, indent=2))
    update_scene(ctx, {"occupancy": {"image": "occupancy.png", "meta": "occupancy.json"}})
    return {"width": meta["width"], "height": meta["height"], **meta["stats"]}
