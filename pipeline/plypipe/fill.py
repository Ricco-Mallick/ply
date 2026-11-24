# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Fill the floor the camera never saw.

A walk-through leaves holes in the floor: under the phone (it never looks straight down at your feet), behind a
chair, at the foot of a wall. In a splat they read as black or see-through patches you can walk into. This fills
them the way a person would patch a photo, with no learned model: copy floor from right next to the hole.

1. floor splats: within FLOOR_BAND of a floor's height, flat (their thinnest axis close to vertical);
2. coverage on a 5 cm raster of the floor: a cell is covered when its splats add up to enough opacity;
3. holes: uncovered cells enclosed by covered floor (a gap that reaches the edge of the floor is outside the room,
   not a hole), from MIN_HOLE to MAX_HOLE in area;
4. every hole cell takes the splats of the nearest covered cell (found breadth-first from the hole's rim), moved over
   by the offset between the two cells, so scale, rotation, colour and view-dependent colour all stay as captured.

Each floor (scene.json levels) is filled on its own; one-floor captures have one floor at y = 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plypipe.clean import read_ply

CELL = 0.05
FLOOR_BAND = 0.06       # metres either side of the floor height
MIN_COVER = 0.6         # summed opacity in a cell that counts as floor
MIN_HOLE, MAX_HOLE = 0.01, 4.0   # m²
MAX_ADD = 300_000


def _normal_y(cols: dict, idx: np.ndarray) -> np.ndarray:
    """|y| of each splat's thinnest axis (its normal for a flat splat)."""
    q = np.stack([cols["rot_0"][idx], cols["rot_1"][idx], cols["rot_2"][idx], cols["rot_3"][idx]], axis=1)  # w, x, y, z
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    w, x, y, z = q.T
    # the rotation's columns are the splat's axes; row y of each column
    cols_y = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1)
    s = np.stack([cols[f"scale_{i}"][idx] for i in range(3)], axis=1)
    return np.abs(cols_y[np.arange(len(idx)), np.argmin(s, axis=1)])


def plan(cols: dict, floor_y: float) -> tuple[np.ndarray, dict]:
    """Indices of the splats to copy (with repeats) and their offsets, for one floor; plus what was found."""
    y = cols["y"]
    near = np.flatnonzero(np.abs(y - floor_y) < FLOOR_BAND)
    if len(near) < 500 or not all(k in cols for k in ("rot_0", "scale_0", "opacity")):
        return np.zeros(0, np.int64), {"floorSplats": int(len(near)), "holes": 0}
    flat = near[_normal_y(cols, near) > 0.8]
    if len(flat) < 500:
        return np.zeros(0, np.int64), {"floorSplats": int(len(flat)), "holes": 0}
    x, z = cols["x"][flat], cols["z"][flat]
    op = 1.0 / (1.0 + np.exp(-cols["opacity"][flat]))
    x0, z0 = np.percentile(x, 0.5) - CELL, np.percentile(z, 0.5) - CELL
    W = int(np.ceil((np.percentile(x, 99.5) - x0) / CELL)) + 2
    H = int(np.ceil((np.percentile(z, 99.5) - z0) / CELL)) + 2
    if W * H > 16_000_000:
        return np.zeros(0, np.int64), {"floorSplats": int(len(flat)), "holes": 0, "why": "floor too large for the raster"}
    i = np.clip(((x - x0) / CELL).astype(int), 0, W - 1)
    j = np.clip(((z - z0) / CELL).astype(int), 0, H - 1)
    k = j * W + i
    cover = np.bincount(k, weights=op, minlength=W * H)
    covered = cover >= MIN_COVER
    # holes: uncovered cells not connected (4-neighbour) to the raster's border through other uncovered cells
    import cv2
    unc = (~covered).reshape(H, W).astype(np.uint8)
    n, lab = cv2.connectedComponents(unc, connectivity=4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist())
    sizes = np.bincount(lab.ravel(), minlength=n) * CELL * CELL
    holes = np.zeros(H * W, bool)
    n_holes = 0
    for c in range(1, n):
        if c in border or not (MIN_HOLE <= sizes[c] <= MAX_HOLE):
            continue
        holes |= (lab == c).ravel()
        n_holes += 1
    if not holes.any():
        return np.zeros(0, np.int64), {"floorSplats": int(len(flat)), "holes": 0}
    # nearest covered cell for every hole cell: OpenCV numbers each covered (zero) cell in scan order and tells every
    # other cell which one is nearest
    _, near_lab = cv2.distanceTransformWithLabels(unc, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    zeros = np.flatnonzero(unc.ravel() == 0)
    donor = zeros[np.clip(near_lab.ravel() - 1, 0, len(zeros) - 1)]
    order = np.argsort(k)
    starts = np.searchsorted(k[order], np.arange(W * H))
    ends = np.searchsorted(k[order], np.arange(W * H), side="right")
    src, off = [], []
    for h in np.flatnonzero(holes):
        d = donor[h]
        members = flat[order[starts[d]:ends[d]]]
        if not len(members):
            continue
        src.append(members)
        off.append(np.repeat([[(h % W - d % W) * CELL, (h // W - d // W) * CELL]], len(members), axis=0))
    if not src:
        return np.zeros(0, np.int64), {"floorSplats": int(len(flat)), "holes": n_holes}
    src_i, off_a = np.concatenate(src), np.concatenate(off)
    info = {"floorSplats": int(len(flat)), "holes": n_holes, "filledM2": round(float(holes.sum()) * CELL * CELL, 2), "added": int(len(src_i))}
    return src_i, {**info, "_off": off_a}


def fill_floors(src: Path, dest: Path, floors: list[float] | None = None) -> dict:
    """Write src with every floor's holes filled to dest. Returns what was done per floor."""
    header, cols = read_ply(src)
    floors = floors or [0.0]
    add_idx, add_off, per = [], [], []
    for fy in floors:
        idx, info = plan(cols, fy)
        off = info.pop("_off", None)
        per.append({"floorY": fy, **info})
        if len(idx):
            add_idx.append(idx)
            add_off.append(off)
    if not add_idx:
        return {"floors": per, "added": 0}
    idx, off = np.concatenate(add_idx), np.concatenate(add_off)
    if len(idx) > MAX_ADD:
        keep = np.random.default_rng(0).choice(len(idx), MAX_ADD, replace=False)
        idx, off = idx[keep], off[keep]
    new = {p: v[idx].copy() for p, v in cols.items()}
    new["x"] = new["x"] + off[:, 0].astype(np.float32)
    new["z"] = new["z"] + off[:, 1].astype(np.float32)
    total = len(cols["x"]) + len(idx)
    out_header = "\n".join(f"element vertex {total}" if ln.startswith("element vertex") else ln for ln in header.splitlines()) + "\n"
    with Path(dest).open("wb") as f:
        f.write(out_header.encode("latin-1"))
        np.stack([cols[p] for p in cols], axis=1).astype(np.float32).tofile(f)
        np.stack([new[p] for p in cols], axis=1).astype(np.float32).tofile(f)
    return {"floors": per, "added": int(len(idx))}
