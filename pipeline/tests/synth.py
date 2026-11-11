"""Synthetic indoor capture for CPU tests: a 6 x 4 m room, 2.8 m walls, a table and a sofa,
cameras walking a loop at 1.45 m. Written as a COLMAP text model in an arbitrary similarity
frame (random rotation, scale, translation), like a real SfM result."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plypipe import colmap_txt, geometry

ROOM = (6.0, 4.0, 2.8)  # x, z, height
# beside the walk, not under it: the occupancy grid trusts the walked corridor (stages/occupancy.py)
TABLE = (2.0, 1.45, 3.2, 2.25, 0.75)  # x0, z0, x1, z1, top
SOFA = (0.4, 2.9, 2.4, 3.6, 0.8)


def room_points(rng: np.random.Generator, n_floor: int = 2500, n_wall: int = 6000, noise: float = 0.01) -> np.ndarray:
    W, D, H = ROOM
    pts = [np.c_[rng.uniform(0, W, n_floor), np.zeros(n_floor), rng.uniform(0, D, n_floor)]]
    per = n_wall // 4
    for (x0, z0, x1, z1) in [(0, 0, W, 0), (W, 0, W, D), (W, D, 0, D), (0, D, 0, 0)]:
        t = rng.uniform(0, 1, per)
        pts.append(np.c_[x0 + (x1 - x0) * t, rng.uniform(0, H, per), z0 + (z1 - z0) * t])
    for (x0, z0, x1, z1, top) in (TABLE, SOFA):
        n = 900
        top_pts = np.c_[rng.uniform(x0, x1, n), np.full(n, top), rng.uniform(z0, z1, n)]
        side = np.c_[rng.uniform(x0, x1, n // 2), rng.uniform(0, top, n // 2), np.where(rng.random(n // 2) < .5, z0, z1)]
        pts += [top_pts, side]
    # ceiling (must never be mistaken for the floor)
    pts.append(np.c_[rng.uniform(0, W, 1500), np.full(1500, H), rng.uniform(0, D, 1500)])
    p = np.concatenate(pts)
    return p + rng.normal(0, noise, p.shape)


def camera_loop(n: int = 60, inward: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Centres on an ellipse inside the room at 1.45 m and cam_from_world rotations, camera y axis
    pointing down (upright phone). Looking outward-ish at the walls, or (inward=True) across the
    room like a real walkthrough."""
    W, D, _ = ROOM
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centers = np.c_[W / 2 + 1.6 * np.cos(ang), np.full(n, 1.45), D / 2 + 0.9 * np.sin(ang)]
    rots = []
    for a, c in zip(ang, centers):
        look = a + (np.pi + 0.5 if inward else 0.6)
        fwd = np.array([np.cos(look), -0.25 if inward else -0.15, np.sin(look)])
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 1, 0])  # y-up world
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        # rows = camera axes expressed in world: x right, y down, z forward
        R_cw = np.stack([right, down, fwd])
        if R_cw[1] @ np.array([0, 1, 0]) > 0:  # y must point down
            R_cw[0] *= -1
            R_cw[1] *= -1
        rots.append(R_cw)
    return centers, np.array(rots)


def random_similarity(rng: np.random.Generator, scale: float = 0.37) -> np.ndarray:
    q = rng.normal(size=4)
    R = geometry.quat_to_mat(q / np.linalg.norm(q))
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = rng.uniform(-5, 5, 3)
    return T


def write_synthetic_model(d: Path, seed: int = 1, scale: float = 0.37, n_cams: int = 60) -> dict:
    rng = np.random.default_rng(seed)
    pts = room_points(rng)
    centers, rots = camera_loop(n_cams)
    T = random_similarity(rng, scale)
    pts_w = geometry.apply(T, pts)
    images = []
    for i, (c, Rcw) in enumerate(zip(centers, rots)):
        t = -Rcw @ c
        q, t2 = geometry.transform_pose(geometry.mat_to_quat(Rcw), t, T)
        images.append(colmap_txt.Image(i + 1, q, t2, 1, f"{i:06d}.jpg", ""))
    n = pts_w.shape[0]
    tails = [f"128 128 128 0.5 {1 + i % n_cams} 0 {1 + (i + 1) % n_cams} 1 {1 + (i + 2) % n_cams} 2" for i in range(n)]
    model = colmap_txt.Model(["# Camera list", "1 PINHOLE 1600 1200 1200 1200 800 600"], images,
                             np.arange(1, n + 1), pts_w, np.full((n, 3), 128, np.uint8), np.full(n, 0.5),
                             np.full(n, 3, np.int32), tails)
    colmap_txt.write_model(model, d)
    return {"T_true": T, "points_room": pts, "centers_room": centers}


# Two 4 x 4 m rooms side by side, joined by a 1 m doorway through a plain partition (no texture, so no points).
# The partition is 0.25 m thick: a 10 cm grid cannot keep a thinner untextured wall, because floor points from both
# sides land in the cells it spans.
TWO_ROOMS = (8.25, 4.0, 2.8)      # x, z, height
PARTITION = (4.0, 4.25)           # x0, x1 of the partition wall
DOOR = (1.6, 2.6)                 # z0, z1 of the doorway


def two_rooms_model(seed: int = 3, strays: tuple = ()) -> tuple[colmap_txt.Model, dict]:
    """An aligned model (scene frame: metres, floor at y = 0) of the two rooms, walked through the doorway.
    Each point is observed by the three nearest cameras that can see it (no sight line through the partition).
    `strays` are extra (x, y, z) points, observed like the rest."""
    rng = np.random.default_rng(seed)
    W, D, H = TWO_ROOMS
    p0, p1 = PARTITION
    floor = np.c_[rng.uniform(0, W, 4000), np.zeros(4000), rng.uniform(0, D, 4000)]
    floor = floor[(floor[:, 0] < p0) | (floor[:, 0] > p1)]
    walls = []
    for (x0, z0, x1, z1) in [(0, 0, W, 0), (W, 0, W, D), (W, D, 0, D), (0, D, 0, 0)]:
        t = rng.uniform(0, 1, 1500)
        walls.append(np.c_[x0 + (x1 - x0) * t, rng.uniform(0, H, 1500), z0 + (z1 - z0) * t])
    pts = np.concatenate([floor, *walls])
    pts[:, [0, 2]] += rng.normal(0, 0.01, (len(pts), 2))
    if len(strays):
        pts = np.concatenate([pts, np.asarray(strays, float).reshape(-1, 3)])

    # out through the doorway and back along the far wall of the second room, at 1.45 m
    route = np.array([[1.0, 0.8], [3.0, 0.8], [3.0, 2.1], [5.2, 2.1], [7.0, 2.1], [7.0, 3.2], [5.0, 3.2]])
    seg = []
    for a, b in zip(route[:-1], route[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.15)))
        seg.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    xz = np.concatenate(seg + [route[-1:]])
    centers = np.c_[xz[:, 0], np.full(len(xz), 1.45), xz[:, 1]]
    images = []
    for i, c in enumerate(centers):
        nxt = centers[min(i + 1, len(centers) - 1)] - centers[max(i - 1, 0)]
        fwd = np.array([nxt[0], -0.2, nxt[2]])
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 1, 0])
        right /= np.linalg.norm(right)
        R_cw = np.stack([right, np.cross(fwd, right), fwd])  # rows: camera x, y, z axes in the world
        if R_cw[1] @ np.array([0, 1, 0]) > 0:  # y must point down
            R_cw[:2] *= -1
        images.append(colmap_txt.Image(i + 1, geometry.mat_to_quat(R_cw), -R_cw @ c, 1, f"{i:06d}.jpg", ""))

    tails = []
    for p in pts:
        seen = np.flatnonzero(_clear_lines(centers, p))
        near = seen[np.argsort(np.linalg.norm(centers[seen] - p, axis=1))[:3]]
        tails.append("128 128 128 0.5 " + " ".join(f"{k + 1} 0" for k in near))
    n = len(pts)
    model = colmap_txt.Model(["# Camera list", "1 PINHOLE 1600 1200 1200 1200 800 600"], images, np.arange(1, n + 1),
                             pts, np.full((n, 3), 128, np.uint8), np.full(n, 0.5), np.full(n, 3, np.int32), tails)
    return model, {"centers": centers}


def _clear_lines(centers: np.ndarray, p: np.ndarray, n: int = 40) -> np.ndarray:
    """Which cameras see p without the sight line passing through the partition beside the doorway."""
    t = np.linspace(0.02, 0.98, n)[None, :, None]
    s = centers[:, None, :] + (p - centers)[:, None, :] * t
    x, z = s[..., 0], s[..., 2]
    solid = (x > PARTITION[0]) & (x < PARTITION[1]) & ((z < DOOR[0]) | (z > DOOR[1]))
    return ~solid.any(axis=1)
