"""Synthetic indoor capture for CPU tests: a 6 x 4 m room, 2.8 m walls, a table and a sofa,
cameras walking a loop at 1.45 m. Written as a COLMAP text model in an arbitrary similarity
frame (random rotation, scale, translation), like a real SfM result."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plypipe import colmap_txt, geometry

ROOM = (6.0, 4.0, 2.8)  # x, z, height
TABLE = (2.0, 1.0, 3.2, 1.8, 0.75)  # x0, z0, x1, z1, top
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
