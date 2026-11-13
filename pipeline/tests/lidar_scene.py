"""A laser-scanned room for CPU tests: synth.py's 6 x 4 x 2.8 m room with its table and sofa as solid boxes, sampled on
every surface a scanner standing at two stations can see (the far sides of the boxes and the floor behind them stay
empty: the scan's shadows), with 2 mm of range noise. Every point keeps the id and the unit normal of its surface."""

from __future__ import annotations

import numpy as np

from synth import ROOM, SOFA, TABLE

STATIONS = np.array([[1.2, 1.4, 1.0], [4.8, 1.4, 3.2]])


def _boxes() -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.array([x0, 0.0, z0]), np.array([x1, top, z1])) for (x0, z0, x1, z1, top) in (TABLE, SOFA)]


def _faces() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """(origin, edge u, edge v, unit normal pointing into the room) of every surface."""
    W, D, H = ROOM
    up = np.array([0.0, 1, 0])
    f = [(np.zeros(3), np.array([W, 0, 0]), np.array([0, 0, D]), up),
         (np.array([0, H, 0.0]), np.array([W, 0, 0]), np.array([0, 0, D]), -up),
         (np.zeros(3), np.array([W, 0, 0]), np.array([0, H, 0]), np.array([0, 0, 1.0])),
         (np.array([0, 0, D]), np.array([W, 0, 0]), np.array([0, H, 0]), np.array([0, 0, -1.0])),
         (np.zeros(3), np.array([0, 0, D]), np.array([0, H, 0]), np.array([1.0, 0, 0])),
         (np.array([W, 0, 0.0]), np.array([0, 0, D]), np.array([0, H, 0]), np.array([-1.0, 0, 0]))]
    for lo, hi in _boxes():
        d = hi - lo
        ex, ey, ez = (np.eye(3)[i] * d[i] for i in range(3))
        f += [(lo + ey, ex, ez, up),
              (lo, ex, ey, np.array([0, 0, -1.0])), (lo + np.array([0, 0, d[2]]), ex, ey, np.array([0, 0, 1.0])),
              (lo, ez, ey, np.array([-1.0, 0, 0])), (lo + np.array([d[0], 0, 0]), ez, ey, np.array([1.0, 0, 0]))]
    return f


def _blocked(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For segments a[i] -> b[i]: does one of the boxes lie between the ends (slab test)?"""
    hit = np.zeros(len(a), bool)
    d = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo, hi in _boxes():
            t0, t1 = (lo - a) / d, (hi - a) / d
            tn, tf = np.minimum(t0, t1), np.maximum(t0, t1)
            tn, tf = np.where(d == 0, np.where((a >= lo) & (a <= hi), -np.inf, np.inf), tn), np.where(d == 0, np.where((a >= lo) & (a <= hi), np.inf, -np.inf), tf)
            near, far = tn.max(1), tf.min(1)
            hit |= (near < far) & (far > 0) & (near < 1)
    return hit


def colour_at(p: np.ndarray) -> np.ndarray:
    """A smooth colour field over the room, uint8: red along x, green up, blue along z."""
    return np.clip(np.stack([p[:, 0] / 6, p[:, 1] / 2.8, p[:, 2] / 4], 1) * 255, 0, 255).astype(np.uint8)


def scanned_room(n: int = 300_000, seed: int = 0, noise: float = 0.002) -> dict:
    """points [N, 3] (y up, the room's own frame), colors, normals (the true surface normal of each point), face ids, stations,
    and `edge`: points within 6 cm of another surface, whose normal is not well defined."""
    rng = np.random.default_rng(seed)
    faces = _faces()
    area = np.array([np.linalg.norm(np.cross(u, v)) for _, u, v, _ in faces])
    pick = rng.choice(len(faces), int(n * 2.2), p=area / area.sum())  # oversampled: about half are hidden from both stations
    uv = rng.random((len(pick), 2))
    o, u, v, nrm = (np.array([f[i] for f in faces])[pick] for i in range(4))
    p = o + uv[:, :1] * u + uv[:, 1:] * v
    seen = np.zeros(len(p), bool)
    for s in STATIONS:
        toward = np.einsum("ij,ij->i", nrm, s - p) > 0
        seen |= toward & ~_blocked(np.broadcast_to(s, p.shape), p + 0.005 * nrm)
    p, nrm, face = p[seen], nrm[seen], pick[seen]
    keep = rng.choice(len(p), min(n, len(p)), replace=False)
    p, nrm, face = p[keep], nrm[keep], face[keep]
    p = p + nrm * rng.normal(0, noise, (len(p), 1))
    from scipy.spatial import cKDTree
    d, idx = cKDTree(p).query(p, k=40)
    edge = ((face[idx] != face[:, None]) & (d <= 0.06)).any(1)
    return {"points": p, "colors": colour_at(p), "normals": nrm, "face": face, "stations": STATIONS.copy(), "edge": edge}


def to_file_frame(scan: dict, z_up: bool = True, flip: bool = False, yaw_deg: float = 23.0, shift=(1234.5, -567.25, 12.0)) -> dict:
    """The scan as a file would hold it: z up (or y up with z_up=False), upside down with flip, turned by yaw about up, far from the origin."""
    def f(a: np.ndarray, translate: bool) -> np.ndarray:
        q = a.copy()
        if flip:
            q[:, 1:] = -q[:, 1:]  # a half turn about x: upside down, still a proper rotation
        if z_up:
            q = np.stack([q[:, 0], -q[:, 2], q[:, 1]], 1)  # y up -> z up, a proper rotation
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        up = 2 if z_up else 1
        i, j = [k for k in range(3) if k != up]
        r = q.copy()
        r[:, i], r[:, j] = c * q[:, i] - s * q[:, j], s * q[:, i] + c * q[:, j]
        return r + np.array(shift) if translate else r
    return {**scan, "points": f(scan["points"], True), "stations": f(scan["stations"], True), "normals": f(scan["normals"], False)}
