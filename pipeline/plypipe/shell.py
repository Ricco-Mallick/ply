"""Measured shell: wall segments, openings and object boxes, the editable layer under the splats.

A Gaussian twin is a photograph you can walk through; it cannot tell you how wide the room is, where the door is, or
which box is the bench. This module recovers that from the solved point cloud: RANSAC lines through the mid-height
band give wall planes, the SAM 3 boxes are attached to the nearest wall as openings, and every labelled object keeps
its measured extents. The viewer draws the result as a measured layer, and an editor can select from it.

Frame: the scene frame (metres, +Y up, floor at y=0), the same one the splats are trained in.
"""

from __future__ import annotations

import math

import numpy as np

BAND = (0.35, 2.35)          # height band that holds walls, not floor or ceiling
TOL = 0.06                   # a point is on a wall within 6 cm
MIN_SUPPORT = 0.02           # a wall needs this share of the band's points (a splat cloud is mostly furniture)
MIN_LEN = 0.9                # metres; shorter fits are furniture edges, not walls
MAX_WALLS = 24
OPENING_KINDS = {"door", "window", "opening", "archway", "gate"}


def _line(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Unit direction, unit normal and offset for the line through p and q (in the xz plane)."""
    d = q - p
    n = math.hypot(float(d[0]), float(d[1]))
    d = d / (n or 1.0)
    normal = np.array([-d[1], d[0]])
    return d, normal, float(normal @ p)


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points (Andrew's monotone chain), counter-clockwise."""
    pts = np.unique(np.round(points, 3), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _corners(poly: np.ndarray, tol_deg: float = 9.0) -> np.ndarray:
    """Split a closed polygon where it actually turns: a hull edge is a wall, a run of collinear vertices is not."""
    n = len(poly)
    if n < 3:
        return poly
    turn = []
    for i in range(n):
        a, b, c = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        d1, d2 = b - a, c - b
        n1, n2 = np.linalg.norm(d1) or 1, np.linalg.norm(d2) or 1
        cos = float(np.clip((d1 @ d2) / (n1 * n2), -1, 1))
        turn.append(math.degrees(math.acos(cos)))
    keep = [i for i, t in enumerate(turn) if t > tol_deg]
    if len(keep) < 3:
        return poly[keep] if len(keep) >= 3 else poly
    return poly[keep]


def fit_walls(pts: np.ndarray, floor_y: float = 0.0, *, band: tuple[float, float] = BAND,
              tol: float = TOL, rng: np.random.Generator | None = None) -> list[dict]:
    """The room's outline as wall segments.

    A room's perimeter is the convex hull of its floor plan — furniture rows and aisle ends sit inside it, so fitting
    lines through the point cloud finds desks before walls (tried: it produced eleven 11 m "walls" down a classroom).
    The hull, simplified, is the outline; each long edge becomes a wall with its length and inward normal.
    """
    rng = rng or np.random.default_rng(7)
    if pts.size == 0:
        return []
    y = pts[:, 1] - floor_y
    band_pts = pts[(y > band[0]) & (y < band[1])][:, [0, 2]]
    if len(band_pts) < 200:
        return []
    # a splat cloud has stray splats a long way out (the hull then spans hundreds of metres): work inside the
    # 1st-99th percentile box, which is where the room actually is
    lo = np.percentile(band_pts, 1.0, axis=0)
    hi = np.percentile(band_pts, 99.0, axis=0)
    inside = np.all((band_pts >= lo) & (band_pts <= hi), axis=1)
    if inside.sum() >= 200:
        band_pts = band_pts[inside]
    if len(band_pts) > 60_000:  # the hull is decided by the extremes: a sample is enough and is 10x faster
        band_pts = band_pts[rng.choice(len(band_pts), size=60_000, replace=False)]

    hull = _hull(band_pts)
    if len(hull) < 3:
        return []
    poly = _corners(hull)
    poly = np.vstack([poly, poly[:1]])  # close the ring so each edge is consecutive
    centre = band_pts.mean(axis=0)

    walls: list[dict] = []
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        length = float(np.linalg.norm(b - a))
        if length < MIN_LEN:
            continue
        d = (b - a) / length
        n = np.array([-d[1], d[0]])
        mid = (a + b) / 2
        if float(n @ (centre - mid)) < 0:  # normals point into the room
            n = -n
        walls.append({
            "a": [round(float(a[0]), 3), round(float(a[1]), 3)],
            "b": [round(float(b[0]), 3), round(float(b[1]), 3)],
            "length": round(length, 3),
            "normal": [round(float(n[0]), 3), round(float(n[1]), 3)],
            "support": int(len(band_pts)),
        })
    walls.sort(key=lambda w: -w["length"])
    for w in walls[:MAX_WALLS]:
        w["angle"] = round(math.degrees(math.atan2(w["b"][1] - w["a"][1], w["b"][0] - w["a"][0])), 1)
    return walls[:MAX_WALLS]


def _merge(walls: list[dict], tol: float = 0.12, angle_tol: float = 4.0) -> list[dict]:
    """Join collinear pieces that overlap or nearly touch (a wall broken by a door comes back as one)."""
    out: list[dict] = []
    for w in sorted(walls, key=lambda x: -x["length"]):
        for o in out:
            # only genuinely collinear pieces join: two parallel walls 6 m apart are not one wall
            if _collinear(o, w, tol, angle_tol):
                o.update(_join(o, w))
                break
        else:
            out.append(dict(w))
    return out


def _angle(w: dict) -> float:
    return math.degrees(math.atan2(w["b"][1] - w["a"][1], w["b"][0] - w["a"][0])) % 180


def _collinear(o: dict, w: dict, tol: float, angle_tol: float) -> bool:
    if abs(_angle(o) - _angle(w)) > angle_tol:
        return False
    a = np.array(o["a"])
    d = np.array(o["b"]) - a
    d = d / (np.linalg.norm(d) or 1)
    n = np.array([-d[1], d[0]])
    return abs(float(n @ (np.array(w["a"]) - a))) < tol and abs(float(n @ (np.array(w["b"]) - a))) < tol


def _join(o: dict, w: dict) -> dict:
    a = np.array(o["a"])
    d = np.array(o["b"]) - a
    d = d / (np.linalg.norm(d) or 1)
    ts = [float(d @ (np.array(p) - a)) for p in (o["a"], o["b"], w["a"], w["b"])]
    a0, a1 = min(ts), max(ts)
    pa, pb = a + d * a0, a + d * a1
    return {"a": [round(float(pa[0]), 3), round(float(pa[1]), 3)], "b": [round(float(pb[0]), 3), round(float(pb[1]), 3)],
            "length": round(a1 - a0, 3), "normal": o["normal"], "support": o["support"] + w["support"]}


def _orient(walls: list[dict], pts: np.ndarray) -> list[dict]:
    """Point each normal at the nearest open space (the camera side), not into the wall."""
    c = np.median(pts, axis=0) if len(pts) else np.zeros(2)
    for w in walls:
        a = np.array(w["a"])
        n = np.array(w["normal"])
        mid = (a + np.array(w["b"])) / 2
        if float(n @ (c - mid)) < 0:
            w["normal"] = [round(float(-n[0]), 3), round(float(-n[1]), 3)]
    return walls


def openings(boxes: list[dict], walls: list[dict], tol: float = 0.45) -> list[dict]:
    """Attach labelled doors and windows to the wall they sit in: a span along the wall and a height range."""
    out: list[dict] = []
    for b in boxes:
        kind = str(b.get("cls") or "").lower()
        if kind not in OPENING_KINDS or not walls:
            continue
        p = np.array(b.get("p") or [0, 0, 0], dtype=float)
        best = None
        for index, w in enumerate(walls):
            a = np.array(w["a"], dtype=float)
            d = np.array(w["b"], dtype=float) - a
            length = np.linalg.norm(d) or 1.0
            u = d / length
            t = float(u @ (p[[0, 2]] - a))
            t = min(max(t, 0.0), length)
            foot = a + u * t
            dist = float(np.linalg.norm(p[[0, 2]] - foot))
            if best is None or dist < best[0]:
                best = (dist, index, t, length)
        if best is None or best[0] > tol:
            continue
        dist, index, t, length = best
        half = float((b.get("half") or [0.5, 1.0, 0.2])[0])
        out.append({
            "kind": kind, "wall": index, "u": round(t, 3), "width": round(min(2 * half, length), 3),
            "sill": round(max(0.0, float(p[1]) - float((b.get("half") or [0.5, 1.0, 0.2])[1])), 3),
            "height": round(2 * float((b.get("half") or [0.5, 1.0, 0.2])[1]), 3),
            "name": b.get("name"),
        })
    return out


def build(pts: np.ndarray, floor_y: float, ceiling_y: float, boxes: list[dict]) -> dict:
    walls = fit_walls(pts, floor_y)
    return {
        "version": 1,
        "frame": "scene, metres, +Y up, floor at 0",
        "floorY": round(float(floor_y), 3),
        "ceilingY": round(float(ceiling_y), 3),
        "walls": walls,
        "openings": openings(boxes, walls),
        "objects": [
            {"id": b.get("id"), "name": b.get("name"), "kind": b.get("cls"),
             "center": [round(float(v), 3) for v in (b.get("p") or [0, 0, 0])],
             "half": [round(float(v), 3) for v in (b.get("half") or [0.2, 0.2, 0.2])]}
            for b in boxes
        ],
    }
