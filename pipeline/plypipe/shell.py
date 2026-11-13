# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
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


# ---------------------------------------------------------------------------------------------------------------------
# The layout from dense geometry (IMPROVEMENT-PLAN 2.5, variant 4b): plypipe.dense's confirmed points and normals, the
# camera path, and optional door/window detections. Same file and fields as build(), plus `source`, `validation` and
# `rejected`; a layout that fails its checks is never the one the viewer is told to draw.

FLAT = 0.85                  # |normal.y| above this: a floor, ceiling or table top; under HORIZONTAL: a wall
HORIZONTAL = 0.5
ABOVE_FURNITURE = 1.1        # the wall band starts here (a sofa back is 0.9 m, a table 0.75 m) and ends 0.3 m under the ceiling
BAND_TOP = 3.0               # and never higher than this above the floor: chapel walls are 8 m and need no more
HOUGH = (2.0, 0.04)          # degrees and metres per bin
SNAP_DEG, SNAP_MOVE = 5.0, 0.2
MERGE_DEG, MERGE_OFFSET = 4.0, 0.15   # the two faces of one partition, seen from either room, are one wall
GAP = 0.3                    # a hole in a wall shorter than this is noise
BIN = 0.05                   # along a wall, for support
DOOR_MIN_WIDTH, DOOR_MIN_HEIGHT, OPENING_MIN = 0.7, 1.9, 0.5
SOLID = 0.5                  # the share of a wall slab's cells that holds a point, above which it is solid
DETECTION_VIEWS = 3
DETECTION_REACH = 0.45       # how far from the wall plane a detection's centre may be and still sit in it
MAX_DENSE_WALLS = 64


def _fit_line(xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unit direction of the total-least-squares line through 2D points."""
    c = xz.mean(0)
    d = xz - c
    return c, np.linalg.eigh(d.T @ d)[1][:, 1]


def _left(d: np.ndarray) -> np.ndarray:
    return np.array([-d[1], d[0]])


def _levels(points: np.ndarray, normals: np.ndarray, cam_y: float, floor_y: float | None = None) -> dict:
    """Floor and ceiling heights: the fullest 5 cm slab of up- or downward-facing points near y = 0 (the aligned frame's
    floor), and the highest slab above the cameras holding at least a quarter of what the fullest does (a lamp or a
    loft edge is a slab too; only the ceiling is a plane the whole room shares)."""
    y, flat = points[:, 1], np.abs(normals[:, 1]) > FLAT

    def slabs(lo, hi):
        sel = flat & (y >= lo) & (y < hi)
        if sel.sum() < 50:
            return None, None
        edges = np.arange(lo, hi + BIN, BIN)
        return np.convolve(np.histogram(y[sel], edges)[0].astype(float), np.ones(3), "same"), edges

    def level(h, e, i):
        return float(np.median(y[flat & (y >= e[i] - BIN) & (y < e[i + 1] + BIN)]))

    floor = 0.0 if floor_y is None else float(floor_y)
    h, e = slabs(-0.6, 0.6) if floor_y is None else (None, None)
    if h is not None:
        floor = level(h, e, int(np.argmax(h)))
    lo, hi = max(cam_y + 0.3, floor + 1.8), float(y.max()) + BIN
    ceiling, seen = float(np.percentile(y, 99.0)), False
    h, e = slabs(lo, hi) if hi > lo + BIN else (None, None)
    if h is not None:
        ceiling, seen = level(h, e, int(np.flatnonzero(h >= 0.25 * h.max())[-1])), True
    return {"floorY": floor, "ceilingY": ceiling, "ceilingSeen": seen}


def _peak(xz: np.ndarray, phi: np.ndarray) -> tuple[float, float] | None:
    """The strongest line in a Hough accumulator over (normal angle, offset). Each point votes for the angles within 6
    degrees of its own normal's, so the lines follow the normals and a noisy normal still lands. (angle, offset)."""
    step, res = HOUGH
    n_theta = int(round(360 / step))
    ks = np.round(np.degrees(phi) / step).astype(np.int64)[:, None] + np.arange(-3, 4)[None]
    theta = np.radians(ks * step)
    r = np.floor((xz[:, 0:1] * np.cos(theta) + xz[:, 1:2] * np.sin(theta)) / res).astype(np.int64)
    r0 = int(r.min())
    width = int(r.max()) - r0 + 3
    acc = np.bincount(((ks % n_theta) * width + (r - r0 + 1)).ravel(), minlength=n_theta * width).reshape(n_theta, width).astype(float)
    acc = acc + np.roll(acc, 1, 1) + np.roll(acc, -1, 1)  # a wall with noise straddles two bins
    t, b = np.unravel_index(int(np.argmax(acc)), acc.shape)
    return (float(np.radians(t * step)), float((b + r0 - 1 + 0.5) * res)) if acc[t, b] >= 1 else None


def extract_lines(xz: np.ndarray, hn: np.ndarray, min_points: int, max_lines: int = MAX_DENSE_WALLS, rng: np.random.Generator | None = None,
                  tol: float = 0.07, sample: int = 150_000) -> list[dict]:
    """Lines through wall points, strongest first: a Hough peak on a sample, then total-least-squares refits on the
    inliers (within `tol`, normal within 25 degrees of the line's), which are taken out. The two faces of one thin wall
    come back as one line. Each is {c, d (unit direction), n (its left normal), idx (inlier indices)}."""
    rng = rng or np.random.default_rng(7)
    phi = np.arctan2(hn[:, 1], hn[:, 0]) % (2 * np.pi)
    alive = np.ones(len(xz), bool)
    lines: list[dict] = []
    for _ in range(max_lines):
        pool = np.flatnonzero(alive)
        if len(pool) < min_points:
            break
        peak = _peak(*(a[pool if len(pool) <= sample else rng.choice(pool, sample, replace=False)] for a in (xz, phi)))
        if peak is None:
            break
        n = np.array([np.cos(peak[0]), np.sin(peak[0])])
        c, ok = n * peak[1], np.zeros(len(pool), bool)
        for k in range(3):  # refit until the inliers follow the wall and not the accumulator's bin
            ok = (np.abs((xz[pool] - c) @ n) < tol) & (np.abs(hn[pool] @ n) > 0.9)
            if ok.sum() < min_points:
                break
            c, d = _fit_line(xz[pool[ok]])
            n = _left(d)
        if ok.sum() < min_points:  # a peak of clutter: spend its points so the next peak is another
            alive[pool[ok]] = False
            if not ok.any():
                break
            continue
        alive[pool[ok]] = False
        lines.append({"c": c, "d": d, "n": n, "idx": pool[ok]})
    return _merge_lines(lines, xz)


def _merge_lines(lines: list[dict], xz: np.ndarray) -> list[dict]:
    """Parallel lines less than MERGE_OFFSET apart are the faces of one wall: refit on both."""
    out: list[dict] = []
    for ln in lines:
        for o in out:
            if math.degrees(math.acos(min(1.0, abs(float(o["n"] @ ln["n"]))))) < MERGE_DEG and abs(float(o["n"] @ (ln["c"] - o["c"]))) < MERGE_OFFSET:
                idx = np.concatenate([o["idx"], ln["idx"]])
                c, d = _fit_line(xz[idx])
                o.update(c=c, d=d, n=_left(d), idx=idx)
                break
        else:
            out.append(dict(ln))
    return out


def _runs(t: np.ndarray) -> list[tuple[float, float]]:
    """Where along a wall there is support: runs of 5 cm bins holding at least 15% of the median occupied bin (and 3
    points), holes shorter than GAP bridged. (start, end) in metres along the line."""
    lo = float(t.min())
    count = np.bincount(np.floor((t - lo) / BIN).astype(np.int64))
    full = count >= max(3.0, 0.15 * float(np.median(count[count > 0])))
    runs: list[list[int]] = []
    start = None
    for i, f in enumerate(np.append(full, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if runs and (start - runs[-1][1]) * BIN < GAP:
                runs[-1][1] = i
            else:
                runs.append([start, i])
            start = None
    return [(lo + a * BIN, lo + b * BIN) for a, b in runs]


def _junction(lines: list[dict], i: int, xz: np.ndarray, g0: float, g1: float, reach: float = 0.5) -> bool:
    """Whether another wall line meets line i inside the gap g0..g1 (give or take 0.2 m), with points of its own near the
    meeting: support thins out where a partition ends on a wall, and that is a corner, not a hole."""
    c, d = lines[i]["c"], lines[i]["d"]
    for j, o in enumerate(lines):
        den = float(d @ o["n"])
        if j == i or abs(den) < 0.5:
            continue
        t = float((o["c"] - c) @ o["n"]) / den
        if g0 - 0.2 <= t <= g1 + 0.2 and (np.linalg.norm(xz[o["idx"]] - (c + d * t), axis=1) < reach).any():
            return True
    return False


def slab_fill(points: np.ndarray, c: np.ndarray, d: np.ndarray, t0: float, t1: float, y0: float, y1: float, thick: float = 0.08) -> float:
    """The share of the 5 cm cells of the wall plane between t0 and t1 along the line and y0..y1 in height that hold a
    point within `thick` of it: about 1 on a solid wall, about 0 in a doorway."""
    xz = points[:, [0, 2]] - c
    sel = (np.abs(xz @ _left(d)) < thick) & (points[:, 1] >= y0) & (points[:, 1] < y1)
    u = xz[sel] @ d
    sel2 = (u >= t0) & (u < t1)
    nu, ny = max(1, int(math.ceil((t1 - t0) / BIN))), max(1, int(math.ceil((y1 - y0) / BIN)))
    if not sel2.any():
        return 0.0
    cell = np.clip(np.floor((u[sel2] - t0) / BIN).astype(np.int64), 0, nu - 1) * ny + np.clip(np.floor((points[sel, 1][sel2] - y0) / BIN).astype(np.int64), 0, ny - 1)
    return len(np.unique(cell)) / (nu * ny)


def _header(points: np.ndarray, c: np.ndarray, d: np.ndarray, t0: float, t1: float, floor: float, ceiling: float) -> float:
    """Height of the top of a doorway: the lowest 5 cm row above 0.3 m where the wall plane over the gap is at least half
    full. The whole height when nothing closes it."""
    xz = points[:, [0, 2]] - c
    u = xz @ d
    sel = (np.abs(xz @ _left(d)) < 0.08) & (u >= t0 + 0.1) & (u < t1 - 0.1) & (points[:, 1] > floor + 0.3) & (points[:, 1] < ceiling)
    cols = max(1, int((t1 - t0 - 0.2) / BIN))
    for row in np.arange(floor + 0.3, ceiling, BIN):
        r = sel & (points[:, 1] >= row) & (points[:, 1] < row + BIN)
        if r.any() and len(np.unique(np.floor((u[r] - t0 - 0.1) / BIN))) >= 0.3 * cols:
            return float(row - floor)
    return float(ceiling - floor)


def _crossings(path: np.ndarray, c: np.ndarray, d: np.ndarray, max_step: float = 1.0) -> np.ndarray:
    """Positions along the line c + t d where the camera path (N, 3) crosses it. A step longer than max_step is a jump
    between clips, not a walk."""
    if len(path) < 2:
        return np.zeros(0)
    p = path[:, [0, 2]]
    s = (p - c) @ _left(d)
    hit = (s[:-1] * s[1:] < 0) & (np.linalg.norm(p[1:] - p[:-1], axis=1) < max_step)
    f = s[:-1][hit] / (s[:-1][hit] - s[1:][hit])
    return (p[:-1][hit] + (p[1:][hit] - p[:-1][hit]) * f[:, None] - c) @ d


def main_axis(walls: list[dict]) -> float | None:
    """The room's main direction in degrees, mod 90: the mean of the wall angles weighted by support, taken four-fold so
    that 0 and 90 agree. None without walls."""
    if not walls:
        return None
    a = np.radians([w["angle"] for w in walls]) * 4
    w = np.array([w["support"] for w in walls], float)
    return math.degrees(math.atan2(float((w * np.sin(a)).sum()), float((w * np.cos(a)).sum())) / 4) % 90


def snap_angle(angle: float, axis: float | None, max_deg: float = SNAP_DEG) -> float | None:
    """The main-axis direction nearest `angle` (degrees) when it is within max_deg, else None."""
    if axis is None:
        return None
    near = axis + 90 * round((angle - axis) / 90)
    return near if abs(angle - near) <= max_deg else None


def _wall_fields(a: np.ndarray, b: np.ndarray, toward: np.ndarray, support: int) -> dict:
    """A wall dict for the segment a-b whose normal is the one of the two that points the way `toward` does."""
    d = b - a
    length = float(np.linalg.norm(d))
    d = d / (length or 1.0)
    n = _left(d) * (1 if float(_left(d) @ toward) >= 0 else -1)
    return {"a": [round(float(a[0]), 3), round(float(a[1]), 3)], "b": [round(float(b[0]), 3), round(float(b[1]), 3)],
            "length": round(length, 3), "normal": [round(float(n[0]), 3), round(float(n[1]), 3)], "support": int(support),
            "angle": round(math.degrees(math.atan2(d[1], d[0])), 1)}


def snap_walls(walls: list[dict], axis: float | None, max_deg: float = SNAP_DEG, max_move: float = SNAP_MOVE) -> list[dict]:
    """Turn each wall about its midpoint onto the room's main axes when it is within max_deg of one, unless that moves
    either end by more than max_move: 5 degrees on a 15 m wall is 65 cm at the ends, and a wall that long is rarely that
    far off. Marks each wall `snapped`."""
    for w in walls:
        a, b = np.array(w["a"], float), np.array(w["b"], float)
        w["snapped"] = False
        want = snap_angle(w["angle"] % 180, axis, max_deg)
        if want is None:
            continue
        turn = math.radians(want - w["angle"] % 180)
        r = np.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
        mid = (a + b) / 2
        a2, b2 = mid + (a - mid) @ r.T, mid + (b - mid) @ r.T
        if max(np.linalg.norm(a2 - a), np.linalg.norm(b2 - b)) <= max_move:
            w.update(_wall_fields(a2, b2, np.array(w["normal"]), w["support"]), snapped=True)
    return walls


def _close_corners(walls: list[dict], reach: float = 0.25) -> None:
    """Move a wall end onto the line of a crossing wall when that is within `reach`: support thins out at a corner and
    two walls that stop 10 cm short leave a hole in the plan."""
    for i, w in enumerate(walls):
        moved = False
        for end in ("a", "b"):
            p = np.array(w[end], float)
            d = np.array(w["b"], float) - np.array(w["a"], float)
            d = d / (np.linalg.norm(d) or 1.0)
            best = None
            for j, o in enumerate(walls):
                q = np.array(o["a"], float)
                e = np.array(o["b"], float) - q
                length = float(np.linalg.norm(e))
                e = e / (length or 1.0)
                cross = d[0] * e[1] - d[1] * e[0]
                if i == j or abs(cross) < 0.5:
                    continue
                s = ((q - p)[0] * e[1] - (q - p)[1] * e[0]) / cross  # p + s d meets the other wall's line
                u = float(e @ (p + s * d - q))
                if abs(s) <= reach and -reach <= u <= length + reach and (best is None or abs(s) < best[0]):
                    best = (abs(s), p + s * d)
            if best is not None:
                w[end], moved = [round(float(best[1][0]), 3), round(float(best[1][1]), 3)], True
        if moved:
            w.update(_wall_fields(np.array(w["a"]), np.array(w["b"]), np.array(w["normal"]), w["support"]))


def _along(w: dict, p: np.ndarray) -> tuple[float, float]:
    """(position along the wall from its a end, distance to its line) of the point p (x, z)."""
    a = np.array(w["a"], float)
    d = (np.array(w["b"], float) - a) / (w["length"] or 1.0)
    return float(d @ (p - a)), abs(float(_left(d) @ (p - a)))


def layout(points: np.ndarray, normals: np.ndarray, path: np.ndarray, *, detections: list[dict] | None = None, floor_y: float | None = None,
           boxes: list[dict] | None = None, rng: np.random.Generator | None = None) -> dict:
    """The shell from dense geometry.

    `points` (P, 3) and `normals` (P, 3): dense.py's confirmed points, scene frame. `path` (N, 3): camera centres in capture
    order. `detections`: optional dicts {cls: "door" | "window", p: [x, y, z], half: [hx, hy, hz], views: int}, the SAM 3
    boxes lifted into the scene; without them only gaps the path crosses become doors and no window is written. `boxes`
    are the labelled objects as build() takes them. The result is build()'s dict plus `source: "dense"`, `validation`
    ({ok, reasons, checks}) and `rejected` (the detections dropped, with the reason)."""
    points, normals, path = np.asarray(points, float), np.asarray(normals, float), np.asarray(path, float).reshape(-1, 3)
    det = [d for d in (detections or []) if str(d.get("cls", "")).lower() in ("door", "window")]
    out: dict = {"version": 1, "frame": "scene, metres, +Y up, floor at 0", "source": "dense", "floorY": 0.0, "ceilingY": 2.7, "walls": [],
                 "openings": [], "objects": _objects(boxes or []), "rejected": []}
    if len(points) < 1000:
        out["validation"] = {"ok": False, "reasons": [f"only {len(points)} dense points"], "checks": {}}
        return out
    cam_y = float(np.percentile(path[:, 1], 95)) if len(path) else 1.45
    lv = _levels(points, normals, cam_y, floor_y)
    floor, ceiling = lv["floorY"], lv["ceilingY"]
    top = min(ceiling - 0.3, floor + BAND_TOP)
    lo = floor + ABOVE_FURNITURE if top - floor - ABOVE_FURNITURE >= 0.4 else floor + 0.5
    band = (points[:, 1] >= lo) & (points[:, 1] < top) & (np.abs(normals[:, 1]) < HORIZONTAL)
    xz = points[band][:, [0, 2]]
    hn = normals[band][:, [0, 2]] / (np.linalg.norm(normals[band][:, [0, 2]], axis=1, keepdims=True) + 1e-9)
    out.update(floorY=round(floor, 3), ceilingY=round(ceiling, 3), ceilingSeen=lv["ceilingSeen"])
    lines = extract_lines(xz, hn, max(150, int(0.005 * len(xz))), rng=rng) if len(xz) >= 200 else []

    used: set[int] = set()
    pieces: list[dict] = []   # a wall before snapping: its line, extent along it, and the doorways it spans
    yb = points[band][:, 1]
    for li, ln in enumerate(lines):
        c, d, n = ln["c"], ln["d"], ln["n"]
        t_all = (xz[ln["idx"]] - c) @ d
        low = yb[ln["idx"]] < floor + DOOR_MIN_HEIGHT - 0.1   # a doorway is empty below its header; a lintel or a window's wall is not
        runs = [r for r in _runs(t_all[low]) if r[1] - r[0] >= GAP] if low.sum() >= 3 else []
        above = _runs(t_all)
        crossed = _crossings(path, c, d)
        toward = n * (1 if float((hn[ln["idx"]] @ n).mean()) >= 0 else -1)  # the points' normals face the room
        groups: list[dict] = []
        for r in runs:
            if groups:
                g0, g1 = groups[-1]["t"][1], r[0]
                door = None
                if ((crossed > g0 - 0.1) & (crossed < g1 + 0.1)).any():
                    door = "path"
                elif g1 - g0 >= DOOR_MIN_WIDTH:
                    mid = (g0 + g1) / 2
                    for k, dt in enumerate(det):
                        if dt["cls"] == "door" and k not in used and int(dt.get("views", DETECTION_VIEWS)) >= DETECTION_VIEWS:
                            p = np.array(dt["p"], float)[[0, 2]] - c
                            if abs(float(p @ n)) <= DETECTION_REACH and g0 - 0.3 <= float(p @ d) <= g1 + 0.3 and abs(float(p @ d) - mid) < (g1 - g0):
                                if _header(points, c, d, g0, g1, floor, ceiling) >= DOOR_MIN_HEIGHT:
                                    door = "detection"
                                    used.add(k)
                                    break
                if door:
                    groups[-1]["t"] = (groups[-1]["t"][0], r[1])
                    groups[-1]["gaps"].append((g0, g1, door))
                    continue
                if any(a0 <= g0 + 0.05 and a1 >= g1 - 0.05 for a0, a1 in above) or _junction(lines, li, xz, g0, g1):
                    groups[-1]["t"] = (groups[-1]["t"][0], r[1])   # a window or a lintel, or a wall that meets this one: still one wall
                    continue
            groups.append({"t": r, "gaps": []})
        for g in groups:
            if g["t"][1] - g["t"][0] >= MIN_LEN:
                pieces.append({"c": c, "d": d, "toward": toward, "t": g["t"], "gaps": g["gaps"], "support": len(ln["idx"])})

    walls = [dict(_wall_fields(p["c"] + p["d"] * p["t"][0], p["c"] + p["d"] * p["t"][1], p["toward"], p["support"]), _piece=p) for p in pieces]
    walls.sort(key=lambda w: -w["length"])
    snap_walls(walls, main_axis(walls))
    _close_corners(walls)
    openings: list[dict] = []
    for i, w in enumerate(walls):  # u is measured from the wall's final a end, so a wall that snapping or a corner moved keeps its doors
        for g0, g1, how in w["_piece"]["gaps"]:
            p = w["_piece"]["c"] + w["_piece"]["d"] * (g0 + g1) / 2
            height = _header(points, w["_piece"]["c"], w["_piece"]["d"], g0, g1, floor, ceiling)
            openings.append({"kind": "door", "wall": i, "u": round(_along(w, p)[0], 3), "width": round(g1 - g0, 3), "sill": 0.0,
                             "height": round(height, 3), "name": "Door", "source": how})
    for dt in det:
        k = next((k for k, x in enumerate(det) if x is dt), -1)
        views = int(dt.get("views", DETECTION_VIEWS))
        if dt["cls"] == "door" and k in used:
            continue
        p = np.array(dt["p"], float)
        half = [float(v) for v in (dt.get("half") or [0.5, 1.0, 0.1])]
        near = min(((_along(w, p[[0, 2]]), i) for i, w in enumerate(walls)), key=lambda x: x[0][1], default=None)
        if views < DETECTION_VIEWS:
            out["rejected"].append({"cls": dt["cls"], "reason": f"seen in {views} views, needs {DETECTION_VIEWS}"})
        elif near is None or near[0][1] > DETECTION_REACH or not -0.1 <= near[0][0] <= walls[near[1]]["length"] + 0.1:
            out["rejected"].append({"cls": dt["cls"], "reason": f"no wall within {DETECTION_REACH} m"})
        elif dt["cls"] == "door":
            w = walls[near[1]]
            d = (np.array(w["b"]) - np.array(w["a"])) / w["length"]
            fill = slab_fill(points, np.array(w["a"], float), d, near[0][0] - 0.3, near[0][0] + 0.3, floor + 0.3, floor + DOOR_MIN_HEIGHT)
            out["rejected"].append({"cls": "door", "reason": "solid wall" if fill > SOLID else f"no gap of {DOOR_MIN_WIDTH} x {DOOR_MIN_HEIGHT} m in the wall",
                                    "fill": round(fill, 2)})
        else:
            w = walls[near[1]]
            d = (np.array(w["b"]) - np.array(w["a"])) / w["length"]
            width = 2 * (abs(half[0] * d[0]) + abs(half[2] * d[1]))
            openings.append({"kind": "window", "wall": near[1], "u": round(near[0][0], 3), "width": round(min(width, w["length"]), 3),
                             "sill": round(max(0.0, float(p[1]) - half[1] - floor), 3), "height": round(2 * half[1], 3), "name": "Window", "source": "detection"})
    out["walls"] = [{k: v for k, v in w.items() if k != "_piece"} for w in walls]
    out["openings"], out["droppedOpenings"] = prune_openings(openings, out["walls"], points, floor)
    out["validation"] = validate(out, points, path)
    return out


def prune_openings(openings: list[dict], walls: list[dict], points: np.ndarray, floor: float) -> tuple[list[dict], list[dict]]:
    """(kept, dropped): an opening under OPENING_MIN in either direction, or a door on a solid wall, is dropped with its reason
    and the layout stands: a bad window says nothing about the walls (the chapel's 12 walls were thrown away for two windows of
    0.44 x 1.39 m and 0.73 x 0.15 m). What makes the walls themselves untrustworthy still fails `validate`."""
    kept, dropped = [], []
    for o in openings:
        why = None
        if o["width"] < OPENING_MIN or o["height"] < OPENING_MIN:
            why = f"{o['width']:.2f} x {o['height']:.2f} m, under {OPENING_MIN} m"
        elif o["kind"] == "door" and 0 <= o["wall"] < len(walls):
            w = walls[o["wall"]]
            a = np.array(w["a"], float)
            d = (np.array(w["b"], float) - a) / (w["length"] or 1.0)
            fill = slab_fill(points, a, d, o["u"] - o["width"] / 2 + 0.1, o["u"] + o["width"] / 2 - 0.1, floor + 0.3, floor + min(o["height"], DOOR_MIN_HEIGHT))
            why = f"stands on a solid wall (the plane is {fill:.0%} full)" if fill > SOLID else None
        (dropped if why else kept).append({"cls": o["kind"], "wall": o["wall"], "reason": why} if why else o)
    return kept, dropped


def _objects(boxes: list[dict]) -> list[dict]:
    return [{"id": b.get("id"), "name": b.get("name"), "kind": b.get("cls"), "center": [round(float(v), 3) for v in (b.get("p") or [0, 0, 0])],
             "half": [round(float(v), 3) for v in (b.get("half") or [0.2, 0.2, 0.2])]} for b in boxes]


def validate(shell: dict, points: np.ndarray, path: np.ndarray) -> dict:
    """Whether a layout can be drawn: {ok, reasons, checks}. It fails when a wall ends outside the area the points and the
    cameras cover (by more than 0.5 m), when the ceiling is not above the cameras (or the room is over 15 m high), when an
    opening is under 0.5 m, when a door stands on a solid wall, or when the camera path crosses a wall where there is
    no opening. A layout with no walls fails: there is nothing to draw."""
    points, path = np.asarray(points, float), np.asarray(path, float).reshape(-1, 3)
    walls, ops = shell.get("walls") or [], shell.get("openings") or []
    floor, ceiling = float(shell.get("floorY") or 0.0), float(shell.get("ceilingY") or 0.0)
    reasons: list[str] = []
    if not walls:
        reasons.append("no walls")
    seen = np.vstack([points[:, [0, 2]], path[:, [0, 2]]]) if len(path) else points[:, [0, 2]]
    lo, hi = np.percentile(seen, 0.5, axis=0) - 0.5, np.percentile(seen, 99.5, axis=0) + 0.5
    for i, w in enumerate(walls):
        if any(not ((lo <= np.array(p)) & (np.array(p) <= hi)).all() for p in (w["a"], w["b"])):
            reasons.append(f"wall {i} ends outside the scanned area")
    cam_top = float(path[:, 1].max()) if len(path) else 1.45
    if ceiling < cam_top + 0.3 or ceiling - floor > 15.0:
        reasons.append(f"ceiling at {ceiling:.2f} m cannot be the one the cameras saw (cameras up to {cam_top:.2f} m)")
    for o in ops:
        if o["width"] < OPENING_MIN or o["height"] < OPENING_MIN:
            reasons.append(f"{o['kind']} on wall {o['wall']} is {o['width']:.2f} x {o['height']:.2f} m, under {OPENING_MIN} m")
    solid = 0
    for o in ops:
        if o["kind"] != "door" or not 0 <= o["wall"] < len(walls) or len(points) == 0:
            continue
        w = walls[o["wall"]]
        a = np.array(w["a"], float)
        d = (np.array(w["b"], float) - a) / (w["length"] or 1.0)
        fill = slab_fill(points, a, d, o["u"] - o["width"] / 2 + 0.1, o["u"] + o["width"] / 2 - 0.1, floor + 0.3, floor + min(o["height"], DOOR_MIN_HEIGHT))
        if fill > SOLID:
            solid += 1
            reasons.append(f"door on wall {o['wall']} stands on a solid wall (the plane is {fill:.0%} full)")
    through = 0
    for i, w in enumerate(walls):
        a = np.array(w["a"], float)
        d = (np.array(w["b"], float) - a) / (w["length"] or 1.0)
        for u in _crossings(path, a, d):
            if 0 <= u <= w["length"] and not any(o["wall"] == i and abs(u - o["u"]) <= o["width"] / 2 + 0.1 for o in ops):
                through += 1
                reasons.append(f"the camera path crosses wall {i} at {u:.2f} m, where there is no opening")
    return {"ok": not reasons, "reasons": reasons[:12], "checks": {"walls": len(walls), "openings": len(ops), "doorsOnSolidWall": solid,
                                                                    "pathCrossings": through, "ceilingSeen": bool(shell.get("ceilingSeen", True))}}
