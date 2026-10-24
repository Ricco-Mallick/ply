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
MIN_SUPPORT = 0.05           # a wall needs this share of the band's points
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


def fit_walls(pts: np.ndarray, floor_y: float = 0.0, *, band: tuple[float, float] = BAND,
              tol: float = TOL, rng: np.random.Generator | None = None) -> list[dict]:
    """Wall segments through the points (N x 3, scene frame). Deterministic for a given seed."""
    rng = rng or np.random.default_rng(7)
    if pts.size == 0:
        return []
    y = pts[:, 1] - floor_y
    band_pts = pts[(y > band[0]) & (y < band[1])][:, [0, 2]]
    if len(band_pts) < 200:
        return []
    floor_share = max(1, round(MIN_SUPPORT * len(band_pts)))

    rest = band_pts.copy()
    walls: list[dict] = []
    for _ in range(MAX_WALLS):
        if len(rest) < floor_share:
            break
        best: tuple[int, np.ndarray, np.ndarray, float] | None = None
        for _ in range(240):
            i, j = rng.integers(0, len(rest), 2)
            if i == j:
                continue
            p, q = rest[i], rest[j]
            if math.hypot(*(q - p)) < 0.8:
                continue
            d, n, off = _line(p, q)
            inl = np.abs(rest @ n - off) < tol
            score = int(inl.sum())
            if best is None or score > best[0]:
                best = (score, d, n, off)
        if best is None or best[0] < floor_share:
            break
        score, d, n, off = best
        seg = rest[np.abs(rest @ n - off) < tol]
        # keep the segment tidy: its extent along the line, trimmed to the 2nd and 98th percentile of the support
        t = seg @ d
        t0, t1 = float(np.percentile(t, 0.5)), float(np.percentile(t, 99.5))
        pad = (t1 - t0) * 0.02  # walls end where the sweep turned, a little past the last inlier
        t0, t1 = t0 - pad, t1 + pad
        if t1 - t0 < MIN_LEN:
            rest = rest[np.abs(rest @ n - off) >= tol]
            continue
        a = d * t0 + n * off
        b = d * t1 + n * off
        # the wall's outward normal must point away from the room centre, decided later by the caller's traffic
        walls.append({
            "a": [round(float(a[0]), 3), round(float(a[1]), 3)],
            "b": [round(float(b[0]), 3), round(float(b[1]), 3)],
            "length": round(float(t1 - t0), 3),
            "normal": [round(float(n[0]), 3), round(float(n[1]), 3)],
            "support": int(score),
        })
        rest = rest[np.abs(rest @ n - off) >= tol]

    walls = _merge(walls)
    walls = _orient(walls, band_pts)
    for w in walls:
        w["angle"] = round(math.degrees(math.atan2(w["b"][1] - w["a"][1], w["b"][0] - w["a"][0])), 1)
    return sorted(walls, key=lambda w: -w["length"])


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
