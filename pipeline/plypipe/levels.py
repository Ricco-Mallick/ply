# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Floors and stairs from one walk through a building.

A person filming holds the phone at about the same height above whatever floor they stand on, so the camera heights
of a two-storey walk fall into two clusters a storey apart, joined by the frames taken on the stairs. That is enough
to find the levels without any learned model:

1. histogram the camera heights (10 cm bins); a level is a peak holding at least MIN_SHARE of the frames, at least
   MIN_STOREY above the last one;
2. the eye height is the median camera height on the level whose floor the aligner put at y = 0; each other level's
   floor starts at its peak minus that eye height and is refined on the points (the densest thin layer within 0.3 m);
3. every frame belongs to the level whose peak is within LEVEL_BAND of its height, else to the stairs; a run of stair
   frames between two levels is a staircase, kept as a polyline of (x, floor y, z) for the viewer to walk on.

One level (the usual capture) returns a single level at y = 0 and no stairs, so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import numpy as np

MIN_SHARE = 0.08     # a level holds at least this share of the frames
MIN_STOREY = 2.0     # metres between two levels' camera heights
LEVEL_BAND = 0.6     # a frame within this of a level's camera height stands on that level
BIN = 0.1


def _peaks(y: np.ndarray) -> list[float]:
    lo, hi = float(y.min()), float(y.max())
    if hi - lo < MIN_STOREY:
        return [float(np.median(y))]
    edges = np.arange(lo - BIN, hi + 2 * BIN, BIN)
    h, e = np.histogram(y, edges)
    hs = np.convolve(h, [1, 2, 1], mode="same") / 4.0  # a little smoothing so one peak does not split in two
    order = np.argsort(-hs)
    peaks: list[float] = []
    for k in order:
        if hs[k] < MIN_SHARE * len(y) / 2:  # smoothed counts are about half a band's frames
            break
        c = float((e[k] + e[k + 1]) / 2)
        if all(abs(c - p) >= MIN_STOREY for p in peaks):
            peaks.append(c)
    # each peak's centre is the median of the frames near it, not the bin edge
    out = []
    for p in sorted(peaks):
        near = y[np.abs(y - p) < LEVEL_BAND]
        if len(near) >= MIN_SHARE * len(y):
            out.append(float(np.median(near)))
    return out or [float(np.median(y))]


def _refine_floor(points: np.ndarray | None, guess: float) -> float:
    if points is None or not len(points):
        return guess
    y = points[:, 1]
    near = y[np.abs(y - guess) < 0.3]
    if len(near) < 200:
        return guess
    h, e = np.histogram(near, np.arange(guess - 0.3, guess + 0.31, 0.02))
    k = int(np.argmax(h))
    return float((e[k] + e[k + 1]) / 2)


def find(cams: np.ndarray, points: np.ndarray | None = None, floor_y: float = 0.0) -> dict:
    """cams: camera centres in capture order (N, 3), scene frame. Returns {levels, stairs, frameLevel}."""
    cams = np.asarray(cams, float).reshape(-1, 3)
    if len(cams) < 10:
        return {"levels": [{"index": 0, "floorY": floor_y, "eyeY": None, "frames": len(cams)}], "stairs": [], "frameLevel": [0] * len(cams)}
    peaks = _peaks(cams[:, 1])
    # the level whose floor the aligner levelled at floor_y: the peak nearest a typical eye height above it
    base = int(np.argmin([abs(p - floor_y - 1.45) for p in peaks]))
    eye = peaks[base] - floor_y
    levels = []
    for i, p in enumerate(peaks):
        fy = floor_y if i == base else _refine_floor(points, p - eye)
        levels.append({"index": i, "floorY": round(fy, 3), "eyeY": round(p, 3)})
    lab = np.full(len(cams), -1)
    for i, p in enumerate(peaks):
        lab[np.abs(cams[:, 1] - p) < LEVEL_BAND] = i
    for lv in levels:
        lv["frames"] = int((lab == lv["index"]).sum())
    stairs = []
    if len(levels) > 1:
        k = 0
        while k < len(cams):
            if lab[k] != -1:
                k += 1
                continue
            j = k
            while j < len(cams) and lab[j] == -1:
                j += 1
            before = lab[k - 1] if k > 0 else -1
            after = lab[j] if j < len(cams) else -1
            if before != -1 and after != -1 and before != after and j - k >= 2:
                seg = cams[max(0, k - 1): min(len(cams), j + 1)]
                path = [[round(float(c[0]), 3), round(float(c[1] - eye), 3), round(float(c[2]), 3)] for c in seg]
                lo, hi = sorted((int(before), int(after)))
                if path[0][1] > path[-1][1]:
                    path.reverse()
                path[0][1] = levels[lo]["floorY"]
                path[-1][1] = levels[hi]["floorY"]
                if not any(s["from"] == lo and s["to"] == hi and _close(s["path"], path) for s in stairs):
                    stairs.append({"from": lo, "to": hi, "path": _thin(path)})
            k = j
    return {"levels": levels, "stairs": stairs, "frameLevel": [int(v) for v in lab]}


def _close(a: list, b: list, tol: float = 1.0) -> bool:
    return np.hypot(a[0][0] - b[0][0], a[0][2] - b[0][2]) < tol and np.hypot(a[-1][0] - b[-1][0], a[-1][2] - b[-1][2]) < tol


def _thin(path: list, step: float = 0.25) -> list:
    """Every ~25 cm along the stairs, ends kept, heights never going down."""
    out = [path[0]]
    for p in path[1:-1]:
        if np.hypot(p[0] - out[-1][0], p[2] - out[-1][2]) >= step:
            out.append([p[0], max(p[1], out[-1][1]), p[2]])
    out.append([path[-1][0], max(path[-1][1], out[-1][1]), path[-1][2]])
    return out
