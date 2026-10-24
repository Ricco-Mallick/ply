# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Cinematic tour: camera keyframes for the viewer's "Play tour", generated from the room and the capture path.

Shots (scene frame: +y up, floor y = 0, metres):
  1. establishing: high in one corner, looking across the room to its centre
  2. a slow walk along the captured path at eye height, looking ahead and a little toward the centre
  3. a reveal from the opposite corner
  4. an overhead-ish wide to finish
Each keyframe: position, look-at target, field of view and seconds to reach it. The viewer interpolates positions and
targets with a smooth spline. Corners are pulled inside the walls so the camera never leaves the room.
"""

from __future__ import annotations

import numpy as np


def _inside(p: np.ndarray, lo: np.ndarray, hi: np.ndarray, margin: float = 0.6) -> np.ndarray:
    return np.clip(p, lo + margin, hi - margin)


def build_tour(cams: np.ndarray, bounds: dict, eye: float = 1.55) -> list[dict]:
    if len(cams) < 4 or not bounds:
        return []
    lo, hi = np.array(bounds["min"], float), np.array(bounds["max"], float)
    top = float(min(hi[1], 4.0))
    c = np.array([(lo[0] + hi[0]) / 2, 1.1, (lo[2] + hi[2]) / 2])
    high = min(top - 0.5, eye + 0.42)  # ~2 m: above furniture, below hanging lamps
    # the two corners of the room farthest apart along its longer diagonal
    corners = [np.array([x, high, z]) for x in (lo[0], hi[0]) for z in (lo[2], hi[2])]
    corners = [_inside(k, lo, hi) for k in corners]
    ref = cams[0]
    a = max(corners, key=lambda k: np.hypot(k[0] - ref[0], k[2] - ref[2]))
    b = max(corners, key=lambda k: np.hypot(k[0] - a[0], k[2] - a[2]))  # the opposite corner
    kf = [{"pos": a, "target": c, "fov": 62, "dur": 0}, {"pos": a + (c - a) * 0.18, "target": c, "fov": 58, "dur": 5}]
    # walk: 4 points spread along the capture path, eye height, looking ahead
    idx = np.linspace(len(cams) * 0.1, len(cams) * 0.9, 4).astype(int)
    for i in idx:
        p = np.array([cams[i][0], eye, cams[i][2]])
        ahead = cams[min(len(cams) - 1, i + max(2, len(cams) // 12))]
        look = np.array([ahead[0], eye - 0.15, ahead[2]]) * 0.6 + c * 0.4
        if np.linalg.norm(look - p) < 0.3:
            look = c
        kf.append({"pos": _inside(p, lo, hi, 0.4), "target": look, "fov": 66, "dur": 4.5})
    kf.append({"pos": b, "target": c, "fov": 60, "dur": 5})
    kf.append({"pos": np.array([c[0], min(top - 0.3, high + 0.15), c[2]]) + (b - c) * np.array([0.35, 0, 0.35]),
               "target": np.array([c[0], 0.0, c[2]]), "fov": 70, "dur": 5})
    return [{"pos": [round(float(v), 3) for v in k["pos"]], "target": [round(float(v), 3) for v in k["target"]],
             "fov": k["fov"], "dur": k["dur"]} for k in kf]
