# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Layout and wall metrics (IMPROVEMENT-PLAN section 5, 'Layout' and the wall-flatness line of 'Mesh').

A reference plan is drawn by hand from the laser mesh: {"walls": [{"a": [x, z], "b": [x, z]}], "doors": [{"p": [x, z]}]}, in the
twin's frame (a shell.json's own walls are accepted as walls, and its door openings are turned into door positions). The
chamfer and F-score of a mesh live in geometry.py; this file holds what that one does not. numpy only, so it runs on the worker.

The plan's layout ship rule: median wall endpoint error at most 10 cm, every real door found, at most one false door per room.
"""
from __future__ import annotations

import numpy as np

SHIP_ENDPOINT_M, SHIP_FALSE_DOORS_PER_ROOM = 0.10, 1.0
DOOR_REACH = 0.5     # a found door is within this of a real one: the plan's door is the leaf's centre, the gap's is the opening's


def _ends(walls: list[dict]) -> np.ndarray:
    return np.array([w[k] for w in walls for k in ("a", "b")], float).reshape(-1, 2)


def door_positions(shell: dict) -> np.ndarray:
    """(x, z) centre of every door opening of a shell.json, from its wall's end point and the opening's position along it."""
    out = []
    for o in shell.get("openings") or []:
        if o.get("kind") != "door" or not 0 <= o["wall"] < len(shell["walls"]):
            continue
        w = shell["walls"][o["wall"]]
        a, b = np.array(w["a"], float), np.array(w["b"], float)
        out.append(a + (b - a) / (np.linalg.norm(b - a) or 1.0) * o["u"])
    return np.array(out, float).reshape(-1, 2)


def endpoint_errors(shell: dict, reference: dict) -> np.ndarray:
    """For each end of each reference wall, the distance to the nearest end of a measured wall. A reference that the shell
    lacks a wall for shows as a large error rather than being skipped."""
    ref, got = _ends(reference.get("walls") or []), _ends(shell.get("walls") or [])
    if len(got) == 0:
        return np.full(len(ref), np.inf)
    return np.linalg.norm(ref[:, None] - got[None], axis=2).min(axis=1)


def doors(shell: dict, reference: dict, reach: float = DOOR_REACH) -> dict:
    """Real doors found (a measured door within `reach` of it, each measured door used once) and false doors (measured
    doors that match no real one)."""
    ref = np.array([d["p"] for d in reference.get("doors") or []], float).reshape(-1, 2)
    got = door_positions(shell)
    free, found = list(range(len(got))), 0
    for r in ref:
        near = sorted(free, key=lambda i: float(np.linalg.norm(got[i] - r)))
        if near and np.linalg.norm(got[near[0]] - r) <= reach:
            free.remove(near[0])
            found += 1
    return {"real": len(ref), "found": found, "false": len(free), "measured": len(got)}


def wall_flatness(points: np.ndarray, shell: dict, reach: float = 0.15, min_points: int = 200) -> list[dict]:
    """RMS distance of surface points to their fitted plane, per wall, in metres. Points are a mesh's vertices or a dense cloud
    (N, 3). A wall's points are those within `reach` of its line, along it (not the last 10 cm at either end, where walls meet)
    and above 0.3 m and under the ceiling less 0.3 m, outside its openings. The plane is fitted twice, the second time without
    points over three times the first RMS, so that a bookshelf against the wall does not count as a bump in it."""
    pts = np.asarray(points, float)
    floor, ceil = float(shell.get("floorY") or 0.0), float(shell.get("ceilingY") or 2.7)
    res = []
    for i, w in enumerate(shell.get("walls") or []):
        a = np.array(w["a"], float)
        d = (np.array(w["b"], float) - a) / (w["length"] or 1.0)
        xz = pts[:, [0, 2]] - a
        u, s = xz @ d, xz @ np.array([-d[1], d[0]])
        keep = (np.abs(s) < reach) & (u > 0.1) & (u < w["length"] - 0.1) & (pts[:, 1] > floor + 0.3) & (pts[:, 1] < ceil - 0.3)
        for o in shell.get("openings") or []:
            if o.get("wall") == i:
                keep &= ~((np.abs(u - o["u"]) < o["width"] / 2 + 0.05) & (pts[:, 1] < floor + o["sill"] + o["height"] + 0.05))
        sel = pts[keep]
        if len(sel) < min_points:
            res.append({"wall": i, "points": int(len(sel)), "rms": None})
            continue
        for _ in range(2):
            c = sel.mean(0)
            n = np.linalg.svd(sel - c, full_matrices=False)[2][-1]
            dist = (sel - c) @ n
            rms = float(np.sqrt((dist ** 2).mean()))
            sel = sel[np.abs(dist) <= max(3 * rms, 0.01)]
        res.append({"wall": i, "points": int(len(sel)), "rms": round(rms, 4)})
    return res


def flatness_summary(per_wall: list[dict]) -> dict:
    """Median and worst RMS over the walls that had enough points. 3b needs no repair (3c, 3d) when the median is within 2 cm."""
    v = [w["rms"] for w in per_wall if w["rms"] is not None]
    return {"walls": len(v), "medianRmsCm": round(100 * float(np.median(v)), 2) if v else None, "worstRmsCm": round(100 * max(v), 2) if v else None}


def score(shell: dict, reference: dict, rooms: int = 1) -> dict:
    """The layout ship rule on one scene: median endpoint error, doors found, false doors per room, and whether it ships."""
    err = endpoint_errors(shell, reference)
    dr = doors(shell, reference)
    med = float(np.median(err)) if len(err) else float("inf")
    per_room = dr["false"] / max(1, rooms)
    return {"medianEndpointCm": round(100 * med, 1) if np.isfinite(med) else None, "maxEndpointCm": round(100 * float(err.max()), 1) if len(err) else None,
            **dr, "falseDoorsPerRoom": round(per_room, 2),
            "ships": bool(len(err) and med <= SHIP_ENDPOINT_M and dr["found"] == dr["real"] and per_room <= SHIP_FALSE_DOORS_PER_ROOM)}
