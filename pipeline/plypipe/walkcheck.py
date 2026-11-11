# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""walkcheck: can a visitor get around a twin in the viewer's Walk mode?

Scores an occupancy grid with the viewer's own rules (app/src/walk/scene.ts, engine.ts): a cell darker than 128 is
blocked, so unknown (127) blocks too; the visitor is a circle of radius 0.22 m tested at its centre and 8 points on
its rim; anything outside the grid is blocked. Positions on a 5 cm lattice that pass that test are joined to their
4 neighbours, and everything is measured from the component the viewer would start in (the spawn, or the nearest
standable point within 6 m, as `unstick()` does).

    cameras   share of capture positions with a standable point within 0.3 m in that component. Someone walked
              there, so anything under 100% is floor the twin wrongly refuses.
    places    share of labelled places (places.json) with a standable point within 1.5 m in that component.

    python -m plypipe.walkcheck <job folder>     # scene.json, occupancy.json/.png, places.json, data/sparse/0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RADIUS = 0.22   # app/src/walk/engine.ts RADIUS
STEP = 0.05
UNSTICK_M = 6.0  # engine.ts unstick() search radius
CAMERA_M = 0.3
PLACE_M = 1.5
PASS_CAMERAS = 0.99


def standable(grid: np.ndarray, meta: dict, radius: float = RADIUS, step: float = STEP):
    """Lattice of positions the viewer accepts: (ok[nz, nx], xs, zs)."""
    H, W = grid.shape
    cell, (ox, oz) = float(meta["cellSize"]), meta["origin"]
    xs = ox + (np.arange(int(np.ceil(W * cell / step))) + 0.5) * step
    zs = oz + (np.arange(int(np.ceil(H * cell / step))) + 0.5) * step
    X, Z = np.meshgrid(xs, zs)
    bad = grid < 128

    def blocked(x, z):
        i = np.floor((x - ox) / cell).astype(int)
        j = np.floor((z - oz) / cell).astype(int)
        inside = (i >= 0) & (j >= 0) & (i < W) & (j < H)
        out = np.ones(x.shape, bool)
        out[inside] = bad[j[inside], i[inside]]
        return out

    ok = ~blocked(X, Z)
    for k in range(8):
        a = k * np.pi / 4
        ok &= ~blocked(X + np.cos(a) * radius, Z + np.sin(a) * radius)
    return ok, xs, zs


def _near(ok_comp: np.ndarray, xs: np.ndarray, zs: np.ndarray, p: np.ndarray, within: float) -> bool:
    i0, i1 = np.searchsorted(xs, [p[0] - within, p[0] + within])
    j0, j1 = np.searchsorted(zs, [p[1] - within, p[1] + within])
    win = ok_comp[j0:j1, i0:i1]
    if not win.any():
        return False
    jj, ii = np.nonzero(win)
    return bool((np.hypot(xs[i0 + ii] - p[0], zs[j0 + jj] - p[1]) <= within).any())


def check(grid: np.ndarray, meta: dict, spawn_xz, cameras_xz=None, places_xz=None) -> dict:
    ok, xs, zs = standable(grid, meta)
    _, labels = cv2.connectedComponents(ok.astype(np.uint8), connectivity=4)
    sx, sz = float(spawn_xz[0]), float(spawn_xz[1])
    jj, ii = np.nonzero(ok)
    if not len(jj):
        return {"radius": RADIUS, "standableM2": 0.0, "cameras": 0.0 if cameras_xz is not None else None,
                "places": 0.0 if places_xz is not None else None, "pass": False}
    d = np.hypot(xs[ii] - sx, zs[jj] - sz)
    k = int(np.argmin(d))
    if d[k] > UNSTICK_M:
        comp = np.zeros_like(ok)
    else:
        comp = labels == labels[jj[k], ii[k]]

    def share(pts, within):
        if pts is None or not len(pts):
            return None
        return round(float(np.mean([_near(comp, xs, zs, np.asarray(p, float), within) for p in pts])), 4)

    cams = share(cameras_xz, CAMERA_M)
    return {"radius": RADIUS, "standableM2": round(float(comp.sum()) * STEP * STEP, 2),
            "cameras": cams, "places": share(places_xz, PLACE_M),
            "pass": bool(cams is not None and cams >= PASS_CAMERAS)}


def load_job(folder: Path) -> dict:
    """Grid, spawn, capture positions and places of a finished job folder, in the scene frame (x, z)."""
    scene = json.loads((folder / "scene.json").read_text())
    meta = json.loads((folder / scene["occupancy"]["meta"]).read_text())
    grid = cv2.imread(str(folder / scene["occupancy"]["image"]), cv2.IMREAD_GRAYSCALE)
    sp = scene.get("spawn", {}).get("position") or [0, 0, 0]
    cams = None
    sparse = folder / "data" / "sparse" / "0"
    if (sparse / "images.txt").exists():
        from plypipe import colmap_txt
        centers, _, _ = colmap_txt.read_model(sparse).cameras()
        cams = centers[:, [0, 2]]
    elif scene.get("cameraPath"):
        cams = np.asarray(scene["cameraPath"], float)[:, [0, 2]]
    places = None
    pf = folder / "places.json"
    if pf.exists():
        places = [p["p"][::2] for p in json.loads(pf.read_text()).get("places", []) if p.get("p")]
    return {"grid": grid, "meta": meta, "spawn": (sp[0], sp[2]), "cameras": cams, "places": places}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    a = ap.parse_args(argv)
    j = load_job(a.folder)
    print(json.dumps(check(j["grid"], j["meta"], j["spawn"], j["cameras"], j["places"])))


if __name__ == "__main__":
    main()
