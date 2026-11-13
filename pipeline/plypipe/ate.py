# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Absolute trajectory error of one camera solve against another: the camera centres of the views both posed, matched by
image name, aligned with the best similarity (scale, rotation, translation), then the distances that remain."""

from __future__ import annotations

import numpy as np

from plypipe import geometry


def centres(rec) -> dict[str, np.ndarray]:
    """name -> camera centre for every posed view of a pycolmap Reconstruction."""
    return {im.name: np.asarray(im.projection_center(), dtype=float) for im in rec.images.values() if im.has_pose}


def ate(est: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> dict:
    """`est` aligned onto `ref` by Umeyama; rmse, median and max are in `ref`'s units (metres for an aligned job, so x100
    for centimetres). `matched` is how many views both solves posed: a score over a small share of the views means little."""
    names = sorted(set(est) & set(ref))
    if len(names) < 3:
        raise ValueError(f"only {len(names)} views are posed in both solves: cannot align them")
    a = np.array([est[n] for n in names])
    b = np.array([ref[n] for n in names])
    s, R, t = geometry.umeyama(a, b)
    d = np.linalg.norm(s * a @ R.T + t - b, axis=1)
    return {"matched": len(names), "estPosed": len(est), "refPosed": len(ref), "scale": s, "rmse": float(np.sqrt((d ** 2).mean())),
            "median": float(np.median(d)), "max": float(d.max())}
