# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Measured depth as training supervision for iPhone LiDAR captures.

gsplat's depth loss (simple_trainer, `depth_loss`) compares the rendered depth with depths at a few hundred pixels per
view: the SfM track points that view saw. A LiDAR capture has far better: a measured depth map for every kept frame
(stages/frames.py writes lidar/ and lidar.json). `install` wraps the trainer's Dataset so that, for a frame with a map,
the points and depths it hands the loss are a few thousand confident LiDAR pixels instead of the sparse tracks. Walls
and floors that SfM found no features on (the white wall, the plain floor) then have a measured surface to sit on,
which is where floaters and fog come from.

Units: the trained scene is the aligned one (metres, from the AR poses), and ARKit depth is metres along the camera's
optical axis, which is what gsplat's depth is. Nothing needs rescaling beyond the file's millimetres. A frame whose
picture does not have the depth map's aspect (an undistortion crop) keeps the sparse points: fail soft, never guess.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NEAR, FAR = 0.15, 6.0      # metres: the iPhone LiDAR's useful range; past ~5 m its depth is interpolated, not measured
SAMPLES = 4096             # pixels per view handed to the loss (gsplat samples the rendered depth at each)
MIN_CONF = 2               # Stray's confidence: 0 low, 1 medium, 2 high


def read_index(lidar_json: Path) -> dict[str, tuple[Path, Path | None]]:
    """frame file name -> (depth map, confidence map or None)."""
    j = json.loads(Path(lidar_json).read_text())
    root = Path(lidar_json).parent / "lidar"
    out = {}
    for name, maps in (j.get("frames") or {}).items():
        d = root / maps["depth"]
        if d.is_file():
            c = root / maps["confidence"] if maps.get("confidence") else None
            out[Path(name).name] = (d, c if c and c.is_file() else None)
    return out


def load_depth(path: Path) -> np.ndarray:
    """Depth in metres (0 where unknown)."""
    if path.suffix == ".npy":
        a = np.load(path)
        return a.astype(np.float32) / (1.0 if a.dtype.kind == "f" and float(np.nanmax(a)) < 100 else 1000.0)
    import cv2
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(f"unreadable depth map {path.name}")
    return a.astype(np.float32) / 1000.0


def sample(depth: np.ndarray, conf: np.ndarray | None, width: int, height: int, n: int = SAMPLES,
           rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Up to n confident depth pixels as (pixel coordinates in a width x height picture, metres). None when the
    picture's aspect is not the map's (it was cropped) or the map has nothing usable."""
    h, w = depth.shape[:2]
    if abs(width / height - w / h) > 0.01 * (w / h):
        return None
    ok = (depth > NEAR) & (depth < FAR) & np.isfinite(depth)
    if conf is not None and conf.shape == depth.shape:
        ok &= conf >= MIN_CONF
    v, u = np.nonzero(ok)
    if len(u) < 64:
        return None
    rng = rng or np.random.default_rng()
    if len(u) > n:
        pick = rng.choice(len(u), n, replace=False)
        u, v = u[pick], v[pick]
    sx, sy = width / w, height / h  # pixel centres map to pixel centres
    pts = np.stack([(u + 0.5) * sx - 0.5, (v + 0.5) * sy - 0.5], axis=1).astype(np.float32)
    return pts, depth[v, u].astype(np.float32)


def install(st, lidar_json: Path, log=print) -> dict:
    """Wrap st.Dataset.__getitem__: frames with a LiDAR map get dense measured depths for the depth loss."""
    import torch
    index = read_index(Path(lidar_json))
    if not index:
        raise ValueError(f"{lidar_json} names no depth maps")
    orig = st.Dataset.__getitem__
    stats = {"maps": len(index), "used": 0, "fallback": 0}

    def getitem(self, i):
        data = orig(self, i)
        if "points" not in data or "depths" not in data:  # depth loss off for this dataset
            return data
        try:
            name = Path(self.parser.image_names[self.indices[i]]).name
            hit = index.get(name)
            if hit is None:
                return data
            img = data["image"]  # H x W x 3
            conf = None
            if hit[1] is not None:
                import cv2
                conf = cv2.imread(str(hit[1]), cv2.IMREAD_UNCHANGED)
            got = sample(load_depth(hit[0]), conf, int(img.shape[1]), int(img.shape[0]))
            if got is None:
                stats["fallback"] += 1
                return data
            data["points"], data["depths"] = torch.from_numpy(got[0]), torch.from_numpy(got[1])
            stats["used"] += 1
        except Exception as e:  # one bad map must not stop a paid training run
            stats["fallback"] += 1
            if stats["fallback"] <= 3:
                log(f"lidar depth: frame {i} keeps its sparse points ({str(e)[:120]})")
        return data

    st.Dataset.__getitem__ = getitem
    return stats
