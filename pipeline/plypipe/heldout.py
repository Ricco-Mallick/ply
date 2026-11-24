# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Held-out photos for the viewer's "Real photo vs twin" check.

Training keeps every Nth view (sorted by name, gsplat's test_every) out of the fit. A few of those are published with
the exact camera they were taken from (scene frame, the same frame the twin is in), so the viewer can stand there,
render the twin and slide between the photo it never saw and its own picture. That is the honest way to show how
faithful a twin is: a frame the model was not allowed to memorise.

scene.json gets `heldOut: [{image, width, height, fy, position, quaternion}]`: position in metres, quaternion
(x, y, z, w) of a three.js camera (looks down -z, +y up) so the viewer applies it as is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plypipe import colmap_txt

MAX_VIEWS = 6
LONG_SIDE = 1280


def qvec_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_xyzw(R: np.ndarray) -> list[float]:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q) / np.linalg.norm(q)
    return [round(float(v), 6) for v in q]


def three_pose(qvec: np.ndarray, tvec: np.ndarray) -> tuple[list[float], list[float]]:
    """COLMAP cam_from_world -> (camera centre, three.js camera quaternion)."""
    R_cw = qvec_to_R(np.asarray(qvec, float))
    C = -R_cw.T @ np.asarray(tvec, float)
    R_three = R_cw.T @ np.diag([1.0, -1.0, -1.0])  # OpenCV camera (x right, y down, z ahead) -> three (y up, -z ahead)
    return [round(float(v), 4) for v in C], R_to_quat_xyzw(R_three)


def pick(names: list[str], holdout: int, n: int = MAX_VIEWS) -> list[str]:
    """The views training never saw (gsplat: every holdout-th by sorted name), spread evenly, at most n."""
    if not holdout or holdout <= 1:
        return []
    val = sorted(names)[::holdout]
    if len(val) <= n:
        return val
    return [val[round(k * (len(val) - 1) / (n - 1))] for k in range(n)]


def publish(data_dir: Path, out: Path, holdout: int) -> list[dict]:
    """Write heldout/*.jpg next to the scene and return the scene.json entries ([] when nothing is held out)."""
    import cv2
    model = colmap_txt.read_model(data_dir / "sparse" / "0")
    by_name = {im.name: im for im in model.images}
    fy = None
    for ln in model.cameras_lines:
        if ln and not ln.startswith("#"):
            p = ln.split()
            if p[1] in ("PINHOLE", "SIMPLE_PINHOLE"):
                fy = float(p[5] if p[1] == "PINHOLE" else p[4])
            break
    if fy is None:
        return []
    dest = out / "heldout"
    dest.mkdir(exist_ok=True)
    entries = []
    for name in pick(list(by_name), holdout):
        img = cv2.imread(str(data_dir / "images" / name))
        if img is None:
            continue
        h, w = img.shape[:2]
        k = min(1.0, LONG_SIDE / max(w, h))
        if k < 1:
            img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        file = f"heldout/{Path(name).stem}.jpg"
        cv2.imwrite(str(out / file), img, [cv2.IMWRITE_JPEG_QUALITY, 86])
        im = by_name[name]
        pos, quat = three_pose(im.qvec, im.tvec)
        entries.append({"image": file, "width": img.shape[1], "height": img.shape[0], "fy": round(fy * k, 2),
                        "position": pos, "quaternion": quat, "name": name})
    return entries
