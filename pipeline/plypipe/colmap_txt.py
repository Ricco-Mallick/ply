# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Read / transform / write COLMAP text models (cameras.txt, images.txt, points3D.txt).

Only numbers that change under a similarity transform are parsed; everything else (camera
params, 2D keypoints, tracks) is copied verbatim, so correspondences survive untouched.

COLMAP >= 4.0 also writes rigs.txt and frames.txt. A transformed model is written without
them: COLMAP's readers then initialise trivial rigs/frames from images.txt, which is exact for
a single phone camera. A model with a multi-sensor rig is refused rather than silently broken.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from plypipe import geometry


@dataclass
class Image:
    image_id: int
    qvec: np.ndarray  # (qw, qx, qy, qz) cam_from_world
    tvec: np.ndarray
    camera_id: int
    name: str
    points2d_line: str  # verbatim


@dataclass
class Model:
    cameras_lines: list[str]
    images: list[Image]
    point_ids: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    error: np.ndarray
    track_len: np.ndarray
    point_tails: list[str] = field(repr=False)  # "R G B ERROR TRACK..." verbatim after XYZ

    @property
    def num_points(self) -> int:
        return int(self.xyz.shape[0])

    def cameras(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World camera centres, up vectors and forward vectors in image-name order."""
        ims = sorted(self.images, key=lambda im: im.name)
        return geometry.camera_centers_and_ups(np.array([i.qvec for i in ims]), np.array([i.tvec for i in ims]))

    def camera_size(self) -> tuple[int, int] | None:
        for ln in self.cameras_lines:
            if ln and not ln.startswith("#"):
                parts = ln.split()
                return int(parts[2]), int(parts[3])
        return None


def _data_lines(path: Path) -> list[str]:
    with path.open() as f:
        return [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]


def read_model(d: str | Path) -> Model:
    d = Path(d)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (d / name).exists():
            raise FileNotFoundError(f"{d / name} missing (expected a COLMAP text model)")
    rigs = d / "rigs.txt"
    if rigs.exists():
        n_sensors = 0
        for ln in _data_lines(rigs):
            parts = ln.split()
            # RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, then sensors
            if len(parts) >= 2 and parts[1].isdigit():
                n_sensors = max(n_sensors, int(parts[1]))
        if n_sensors > 1:
            raise ValueError("multi-sensor rig models are not supported by the text transformer")

    cameras_lines = (d / "cameras.txt").read_text().splitlines()

    images: list[Image] = []
    with (d / "images.txt").open() as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        p = lines[i].split()
        pts_line = lines[i + 1] if i + 1 < len(lines) else ""
        images.append(Image(int(p[0]), np.array(list(map(float, p[1:5]))), np.array(list(map(float, p[5:8]))),
                            int(p[8]), " ".join(p[9:]), pts_line))
        i += 2

    ids, xyz, rgb, err, tl, tails = [], [], [], [], [], []
    for ln in _data_lines(d / "points3D.txt"):
        p = ln.split()
        ids.append(int(p[0]))
        xyz.append([float(v) for v in p[1:4]])
        rgb.append([int(v) for v in p[4:7]])
        err.append(float(p[7]))
        tl.append((len(p) - 8) // 2)
        tails.append(" ".join(p[4:]))
    return Model(cameras_lines, images, np.array(ids, dtype=np.int64), np.array(xyz, dtype=np.float64).reshape(-1, 3),
                 np.array(rgb, dtype=np.uint8).reshape(-1, 3), np.array(err), np.array(tl, dtype=np.int32), tails)


def write_model(model: Model, d: str | Path) -> None:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for stale in ("rigs.txt", "frames.txt", "rigs.bin", "frames.bin", "cameras.bin", "images.bin", "points3D.bin"):
        if (d / stale).exists():
            (d / stale).unlink()
    (d / "cameras.txt").write_text("\n".join(model.cameras_lines) + "\n")
    with (d / "images.txt").open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(model.images)}\n")
        for im in model.images:
            q, t = im.qvec, im.tvec
            f.write(f"{im.image_id} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} {t[0]:.12g} {t[1]:.12g} {t[2]:.12g} {im.camera_id} {im.name}\n")
            f.write(im.points2d_line + "\n")
    with (d / "points3D.txt").open("w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {model.num_points}\n")
        for pid, p, tail in zip(model.point_ids, model.xyz, model.point_tails):
            f.write(f"{pid} {p[0]:.9g} {p[1]:.9g} {p[2]:.9g} {tail}\n")


CAMERA_MODEL_IDS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3, "OPENCV": 4, "OPENCV_FISHEYE": 5,
                    "FULL_OPENCV": 6, "FOV": 7, "SIMPLE_RADIAL_FISHEYE": 8, "RADIAL_FISHEYE": 9, "THIN_PRISM_FISHEYE": 10}


def write_model_bin(model: Model, d: str | Path) -> None:
    """The same model in COLMAP's classic binary layout (cameras.bin, images.bin, points3D.bin).

    gsplat <= 1.5.x loads models through rmbrualla's pycolmap, whose text reader is broken on Python 3 but whose
    binary reader works; it prefers .bin when present.
    """
    import struct

    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    cams = [ln.split() for ln in model.cameras_lines if ln.strip() and not ln.startswith("#")]
    with (d / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for c in cams:
            params = [float(x) for x in c[4:]]
            f.write(struct.pack("<iiQQ", int(c[0]), CAMERA_MODEL_IDS[c[1]], int(c[2]), int(c[3])))
            f.write(struct.pack(f"<{len(params)}d", *params))
    with (d / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(model.images)))
        for im in model.images:
            vals = im.points2d_line.split()
            pts = [(float(vals[i]), float(vals[i + 1]), int(vals[i + 2])) for i in range(0, len(vals) - 2, 3)]
            f.write(struct.pack("<I4d3dI", im.image_id, *map(float, im.qvec), *map(float, im.tvec), im.camera_id))
            f.write(im.name.encode() + b"\x00")
            f.write(struct.pack("<Q", len(pts)))
            for x, y, pid in pts:
                f.write(struct.pack("<ddq", x, y, pid))
    with (d / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", model.num_points))
        for pid, p, tail in zip(model.point_ids, model.xyz, model.point_tails):
            t = tail.split()
            track = [(int(t[i]), int(t[i + 1])) for i in range(4, len(t) - 1, 2)]
            f.write(struct.pack("<Q3d3Bd", int(pid), *map(float, p), int(t[0]), int(t[1]), int(t[2]), float(t[3])))
            f.write(struct.pack("<Q", len(track)))
            for img, idx in track:
                f.write(struct.pack("<ii", img, idx))


def transformed(model: Model, T: np.ndarray) -> Model:
    """Return a copy of the model with the world mapped by the similarity T."""
    images = []
    for im in model.images:
        q2, t2 = geometry.transform_pose(im.qvec, im.tvec, T)
        images.append(Image(im.image_id, q2, t2, im.camera_id, im.name, im.points2d_line))
    xyz = geometry.apply(T, model.xyz) if model.num_points else model.xyz.copy()
    return Model(list(model.cameras_lines), images, model.point_ids.copy(), xyz, model.rgb.copy(), model.error.copy(),
                 model.track_len.copy(), list(model.point_tails))


def copy_model(src: str | Path, dst: str | Path) -> None:
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)
