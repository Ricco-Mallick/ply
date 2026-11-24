# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""iPhone LiDAR captures recorded with Stray Scanner (open source, github.com/strayrobots/scanner).

A scan is a folder (usually uploaded as a zip):

    rgb.mp4            HEVC video, one frame per row of odometry.csv
    odometry.csv       timestamp, frame, x, y, z, qx, qy, qz, qw[, fx, fy, cx, cy, ...]: ARKit's camera-to-world pose
                       in metres with +y up (gravity), from where the session started
    camera_matrix.csv  3 x 3 intrinsics of the full-resolution video frame (older exports have no per-frame columns)
    depth/000123.png   LiDAR depth, 16-bit millimetres, 256 x 192 (older exports: .npy)
    confidence/...     0, 1 or 2 per depth pixel

`to_ar_tar` turns one scan into the tar the capture app's AR scan uploads (keyframe JPEGs plus poses.json), so the rest
of the pipeline treats it exactly like an Android AR scan: frames picks the sharpest keyframes, align fits the solve
onto the tracked positions (true metres, level floor). The depth maps of the keyframes kept go to lidar/ for the
stages that can use measured depth.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import numpy as np

VIDEO_NAMES = ("rgb.mp4", "rgb.mov")


def is_scan(d: Path) -> bool:
    return d.is_dir() and (d / "odometry.csv").is_file() and any((d / v).is_file() for v in VIDEO_NAMES)


def unzip(zp: Path, dest: Path, max_bytes: int = 12 * 2**30) -> Path:
    """Extract a zip safely (no paths outside dest, no symlinks, a size ceiling); returns dest."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zp) as z:
        total = sum(i.file_size for i in z.infolist())
        if total > max_bytes:
            raise ValueError(f"{zp.name} unpacks to {total / 2**30:.1f} GB, over the {max_bytes / 2**30:.0f} GB limit")
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            if name.startswith("__MACOSX/") or name.endswith("/.DS_Store") or name.split("/")[-1].startswith("._"):
                continue
            target = (root / name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"{zp.name}: unsafe path {info.filename!r}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                continue  # a symlink inside the archive: skip it
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
    return dest


def find_scans(root: Path) -> list[Path]:
    """Scan folders under root (the root itself, or any depth below it), in name order."""
    if is_scan(root):
        return [root]
    return sorted({p.parent for p in root.rglob("odometry.csv") if is_scan(p.parent)})


def read_odometry(d: Path) -> list[dict]:
    rows = []
    with open(d / "odometry.csv", newline="") as f:
        r = csv.reader(f)
        head = [h.strip().lower() for h in next(r)]
        idx = {h: i for i, h in enumerate(head)}
        for i, h in enumerate(("timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw")):
            idx.setdefault(h, i)  # headerless or renamed columns: the documented order
        for line in r:
            if not line or not line[0].strip():
                continue
            def val(k):
                j = idx.get(k)
                s = line[j].strip() if j is not None and j < len(line) else ""
                return float(s) if s else None
            fr = val("frame")
            rows.append({"t": val("timestamp"), "frame": int(fr) if fr is not None else len(rows),
                         "p": [val("x"), val("y"), val("z")], "q": [val("qx"), val("qy"), val("qz"), val("qw")],
                         "K": [val("fx"), val("fy"), val("cx"), val("cy")]})
    if not rows:
        raise ValueError(f"{d.name}: odometry.csv has no poses")
    return rows


def camera_matrix(d: Path) -> list[float] | None:
    f = d / "camera_matrix.csv"
    if not f.is_file():
        return None
    m = np.loadtxt(f, delimiter=",").reshape(3, 3)
    return [float(m[0, 0]), float(m[1, 1]), float(m[0, 2]), float(m[1, 2])]


def quat_matrix(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_m(row: dict) -> list[float]:
    """Camera-to-world as a column-major 4x4 (the layout of the capture app's poses.json: translation at 12:15)."""
    T = np.eye(4)
    if None not in row["q"]:
        T[:3, :3] = quat_matrix(row["q"])
    T[:3, 3] = row["p"]
    return [float(v) for v in T.T.reshape(-1)]


def _probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets:format=duration",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = (j.get("streams") or [{}])[0]
    num, _, den = str(st.get("r_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {"width": int(st["width"]), "height": int(st["height"]), "fps": fps,
            "frames": int(st.get("nb_read_packets") or 0), "duration": float(j.get("format", {}).get("duration") or 0)}


def to_ar_tar(d: Path, out_tar: Path, candidates_per_s: float, long_side: int, work: Path) -> dict:
    """One scan -> a capture-app AR tar of candidate keyframes (every Nth video frame, resized) with ARKit poses.

    Returns a summary with how many candidates were written and the depth files they map to."""
    video = next(d / v for v in VIDEO_NAMES if (d / v).is_file())
    rows = read_odometry(d)
    info = _probe(video)
    fps = info["fps"] or (len(rows) / info["duration"] if info["duration"] else 30.0)
    step = max(1, int(round(fps / max(candidates_per_s, 0.1))))
    K0 = camera_matrix(d)
    s = min(1.0, long_side / max(info["width"], info["height"]))
    tmp = work / f"_stray_{d.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    vf = f"select='not(mod(n\\,{step}))'"
    if s < 1:
        vf += f",scale={round(info['width'] * s / 2) * 2}:{round(info['height'] * s / 2) * 2}:flags=lanczos"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-noautorotate", "-i", str(video), "-vf", vf,
                    "-fps_mode", "passthrough", "-q:v", "2", str(tmp / "%06d.jpg")], check=True)
    jpgs = sorted(tmp.glob("*.jpg"))
    by_frame = {r["frame"]: r for r in rows}
    depth_dir = d / "depth"
    keyframes, depth_map = [], {}
    for j, jp in enumerate(jpgs):
        n = j * step  # the nth video frame; odometry row n describes it
        row = by_frame.get(n) or (rows[n] if n < len(rows) else None)
        if row is None or None in row["p"]:
            continue
        K = row["K"] if None not in row["K"] else K0
        if K is None:
            raise ValueError(f"{d.name}: no intrinsics (neither per-frame columns nor camera_matrix.csv)")
        # intrinsics are for the full-resolution frame; the video can be smaller than that
        ref_w = max(info["width"], round(2 * K[2]))
        k = (info["width"] / ref_w) * s
        name = f"kf-{n:06d}.jpg"
        keyframes.append({"file": name, "m": pose_m(row), "t": round(1000 * ((row["t"] or 0) - (rows[0]["t"] or 0)), 1),
                          "fx": K[0] * k, "fy": K[1] * k, "cx": K[2] * k, "cy": K[3] * k, "src": f"{row['frame']:06d}"})
        for ext in (".png", ".npy"):
            if (depth_dir / f"{row['frame']:06d}{ext}").is_file():
                depth_map[f"{row['frame']:06d}"] = f"{row['frame']:06d}{ext}"
                break
    if not keyframes:
        raise ValueError(f"{d.name}: no video frame matched a pose")
    walked = float(np.sum(np.linalg.norm(np.diff(np.array([r["p"] for r in rows if None not in r["p"]]), axis=0), axis=1)))
    meta = {"keyframes": keyframes, "device": "iPhone LiDAR (Stray Scanner)", "source": "stray", "space": "arkit",
            "video": info, "step": step, "walkedM": round(walked, 2), "depthFrames": len(depth_map)}
    with tarfile.open(out_tar, "w") as tf:
        tf.add(str(_write(tmp / "poses.json", json.dumps(meta))), arcname="poses.json")
        for jp in jpgs:
            n = (int(jp.stem) - 1) * step
            tf.add(str(jp), arcname=f"kf-{n:06d}.jpg")
    shutil.rmtree(tmp, ignore_errors=True)
    return {"scan": d.name, "candidates": len(keyframes), "step": step, "video": info, "walkedM": round(walked, 2),
            "depthFrames": len(depth_map), "intrinsics": "per-frame" if rows[0]["K"][0] is not None else "camera_matrix.csv"}


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


def depth_index(d: Path, frames: list[dict]) -> dict:
    """For the keyframes frames kept (their 'src' ids), where the scan's depth and confidence maps are."""
    out = {}
    for fr in frames:
        sid = fr.get("src")
        if not sid:
            continue
        for ext in (".png", ".npy"):
            p = d / "depth" / f"{sid}{ext}"
            if p.is_file():
                c = d / "confidence" / f"{sid}.png"
                out[fr["frame"]] = {"depth": str(p), "confidence": str(c) if c.is_file() else None}
                break
    return out
