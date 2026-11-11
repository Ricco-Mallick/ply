# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""topview: the twin seen from straight above, ceiling cut away, as a transparent image (a dollhouse floor plan).

Reads a gsplat PLY or an SPZ v2/v3 and splats every Gaussian orthographically onto the floor plane on the CPU. A splat
is drawn with its real footprint (the x-z block of R S² Rᵀ), its opacity and its colour. Height decides what hides
what: the scene is cut at `cutY` (everything above it is dropped: ceiling, vault, lamps) and the rest is composited
from the top down, in thin height layers. Inside a layer splats add up as absorbers (optical depth, so their order
does not matter); between layers it is the usual front-to-back "over".

The image uses the occupancy grid's convention: row = +Z, column = +X, pixel (c, r) covers
x = origin[0] + (c + .5) * metersPerPixel, z = origin[1] + (r + .5) * metersPerPixel. Unmirrored, as seen from above.

    python -m plypipe.topview <twin folder | splat file> [--out DIR] [--cut M] [--mpp M] [--write-scene]
"""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPZ_MAGIC = 0x5053474E  # "NGSP"
SPZ_SH = {0: 0, 1: 3, 2: 8, 3: 15}
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435)
# SH coefficients (index into sh[1..15]) that survive for a camera looking straight down, dir = (0, -1, 0), and
# their signs there: band 1 sh[1]; band 2 sh[6], sh[8]; band 3 sh[9], sh[11]
DOWN_SH = ((0, C1), (5, -C2[2]), (7, -C2[4]), (8, C3[0]), (10, C3[2]))
CHUNK = 250_000


@dataclass
class Splats:
    """What the top view needs of each Gaussian. Metres, +Y up; cov = x-z block of the world covariance."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sxx: np.ndarray
    szz: np.ndarray
    sxz: np.ndarray
    syy: np.ndarray
    alpha: np.ndarray
    rgb: np.ndarray  # (n, 3) float32, 0..1, DC term plus the SH seen from straight above

    def __len__(self) -> int:
        return len(self.x)

    def take(self, keep: np.ndarray) -> "Splats":
        return Splats(*(a[keep] for a in (self.x, self.y, self.z, self.sxx, self.szz, self.sxz, self.syy, self.alpha, self.rgb)))


def _cov(scale: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, ...]:
    """x-z block and y variance of R diag(s²) Rᵀ from scales (metres, (n,3)) and unit quaternions (w, x, y, z)."""
    q = quat_wxyz / np.maximum(np.linalg.norm(quat_wxyz, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    r0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1)
    r1 = np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1)
    r2 = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)
    s2 = scale * scale
    return (r0 * r0 * s2).sum(1), (r2 * r2 * s2).sum(1), (r0 * r2 * s2).sum(1), (r1 * r1 * s2).sum(1)


def _splats(xyz, scale, quat, alpha, dc, sh_down) -> Splats:
    sxx, szz, sxz, syy = _cov(scale.astype(np.float32), quat.astype(np.float32))
    rgb = np.clip(dc * C0 + 0.5 + sh_down, 0.0, 1.0)
    return Splats(xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy(), sxx.astype(np.float32), szz.astype(np.float32),
                  sxz.astype(np.float32), syy.astype(np.float32), alpha.astype(np.float32), rgb.astype(np.float32))


def _join(parts: list[Splats]) -> Splats:
    return Splats(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                    ("x", "y", "z", "sxx", "szz", "sxz", "syy", "alpha", "rgb")))


def _unpack_quat_v3(r: np.ndarray) -> np.ndarray:
    """SPZ v3 'smallest three' rotations ((n, 4) uint8) -> (w, x, y, z) float32."""
    comp = r[:, 0].astype(np.uint32) | (r[:, 1].astype(np.uint32) << 8) | (r[:, 2].astype(np.uint32) << 16) | (
        r[:, 3].astype(np.uint32) << 24)
    largest = (comp >> 30).astype(np.int64)
    out = np.zeros((len(r), 4), np.float32)
    total = np.zeros(len(r), np.float32)
    for i in (3, 2, 1, 0):
        use = largest != i
        mag = (comp & 511).astype(np.float32)
        neg = ((comp >> 9) & 1).astype(bool)
        val = np.where(neg, -mag, mag) * (np.float32(np.sqrt(0.5)) / 511.0)
        out[use, i] = val[use]
        total += np.where(use, val * val, 0)
        comp = np.where(use, comp >> 10, comp)
    out[np.arange(len(r)), largest] = np.sqrt(np.maximum(1.0 - total, 0.0))
    return out[:, [3, 0, 1, 2]]  # stored x, y, z, w


def read_spz(path: Path) -> Splats:
    raw = gzip.decompress(Path(path).read_bytes())
    magic, version, n, degree, frac, _flags, _ = struct.unpack("<IIIBBBB", raw[:16])
    k = SPZ_SH.get(degree)
    if magic != SPZ_MAGIC or version not in (2, 3) or k is None:
        raise ValueError(f"{path}: unsupported SPZ (version {version}, SH degree {degree})")
    rot = 4 if version == 3 else 3
    o_pos, o_a = 16, 16 + 9 * n
    o_col, o_sc = o_a + n, o_a + 4 * n
    o_rot = o_sc + 3 * n
    o_sh = o_rot + rot * n
    if len(raw) < o_sh + n * k * 3:
        raise ValueError(f"{path}: truncated SPZ")
    buf = np.frombuffer(raw, np.uint8)
    parts = []
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        pos = buf[o_pos + 9 * lo:o_pos + 9 * hi].reshape(-1, 3, 3).astype(np.int32)
        fixed = pos[..., 0] | (pos[..., 1] << 8) | (pos[..., 2] << 16)
        fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
        xyz = fixed.astype(np.float32) / np.float32(1 << frac)
        alpha = buf[o_a + lo:o_a + hi].astype(np.float32) / 255.0
        dc = (buf[o_col + 3 * lo:o_col + 3 * hi].reshape(-1, 3).astype(np.float32) / 255.0 - 0.5) / 0.15
        scale = np.exp(buf[o_sc + 3 * lo:o_sc + 3 * hi].reshape(-1, 3).astype(np.float32) / 16.0 - 10.0)
        r = buf[o_rot + rot * lo:o_rot + rot * hi].reshape(-1, rot)
        if version == 3:
            quat = _unpack_quat_v3(r)
        else:
            xyz3 = r.astype(np.float32) / 127.5 - 1.0
            quat = np.concatenate([np.sqrt(np.maximum(1.0 - (xyz3 ** 2).sum(1, keepdims=True), 0.0)), xyz3], 1)
        down = np.zeros((hi - lo, 3), np.float32)
        if k:
            sh = buf[o_sh + 3 * k * lo:o_sh + 3 * k * hi].reshape(-1, k, 3)
            for j, c in DOWN_SH:
                if j < k:
                    down += c * (sh[:, j, :].astype(np.float32) - 128.0) / 128.0
        parts.append(_splats(xyz, scale, quat, alpha, dc, down))
    return _join(parts)


def _ply_header(path: Path) -> tuple[list[str], int, int]:
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            header += line
    lines = header.decode("ascii", "replace").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"{path}: not a little-endian binary PLY")
    props = [ln.split() for ln in lines if ln.startswith("property")]
    if any(p[1] != "float" for p in props):
        raise ValueError(f"{path}: expected float properties only")
    n = next(int(ln.split()[-1]) for ln in lines if ln.startswith("element vertex"))
    return [p[-1] for p in props], n, len(header)


def read_ply(path: Path) -> Splats:
    names, n, offset = _ply_header(Path(path))
    col = {nm: i for i, nm in enumerate(names)}
    need = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [c for c in need if c not in col]
    if missing:
        raise ValueError(f"{path}: not a gsplat PLY (missing {', '.join(missing)})")
    k = len([nm for nm in names if nm.startswith("f_rest_")]) // 3
    parts = []
    with open(path, "rb") as f:
        f.seek(offset)
        for lo in range(0, n, CHUNK):
            m = min(CHUNK, n - lo)
            data = np.fromfile(f, dtype="<f4", count=m * len(names)).reshape(m, len(names))
            xyz = data[:, [col["x"], col["y"], col["z"]]]
            dc = data[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]]
            alpha = 1.0 / (1.0 + np.exp(-data[:, col["opacity"]]))
            scale = np.exp(data[:, [col["scale_0"], col["scale_1"], col["scale_2"]]])
            quat = data[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]]
            down = np.zeros((m, 3), np.float32)
            for j, c in DOWN_SH:  # f_rest is channel-major: all R coefficients, then G, then B
                if j < k:
                    down += c * data[:, [col[f"f_rest_{ch * k + j}"] for ch in range(3)]]
            parts.append(_splats(xyz, scale, quat, alpha, dc, down))
    return _join(parts)


def read_splats(path: Path) -> Splats:
    path = Path(path)
    return read_spz(path) if path.suffix.lower() == ".spz" else read_ply(path)
