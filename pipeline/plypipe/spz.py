# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""spz: full-precision spherical harmonics in an SPZ v3 file.

SPZ stores every SH coefficient as a byte ((sh * 128) + 128), but the reference encoder, which splat-transform
follows, then rounds SH1 to 5 bits and SH2/SH3 to 4 bits so the file gzips smaller. On Dr Johnson that rounding was
nearly all of the SPZ loss: 31.0 dB trained, 30.0 dB through the SPZ, 30.1 dB with only the SH rounded. Readers
(Spark included) decode the plain byte, so writing the unrounded byte back costs a larger file and nothing else.

restore_sh rewrites the SH block of an SPZ that splat-transform wrote from a PLY, using that PLY's coefficients.
It checks first that both hold the same splats in the same order (count, SH degree, positions) and leaves the
file alone if not.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x5053474E  # "NGSP"
SH_COEFFS = {0: 0, 1: 3, 2: 8, 3: 15}


def _read_ply(path: Path) -> tuple[list[str], np.ndarray]:
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
        n = next(int(ln.split()[-1]) for ln in lines if ln.startswith("element vertex"))
        props = [ln.split() for ln in lines if ln.startswith("property")]
        if any(p[1] != "float" for p in props):
            raise ValueError(f"{path}: expected float properties only")
        names = [p[-1] for p in props]
        data = np.fromfile(f, dtype="<f4", count=n * len(names)).reshape(n, len(names))
    return names, data


def restore_sh(spz: Path, ply: Path, bits: int = 8, level: int = 9) -> dict:
    """Rewrite spz's SH bytes from ply, rounded to `bits` (8 = the full byte). Returns what it did (and why not)."""
    raw = gzip.decompress(Path(spz).read_bytes())
    magic, version, n, degree, frac, _flags, _ = struct.unpack("<IIIBBBB", raw[:16])
    k = SH_COEFFS.get(degree)
    if magic != MAGIC or version not in (2, 3) or k is None:
        return {"restored": False, "why": f"unsupported SPZ (version {version}, SH degree {degree})"}
    if k == 0:
        return {"restored": False, "why": "no SH bands"}
    rot_bytes = 4 if version == 3 else 3
    sh_off = 16 + n * (9 + 1 + 3 + 3 + rot_bytes)
    if len(raw) != sh_off + n * k * 3:
        return {"restored": False, "why": "unexpected SPZ size"}
    names, data = _read_ply(Path(ply))
    rest = [nm for nm in names if nm.startswith("f_rest_")]
    k_ply = len(rest) // 3
    if len(data) != n or k_ply < k:
        return {"restored": False, "why": f"PLY has {len(data)} splats / {k_ply} SH coefficients, SPZ {n} / {k}"}
    # same splats in the same order: the SPZ's 24-bit fixed-point positions must match the PLY's
    pos = np.frombuffer(raw, np.uint8, n * 9, 16).reshape(n, 3, 3).astype(np.int32)
    fixed = pos[..., 0] | (pos[..., 1] << 8) | (pos[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    xyz = data[:, [names.index("x"), names.index("y"), names.index("z")]]
    in_range = np.abs(xyz).max(axis=1) < (1 << 23) / (1 << frac)  # far splats saturate the fixed point
    err = np.abs(fixed / float(1 << frac) - xyz)[in_range]
    if err.size and err.max() > 2.0 / (1 << frac):
        return {"restored": False, "why": "the SPZ and the PLY hold different splats"}
    first = names.index(rest[0])
    # PLY: f_rest channel-major (all R coefficients, then G, then B); SPZ: per splat, coefficient-major, RGB inside
    sh = data[:, first:first + 3 * k_ply].reshape(n, 3, k_ply)[:, :, :k].transpose(0, 2, 1)
    step = float(1 << (8 - max(1, min(8, bits))))
    sh_bytes = np.clip(np.round(sh * 128.0 / step) * step + 128.0, 0, 255).astype(np.uint8)
    out = bytearray(raw)
    out[sh_off:] = sh_bytes.tobytes()
    before = Path(spz).stat().st_size
    Path(spz).write_bytes(gzip.compress(bytes(out), compresslevel=level, mtime=0))
    return {"restored": True, "shDegree": degree, "bits": bits, "mbBefore": round(before / 2**20, 2),
            "mbAfter": round(Path(spz).stat().st_size / 2**20, 2)}
