# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""topview: the twin seen from straight above, ceiling cut away, as a transparent image (a dollhouse floor plan).

Reads a gsplat PLY or an SPZ v2/v3 and splats every Gaussian orthographically onto the floor plane on the CPU. A splat
is drawn with its real footprint (the x-z block of R S² Rᵀ), its opacity and its colour (the SH DC term: the higher
bands are fitted to the horizontal walk and, evaluated for a camera looking straight down, only muddied the floor and
darkened the furniture). Height decides what hides what: the scene is cut at `cutY` (everything above it is dropped:
ceiling, vault, lamps) and the rest is composited from the top down in thin height layers. Inside a layer splats add up
as absorbers (optical depth, so their order does not matter); between layers it is the usual front-to-back "over".

What is drawn is limited to the room: the walls show as a ring of small, dense splats, and the model's fog of large
streaks beyond them is cut away by filling that ring. The image uses the occupancy grid's convention: row = +Z,
column = +X, pixel (c, r) covers x = origin[0] + (c + .5) * metersPerPixel, z = origin[1] + (r + .5) * metersPerPixel.
That is a true, unmirrored view from above.

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

import cv2
import numpy as np

SPZ_MAGIC = 0x5053474E  # "NGSP"
SPZ_SH = {0: 0, 1: 3, 2: 8, 3: 15}
C0 = 0.28209479177387814
CHUNK = 250_000

CELL = 0.1           # footprint grid, metres
RESOLVED = 0.05      # a splat whose footprint is smaller than this (sigma, metres) sits on a real surface
LAYER = 0.025        # height layers, metres
MIN_ALPHA = 0.02     # splats fainter than this draw nothing worth keeping
LOWPASS = 0.3        # px² added to each footprint's covariance (the 3DGS anti-aliasing term): keeps a wall's
                     # edge-on sliver at about a pixel instead of letting it vanish
LEVELS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])  # footprint half-sizes the stamps come in
PAIRS = 2_000_000    # splat-pixel pairs worked on at once
FLOOR_LO = -0.4      # the floor is at y = 0; splats under it are reflections and noise
DEFAULT_CUT = 1.8
IMAGE = "topview.webp"


@dataclass
class Splats:
    """What the top view needs of each Gaussian. Metres, +Y up; s** is the x-z block of the world covariance."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sxx: np.ndarray
    szz: np.ndarray
    sxz: np.ndarray
    syy: np.ndarray
    alpha: np.ndarray
    rgb: np.ndarray  # (n, 3) float32, 0..1

    def __len__(self) -> int:
        return len(self.x)


_FIELDS = ("x", "y", "z", "sxx", "szz", "sxz", "syy", "alpha", "rgb")


def _splats(xyz, scale, quat_wxyz, alpha, dc) -> Splats:
    q = quat_wxyz / np.maximum(np.linalg.norm(quat_wxyz, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    r0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1)
    r1 = np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1)
    r2 = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)
    s2 = scale * scale
    f32 = np.float32
    return Splats(xyz[:, 0].astype(f32), xyz[:, 1].astype(f32), xyz[:, 2].astype(f32),
                  (r0 * r0 * s2).sum(1).astype(f32), (r2 * r2 * s2).sum(1).astype(f32), (r0 * r2 * s2).sum(1).astype(f32),
                  (r1 * r1 * s2).sum(1).astype(f32), alpha.astype(f32), np.clip(dc * C0 + 0.5, 0, 1).astype(f32))


def _join(parts: list[Splats]) -> Splats:
    return Splats(*(np.concatenate([getattr(p, f) for p in parts]) for f in _FIELDS))


def _unpack_quat_v3(r: np.ndarray) -> np.ndarray:
    """SPZ v3 'smallest three' rotations ((n, 4) uint8: x, y, z, w) -> (w, x, y, z) float32."""
    comp = r[:, 0].astype(np.uint32) | (r[:, 1].astype(np.uint32) << 8) | (r[:, 2].astype(np.uint32) << 16) | (
        r[:, 3].astype(np.uint32) << 24)
    largest = (comp >> 30).astype(np.int64)
    out = np.zeros((len(r), 4), np.float32)
    total = np.zeros(len(r), np.float32)
    for i in (3, 2, 1, 0):
        use = largest != i
        mag = (comp & 511).astype(np.float32)
        val = np.where((comp >> 9) & 1 == 1, -mag, mag) * (np.float32(np.sqrt(0.5)) / 511.0)
        out[use, i] = val[use]
        total += np.where(use, val * val, 0)
        comp = np.where(use, comp >> 10, comp)
    out[np.arange(len(r)), largest] = np.sqrt(np.maximum(1.0 - total, 0.0))
    return out[:, [3, 0, 1, 2]]


def read_spz(path: Path) -> Splats:
    raw = gzip.decompress(Path(path).read_bytes())
    magic, version, n, degree, frac, _flags, _ = struct.unpack("<IIIBBBB", raw[:16])
    if magic != SPZ_MAGIC or version not in (2, 3) or degree not in SPZ_SH:
        raise ValueError(f"{path}: unsupported SPZ (version {version}, SH degree {degree})")
    rot = 4 if version == 3 else 3
    o_pos, o_a = 16, 16 + 9 * n
    o_col, o_sc = o_a + n, o_a + 4 * n
    o_rot = o_sc + 3 * n
    if len(raw) < o_rot + rot * n:
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
            v = r.astype(np.float32) / 127.5 - 1.0
            quat = np.concatenate([np.sqrt(np.maximum(1.0 - (v * v).sum(1, keepdims=True), 0.0)), v], 1)
        parts.append(_splats(xyz, scale, quat, alpha, dc))
    return _join(parts)


def read_ply(path: Path) -> Splats:
    path = Path(path)
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
        names = [p[-1] for p in props]
        n = next(int(ln.split()[-1]) for ln in lines if ln.startswith("element vertex"))
        col = {nm: i for i, nm in enumerate(names)}
        need = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"]
        if any(c not in col for c in need):
            raise ValueError(f"{path}: not a gsplat PLY (needs {', '.join(need)})")
        ix = [[col[c] for c in need[i:j]] for i, j in ((0, 3), (3, 6), (6, 7), (7, 10), (10, 14))]
        parts = []
        for lo in range(0, n, CHUNK):
            m = min(CHUNK, n - lo)
            data = np.fromfile(f, dtype="<f4", count=m * len(names)).reshape(m, len(names))
            xyz, dc, op, scale, quat = (data[:, i] for i in ix)
            parts.append(_splats(xyz, np.exp(scale), quat, 1.0 / (1.0 + np.exp(-op[:, 0])), dc))
    return _join(parts)


def read_splats(path: Path) -> Splats:
    path = Path(path)
    return read_spz(path) if path.suffix.lower() == ".spz" else read_ply(path)


# ------------------------------------------------------------------ the room's footprint

def _sigma_major(s: Splats) -> np.ndarray:
    return np.sqrt(0.5 * (s.sxx + s.szz) + np.sqrt(0.25 * (s.sxx - s.szz) ** 2 + s.sxz ** 2))


class Footprint:
    """Cells (CELL metres) that hold the room: walls, floor and what stands on it. Everything else is the model's fog."""

    def __init__(self, s: Splats, cut: float, hint: dict | None = None):
        lo, hi = self._window(s, hint)
        self.x0, self.z0 = float(lo[0]), float(lo[1])
        self.w, self.h = (int(np.ceil((hi[i] - lo[i]) / CELL)) + 1 for i in (0, 1))
        near = (s.y < cut) & (s.y >= FLOOR_LO) & (s.x >= lo[0]) & (s.x < hi[0]) & (s.z >= lo[1]) & (s.z < hi[1])
        solid = near & (s.alpha > 0.1)
        small = solid & (_sigma_major(s) < RESOLVED)
        self.cells = self._fill(self._count(s, small) >= 2)
        if self.cells.sum() < 25:  # a model of big splats only: take them all
            self.cells = self._fill(self._count(s, solid) >= 2)

    @staticmethod
    def _window(s: Splats, hint: dict | None):
        """Search window in x-z: the published bounds plus a metre, else the central 99 % of the splats plus a metre."""
        good = (s.alpha > 0.1) & (s.y >= FLOOR_LO)
        if hint and hint.get("bounds"):
            b = hint["bounds"]
            return np.array([b["min"][0], b["min"][2]]) - 1.0, np.array([b["max"][0], b["max"][2]]) + 1.0
        px, pz = (np.percentile(a[good], [0.5, 99.5]) for a in (s.x, s.z))
        return np.array([px[0], pz[0]]) - 1.0, np.array([px[1], pz[1]]) + 1.0

    def _count(self, s: Splats, sel: np.ndarray) -> np.ndarray:
        ix = ((s.x[sel] - self.x0) / CELL).astype(np.int64)
        iz = ((s.z[sel] - self.z0) / CELL).astype(np.int64)
        ok = (ix >= 0) & (ix < self.w) & (iz >= 0) & (iz < self.h)
        return np.bincount(iz[ok] * self.w + ix[ok], minlength=self.w * self.h).reshape(self.h, self.w)

    def _fill(self, occ: np.ndarray) -> np.ndarray:
        """Specks dropped, door-sized gaps in the wall ring closed, the inside of the ring filled."""
        occ = occ.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(occ, np.ones((3, 3), np.uint8)), connectivity=8)
        if n <= 1:
            return np.zeros_like(occ, bool)
        area = stats[1:, cv2.CC_STAT_AREA]
        keep = np.flatnonzero(area >= max(6, 0.05 * area.max())) + 1
        blob = np.isin(labels, keep).astype(np.uint8)
        pad = 8
        blob = np.pad(blob, pad)
        k = int(round(0.5 / CELL))
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        flood = blob.copy()
        cv2.floodFill(flood, np.zeros((blob.shape[0] + 2, blob.shape[1] + 2), np.uint8), (0, 0), 1)
        filled = blob | (flood == 0)
        filled = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return filled[pad:-pad, pad:-pad].astype(bool)

    def extent(self, margin: float = 0.15) -> tuple[float, float, float, float]:
        """(x0, z0, x1, z1) of the room, metres."""
        rows, cols = np.flatnonzero(self.cells.any(1)), np.flatnonzero(self.cells.any(0))
        if not len(rows):
            raise ValueError("no room found in the splats")
        return (self.x0 + cols[0] * CELL - margin, self.z0 + rows[0] * CELL - margin,
                self.x0 + (cols[-1] + 1) * CELL + margin, self.z0 + (rows[-1] + 1) * CELL + margin)

    def mask(self, x0: float, z0: float, mpp: float, width: int, height: int) -> np.ndarray:
        """Footprint as a soft 0..1 image on the pixel grid (pixel centres sampled bilinearly from the cells)."""
        gx = ((x0 + (np.arange(width) + 0.5) * mpp - self.x0) / CELL - 0.5).astype(np.float32)
        gz = ((z0 + (np.arange(height) + 0.5) * mpp - self.z0) / CELL - 0.5).astype(np.float32)
        cells = cv2.copyMakeBorder(self.cells.astype(np.float32), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        return cv2.remap(cells, gx[None, :].repeat(height, 0) + 1, gz[:, None].repeat(width, 1) + 1, cv2.INTER_LINEAR)


# ------------------------------------------------------------------ the cut

def choose_cut(s: Splats, fp: Footprint, scene: dict | None = None) -> float:
    """Height of the cutting plane. 1.8 m keeps furniture and walls and opens doorways (door frames end near 2 m); a
    low room is cut 0.5 m under its ceiling, found as the strongest flat, horizontal slab of splats above 1.8 m."""
    x0, z0, x1, z1 = fp.extent(0.0)
    here = (s.alpha > 0.2) & (s.x > x0) & (s.x < x1) & (s.z > z0) & (s.z < z1) & (s.y > 1.0)
    flat = here & (s.syy < 0.05 * 0.5 * (s.sxx + s.szz))
    if not flat.any():
        return DEFAULT_CUT
    mass = s.alpha * np.sqrt(np.maximum(s.sxx * s.szz - s.sxz ** 2, 1e-10))
    edges = np.arange(1.0, max(float(s.y[flat].max()), 2.0) + 0.1, 0.1)
    hist, _ = np.histogram(s.y[flat], edges, weights=mass[flat])
    ceiling = edges[int(np.argmax(hist[edges[:-1] >= 1.8])) + int(np.sum(edges[:-1] < 1.8))] + 0.05 \
        if (edges[:-1] >= 1.8).any() else None
    if ceiling is None:
        return DEFAULT_CUT
    return float(np.clip(min(DEFAULT_CUT, ceiling - 0.5), 1.4, DEFAULT_CUT))


# ------------------------------------------------------------------ the rasteriser

def _stamp_level(r: np.ndarray) -> np.ndarray:
    return np.minimum(np.searchsorted(LEVELS, np.ceil(r) - 1e-6), len(LEVELS) - 1)


def rasterize(s: Splats, x0: float, z0: float, mpp: float, width: int, height: int, cut: float) -> tuple[np.ndarray, np.ndarray]:
    """(rgb (H, W, 3) float32 un-premultiplied, alpha (H, W) float32) of the splats below `cut`, seen from above."""
    hw = width * height
    inv = 1.0 / (mpp * mpp)
    u = (s.x - x0) / mpp - 0.5
    v = (s.z - z0) / mpp - 0.5
    reach = LEVELS[-1] + 2  # a splat further outside the image than its largest stamp cannot touch it
    keep = (s.alpha >= MIN_ALPHA) & (s.y < cut) & (s.y >= FLOOR_LO) & (u > -reach) & (u < width + reach) \
        & (v > -reach) & (v < height + reach)
    sel = np.flatnonzero(keep)
    layer = ((cut - s.y[sel]) / LAYER).astype(np.int32)
    order = np.argsort(layer, kind="stable")
    sel, layer = sel[order], layer[order]
    first = np.searchsorted(layer, np.arange(layer[-1] + 2)) if len(sel) else np.zeros(1, np.int64)

    trans = np.ones(hw, np.float32)
    premult = np.zeros((hw, 3), np.float32)
    for li in range(len(first) - 1):
        ids = sel[first[li]:first[li + 1]]
        if not len(ids):
            continue
        tau, tau_c = _layer(s, ids, u[ids], v[ids], inv, width, height)
        if tau is None:
            continue
        t = tau.astype(np.float32)
        premult += (trans * (1.0 - np.exp(-t)))[:, None] * (tau_c.astype(np.float32) / np.maximum(t, 1e-6)[:, None])
        trans *= np.exp(-t)
    alpha = 1.0 - trans
    rgb = premult / np.maximum(alpha, 1e-6)[:, None]
    return rgb.reshape(height, width, 3), alpha.reshape(height, width)


def _layer(s: Splats, ids: np.ndarray, u: np.ndarray, v: np.ndarray, inv: float, width: int, height: int):
    """Optical depth and tau-weighted colour that the splats `ids` (one height layer) add to every pixel."""
    sxx = s.sxx[ids] * inv + LOWPASS
    szz = s.szz[ids] * inv + LOWPASS
    sxz = s.sxz[ids] * inv
    det = sxx * szz - sxz * sxz
    ca, cb, cc = szz / det, -sxz / det, sxx / det
    code = _stamp_level(3 * np.sqrt(sxx)) * len(LEVELS) + _stamp_level(3 * np.sqrt(szz))
    alpha, rgb = s.alpha[ids], s.rgb[ids]
    cu, cv = np.rint(u).astype(np.int32), np.rint(v).astype(np.int32)
    hw = width * height
    tau_acc = np.zeros(hw)
    col_acc = np.zeros((hw, 3))
    pend: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    pending = 0

    def flush():
        nonlocal pending
        if not pend:
            return
        idx = np.concatenate([p[0] for p in pend])
        tau = np.concatenate([p[1] for p in pend])
        col = np.concatenate([p[2] for p in pend])
        tau_acc.__iadd__(np.bincount(idx, tau, minlength=hw))
        for ch in range(3):
            col_acc[:, ch] += np.bincount(idx, tau * col[:, ch], minlength=hw)
        pend.clear()
        pending = 0

    for c in np.unique(code):
        members = np.flatnonzero(code == c)
        rx, ry = int(LEVELS[c // len(LEVELS)]), int(LEVELS[c % len(LEVELS)])
        step = max(1, PAIRS // ((2 * rx + 1) * (2 * ry + 1)))
        dx = np.arange(-rx, rx + 1, dtype=np.int32)[None, :, None]
        dy = np.arange(-ry, ry + 1, dtype=np.int32)[None, None, :]
        for lo in range(0, len(members), step):
            m = members[lo:lo + step]
            col_px = cu[m][:, None, None] + dx
            row_px = cv[m][:, None, None] + dy
            ddx = col_px - u[m][:, None, None]
            ddy = row_px - v[m][:, None, None]
            power = -0.5 * (ca[m][:, None, None] * ddx * ddx + cc[m][:, None, None] * ddy * ddy) \
                - cb[m][:, None, None] * ddx * ddy
            w = alpha[m][:, None, None] * np.exp(power)
            ok = (w > 1.0 / 255) & (col_px >= 0) & (col_px < width) & (row_px >= 0) & (row_px < height)
            owner = np.broadcast_to(np.arange(len(m))[:, None, None], ok.shape)[ok]
            pend.append(((row_px * width + col_px)[ok], -np.log1p(-np.minimum(w[ok], 0.995)), rgb[m][owner]))
            pending += len(pend[-1][0])
            if pending >= PAIRS:
                flush()
    flush()
    return (tau_acc, col_acc) if tau_acc.any() else (None, None)


# ------------------------------------------------------------------ the picture

def finish(rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGBA uint8 (BGRA channel order, for OpenCV): inside the footprint every pixel is opaque, where the splats left
    a hole the colour is inpainted from its surroundings; outside the footprint it is transparent."""
    bgr = np.clip(rgb[..., ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    inside = mask > 0.5
    known = (alpha >= 0.35) & inside
    hole = inside & ~known
    if hole.any() and known.any():
        bgr = cv2.inpaint(bgr, hole.astype(np.uint8) * 255, 4, cv2.INPAINT_TELEA)
    soft = np.clip((mask - 0.25) * 2.0, 0.0, 1.0)  # a one-pixel feathered edge
    return np.dstack([bgr, (soft * 255.0 + 0.5).astype(np.uint8)])


def render(splats: Splats, scene: dict | None = None, cut: float | None = None, mpp: float | None = None) -> tuple[np.ndarray, dict]:
    """(BGRA image, metadata without file names) of the splats seen from above."""
    scene = scene or {}
    cut = float(cut) if cut is not None else None
    fp = Footprint(splats, cut if cut is not None else 3.0, scene)
    cut_used = cut if cut is not None else choose_cut(splats, fp, scene)
    if cut is None:  # the footprint was found with everything up to 3 m, the rendering is limited by the cut
        fp = Footprint(splats, cut_used, scene)
    x0, z0, x1, z1 = fp.extent()
    mpp = float(mpp) if mpp else max(0.02, max(x1 - x0, z1 - z0) / 2048)
    width, height = int(np.ceil((x1 - x0) / mpp)), int(np.ceil((z1 - z0) / mpp))
    rgb, alpha = rasterize(splats, x0, z0, mpp, width, height, cut_used)
    image = finish(rgb, alpha, fp.mask(x0, z0, mpp, width, height))
    meta = {"version": 1, "origin": [round(x0, 4), round(z0, 4)], "metersPerPixel": round(mpp, 5), "width": width,
            "height": height, "cutY": round(cut_used, 3), "splats": len(splats)}
    return image, meta


def write_image(image: np.ndarray, path: Path) -> None:
    ok, buf = cv2.imencode(path.suffix, image, [cv2.IMWRITE_WEBP_QUALITY, 90] if path.suffix == ".webp" else [])
    if not ok:
        raise RuntimeError(f"could not encode {path.name}")
    path.write_bytes(buf.tobytes())


def find_splats(folder: Path) -> Path:
    for name in ("scene_clean.ply", "scene.spz", "scene.ply", "scene_lite.spz"):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"{folder}: no scene.spz or scene.ply")


def build(source: Path, out: Path, scene: dict | None = None, cut: float | None = None, mpp: float | None = None,
          name: str = IMAGE) -> dict:
    """Render `source` and write the image into the folder `out`. Returns the metadata block for scene.json."""
    image, meta = render(read_splats(source), scene, cut, mpp)
    out.mkdir(parents=True, exist_ok=True)
    write_image(image, out / name)
    return {**meta, "image": name, "source": source.name}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="plypipe.topview", description="Top view (floor plan image) of a splat twin.")
    ap.add_argument("source", type=Path, help="a twin folder (scene.json + scene.spz or scene.ply) or a splat file")
    ap.add_argument("--out", type=Path, help="output folder (default: next to the source)")
    ap.add_argument("--cut", type=float, help="cutting height in metres (default: chosen from the model)")
    ap.add_argument("--mpp", type=float, help="metres per pixel (default: about 2 cm, at most 2048 px on the long side)")
    ap.add_argument("--write-scene", action="store_true", help="record the result as scene.json's `topview` block")
    a = ap.parse_args(argv)
    folder = a.source if a.source.is_dir() else a.source.parent
    src = find_splats(a.source) if a.source.is_dir() else a.source
    scene_path = folder / "scene.json"
    scene = json.loads(scene_path.read_text()) if scene_path.exists() else {}
    meta = build(src, a.out or folder, scene, a.cut, a.mpp)
    if a.write_scene:
        if not scene_path.exists():
            print(f"{scene_path} not found: nothing to write into", file=sys.stderr)
            return 2
        scene["topview"] = meta
        scene_path.write_text(json.dumps(scene, indent=2))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
