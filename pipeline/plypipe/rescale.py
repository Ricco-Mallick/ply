# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""rescale: change a finished twin's metric scale without retraining (plan section 9, 2r, the reference measurement).

A twin's scale is a guess (the camera 1.45 m above the floor) until someone measures something in it. Given a known
length and the same length measured in the twin, factor F = true / measured, and every metric thing in the twin
moves together about the scene origin (the floor point under the median camera, so the floor stays at y = 0):

    splats          positions x F, log-scales + ln F; rotations, opacities, colours and SH untouched (scene.spz,
                    scene_lite.spz, and in a job folder scene.ply, scene_clean.ply, scene_lite.ply, mesh.ply)
    scene.json      scale, transform, bounds, cameraPath, spawn and tour (x and z), the topview block (origin,
                    metres per pixel, cut height); scaleSource becomes "reference" and a `reference` block records F
    occupancy.json  origin, cellSize, the height bands, the walked radius, ranges and areas; the image is not touched:
                    a cell holds the same splats, it is just F times wider
    shell.json      wall ends and lengths, openings, ceiling, objects          places.json  every `p` and `half`
    job folders     data/sparse/0 (cameras, points; text and binary), dense_points.npz (points, voxel),
                    report.json and .done/align.json (the align block's scale, source and bounds)

What is not metric and stays: rotations, colours, opacities, SH, pixels (topview.webp, occupancy.png, width and height),
floorY (0), the eye height in spawn and the heights in tour (a person's, not the room's; the viewer reads the spawn's y
as the eye height), the walker radius, counts, `cameraHeight` (the prior that was used), reprojection errors, camera
intrinsics, dense/ (the frame before alignment, whose units are arbitrary), ar_poses.json (the phone's own metres),
train/ checkpoints.

SPZ stores log-scales as bytes in steps of 1/16: a splat's size moves by the nearest whole step, so sizes land within
3.1% of the exact factor (a systematic +3.0% at F = 1.1), and F then 1/F gives the original bytes back. Positions are
24-bit fixed point (1/4096 m by default): F then 1/F is within one step. PLY files are float32 and exact to rounding.

A rescale is atomic per folder: every new file is written under <folder>/.rescale/, a commit record is written, then
each file is renamed into place. A run that stops before the record leaves the original (the next run discards the
staging); one that stops after it is finished by the next run (`recover`), so a twin is never half-scaled.

    python -m plypipe rescale <folder> --factor 1.1  |  --known 2.04 --measured 1.85   [--force] [--dry-run]
"""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Callable

import numpy as np

from plypipe import colmap_txt, spz

LIMITS = (0.2, 5.0)  # a factor outside this is a mistyped length, not a correction (the camera-height prior is within 2x)
STAGING = ".rescale"
_FLOAT_DIGITS = 6


def factor_from(known_m: float, measured: float) -> float:
    """True length / the same length in the twin."""
    if not (known_m > 0 and measured > 0 and math.isfinite(known_m) and math.isfinite(measured)):
        raise ValueError(f"lengths must be positive numbers (known {known_m}, measured {measured})")
    return float(known_m) / float(measured)


def measure(p, q) -> float:
    """The distance between two points picked in the twin, in twin units: the `measured` of `--measured`, and all the
    viewer needs from this module (two taps, then the true length typed in, then factor_from)."""
    return float(np.linalg.norm(np.asarray(p, float) - np.asarray(q, float)))


# ---------------------------------------------------------------- json

def _r(v: float) -> float:
    return round(float(v), _FLOAT_DIGITS)


def _vec(v, f: float, keep: tuple[int, ...] = ()):  # noqa: ANN001
    return [x if i in keep else _r(x * f) for i, x in enumerate(v)]


def scene_json(scene: dict, f: float, reference: dict | None = None) -> dict:
    """scene.json after a rescale by f (a new dict)."""
    s = json.loads(json.dumps(scene))
    old = s.get("reference") if s.get("scaleSource") == "reference" else None
    first_source, first_scale = (old.get("previousSource"), old.get("previousScale")) if old else (s.get("scaleSource"), s.get("scale"))
    if isinstance(s.get("scale"), (int, float)):
        s["scale"] = s["scale"] * f
    if s.get("transform"):  # world -> scene: the scaling is applied after it, so rows 0-2 gain f
        s["transform"] = [[v * f for v in row] if i < 3 else list(row) for i, row in enumerate(s["transform"])]
    if s.get("bounds"):
        s["bounds"] = {k: _vec(v, f) for k, v in s["bounds"].items()}
    if s.get("cameraPath"):
        s["cameraPath"] = [_vec(p, f) for p in s["cameraPath"]]
    sp = s.get("spawn")
    if sp:  # y is the eye height the viewer reads (1.6 m), not a place in the room
        s["spawn"] = {k: (_vec(v, f, keep=(1,)) if isinstance(v, list) else v) for k, v in sp.items()}
    if s.get("tour"):
        s["tour"] = [{k: (_vec(v, f, keep=(1,)) if k in ("pos", "target") else v) for k, v in kf.items()} for kf in s["tour"]]
    top = s.get("topview")
    if top:
        for k in ("metersPerPixel", "cutY"):
            if k in top:
                top[k] = _r(top[k] * f)
        if "origin" in top:
            top["origin"] = _vec(top["origin"], f)
    ref = {"factor": _r(f * (old["factor"] if old else 1.0)), "appliedFactor": _r(f), "previousSource": first_source, "previousScale": first_scale,
           "scale": s.get("scale"), **(reference or {})}
    s["scaleSource"] = "reference"
    s["reference"] = ref
    return s


_LENGTHS = ("cellSize", "walkedRadius", "below", "maxRange", "voxel")
_LENGTH_LISTS = ("origin", "obstacleBand", "floorBand")
_AREAS = ("holeM2", "standableM2")


def occupancy_json(meta, f: float):  # noqa: ANN001
    """occupancy.json after a rescale: lengths x f, areas x f^2, at any depth (the walk and depth blocks repeat them).
    `radius` is the walker's, `width`/`height` are cells."""
    if isinstance(meta, dict):
        out = {}
        for k, v in meta.items():
            if k in _LENGTHS and isinstance(v, (int, float)):
                out[k] = _r(v * f)
            elif k in _AREAS and isinstance(v, (int, float)):
                out[k] = _r(v * f * f)
            elif k in _LENGTH_LISTS and isinstance(v, list):
                out[k] = [_r(x * f) for x in v]
            else:
                out[k] = occupancy_json(v, f)
        return out
    if isinstance(meta, list):
        return [occupancy_json(v, f) for v in meta]
    return meta


def shell_json(shell: dict, f: float) -> dict:
    s = json.loads(json.dumps(shell))
    for k in ("floorY", "ceilingY"):
        if isinstance(s.get(k), (int, float)):
            s[k] = _r(s[k] * f)
    for w in s.get("walls") or []:
        w["a"], w["b"] = _vec(w["a"], f), _vec(w["b"], f)
        if "length" in w:
            w["length"] = _r(w["length"] * f)
    for o in s.get("openings") or []:
        for k in ("u", "width", "sill", "height"):
            if isinstance(o.get(k), (int, float)):
                o[k] = _r(o[k] * f)
    for o in s.get("objects") or []:
        for k in ("center", "half"):
            if o.get(k) is not None:
                o[k] = _vec(o[k], f)
    return s


def places_json(places: dict, f: float) -> dict:
    s = json.loads(json.dumps(places))
    for pl in s.get("places") or []:
        if pl.get("p") is not None:
            pl["p"] = _vec(pl["p"], f)
        if (pl.get("ai") or {}).get("half") is not None:
            pl["ai"]["half"] = _vec(pl["ai"]["half"], f)
    return s


def align_report(a: dict, f: float, scene: dict) -> dict:
    """The align stage's report after a rescale: scale, source, bounds and floor residual, kept true to the files."""
    a = json.loads(json.dumps(a))
    a.update({"scale": scene.get("scale"), "scaleSource": "reference"})
    if a.get("bounds"):
        a["bounds"] = {k: _vec(v, f) for k, v in a["bounds"].items()}
    if isinstance(a.get("floorResidualMedianM"), (int, float)):
        a["floorResidualMedianM"] = a["floorResidualMedianM"] * f
    return a


# ---------------------------------------------------------------- binary files

def _spz(src: Path, dst: Path, f: float) -> None:
    raw = bytearray(gzip.decompress(src.read_bytes()))
    magic, version, n, degree, frac, _flags, _ = struct.unpack("<IIIBBBB", bytes(raw[:16]))
    if magic != spz.MAGIC or version not in (2, 3):
        raise ValueError(f"{src}: unsupported SPZ (version {version})")
    if n:
        b = np.frombuffer(bytes(raw[16:16 + 9 * n]), np.uint8).reshape(n, 3, 3).astype(np.int32)
        fixed = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
        x = np.where(fixed & 0x800000, fixed - (1 << 24), fixed) / float(1 << frac) * f
        while frac > 0 and np.abs(x).max() * (1 << frac) > (1 << 23) - 1:  # a far splat would saturate: fewer fraction bits
            frac -= 1
        q = np.clip(np.round(x * (1 << frac)), -(1 << 23), (1 << 23) - 1).astype(np.int64) & 0xFFFFFF
        pos = np.stack([q & 255, (q >> 8) & 255, (q >> 16) & 255], axis=-1).astype(np.uint8)
        raw[16:16 + 9 * n] = pos.tobytes()
        raw[13] = frac
        o = 16 + 13 * n  # positions 9n, alphas n, colours 3n, then the log-scales: (byte / 16) - 10
        sc = np.frombuffer(bytes(raw[o:o + 3 * n]), np.uint8).astype(np.float64)
        raw[o:o + 3 * n] = np.clip(np.round(sc + 16.0 * math.log(f)), 0, 255).astype(np.uint8).tobytes()
    dst.write_bytes(gzip.compress(bytes(raw), compresslevel=9, mtime=0))


_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _ply(src: Path, dst: Path, f: float) -> None:
    """Binary little-endian PLY (a splat file, or a mesh whose faces follow the vertices): x y z and scale_0..2 change."""
    with open(src, "rb") as fh:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = fh.readline()
            if not line:
                raise ValueError(f"{src}: no end_header")
            header += line
        lines = header.decode("ascii", "replace").splitlines()
        if "format binary_little_endian 1.0" not in lines:
            raise ValueError(f"{src}: not a little-endian binary PLY")
        first = next((i for i, ln in enumerate(lines) if ln.startswith("element")), None)
        if first is None or lines[first].split()[1] != "vertex":
            raise ValueError(f"{src}: the vertex element must come first")
        n = int(lines[first].split()[2])
        fields = []
        for ln in lines[first + 1:]:
            if ln.startswith("element"):
                break
            if ln.startswith("property"):
                p = ln.split()
                if p[1] == "list" or p[1] not in _PLY_TYPES:
                    raise ValueError(f"{src}: unsupported vertex property {ln!r}")
                fields.append((p[2], "<" + _PLY_TYPES[p[1]]))
        data = np.fromfile(fh, dtype=np.dtype(fields), count=n)
        tail = fh.read()
    names = data.dtype.names or ()
    for k in ("x", "y", "z"):
        if k in names:
            data[k] = data[k].astype(np.float64) * f
    for k in ("scale_0", "scale_1", "scale_2"):
        if k in names:
            data[k] = data[k].astype(np.float64) + math.log(f)
    dst.write_bytes(header + data.tobytes() + tail)


def _dense_points(src: Path, dst: Path, f: float) -> None:
    z = np.load(src)
    arrays = {k: z[k] for k in z.files}
    arrays["points"] = (arrays["points"].astype(np.float64) * f).astype(arrays["points"].dtype)
    if "voxel" in arrays:
        arrays["voxel"] = (arrays["voxel"].astype(np.float64) * f).astype(arrays["voxel"].dtype)
    with open(dst, "wb") as fh:
        np.savez_compressed(fh, **arrays)


def _sparse(src: Path, dst: Path, f: float) -> None:
    model = colmap_txt.transformed(colmap_txt.read_model(src), np.diag([f, f, f, 1.0]))
    colmap_txt.write_model(model, dst)
    colmap_txt.write_model_bin(model, dst)


# ---------------------------------------------------------------- the folder

def _json_file(fn: Callable[[dict, float], dict]) -> Callable[[Path, Path, float, dict], None]:
    return lambda src, dst, f, ctx: dst.write_text(json.dumps(fn(json.loads(src.read_text()), f), indent=2))


# name (relative to the folder) -> how to write the rescaled copy. scene.json and report.json need the folder's context
# and are handled in rescale(). A new metric file (a layout, a mesh) gets one line here.
HANDLERS: dict[str, Callable[[Path, Path, float, dict], None]] = {
    "scene.spz": lambda s, d, f, c: _spz(s, d, f),
    "scene_lite.spz": lambda s, d, f, c: _spz(s, d, f),
    "scene.ply": lambda s, d, f, c: _ply(s, d, f),
    "scene_clean.ply": lambda s, d, f, c: _ply(s, d, f),
    "scene_lite.ply": lambda s, d, f, c: _ply(s, d, f),
    "mesh.ply": lambda s, d, f, c: _ply(s, d, f),
    "occupancy.json": _json_file(occupancy_json),
    "shell.json": _json_file(shell_json),
    "places.json": _json_file(places_json),
    "dense_points.npz": lambda s, d, f, c: _dense_points(s, d, f),
    "data/sparse/0": lambda s, d, f, c: _sparse(s, d, f),
}


def _commit(folder: Path, rels: list[str]) -> None:
    """Move each staged item over its original. Idempotent: an item already moved has nothing staged any more."""
    st = folder / STAGING
    for rel in rels:
        new, cur = st / "new" / rel, folder / rel
        if not (new.exists() or new.is_symlink()):
            continue
        cur.parent.mkdir(parents=True, exist_ok=True)
        if cur.is_dir():  # a directory cannot be renamed over a non-empty one: park the old one first
            old = st / "old" / rel
            old.parent.mkdir(parents=True, exist_ok=True)
            if old.exists():
                shutil.rmtree(old)
            os.replace(cur, old)
        os.replace(new, cur)


def recover(folder: Path) -> str | None:
    """Finish an interrupted rescale (its commit record exists) or drop a half-written staging. Returns what it did."""
    folder = Path(folder)
    st = folder / STAGING
    if not st.exists():
        return None
    record = st / "commit.json"
    if record.exists():
        _commit(folder, json.loads(record.read_text())["items"])
        shutil.rmtree(st)
        return "finished the interrupted rescale"
    shutil.rmtree(st)
    return "discarded an unfinished rescale"


def items(folder: Path) -> list[str]:
    """The metric files this folder has, by relative name (scene.json and report.json are added by rescale())."""
    folder = Path(folder)
    return [rel for rel in HANDLERS if (folder / rel).exists()]


def rescale(folder: Path, factor: float, *, force: bool = False, reference: dict | None = None, dry_run: bool = False) -> dict:
    """Rescale the twin in `folder` by `factor` about the scene origin. `reference` is recorded in scene.json
    ({"knownM": ..., "measuredM": ...}). Returns {"factor", "files", "scale", "previousSource", ...}."""
    folder = Path(folder)
    f = float(factor)
    if not (math.isfinite(f) and f > 0):
        raise ValueError(f"the factor must be a positive number, not {factor!r}")
    if not (LIMITS[0] <= f <= LIMITS[1]) and not force:
        raise ValueError(f"factor {f:.4g} is outside {LIMITS[0]} to {LIMITS[1]}: a mistyped length? --force applies it anyway")
    scene_p = folder / "scene.json"
    if not scene_p.exists():
        raise FileNotFoundError(f"{scene_p} not found: not a twin folder")
    note = None if dry_run else recover(folder)
    scene = json.loads(scene_p.read_text())
    new_scene = scene_json(scene, f, reference)
    names = items(folder)
    files = ["scene.json", *names, *(["report.json"] if (folder / "report.json").exists() else []),
             *([".done/align.json"] if (folder / ".done" / "align.json").exists() else [])]
    out = {"factor": f, "files": files, "scale": new_scene.get("scale"), "previousScale": scene.get("scale"), "previousSource": scene.get("scaleSource"),
           "scaleSource": "reference", **({"recovered": note} if note else {})}
    if dry_run:
        return out
    st = folder / STAGING
    new = st / "new"
    try:
        new.mkdir(parents=True)
        ctx = {"scene": scene}
        for rel in names:
            dst = new / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            HANDLERS[rel](folder / rel, dst, f, ctx)
        (new / "scene.json").write_text(json.dumps(new_scene, indent=2))
        for rel in files[1 + len(names):]:
            (new / rel).parent.mkdir(parents=True, exist_ok=True)
            doc = json.loads((folder / rel).read_text())
            if rel == "report.json":
                if isinstance(doc.get("align"), dict):
                    doc["align"] = align_report(doc["align"], f, new_scene)
                doc.setdefault("rescale", []).append({"factor": f, "scale": new_scene.get("scale"), **(reference or {})})
            else:
                doc = align_report(doc, f, new_scene)
            (new / rel).write_text(json.dumps(doc, indent=2, default=str))
        tmp = st / "commit.tmp"
        tmp.write_text(json.dumps({"factor": f, "items": files}))
        os.replace(tmp, st / "commit.json")  # from here on the rescale is finished, by this run or the next
    except BaseException:
        shutil.rmtree(st, ignore_errors=True)
        raise
    _commit(folder, files)
    shutil.rmtree(st)
    return out


def main(args) -> int:  # noqa: ANN001
    """The `plypipe rescale` subcommand: prints what changed as JSON."""
    known, measured = args.known, args.measured
    if args.factor is not None and (known is not None or measured is not None):
        raise SystemExit("rescale: give --factor, or --known with --measured, not both")
    if args.factor is None:
        if known is None or measured is None:
            raise SystemExit("rescale: give --factor F, or --known M_true --measured M_twin")
        factor = factor_from(known, measured)
        reference = {"knownM": known, "measuredM": measured}
    else:
        factor, reference = args.factor, None
    try:
        res = rescale(args.folder, factor, force=args.force, reference=reference, dry_run=args.dry_run)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"rescale: {e}")
    print(json.dumps(res, indent=2))
    return 0
