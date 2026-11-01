# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""clean: floaters out, with statistics that cannot silently skip.

splat-transform's `-F` pass needs a WebGPU adapter. A GPU container that lacks one does not fail the
export, it just leaves the blobs in, which is how the chapel shipped with the floaters the operator
then had to look at. This pass is numpy only: it runs everywhere, records what it removed, and the
export stage refuses to treat it as optional.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def read_ply(path: Path) -> tuple[str, dict[str, np.ndarray]]:
    """Header text and the named float properties of a binary little-endian PLY (vertex only)."""
    with Path(path).open("rb") as f:
        head = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: not a PLY")
            head.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(head).decode("latin-1")
    if "format binary_little_endian" not in header:
        raise ValueError(f"{path}: only binary little-endian PLY is supported")
    props = [ln.split()[-1] for ln in header.splitlines() if ln.startswith("property float ")]
    count = int(next(ln.split()[-1] for ln in header.splitlines() if ln.startswith("element vertex")))
    offset = sum(len(ln) for ln in head)
    data = np.fromfile(path, dtype=np.float32, offset=offset).reshape(count, len(props))
    return header, {p: data[:, i] for i, p in enumerate(props)}


def floater_filter(src: Path, dest: Path, *, cell: float = 0.12, min_neighbours: int = 3,
                   min_opacity: float = 0.05) -> dict:
    """Drop splats that have no company.

    A real surface is sampled many times over a few centimetres; a floater is a handful of splats
    hanging in space. Counting occupants of a 12 cm cell separates the two without touching scale or
    colour, so thin geometry (rails, grilles, a vault rib) survives as long as something else is near it.
    """
    header, cols = read_ply(src)
    n = len(cols["x"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1)
    opacity = 1.0 / (1.0 + np.exp(-cols["opacity"])) if "opacity" in cols else np.ones(n, np.float32)

    keys = np.floor(xyz / cell).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    crowd = counts[inverse]

    keep = (crowd >= min_neighbours) & (opacity >= min_opacity)
    removed_isolated = int((crowd < min_neighbours).sum())
    removed_faint = int(((crowd >= min_neighbours) & (opacity < min_opacity)).sum())

    out = {p: v[keep] for p, v in cols.items()}
    header = "\n".join(
        f"element vertex {int(keep.sum())}" if ln.startswith("element vertex") else ln
        for ln in header.splitlines()
    ) + "\n"
    with Path(dest).open("wb") as f:
        f.write(header.encode("latin-1"))
        np.stack([out[p] for p in cols], axis=1).astype(np.float32).tofile(f)
    return {"method": "cpu-grid", "cell": cell, "minNeighbours": min_neighbours, "minOpacity": min_opacity,
            "in": n, "kept": int(keep.sum()), "removedIsolated": removed_isolated, "removedFaint": removed_faint,
            "keptFraction": round(float(keep.sum()) / n, 3) if n else None}


def clipped_fraction(path: Path, threshold: float = 0.985) -> float | None:
    """Share of splats whose colour sits at the top of the range (the glare the viewer cannot undo)."""
    _, cols = read_ply(path)
    if not all(k in cols for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        return None
    c0 = 0.28209479177387814
    rgb = np.clip(0.5 + c0 * np.stack([cols["f_dc_0"], cols["f_dc_1"], cols["f_dc_2"]], axis=1), 0, 1)
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    return round(float((lum >= threshold).mean()), 4)
