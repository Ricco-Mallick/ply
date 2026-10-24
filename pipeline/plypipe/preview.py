# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Coverage preview: what camera solving made of a capture, small enough to send to a phone.

    preview.bin   uint32 n, then n x float32 xyz, n x uint8 rgb, n x uint8 views (images that saw the point, capped 255)
    preview.json  frames (clip, second, registered?), camera centres, counts

Built from the aligned model (scene frame: +y up, metres) when align ran, else from the raw SfM model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from plypipe import colmap_txt

MAX_POINTS = 80_000


def _write(path: Path, data: bytes) -> None:
    """Atomic: readers (an upload on another thread) never see a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def build(out: Path, dest: Path | None = None) -> list[Path]:
    """Read the job folder `out`, write preview.bin / preview.json into `dest` (default `out`)."""
    dest = dest or out
    dest.mkdir(parents=True, exist_ok=True)
    for d in (out / "data" / "sparse" / "0", out / "dense" / "sparse_txt"):
        if (d / "points3D.txt").exists():
            model, aligned = colmap_txt.read_model(d), d.parent.name == "sparse"
            break
    else:
        raise RuntimeError("no sparse model to preview")
    keep = np.arange(model.num_points)
    if len(keep) > MAX_POINTS:  # keep the best-seen points first
        keep = np.argsort(-model.track_len)[:MAX_POINTS]
    xyz = model.xyz[keep].astype(np.float32)
    rgb = model.rgb[keep].astype(np.uint8)
    views = np.minimum(model.track_len[keep], 255).astype(np.uint8)
    _write(dest / "preview.bin", np.array([len(keep)], np.uint32).tobytes() + xyz.tobytes() + rgb.tobytes() + views.tobytes())
    names = sorted(im.name for im in model.images)
    centers, _, _ = model.cameras()
    reg = {n: [round(float(v), 3) for v in c] for n, c in zip(names, centers)}
    idx = json.loads((out / "frames_index.json").read_text()) if (out / "frames_index.json").exists() else []
    all_frames = sorted(p.name for p in (out / "frames").glob("*.jpg")) if (out / "frames").exists() else names
    src = {e["frame"]: e for e in idx}
    frames = [{"frame": f, "clip": src.get(f, {}).get("clip"), "t": src.get(f, {}).get("t"), "registered": f in reg, "c": reg.get(f)} for f in all_frames]
    meta = {"version": 1, "aligned": aligned, "points": int(len(keep)), "frames": frames,
            "registered": sum(1 for f in frames if f["registered"]), "total": len(frames)}
    _write(dest / "preview.json", json.dumps(meta).encode())
    return [dest / "preview.bin", dest / "preview.json"]
