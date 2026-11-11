# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""A small aligned scene gsplat's own parser accepts: synth.py's room with real 2D observations (the synthetic model
has none, and pycolmap's reader rejects tracks that point past an empty image), gradient placeholder photos, the
data/images link the align stage makes, and a frames_index.json of two clips with some frames the 'solve' dropped."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from plypipe import colmap_txt, geometry
from synth import write_synthetic_model

SIZE = (320, 240)  # width, height of the placeholder photos


def write_trainable_scene(root: Path, n_cams: int = 40, clips: tuple[int, ...] = (28, 16), dropped: tuple[int, ...] = (30,),
                          seed: int = 1) -> list[str]:
    """Write <root>/data/{sparse/0,images}, <root>/dense/images and <root>/frames_index.json. Frames are numbered
    0..sum(clips)-1 in clip order; the model registers the first n_cams that are not in `dropped`, and the index also
    lists the dropped ones. Returns the registered image names."""
    import cv2

    w, h = SIZE
    sparse = root / "data" / "sparse" / "0"
    write_synthetic_model(sparse, seed=seed, n_cams=n_cams)
    raw = colmap_txt.read_model(sparse)
    f, cx, cy = 240.0, w / 2, h / 2
    keep = [i for i in range(sum(clips)) if i not in dropped][:n_cams]
    obs: dict[int, list[tuple[float, float, int]]] = {im.image_id: [] for im in raw.images}
    tracks: dict[int, list[tuple[int, int]]] = {}
    for pid, xyz in zip(raw.point_ids, raw.xyz):
        for im in raw.images:
            xc = geometry.quat_to_mat(im.qvec) @ xyz + im.tvec
            if xc[2] < 0.3:
                continue
            u, v = f * xc[0] / xc[2] + cx, f * xc[1] / xc[2] + cy
            if 0 <= u < w and 0 <= v < h:
                tracks.setdefault(int(pid), []).append((im.image_id, len(obs[im.image_id])))
                obs[im.image_id].append((u, v, int(pid)))
    seen = [pid for pid, t in tracks.items() if len(t) >= 2]
    seen_set = set(seen)
    index_of = {int(p): i for i, p in enumerate(raw.point_ids)}
    sel = [index_of[p] for p in seen]
    images = [colmap_txt.Image(im.image_id, im.qvec, im.tvec, 1, f"{keep[im.image_id - 1]:06d}.jpg",
                               " ".join(f"{u:.3f} {v:.3f} {p if p in seen_set else -1}" for u, v, p in obs[im.image_id]))
              for im in raw.images]
    tails = ["128 128 128 0.5 " + " ".join(f"{a} {b}" for a, b in tracks[p]) for p in seen]
    model = colmap_txt.Model([f"1 PINHOLE {w} {h} {f} {f} {cx} {cy}"], images, np.array(seen), raw.xyz[sel], raw.rgb[sel],
                             np.full(len(sel), 0.5), np.array([len(tracks[p]) for p in seen], np.int32), tails)
    colmap_txt.write_model(model, sparse)
    colmap_txt.write_model_bin(model, sparse)

    dense = root / "dense" / "images"
    dense.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:h, 0:w]
    for im in images:
        rng = np.random.default_rng(im.image_id)
        px = np.stack([xx / w * 200, yy / h * 200, np.full_like(xx, 100)], -1) + rng.random((h, w, 3)) * 30
        cv2.imwrite(str(dense / im.name), px.astype("uint8"))
    link = root / "data" / "images"
    if not link.is_symlink():
        os.symlink(os.path.relpath(dense, root / "data"), link)

    index, first = [], 0
    for c, n in enumerate(clips):
        index += [{"frame": f"{i:06d}.jpg", "clip": f"clip_{c}.mp4", "t": round(0.5 * (i - first), 2)} for i in range(first, first + n)]
        first += n
    (root / "frames_index.json").write_text(json.dumps(index))
    return sorted(im.name for im in images)
