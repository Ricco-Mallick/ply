# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""ALIKED features + LightGlue matches in PyTorch, written into COLMAP's database.

COLMAP 4 runs ALIKED and LightGlue through ONNX Runtime, and neither pycolmap wheel on PyPI (4.2.1, CPU or CUDA 12)
is built with it: asking for ALIKED aborts the process. The same networks ship as PyTorch modules in cvg/LightGlue,
so this extracts and matches there, writes keypoints and matches into the database, and lets COLMAP verify them
geometrically and map as usual.

Pairs, for ordered video frames:
  * within each clip, every frame with its next `overlap` frames and with frames 2^k ahead (COLMAP's sequential
    pairing with quadratic overlap);
  * across the whole capture, each frame with its `retrieve` most similar frames by a DINOv2 image descriptor
    (loop closure and clip-to-clip links, which sequential pairing cannot see).
Photo sets have no order, so they are paired by retrieval only (or exhaustively while small).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

DINO = "facebook/dinov2-small"


def available() -> bool:
    """A CUDA GPU and the LightGlue package (the CPU path is ~3 s per frame: not worth it)."""
    import importlib.util
    if importlib.util.find_spec("lightglue") is None or importlib.util.find_spec("torch") is None:
        return False
    import torch
    return bool(torch.cuda.is_available())


def make_pairs(names: list[str], clip_of: dict[str, str] | None, desc: np.ndarray | None, overlap: int = 20,
               retrieve: int = 20, ordered: bool = True, exhaustive_max: int = 150) -> list[tuple[int, int]]:
    """Index pairs (i < j) into `names` (sorted). `desc`: one L2-normalised row per name, or None."""
    n = len(names)
    out: set[tuple[int, int]] = set()
    if not ordered and n <= exhaustive_max:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if ordered:
        clip = [clip_of.get(nm) if clip_of else None for nm in names]
        for i in range(n):
            ahead = list(range(i + 1, min(n, i + overlap + 1)))
            k = int(np.log2(overlap)) + 1 if overlap > 1 else 1
            while i + 2 ** k < n:
                ahead.append(i + 2 ** k)
                k += 1
            out.update((i, j) for j in ahead if clip[j] == clip[i])
    if desc is not None and retrieve > 0 and n > 1:
        sim = desc @ desc.T
        np.fill_diagonal(sim, -np.inf)
        k = min(retrieve, n - 1)
        for i in range(n):
            for j in np.argpartition(-sim[i], k - 1)[:k]:
                out.add((min(i, int(j)), max(i, int(j))))
    return sorted(out)


def _dino_descriptors(paths: list[Path], device: str) -> np.ndarray | None:
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(DINO)
        model = AutoModel.from_pretrained(DINO).to(device).eval()
    except Exception:
        return None
    rows = []
    with torch.inference_mode():
        for s in range(0, len(paths), 32):
            batch = [Image.open(p).convert("RGB") for p in paths[s:s + 32]]
            hid = model(**proc(images=batch, return_tensors="pt").to(device)).last_hidden_state
            d = torch.cat([hid[:, 0], hid[:, 1:].mean(1)], 1)
            rows.append(torch.nn.functional.normalize(d, dim=1).float().cpu().numpy())
    return np.concatenate(rows)


def extract_and_match(pycolmap, db_path: Path, images: Path, camera_mode, reader, clip_of: dict[str, str] | None,
                      ordered: bool = True, max_keypoints: int = 4096, overlap: int = 20, retrieve: int = 20,
                      log=None, device: str = "cuda") -> dict:
    """Fill a fresh database at db_path: images, ALIKED keypoints, LightGlue matches, verified two-view geometry."""
    import torch
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image

    say = log or (lambda *a, **k: None)
    dev = device
    t0 = time.time()
    pycolmap.Database.open(db_path).close()  # import_images needs the file to exist
    pycolmap.import_images(db_path, images, camera_mode=camera_mode, options=reader)
    db = pycolmap.Database.open(db_path)
    ids = {im.name: im.image_id for im in db.read_all_images()}
    names = sorted(ids)
    extractor = ALIKED(max_num_keypoints=max_keypoints).eval().to(dev)
    feats = []
    with torch.inference_mode():
        for i, nm in enumerate(names):
            f = extractor.extract(load_image(images / nm).to(dev))
            kp = f["keypoints"][0].cpu().numpy().astype(np.float32)
            db.write_keypoints(ids[nm], kp + 0.5)  # COLMAP puts the origin at the corner of the first pixel
            feats.append({"keypoints": f["keypoints"], "descriptors": f["descriptors"].half(), "image_size": f["image_size"]})
            if i % 200 == 0:
                say(f"ALIKED features {i}/{len(names)}", done=i, total=len(names))
    t_ext = time.time() - t0
    desc = _dino_descriptors([images / nm for nm in names], dev) if retrieve else None
    pairs = make_pairs(names, clip_of, desc, overlap=overlap, retrieve=retrieve if desc is not None else 0, ordered=ordered)
    if desc is None and ordered and clip_of and len(set(clip_of.values())) > 1:
        # no retrieval model: link clips by every 8th frame against every 8th frame of the other clips
        sub = list(range(0, len(names), 8))
        pairs = sorted(set(pairs) | {(i, j) for i in sub for j in sub if i < j and clip_of.get(names[i]) != clip_of.get(names[j])})
    say(f"LightGlue matching {len(pairs)} pairs" + ("" if desc is not None else " (no retrieval model)"))
    matcher = LightGlue(features="aliked", mp=dev == "cuda").eval().to(dev)
    t1, n_matches = time.time(), 0
    with torch.inference_mode():
        for k, (i, j) in enumerate(pairs):
            a, b = feats[i], feats[j]
            r = matcher({"image0": {**a, "descriptors": a["descriptors"].float()}, "image1": {**b, "descriptors": b["descriptors"].float()}})
            m = r["matches"][0].cpu().numpy().astype(np.uint32)
            if len(m) >= 15:
                db.write_matches(ids[names[i]], ids[names[j]], m)
                n_matches += len(m)
            if k % 2000 == 0:
                say(f"LightGlue {k}/{len(pairs)}", done=k, total=len(pairs))
    db.close()
    t_match = time.time() - t1
    say("geometric verification")
    t2 = time.time()
    pycolmap.geometric_verification(db_path)
    return {"pairs": len(pairs), "retrieval": desc is not None, "matches": n_matches, "keypointsMax": max_keypoints,
            "secondsExtract": round(t_ext, 1), "secondsMatch": round(t_match, 1), "secondsVerify": round(time.time() - t2, 1)}
