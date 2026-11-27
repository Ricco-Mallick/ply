# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""growth: as many splats as the twin needs, found by measuring, not by a number.

There is no splat ceiling. Training starts at the plan's estimate (plypipe.plan: surface area and views), then grows
one round at a time, and each round has to earn its place on views the twin never trained on:

  round  = continue from the best model so far with ~1.5x the splats (MCMC adds them where the image error is) and a
           quarter of the base run's steps, same held-out views, same settings
  keep   = held-out PSNR (colour-corrected when there is one) up by >= MIN_GAIN_DB and LPIPS no worse than LPIPS_TOL,
           or no worse in dB and clearly fewer soft patches (PATCH_DROP)
  stop   = a round that does not earn it (the previous model ships), a round that diverged (same), or the GPU is
           full: then the report says the twin was still improving and wanted a bigger GPU (`needsBiggerGpu`)

The rules are written to plan.json["growth"] before the first round, so a run can be audited against them.

"Soft patches" are measured, not guessed: every held-out view is cut into tiles and each tile scored against the photo
(gsplat's eval canvases: photo on the left, render on the right). A tile more than PATCH_DB under its view's median is
a patch; tiles a person covered (plypipe.people masks) and flat tiles (a blank wall scores anything) are left out.

Density tells the two causes of a patch apart. Per floor cell: splats, and how many cameras looked from nearby.
  under-dense  many cameras, few splats  -> the model can do better: more splats / steps fix it (growth)
  under-filmed few cameras               -> no number of splats fixes it: film that spot again (listed, with where)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIN_GAIN_DB = 0.15      # above the noise of 10-30 held-out views
LPIPS_TOL = 0.002
PATCH_DROP = 0.15       # a round that cuts the patch share by 15 % (and loses no dB) is kept too
PATCH_DB = 4.0
GROW_FACTOR = 1.5
ROUND_STEP_SHARE = 0.25
ROUND_MIN_STEPS = 5_000
TILE = 64
FLAT_STD = 0.02         # a tile whose photo varies less than this (0-1 scale) is too flat to judge
CELL_M = 0.5
SEE_M = 4.0             # a camera "sees" floor cells within this distance
UNDER_DENSE = 0.25      # share of the median splats per well-seen cell
WELL_SEEN = 6           # cameras near a cell for it to count as filmed


def rules() -> dict:
    return {"minGainDb": MIN_GAIN_DB, "lpipsTol": LPIPS_TOL, "patchDrop": PATCH_DROP, "patchDb": PATCH_DB,
            "growFactor": GROW_FACTOR, "roundStepShare": ROUND_STEP_SHARE, "roundMinSteps": ROUND_MIN_STEPS,
            "ceiling": "none (this GPU's memory is the only bound)"}


# ---- patches in the held-out views ------------------------------------------------------------------------------

def tile_psnr(photo: np.ndarray, render: np.ndarray, keep: np.ndarray | None = None, tile: int = TILE) -> np.ndarray:
    """PSNR per tile (rows x cols), NaN where a tile is too flat to judge or mostly masked. Images are 0-1 floats."""
    h, w = photo.shape[:2]
    rows, cols = h // tile, w // tile
    out = np.full((rows, cols), np.nan)
    for r in range(rows):
        for c in range(cols):
            sl = (slice(r * tile, (r + 1) * tile), slice(c * tile, (c + 1) * tile))
            p, q = photo[sl], render[sl]
            if keep is not None and keep[sl].mean() < 0.7:
                continue
            if p.std() < FLAT_STD:
                continue
            mse = float(np.mean((p - q) ** 2))
            out[r, c] = 99.0 if mse <= 1e-10 else 10 * np.log10(1.0 / mse)
    return out


def view_patches(tiles: np.ndarray, patch_db: float = PATCH_DB) -> dict:
    vals = tiles[np.isfinite(tiles)]
    if vals.size == 0:
        return {"tiles": 0, "patches": 0, "share": 0.0, "medianDb": None}
    med = float(np.median(vals))
    bad = int(np.sum(vals < med - patch_db))
    return {"tiles": int(vals.size), "patches": bad, "share": round(bad / vals.size, 4), "medianDb": round(med, 2)}


def split_canvas(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """gsplat's eval canvas: the photo on the left half, the render on the right."""
    w = img.shape[1] // 2
    return img[:, :w], img[:, w:2 * w]


def patches(result: Path, masks: list[Path | None] | None = None, stage: str = "val") -> dict | None:
    """Patch share over the newest held-out canvases in <result>/renders. `masks` (keep masks in view order) optional."""
    import cv2
    stats = sorted((result / "stats").glob(f"{stage}_step*.json"), key=lambda p: int(p.stem.rsplit("step", 1)[1]))
    if not stats:
        return None
    step = stats[-1].stem.rsplit("step", 1)[1].lstrip("0") or "0"
    pngs = sorted((result / "renders").glob(f"{stage}_step{step}_*.png"))
    if not pngs:
        return None
    per = []
    for i, png in enumerate(pngs):
        img = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if img is None:
            continue
        photo, render = split_canvas(img.astype(np.float32) / 255.0)
        keep = None
        if masks and i < len(masks) and masks[i] is not None and Path(masks[i]).is_file():
            k = cv2.imread(str(masks[i]), cv2.IMREAD_GRAYSCALE)
            if k is not None:
                keep = (cv2.resize(k, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_NEAREST) > 127)[..., None].astype(np.float32)
        per.append({"view": png.name, **view_patches(tile_psnr(photo, render, keep))})
    if not per:
        return None
    total = sum(v["tiles"] for v in per)
    bad = sum(v["patches"] for v in per)
    worst = sorted(per, key=lambda v: -v["share"])[:5]
    return {"views": len(per), "tiles": total, "patches": bad, "share": round(bad / total, 4) if total else 0.0,
            "worst": [{"view": v["view"], "share": v["share"]} for v in worst]}


# ---- density: under-dense (grow) versus under-filmed (film again) ------------------------------------------------

def density(means: np.ndarray, cameras: np.ndarray, floor_y: float = 0.0, cell: float = CELL_M,
            see_m: float = SEE_M, well_seen: int = WELL_SEEN, under: float = UNDER_DENSE) -> dict:
    """Splats per floor cell next to how many cameras were near it. `means` and `cameras` are N x 3 in the scene frame
    (+Y up, metres)."""
    if len(means) == 0 or len(cameras) == 0:
        return {"cells": 0}
    xz = means[:, [0, 2]]
    lo = np.floor(xz.min(axis=0) / cell) * cell
    ij = np.floor((xz - lo) / cell).astype(int)
    shape = ij.max(axis=0) + 1
    counts = np.zeros(shape, dtype=np.int64)
    np.add.at(counts, (ij[:, 0], ij[:, 1]), 1)
    centres = lo + (np.argwhere(np.ones(shape, bool)) + 0.5) * cell
    cam_xz = cameras[:, [0, 2]]
    seen = np.zeros(len(centres), dtype=np.int64)
    for k in range(0, len(cam_xz), 256):  # cameras within see_m of each cell centre
        d = np.linalg.norm(centres[:, None, :] - cam_xz[None, k:k + 256, :], axis=2)
        seen += (d <= see_m).sum(axis=1)
    seen = seen.reshape(shape)
    occupied = counts > 0
    well = occupied & (seen >= well_seen)
    if not well.any():
        return {"cells": int(occupied.sum()), "wellSeen": 0}
    med = float(np.median(counts[well]))
    thin = well & (counts < under * med)
    unfilmed = occupied & (seen < well_seen) & (counts < under * med)

    def where(mask):
        idx = np.argwhere(mask)
        return [[round(float(lo[0] + (i + 0.5) * cell), 2), round(float(lo[1] + (j + 0.5) * cell), 2)] for i, j in idx[:20]]

    return {"cells": int(occupied.sum()), "wellSeen": int(well.sum()), "medianSplatsPerCell": med,
            "underDense": int(thin.sum()), "underDenseShare": round(float(thin.sum() / well.sum()), 4),
            "underFilmed": int(unfilmed.sum()), "underFilmedAt": where(unfilmed), "underDenseAt": where(thin),
            "cellM": cell}


def ply_means(path: Path) -> np.ndarray:
    from plypipe.clean import read_ply
    _, cols = read_ply(path)
    return np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)


def cameras_from_scene(out: Path) -> np.ndarray:
    try:
        path = json.loads((out / "scene.json").read_text()).get("cameraPath") or []
        return np.asarray(path, dtype=np.float64).reshape(-1, 3)
    except (OSError, ValueError):
        return np.zeros((0, 3))


# ---- the decision ------------------------------------------------------------------------------------------------

def score_db(ev: dict | None) -> float | None:
    if not ev:
        return None
    v = ev.get("ccPsnr", ev.get("psnr"))
    return float(v) if v is not None else None


def judge(best: dict, cand: dict) -> tuple[bool, str]:
    """Is the candidate round better than the best so far? Each is {"eval": {...}, "patches": {...}}."""
    b, c = score_db(best.get("eval")), score_db(cand.get("eval"))
    if c is None or b is None:
        return False, "no held-out score to compare"
    gain = c - b
    bl, cl = (best.get("eval") or {}).get("lpips"), (cand.get("eval") or {}).get("lpips")
    lp_ok = bl is None or cl is None or cl <= bl + LPIPS_TOL
    bp, cp = (best.get("patches") or {}).get("share"), (cand.get("patches") or {}).get("share")
    if gain >= MIN_GAIN_DB and lp_ok:
        return True, f"+{gain:.2f} dB on held-out views"
    if gain >= 0 and lp_ok and bp and cp is not None and cp <= bp * (1 - PATCH_DROP):
        return True, f"{gain:+.2f} dB and soft patches {bp:.1%} -> {cp:.1%}"
    why = f"{gain:+.2f} dB (needs +{MIN_GAIN_DB})"
    if not lp_ok:
        why += f", LPIPS {bl:.3f} -> {cl:.3f}"
    return False, why


def next_cap(splats: int, device_max: int | None) -> tuple[int | None, str]:
    want = int(splats * GROW_FACTOR)
    if device_max and want > device_max:
        if device_max <= splats * 1.05:
            return None, f"this GPU holds about {device_max:,} splats and the model has {splats:,}"
        return int(device_max), f"{want:,} wanted, {device_max:,} fit this GPU"
    return want, f"{splats:,} -> {want:,}"


def round_steps(base_steps: int) -> int:
    return max(ROUND_MIN_STEPS, int(base_steps * ROUND_STEP_SHARE))
