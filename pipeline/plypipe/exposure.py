# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""exposure: bring a model trained with an appearance model back to the photos' colours (IMPROVEMENT-PLAN.md, variant Q1).

Photos of one room differ in exposure and white balance. Trained without an appearance model, the splats explain that
with geometry and view-dependent colour: haze and floaters at the windows. Trained with one (gsplat's bilateral grid,
a 3x4 affine colour transform per grid cell and photo), nothing pins the splats' own brightness, and the viewer, which
applies no appearance model, showed the twin x1.20 brighter than the photos (docs/BENCHMARK.md). What the grid leaves
free is an affine colour map, so one is fitted, checked and folded into the splats:

  fit     M (3x3) and b that take the model's raw render r, as the viewer draws it, to the photo p: least squares
          over training views, min sum |M r + alpha b - p|^2. Held-out views are never read. Only pixels the model
          covers (alpha >= 0.99) and the photo did not clip (every channel in 0.02..0.98) count, so glare and crushed
          blacks do not bias the gain. A gain per channel alone would leave a colour cast: the grid mixes channels.
  guard   too few views or pixels, renders whose colours do not determine the map, or an implausible map: no fold,
          the model stays as trained and the reason is recorded.
  fold    a splat's colour is max(0.5 + C0 sh0 + sum_k basis_k(direction) shN_k, 0) per channel, so M c + b is every
          SH coefficient (a vector over RGB) times M, plus (M 0.5 + b - 0.5) / C0 on sh0. Blending is linear: the
          folded model renders M r + alpha b, exactly, wherever no splat's colour is clamped at zero.

The train stage then trains on from the folded model with no appearance model (stages/train.py), so the splats settle
in the colours the viewer shows. Everything here is numpy except `_renderer`, which needs CUDA and gsplat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plypipe import clean

C0 = 0.28209479177387814      # the constant SH basis function
ALPHA_MIN = 0.99              # below it the pixel shows the black background, not a colour to match
PHOTO_RANGE = (0.02, 0.98)    # a photo pixel outside it in any channel is clipped: it says nothing about the gain
# The fit reads at most MAX_VIEWS training views, evenly spread, at SCALE of their resolution. It has 12 unknowns and
# millions of pixels: what limits it is how well the views sample the capture's exposures, not the pixel count.
MAX_VIEWS, SCALE = 96, 0.5
MIN_VIEWS, MIN_PIXELS = 8, 100_000
MAX_CONDITION = 1e8           # of the Gram matrix of (r, g, b, alpha): a capture without colour leaves the mixing free
GAIN = (0.5, 2.0)             # each channel's gain on a neutral colour (a row sum of M); the drift measured is x1.2-1.3
STRETCH = 4.0                 # no colour direction is scaled by more than this, or by less than 1 / this
OFFSET = 0.25                 # largest |b|, in the photos' 0..1 range


def sums() -> dict:
    """Empty normal equations: the 4x4 Gram matrix of (r, g, b, alpha), shared by the three photo channels."""
    return {"gram": np.zeros((4, 4)), "cross": np.zeros((4, 3)), "views": 0, "pixels": 0, "seen": 0}


def usable(render: np.ndarray, alpha: np.ndarray, photo: np.ndarray) -> np.ndarray:
    lo, hi = PHOTO_RANGE
    return (alpha >= ALPHA_MIN) & (photo >= lo).all(-1) & (photo <= hi).all(-1) & np.isfinite(render).all(-1)


def accumulate(acc: dict, render: np.ndarray, alpha: np.ndarray, photo: np.ndarray) -> None:
    """Add one view: `render` [H, W, 3] as the rasteriser returns it (not clamped), its `alpha` [H, W], and the
    `photo` [H, W, 3] in 0..1. The offset's column is alpha: the fold adds b to every splat, so a pixel gets alpha b."""
    keep = usable(render, alpha, photo)
    x = np.concatenate([render[keep], alpha[keep][:, None]], 1).astype(np.float64)
    acc["gram"] += x.T @ x
    acc["cross"] += x.T @ photo[keep].astype(np.float64)
    acc["pixels"] += len(x)
    acc["views"] += bool(len(x))
    acc["seen"] += alpha.size


def implausible(m: np.ndarray, b: np.ndarray) -> str | None:
    """Why this colour map must not be folded into a model, or None."""
    if not (np.isfinite(m).all() and np.isfinite(b).all()):
        return "the fitted colour map is not finite"
    det = float(np.linalg.det(m))
    if det <= 0:
        return f"the fitted matrix mirrors or flattens colour (determinant {det:.3g})"
    gains = m.sum(1)
    if gains.min() < GAIN[0] or gains.max() > GAIN[1]:
        return f"channel gains of {', '.join(f'{g:.2f}' for g in gains)}: outside {GAIN[0]} to {GAIN[1]}"
    stretch = np.linalg.svd(m, compute_uv=False)
    if stretch.max() > STRETCH or stretch.min() < 1 / STRETCH:
        return f"the fitted matrix scales a colour direction by {stretch.max():.2f} and another by {stretch.min():.2f}: outside 1/{STRETCH:g} to {STRETCH:g}"
    if np.abs(b).max() > OFFSET:
        return f"an offset of {b[np.abs(b).argmax()]:.2f}: beyond {OFFSET}"
    return None


def solve(acc: dict, min_views: int = MIN_VIEWS, min_pixels: int = MIN_PIXELS) -> dict:
    """The colour map of the summed views: "matrix" (M, rows are the photo's channels), "offset" (b) and "gains", with
    what it rests on. "why" is set when it must not be folded. The map is rounded to 6 decimals: what the record
    says is what gets folded."""
    out = {"views": acc["views"], "pixels": acc["pixels"], "usable": round(acc["pixels"] / max(acc["seen"], 1), 4)}
    if acc["views"] < min_views or acc["pixels"] < min_pixels:
        return {**out, "why": f"too little to fit on: {acc['pixels']} usable pixels in {acc['views']} views (needs {min_pixels} in {min_views})"}
    gram, cross = acc["gram"] / acc["pixels"], acc["cross"] / acc["pixels"]
    cond = float(np.linalg.cond(gram))
    if not np.isfinite(cond) or cond > MAX_CONDITION:
        return {**out, "why": f"the renders' colours do not determine a colour map (condition number {cond:.3g}, limit {MAX_CONDITION:g}): "
                              "a capture without colour?"}
    a = np.round(np.linalg.solve(gram, cross), 6)  # [4, 3]: column j takes (r, g, b, alpha) to the photo's channel j
    m, b = a[:3].T, a[3]
    out["condition"] = round(cond, 1)
    if np.isfinite(a).all():
        out.update(matrix=m.tolist(), offset=b.tolist(), gains=np.round(m.sum(1), 4).tolist())
    why = implausible(m, b)
    return {**out, "why": why} if why else out


def measure(views, m, b) -> dict:
    """What the map changes on `views` (render, alpha, photo), as the viewer shows them (clamped to 0..1): the mean
    absolute error on the pixels the fit used, and brightness as viewer_eval scores it, mean render / mean photo over
    every pixel of a view, averaged over the views."""
    m, b = np.asarray(m, np.float64), np.asarray(b, np.float64)
    err, n, bright = np.zeros(2), 0, []
    for render, alpha, photo in views:
        shown = [np.clip(r, 0, 1) for r in (render, render @ m.T + alpha[..., None] * b)]
        keep = usable(render, alpha, photo)
        err += [np.abs(s[keep] - photo[keep]).sum() for s in shown]
        n += 3 * int(keep.sum())
        bright.append([np.nanmean(s) / max(float(photo.mean()), 1e-6) for s in shown])
    if not n:
        return {}
    before, after = np.nanmean(np.array(bright), axis=0)
    return {"maeBefore": round(float(err[0] / n), 5), "maeAfter": round(float(err[1] / n), 5),
            "brightnessBefore": round(float(before), 4), "brightnessAfter": round(float(after), 4)}


def fold(sh0: np.ndarray, shN: np.ndarray, m, b) -> tuple[np.ndarray, np.ndarray]:
    """SH coefficients (gsplat's layout: [N, 1, 3] and [N, K, 3]) of the colours M c + b."""
    m, b = np.asarray(m, np.float64), np.asarray(b, np.float64)
    shift = (m @ np.full(3, 0.5) + b - 0.5) / C0
    return (sh0 @ m.T + shift).astype(sh0.dtype), (shN @ m.T).astype(shN.dtype)


def _colours(path: Path, cols: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """sh0 [N, 1, 3], shN [N, K, 3] and the f_rest names in order. f_rest is channel-major in the file."""
    if not all(f"f_dc_{i}" in cols for i in range(3)):
        raise ValueError(f"{path}: not a 3DGS model (no f_dc_0..2)")
    rest = sorted((k for k in cols if k.startswith("f_rest_")), key=lambda k: int(k.rsplit("_", 1)[1]))
    n, k = len(cols["f_dc_0"]), len(rest) // 3
    sh0 = np.stack([cols[f"f_dc_{i}"] for i in range(3)], 1)[:, None, :]
    shN = np.stack([cols[r] for r in rest], 1).reshape(n, 3, k).transpose(0, 2, 1) if k else np.zeros((n, 0, 3), np.float32)
    return sh0, shN, rest


def fold_ply(src: Path, dest: Path, m, b) -> dict:
    """Write the model in `src` with the colours M c + b to `dest`: same header, same splats, only f_dc and f_rest change."""
    header, cols = clean.read_ply(Path(src))
    sh0, shN, rest = _colours(Path(src), cols)
    sh0, shN = fold(sh0, shN, m, b)
    out = {**cols, **{f"f_dc_{i}": sh0[:, 0, i] for i in range(3)}, **dict(zip(rest, shN.transpose(0, 2, 1).reshape(len(sh0), -1).T))}
    with Path(dest).open("wb") as f:
        f.write(header.encode("latin-1"))
        np.stack([out[p] for p in cols], axis=1).astype(np.float32).tofile(f)
    return {"splats": len(sh0), "shDegree": int(round(np.sqrt(shN.shape[1] + 1))) - 1}


def _renderer(ply: Path, antialiased: bool):
    """render(view, K, width, height) -> (rgb [H, W, 3] not clamped, alpha [H, W]): the model in `ply` through gsplat's
    rasteriser with its SH colours and nothing else, which is what the viewer draws. None without CUDA and gsplat."""
    try:
        import torch
        from gsplat import rasterization
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    _, c = clean.read_ply(Path(ply))
    sh0, shN, _ = _colours(Path(ply), c)
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device="cuda")
    means = t(np.stack([c["x"], c["y"], c["z"]], 1))
    quats = t(np.stack([c[f"rot_{i}"] for i in range(4)], 1))
    scales = torch.exp(t(np.stack([c[f"scale_{i}"] for i in range(3)], 1)))
    opac = torch.sigmoid(t(c["opacity"]))
    colors = t(np.concatenate([sh0, shN], 1))
    degree = int(round(np.sqrt(colors.shape[1]))) - 1

    def render(view: np.ndarray, K: np.ndarray, width: int, height: int):
        with torch.no_grad():
            rgb, alpha, _ = rasterization(means, quats, scales, opac, colors, t(view)[None], t(K)[None], width, height,
                                          sh_degree=degree, rasterize_mode="antialiased" if antialiased else "classic")
        return rgb[0].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()

    return render


def fit_views(ply: Path, data_dir: Path, every: int = 30, block=(), max_views: int = MAX_VIEWS, scale: float = SCALE,
              antialiased: bool = True) -> dict | None:
    """Fit the colour map of the model in `ply` on the views it was trained on: never the every-`every`-th views, never
    the held-out `block` (image names). Two passes over the same views: the normal equations, then what the map
    changes there (`measure`). Returns `solve`'s record; None without CUDA and gsplat or with cameras that are not PINHOLE."""
    import cv2
    from plypipe import colmap_txt, viewer_eval

    data_dir = Path(data_dir)
    model = colmap_txt.read_model(data_dir / "sparse" / "0")
    cams = {}
    for ln in model.cameras_lines:
        if ln and not ln.startswith("#"):
            p = ln.split()
            if p[1] != "PINHOLE":
                return None
            cams[int(p[0])] = (int(p[2]), int(p[3]), *map(float, p[4:8]))
    render = _renderer(Path(ply), antialiased)
    if render is None:
        return None
    ims = sorted(model.images, key=lambda im: im.name)
    train = viewer_eval.training_views([im.name for im in ims], every, block)
    picks = train[::max(1, -(-len(train) // max(1, max_views)))]

    def views():
        for i in picks:
            im = ims[i]
            w, h, fx, fy, cx, cy = cams[im.camera_id]
            photo = cv2.imread(str(data_dir / "images" / im.name), cv2.IMREAD_COLOR)
            if photo is None:
                continue
            size = max(1, int(w * scale)), max(1, int(h * scale))
            photo = cv2.cvtColor(cv2.resize(photo, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            view = np.eye(4)
            view[:3, :3], view[:3, 3] = viewer_eval._quat_to_R(im.qvec), im.tvec
            K = np.array([[fx * scale, 0, cx * scale], [0, fy * scale, cy * scale], [0, 0, 1]])
            yield (*render(view, K, *size), photo)

    acc = sums()
    for view in views():
        accumulate(acc, *view)
    fit = solve(acc)
    return {**fit, **measure(views(), fit["matrix"], fit["offset"])} if "why" not in fit else fit
