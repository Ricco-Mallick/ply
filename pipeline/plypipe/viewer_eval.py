# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""viewer_eval: score the twin the way the viewer shows it.

gsplat's held-out scores render through the training-time appearance model (PPISP's controller), which the web viewer
never applies. A PPISP twin can score well there and still open ~20 % brighter than the photos. This pass renders the
exported, filtered splats (scene_clean.ply) at the held-out cameras with nothing but the rasterizer and measures:

  psnr / ssim        against the photo, as the viewer would show it
  brightness         mean render / mean photo (1.0 = same exposure)
  holes              share of pixels where the splats cover less than 90 % (see-through gaps, dark patches)

Needs CUDA and gsplat; anywhere else it returns None and the export stage carries on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def held_out(names: list[str], every: int) -> list[int]:
    """gsplat's test split: positions (in name order) divisible by test_every."""
    return [i for i in range(len(names)) if every > 1 and i % every == 0]


def evaluate(ply: Path, data_dir: Path, every: int = 30, antialiased: bool = True, max_views: int = 12) -> dict | None:
    try:
        import cv2
        import torch
        from gsplat import rasterization
    except ImportError:
        return None
    if not torch.cuda.is_available() or every <= 1:
        return None
    from plypipe import clean, colmap_txt

    model = colmap_txt.read_model(data_dir / "sparse" / "0")
    cams = {}
    for ln in model.cameras_lines:
        if ln and not ln.startswith("#"):
            p = ln.split()
            if p[1] != "PINHOLE":
                return None
            cams[int(p[0])] = (int(p[2]), int(p[3]), *map(float, p[4:8]))
    ims = sorted(model.images, key=lambda im: im.name)
    picks = held_out([im.name for im in ims], every)
    if not picks:
        return None
    picks = picks[:: max(1, len(picks) // max_views)][:max_views]

    _, c = clean.read_ply(ply)
    dev = "cuda"
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    means = t(np.stack([c["x"], c["y"], c["z"]], 1))
    quats = t(np.stack([c[f"rot_{i}"] for i in range(4)], 1))
    scales = torch.exp(t(np.stack([c[f"scale_{i}"] for i in range(3)], 1)))
    opac = torch.sigmoid(t(c["opacity"]))
    rest = sorted((k for k in c if k.startswith("f_rest_")), key=lambda k: int(k.rsplit("_", 1)[1]))
    dc = np.stack([c[f"f_dc_{i}"] for i in range(3)], 1)[:, None, :]
    if rest:  # PLY order is channel-major: [R coeffs..., G coeffs..., B coeffs...]
        k = len(rest) // 3
        sh = np.concatenate([dc, np.stack([c[r] for r in rest], 1).reshape(-1, 3, k).transpose(0, 2, 1)], 1)
    else:
        sh = dc
    colors = t(sh)
    degree = int(round(np.sqrt(sh.shape[1]))) - 1

    psnrs, ssims, bright, holes = [], [], [], []
    for i in picks:
        im = ims[i]
        w, h, fx, fy, cx, cy = cams[im.camera_id]
        photo = cv2.imread(str(data_dir / "images" / im.name), cv2.IMREAD_COLOR)
        if photo is None:
            continue
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        view = np.eye(4)
        view[:3, :3], view[:3, 3] = _quat_to_R(im.qvec), im.tvec
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        with torch.no_grad():
            rgb, alpha, _ = rasterization(means, quats, scales, opac, colors, t(view)[None], t(K)[None], w, h,
                                          sh_degree=degree, rasterize_mode="antialiased" if antialiased else "classic")
        r = rgb[0].clamp(0, 1).cpu().numpy()
        a = alpha[0, ..., 0].cpu().numpy()
        mse = float(((r - photo) ** 2).mean())
        psnrs.append(10 * np.log10(1.0 / max(mse, 1e-10)))
        g1 = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        ssims.append(_ssim(g1, g2))
        bright.append(float(r.mean() / max(photo.mean(), 1e-6)))
        holes.append(float((a < 0.9).mean()))
    if not psnrs:
        return None
    return {"views": len(psnrs), "psnr": round(float(np.mean(psnrs)), 3), "ssim": round(float(np.mean(ssims)), 4),
            "brightness": round(float(np.mean(bright)), 3), "holes": round(float(np.mean(holes)), 4),
            "worstHoles": round(float(np.max(holes)), 4)}


def training_weights(ply: Path, data_dir: Path, every: int = 30, step: int = 5, scale: float = 0.5,
                     antialiased: bool = True) -> np.ndarray | None:
    """Each splat's blend weight summed over the training views, in full-resolution pixels: how much of the
    photographs it actually draws. Held-out views (every `every`-th) are never used, so a cleanup step chosen with
    these weights is still judged fairly on them. Every `step`-th training view at `scale` resolution; the chapel's
    2M splats over 237 views took 1.3 s on an L4. With colours fixed at 1 the render is the sum of the blend weights,
    so its gradient with respect to each splat's colour is that splat's weight. None without CUDA and gsplat."""
    try:
        import torch
        from gsplat import rasterization
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    from plypipe import clean, colmap_txt

    model = colmap_txt.read_model(data_dir / "sparse" / "0")
    cams = {}
    for ln in model.cameras_lines:
        if ln and not ln.startswith("#"):
            p = ln.split()
            if p[1] != "PINHOLE":
                return None
            cams[int(p[0])] = (int(p[2]), int(p[3]), *map(float, p[4:8]))
    ims = sorted(model.images, key=lambda im: im.name)
    held = set(held_out([im.name for im in ims], every))
    train = [i for i in range(len(ims)) if i not in held][::max(1, step)]
    _, c = clean.read_ply(ply)
    dev = "cuda"
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    means = t(np.stack([c["x"], c["y"], c["z"]], 1))
    quats = t(np.stack([c[f"rot_{i}"] for i in range(4)], 1))
    scales = torch.exp(t(np.stack([c[f"scale_{i}"] for i in range(3)], 1)))
    opac = torch.sigmoid(t(c["opacity"]))
    weights = torch.zeros(len(means), device=dev)
    for i in train:
        im = ims[i]
        w, h, fx, fy, cx, cy = cams[im.camera_id]
        view = np.eye(4)
        view[:3, :3], view[:3, 3] = _quat_to_R(im.qvec), im.tvec
        K = np.array([[fx * scale, 0, cx * scale], [0, fy * scale, cy * scale], [0, 0, 1]])
        ones = torch.ones(len(means), 3, device=dev, requires_grad=True)
        rgb, _, _ = rasterization(means, quats, scales, opac, ones, t(view)[None], t(K)[None],
                                  max(1, int(w * scale)), max(1, int(h * scale)),
                                  rasterize_mode="antialiased" if antialiased else "classic")
        rgb[..., 0].sum().backward()
        weights += ones.grad[:, 0]
    return (weights / (scale * scale)).cpu().numpy()


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    import cv2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    return float((((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean())
