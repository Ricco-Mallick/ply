# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""priors: the scene's metric scale from a monocular depth model (plan section 9, 2b), and its depth kept for later.

A camera solve has no units. MoGe-2 predicts metric depth from one photo; where a frame's SfM track points have depth
in the solve's units, the ratio (predicted metres / SfM units) at those points is that frame's metres per unit. About
150 frames spread along the path are measured, the scene's scale is the median over frames, and how much frames
disagree says whether to believe it: robust spread (1.4826 x MAD / median) above 5% reports "inconsistent", and `align`
then keeps the camera-height prior (2a) and says why in scene.json (`scaleFallback`).

Per frame, a track point counts when it is in front of the camera, not within 5% of an image edge, the predictor has
a finite positive depth there (and confidence >= 0.5 when it gives one); the frame's ratio is the median of the points'
ratios after dropping those more than 3 robust sigmas (log scale) from it, and the frame needs 20 such points.

The depth model is a callable, Frame -> Depth (metric depth, optional normals and confidence at the frame's size), so
tests pass a stand-in; `real_predictor` is MoGe-2 behind lazy imports and needs a GPU. Depth (float16 metres) and
normals (int8) are written at half resolution to priors/depth_half.npz for the depth-supervised experiments, with the
result and the seconds it took in priors/priors.json. Nothing here runs unless asked for (`--priors`, or
`plypipe priors <folder>`); a run without priors.json behaves exactly as before.

    python -m plypipe priors <folder> [--model moge2-l|moge2-b] [--frames 150]   # GPU; prints the scale and the spread
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from plypipe import colmap_txt, geometry

RESULT = "priors/priors.json"
DEPTH_FILE = "priors/depth_half.npz"
MAX_SPREAD = 0.05      # plan section 9: frames that disagree by more than 5% mean the scale is not to be trusted
MIN_POINTS = 20        # track points a frame needs for its own ratio
MIN_FRAMES = 10        # frames with a ratio the scene needs
EDGE = 0.05
MIN_CONFIDENCE = 0.5
MODELS = {"moge2-l": "Ruicheng/moge-2-vitl-normal", "moge2-b": "Ruicheng/moge-2-vitb-normal"}  # 2b, and 2c


@dataclass
class Frame:
    name: str
    image: np.ndarray            # H x W x 3, uint8 RGB
    fov_x: float | None = None   # degrees, from the solve's focal length


@dataclass
class Depth:
    depth: np.ndarray                          # H x W metres; 0, inf or nan where there is none
    normals: np.ndarray | None = None          # H x W x 3, unit, camera axes
    confidence: np.ndarray | None = None       # H x W, 0..1


Predictor = Callable[[Frame], Depth]


def choose_frames(centers: np.ndarray, n: int = 150) -> list[int]:
    """Indices (into `centers`, in path order) of n frames evenly spaced along the path's length, so a long pause
    costs one frame instead of fifty. All of them when there are n or fewer."""
    k = len(centers)
    if k <= n:
        return list(range(k))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))])
    if s[-1] <= 0:
        return sorted({int(i) for i in np.linspace(0, k - 1, n)})
    chosen: list[int] = []
    for t in np.linspace(0, s[-1], n):
        i = int(np.argmin(np.abs(s - t)))
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen)


def _interleave(chosen: list[int]) -> list[int]:
    """The order to predict in, so a run stopped by its time budget has frames from the whole path."""
    bits = max(1, (len(chosen) - 1).bit_length())
    rank = lambda i: int(format(i, f"0{bits}b")[::-1], 2)  # noqa: E731
    return [chosen[i] for i in sorted(range(len(chosen)), key=rank)]


def frame_ratio(pred: np.ndarray, uv: np.ndarray, z: np.ndarray, conf: np.ndarray | None = None) -> tuple[float, int] | None:
    """(metres per SfM unit, points used) for one frame, or None. `uv` are the track points' pixels in pred's grid
    (x right, y down), `z` their depth in SfM units."""
    h, w = pred.shape
    x, y = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    ok = (z > 0) & (uv[:, 0] > EDGE * w) & (uv[:, 0] < (1 - EDGE) * w) & (uv[:, 1] > EDGE * h) & (uv[:, 1] < (1 - EDGE) * h)
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    d = pred[y, x].astype(np.float64)
    ok &= np.isfinite(d) & (d > 0)
    if conf is not None:
        ok &= conf[y, x] >= MIN_CONFIDENCE
    if ok.sum() < MIN_POINTS:
        return None
    lr = np.log(d[ok] / z[ok])
    m = np.median(lr)
    sigma = max(1.4826 * np.median(np.abs(lr - m)), 0.02)
    keep = np.abs(lr - m) <= 3 * sigma
    if keep.sum() < MIN_POINTS:
        return None
    return float(np.exp(np.median(lr[keep]))), int(keep.sum())


def solve_scale(ratios: list[float | None], max_spread: float = MAX_SPREAD, min_frames: int = MIN_FRAMES) -> dict:
    """The scene's scale from the frames' ratios: the median, the robust relative spread, and a verdict."""
    r = np.array([v for v in ratios if v is not None and math.isfinite(v) and v > 0], float)
    out = {"frames": int(len(r)), "tried": len(ratios)}
    if len(r) < min_frames:
        return {**out, "status": "too-few-frames", "scale": None, "spread": None,
                "reason": f"only {len(r)} frames gave a ratio (needs {min_frames})"}
    scale = float(np.median(r))
    spread = float(1.4826 * np.median(np.abs(r - scale)) / scale)
    inliers = int((np.abs(r - scale) <= 3 * max(spread * scale, 1e-12)).sum())
    out.update({"scale": scale, "spread": spread, "inliers": inliers, "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90))})
    if spread > max_spread:
        return {**out, "status": "inconsistent", "reason": f"frames disagree: spread {spread:.1%} over {len(r)} frames (limit {max_spread:.0%})"}
    return {**out, "status": "ok"}


def _intrinsics(model: colmap_txt.Model) -> dict[int, tuple[int, int, float, float, float, float]]:
    cams = {}
    for ln in model.cameras_lines:
        if ln and not ln.startswith("#"):
            p = ln.split()
            v = list(map(float, p[4:]))
            if p[1] == "PINHOLE":
                cams[int(p[0])] = (int(p[2]), int(p[3]), v[0], v[1], v[2], v[3])
            elif p[1] == "SIMPLE_PINHOLE":
                cams[int(p[0])] = (int(p[2]), int(p[3]), v[0], v[0], v[1], v[2])
            else:
                raise ValueError(f"camera model {p[1]}: the depth ratio needs the undistorted PINHOLE model (dense/sparse_txt)")
    return cams


def observations(model: colmap_txt.Model, im: colmap_txt.Image, good: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The image's track points as (pixels, depth in SfM units), for the points `good` selects."""
    vals = np.array(im.points2d_line.split(), float).reshape(-1, 3) if im.points2d_line.strip() else np.zeros((0, 3))
    if not len(vals):
        return np.zeros((0, 2)), np.zeros(0)
    order = np.argsort(model.point_ids)
    ids = vals[:, 2].astype(np.int64)
    pos = np.searchsorted(model.point_ids, ids, sorter=order)
    pos = order[np.clip(pos, 0, len(order) - 1)]
    found = (ids >= 0) & (model.point_ids[pos] == ids) & good[pos]
    R = geometry.quat_to_mat(np.asarray(im.qvec, float))
    z = (model.xyz[pos[found]] @ R.T + np.asarray(im.tvec, float))[:, 2]
    return vals[found, :2], z


def _half(a: np.ndarray, interpolation: int) -> np.ndarray:
    import cv2
    return cv2.resize(a, (max(1, a.shape[1] // 2), max(1, a.shape[0] // 2)), interpolation=interpolation)


def measure(model: colmap_txt.Model, images_dir: Path, predictor: Predictor, n_frames: int = 150, max_seconds: float | None = None,
            log=None, keep_depth: bool = True) -> tuple[dict, dict]:  # noqa: ANN001
    """Run the predictor on the chosen frames and solve the scale. Returns (result, arrays to save)."""
    import cv2
    cams = _intrinsics(model)
    ims = sorted(model.images, key=lambda im: im.name)
    centers, _, _ = model.cameras()
    chosen = choose_frames(centers, n_frames)
    good = (model.track_len >= 3) & (model.error < 2.0)
    if good.sum() < 200:
        good = np.ones(model.num_points, bool)
    t0 = time.monotonic()
    ratios: dict[int, float | None] = {}
    saved: dict[int, dict] = {}
    spent, stopped = 0.0, None
    for done, i in enumerate(_interleave(chosen)):
        if max_seconds is not None and done >= MIN_FRAMES and time.monotonic() - t0 > max_seconds:
            stopped = f"time budget of {max_seconds:.0f} s reached after {done} of {len(chosen)} frames"
            break
        im = ims[i]
        w, h, fx, fy, cx, cy = cams[im.camera_id]
        bgr = cv2.imread(str(images_dir / im.name), cv2.IMREAD_COLOR)
        if bgr is None:
            ratios[i] = None
            continue
        t1 = time.monotonic()
        out = predictor(Frame(im.name, bgr[:, :, ::-1].copy(), math.degrees(2 * math.atan(w / (2 * fx)))))
        spent += time.monotonic() - t1
        depth = np.asarray(out.depth, np.float32)
        conf = None if out.confidence is None else np.asarray(out.confidence, np.float32)
        uv, z = observations(model, im, good)
        uv = uv * np.array([depth.shape[1] / w, depth.shape[0] / h])  # the camera's pixels onto the prediction's grid
        got = frame_ratio(depth, uv, z, conf)
        ratios[i] = got[0] if got else None
        if keep_depth:
            valid = np.isfinite(depth) & (depth > 0)
            entry = {"depth": _half(np.where(valid, depth, 0).astype(np.float32), cv2.INTER_NEAREST).astype(np.float16),
                     "K": np.array([[fx / 2 * depth.shape[1] / w, 0, cx / 2 * depth.shape[1] / w], [0, fy / 2 * depth.shape[0] / h, cy / 2 * depth.shape[0] / h],
                                    [0, 0, 1]], np.float32), "ratio": np.float32(got[0] if got else 0.0)}
            if out.normals is not None:
                nrm = _half(np.asarray(out.normals, np.float32), cv2.INTER_AREA)
                entry["normals"] = np.round(np.clip(nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6), -1, 1) * 127).astype(np.int8)
            if conf is not None:
                entry["confidence"] = np.round(np.clip(_half(conf, cv2.INTER_AREA), 0, 1) * 255).astype(np.uint8)
            saved[i] = entry
        if log and done % 10 == 0:
            log.emit("priors", "progress", f"frame {done + 1}/{len(chosen)}", done=done + 1, total=len(chosen))
    solved = solve_scale([ratios.get(i) for i in sorted(ratios)])
    result = {"version": 1, **solved, "chosen": len(chosen), "predicted": len(ratios), "frame": None,
              "predictSeconds": round(spent, 2), "secondsPerFrame": round(spent / max(1, len(ratios)), 3),
              "world": {"points": model.num_points, "images": len(model.images)}, **({"stopped": stopped} if stopped else {})}
    arrays: dict = {}
    if saved:
        idx = sorted(saved)
        arrays = {"names": np.array([ims[i].name for i in idx]), "depth": np.stack([saved[i]["depth"] for i in idx]),
                  "K": np.stack([saved[i]["K"] for i in idx]), "ratio": np.array([saved[i]["ratio"] for i in idx], np.float32)}
        for k in ("normals", "confidence"):
            if all(k in saved[i] for i in idx):
                arrays[k] = np.stack([saved[i][k] for i in idx])
    return result, arrays


def source_model(out: Path) -> tuple[colmap_txt.Model, str]:
    """The camera solve the ratio is measured on: the world frame `align` reads, else the aligned one (a finished twin:
    the ratio is then a correction to its scale, and `align` does not use it)."""
    if (out / "dense" / "sparse_txt").exists():
        return colmap_txt.read_model(out / "dense" / "sparse_txt"), "world"
    return colmap_txt.read_model(out / "data" / "sparse" / "0"), "scene"


def run_stage(ctx, predictor: Predictor | None = None) -> dict:  # noqa: ANN001
    """Measure the scale on ctx.out's solve and write priors/priors.json (and the depth). Returns the report."""
    t0 = time.monotonic()
    model, frame = source_model(ctx.out)
    images = ctx.dense_dir / "images"
    if not images.exists():
        images = ctx.data_dir / "images"
    load = 0.0
    if predictor is None:
        t1 = time.monotonic()
        predictor = real_predictor(str(ctx.opt("priors_model", "moge2-l")))
        load = time.monotonic() - t1
    result, arrays = measure(model, images, predictor, int(ctx.opt("priors_frames", 150)), ctx.opt("priors_max_seconds"), ctx.log)
    result["frame"] = frame
    result["predictor"] = getattr(predictor, "name", getattr(predictor, "__name__", type(predictor).__name__))
    result["loadSeconds"] = round(float(getattr(predictor, "load_seconds", load)), 2)
    if arrays:
        (ctx.out / "priors").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ctx.out / DEPTH_FILE, **arrays)
        result["depthFile"] = DEPTH_FILE
        result["depthMB"] = round((ctx.out / DEPTH_FILE).stat().st_size / 2**20, 2)
    result["seconds"] = round(time.monotonic() - t0, 1)
    (ctx.out / "priors").mkdir(parents=True, exist_ok=True)
    (ctx.out / RESULT).write_text(json.dumps(result, indent=2))
    ctx.log.emit("priors", "done" if result["status"] == "ok" else "warn",
                 f"scale {result['scale']:.4f} m/unit, spread {result['spread']:.1%} over {result['frames']} frames ({result['status']})"
                 if result.get("scale") else result["reason"])
    return result


def read(out: Path) -> dict | None:
    f = Path(out) / RESULT
    return json.loads(f.read_text()) if f.exists() else None


def for_align(out: Path, model: colmap_txt.Model) -> dict | None:
    """What `align` takes from priors/priors.json: {"scale", "spread", "frames"} when it can be used, else
    {"fallback": why} (a result exists but cannot be used), else None (no priors: align runs as it always did)."""
    r = read(out)
    if r is None:
        return None
    if r.get("frame") != "world":
        return {"fallback": "priors were measured on an aligned model, not the camera solve align reads"}
    w = r.get("world") or {}
    if w and (w.get("points"), w.get("images")) != (model.num_points, len(model.images)):
        return {"fallback": "priors were measured on a different camera solve"}
    if r.get("status") != "ok" or not r.get("scale"):
        return {"fallback": f"monocular depth {r.get('status')}: {r.get('reason')}", "spread": r.get("spread")}
    return {"scale": float(r["scale"]), "spread": float(r["spread"]), "frames": int(r["frames"])}


def readout(out: Path) -> dict:
    """What a measurement run reads off a job: the twin's scale under 2a (what scene.json holds) and under 2b."""
    r, scene_p = read(out), Path(out) / "scene.json"
    scene = json.loads(scene_p.read_text()) if scene_p.exists() else {}
    res = {"scene": {"scale": scene.get("scale"), "scaleSource": scene.get("scaleSource"), "scaleSpread": scene.get("scaleSpread"),
                     "scaleFallback": scene.get("scaleFallback")}}
    if r and r.get("scale"):
        base = scene.get("scale") if r.get("frame") == "scene" else 1.0
        res["priors"] = {"status": r["status"], "ratio": r["scale"], "spread": r["spread"], "frames": r["frames"], "frame": r["frame"],
                         "twinScale": None if base is None else base * r["scale"], "seconds": r.get("seconds"), "secondsPerFrame": r.get("secondsPerFrame")}
    return res


class _Moge:
    """MoGe-2 (Ruicheng/moge-2-vitl-normal, or the ViT-B) on a GPU. Metric depth, normals, and its own valid-pixel mask as the confidence."""

    def __init__(self, repo: str, name: str):
        import torch
        from moge.model.v2 import MoGeModel  # pip install git+https://github.com/microsoft/MoGe.git
        if not torch.cuda.is_available():
            raise RuntimeError("MoGe-2 needs a GPU")
        t0 = time.monotonic()
        self.torch, self.name, self.device = torch, name, torch.device("cuda")
        self.model = MoGeModel.from_pretrained(repo).to(self.device).eval()
        self.load_seconds = time.monotonic() - t0

    def __call__(self, frame: Frame) -> Depth:
        t = self.torch.from_numpy(frame.image).to(self.device).float().permute(2, 0, 1) / 255.0
        o = self.model.infer(t, fov_x=frame.fov_x, use_fp16=True)
        depth = o["depth"].float().cpu().numpy()
        mask = o["mask"].float().cpu().numpy() if "mask" in o else np.isfinite(depth).astype(np.float32)
        normals = o["normal"].float().cpu().numpy() if "normal" in o else None
        return Depth(np.where(np.isfinite(depth), depth, 0.0), normals, mask)


def real_predictor(model: str = "moge2-l") -> Predictor:
    if model not in MODELS:
        raise ValueError(f"unknown depth model {model!r}: one of {', '.join(MODELS)}")
    return _Moge(MODELS[model], model)
