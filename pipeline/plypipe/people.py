# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""People out of the twin: SAM 3 finds every person in every kept frame, and nothing learns from those pixels.

A person who walks through the shot is in a different place in every frame. Trained as if they were part of the room
they become ghosts and floaters, and their feature matches pull the camera solve off. So, before the camera solve:

1. SAM 3 (facebook/sam3 through transformers, the text prompt "person") runs on every kept frame. Its instance masks
   are joined and grown by about 1% of the frame's long side, to cover motion blur and hair.
2. A frame that is more than --people-max-share person (someone walked right past the lens) teaches nothing about the
   room: it moves to frames_people/ and the solve never sees it.
3. masks/<frame>.jpg.png: white where the room is, black on people. That is COLMAP's own mask layout
   (ImageReaderOptions.mask_path), so no features are found on people and they never become 3D points.
4. After the solve, the masks are warped into the undistorted PINHOLE views that training reads (dense/masks/), and
   the trainer's colour loss skips the black pixels (gsplat simple_trainer: l1 on colors[mask], SSIM with both sides
   zeroed there). Depth samples that land on a person are dropped too.

Someone who sat still for the whole take is masked in every view, so the twin has no picture of what was behind them:
expect a soft patch there instead of a person. Model: SAM 3, not 3.1. SAM 3.1's gain is multi-object tracking through
video; its per-image detector is SAM 3's, and it ships no transformers port. The frames here are ~0.5 s apart and
looked at one by one, which is SAM 3's image path.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

PROMPT = "person"
THRESHOLD = 0.4          # SAM 3 instance score; "person" is its best-trained concept, so this stays permissive
GROW = 0.01              # dilation as a share of the long side
MAX_SHARE = 0.45         # above this share of person pixels the frame is dropped, not masked
MASKS = "masks"          # <out>/masks/<frame>.jpg.png (COLMAP layout); <out>/dense/masks/<frame>.jpg.png after undistortion
DROPPED = "frames_people"


def mask_name(image_name: str) -> str:
    """COLMAP's mask file for an image: the image's own name plus .png."""
    return f"{image_name}.png"


def grow(person: np.ndarray, px: int) -> np.ndarray:
    """Dilate a boolean person mask by px pixels (an ellipse, so corners stay round)."""
    if px <= 0 or not person.any():
        return person
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(person.astype(np.uint8), k) > 0


def grow_px(width: int, height: int, share: float = GROW) -> int:
    return max(3, int(round(share * max(width, height))))


def write_mask(path: Path, person: np.ndarray) -> None:
    """White (255) where training and features may look, black (0) on people."""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.where(person, 0, 255).astype(np.uint8))


def read_keep(path: Path) -> np.ndarray | None:
    """The mask as a boolean 'keep' array (True = room), or None when there is no file."""
    import cv2
    if not path.is_file():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


# ---- detection -------------------------------------------------------------------------------------------------

def sam3_dtype(torch):
    """bf16 on Ampere and newer; fp32 below (a T4 has no bf16 units, and SAM 3 in bf16 there gives wrong masks)."""
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32


def load_sam3():
    """(model, processor, torch) or raises: the same weights the label stage uses."""
    import torch
    from transformers import Sam3Model, Sam3Processor
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU")
    model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=sam3_dtype(torch)).to("cuda").eval()
    return model, Sam3Processor.from_pretrained("facebook/sam3"), torch


def sam3_detector(model, proc, torch, threshold: float = THRESHOLD):
    """A function RGB array (H x W x 3) -> boolean person mask (H x W), all instances joined."""
    from PIL import Image
    text = proc(text=PROMPT, return_tensors="pt").to("cuda")

    def detect(rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        img = Image.fromarray(rgb)
        inputs = proc(images=img, return_tensors="pt").to("cuda")
        dt = sam3_dtype(torch)
        with torch.no_grad(), torch.autocast("cuda", dtype=dt, enabled=dt != torch.float32):
            ve = model.get_vision_features(pixel_values=inputs.pixel_values.to(dt))
            outp = model(vision_embeds=ve, input_ids=text.input_ids, attention_mask=text.get("attention_mask"))
        res = proc.post_process_instance_segmentation(outp, threshold=threshold, mask_threshold=0.5, target_sizes=[[H, W]])[0]
        out = np.zeros((H, W), bool)
        for m in res["masks"]:
            out |= m.bool().cpu().numpy()
        return out
    return detect


def mask_frames(frames: Path, out: Path, detect, max_share: float = MAX_SHARE, log=None) -> dict:
    """Run `detect` on every frame in `frames`; write <out>/masks/, move frames that are mostly person to
    <out>/frames_people/. Returns the report (also written to <out>/masks/people.json)."""
    import cv2
    mdir = out / MASKS
    if mdir.exists():
        shutil.rmtree(mdir)
    mdir.mkdir(parents=True)
    files = sorted(frames.glob("*.jpg"))
    t0 = time.time()
    with_people, dropped, shares = 0, [], []
    for n, f in enumerate(files):
        bgr = cv2.imread(str(f))
        if bgr is None:
            raise ValueError(f"{f.name}: unreadable frame")
        h, w = bgr.shape[:2]
        person = detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if person.shape != (h, w):
            raise ValueError(f"{f.name}: the detector returned a {person.shape} mask for a {h}x{w} frame")
        person = grow(person, grow_px(w, h))
        share = float(person.mean())
        if share > 0:
            with_people += 1
            shares.append(share)
        if share > max_share:
            dropped.append({"frame": f.name, "share": round(share, 3)})
            (out / DROPPED).mkdir(exist_ok=True)
            shutil.move(str(f), out / DROPPED / f.name)
            continue
        write_mask(mdir / mask_name(f.name), person)
        if log and n % 25 == 0:
            log(f"looking for people: frame {n + 1}/{len(files)}", done=n + 1, total=len(files))
    report = {"model": "facebook/sam3", "prompt": PROMPT, "frames": len(files), "framesWithPeople": with_people,
              "dropped": dropped, "meanShareWhenPresent": round(float(np.mean(shares)), 4) if shares else 0.0,
              "maxShareKept": max_share, "seconds": round(time.time() - t0, 1)}
    (mdir / "people.json").write_text(json.dumps(report, indent=1))
    return report


def cached(out: Path, frames: Path) -> dict | None:
    """The earlier run's report when every frame still in `frames` has its mask (an sfm retry reuses them)."""
    try:
        rep = json.loads((out / MASKS / "people.json").read_text())
    except (OSError, ValueError):
        return None
    names = [f.name for f in frames.glob("*.jpg")]
    if names and all((out / MASKS / mask_name(n)).is_file() for n in names):
        return rep
    return None


def prepare(ctx) -> dict | None:
    """The sfm stage's first step. Returns the report, or None when masking is off or cannot run (logged): people
    are then trained like the room, as before this existed."""
    if ctx.opt("no_people_mask"):
        return None
    rep = cached(ctx.out, ctx.frames_dir)
    if rep is not None:
        ctx.log.emit("sfm", "progress", f"people masks reused: {rep['framesWithPeople']} of {rep['frames']} frames had someone in them")
        return rep
    try:
        model, proc, torch = load_sam3()
    except Exception as e:
        ctx.log.emit("sfm", "warn", f"people are not masked (SAM 3 unavailable: {str(e)[:160]}); anyone walking through will leave ghosts")
        return None
    try:
        detect = sam3_detector(model, proc, torch, float(ctx.opt("people_threshold", THRESHOLD)))
        rep = mask_frames(ctx.frames_dir, ctx.out, detect, float(ctx.opt("people_max_share", MAX_SHARE)),
                          log=lambda msg, **kw: ctx.log.emit("sfm", "progress", msg, **kw))
    except Exception as e:  # OOM, an API change: a twin with ghosts beats no twin
        shutil.rmtree(ctx.out / MASKS, ignore_errors=True)
        ctx.log.emit("sfm", "warn", f"people masking stopped ({type(e).__name__}: {str(e)[:160]}); training on whole frames")
        return None
    finally:
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    ctx.log.emit("sfm", "progress", f"people: {rep['framesWithPeople']} of {rep['frames']} frames had someone in them, "
                 f"{len(rep['dropped'])} dropped as mostly person")
    return rep


# ---- masks into the undistorted views ----------------------------------------------------------------------------

def distort(model: str, params: list[float], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised camera coordinates -> COLMAP pixel coordinates (pixel centres at +0.5) for the lens models the
    pipeline solves with."""
    p = [float(v) for v in params]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        return f * x + cx, f * y + cy
    if model == "PINHOLE":
        fx, fy, cx, cy = p
        return fx * x + cx, fy * y + cy
    if model in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy, k1 = p[:4]
        k2 = p[4] if model == "RADIAL" else 0.0
        r2 = x * x + y * y
        d = 1 + k1 * r2 + k2 * r2 * r2
        return f * x * d + cx, f * y * d + cy
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        r2 = x * x + y * y
        d = 1 + k1 * r2 + k2 * r2 * r2
        xd = x * d + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * d + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return fx * xd + cx, fy * yd + cy
    if model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, k1, k2, k3, k4 = p
        r = np.sqrt(x * x + y * y)
        th = np.arctan(r)
        thd = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
        s = np.where(r > 1e-9, thd / np.maximum(r, 1e-9), 1.0)
        return fx * x * s + cx, fy * y * s + cy
    raise ValueError(f"no mask warp for camera model {model}")


def warp_mask(keep: np.ndarray, src_model: str, src_params: list[float], dst_params: list[float], dst_w: int, dst_h: int) -> np.ndarray:
    """A keep-mask in the original (distorted) frame -> the same mask in the undistorted PINHOLE view
    (dst_params = fx, fy, cx, cy). Nearest neighbour, so it stays binary; outside the original frame counts as room."""
    import cv2
    fx, fy, cx, cy = [float(v) for v in dst_params]
    u, v = np.meshgrid(np.arange(dst_w, dtype=np.float64) + 0.5, np.arange(dst_h, dtype=np.float64) + 0.5)
    su, sv = distort(src_model, src_params, (u - cx) / fx, (v - cy) / fy)
    mx, my = (su - 0.5).astype(np.float32), (sv - 0.5).astype(np.float32)  # cv2: pixel centres at integers
    out = cv2.remap(keep.astype(np.uint8) * 255, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return out > 127


def undistort_masks(out: Path, src_rec, dst_rec, dest: Path) -> dict:
    """Warp <out>/masks/ into `dest` for every image of the undistorted model dst_rec (pycolmap Reconstructions;
    src_rec is the solved, distorted one). Returns counts."""
    mdir = out / MASKS
    src_by_name = {im.name: im for im in src_rec.images.values()}
    dest.mkdir(parents=True, exist_ok=True)
    n, missing = 0, 0
    for im in dst_rec.images.values():
        keep = read_keep(mdir / mask_name(im.name))
        s = src_by_name.get(im.name)
        if keep is None or s is None:
            missing += 1
            continue
        sc, dc = src_rec.cameras[s.camera_id], dst_rec.cameras[im.camera_id]
        if keep.shape != (sc.height, sc.width):
            missing += 1
            continue
        if keep.all():  # nobody in this frame: an all-white file, no warp needed
            warped = np.ones((dc.height, dc.width), bool)
        else:
            warped = warp_mask(keep, sc.model.name, list(sc.params), list(dc.params), dc.width, dc.height)
        write_mask(dest / mask_name(im.name), ~warped)
        n += 1
    return {"masks": n, "withoutMask": missing}


# ---- training ----------------------------------------------------------------------------------------------------

def install(st, masks: Path, log=print) -> dict:
    """Wrap st.Dataset.__getitem__: each view carries its keep-mask as data['mask'] ([H, W] bool, True = train on
    it), joined with any mask gsplat already set, and depth samples on people are dropped."""
    import torch
    masks = Path(masks)
    files = {p.name[:-4]: p for p in masks.glob("*.png")}  # image name -> mask file
    if not files:
        raise ValueError(f"{masks} has no masks")
    orig = st.Dataset.__getitem__
    stats = {"masks": len(files), "applied": 0, "skipped": 0}

    def getitem(self, i):
        data = orig(self, i)
        try:
            name = Path(self.parser.image_names[self.indices[i]]).name
            f = files.get(name)
            if f is None:
                return data
            keep = read_keep(f)
            h, w = int(data["image"].shape[0]), int(data["image"].shape[1])
            if keep is None or keep.shape != (h, w):  # a crop (patch_size) or a resized view: train it whole
                stats["skipped"] += 1
                return data
            if keep.all():
                return data
            t = torch.from_numpy(keep)
            data["mask"] = (data["mask"].bool() & t) if "mask" in data else t
            if "points" in data and "depths" in data and len(data["points"]):
                pts = data["points"]
                xi = pts[:, 0].round().long().clamp(0, w - 1)
                yi = pts[:, 1].round().long().clamp(0, h - 1)
                ok = data["mask"][yi, xi]
                if ok.sum() >= 8:  # gsplat's depth loss needs some points; keep all rather than none
                    data["points"], data["depths"] = pts[ok], data["depths"][ok]
            stats["applied"] += 1
        except Exception as e:  # one bad mask must not stop a paid training run
            stats["skipped"] += 1
            if stats["skipped"] <= 3:
                log(f"people mask: view {i} trains whole ({str(e)[:120]})")
        return data

    st.Dataset.__getitem__ = getitem
    return stats


def hide(img: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Blur people out of a picture that gets published (held-out photos): a heavy pixelate inside the mask."""
    import cv2
    if keep.all():
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 40), max(1, h // 40)), interpolation=cv2.INTER_AREA)
    coarse = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 6)
    out = img.copy()
    out[~keep] = coarse[~keep]
    return out


def review_sheet(out: Path, count: int = 24, thumb: int = 320) -> Path | None:
    """<out>/people-check.jpg: the frames with the most person in them, people blurred out (hide) and the mask edge
    drawn in red, so a reviewer can see what training will skip without anyone's face being published."""
    import cv2
    mdir = out / MASKS
    rows = []
    for m in sorted(mdir.glob("*.jpg.png")):
        keep = read_keep(m)
        if keep is not None and not keep.all():
            rows.append((float((~keep).mean()), m))
    if not rows:
        return None
    rows = sorted(rows, reverse=True)[:count]
    tiles = []
    for _, m in sorted(rows, key=lambda r: r[1].name):
        img = cv2.imread(str(out / "frames" / m.name[:-4]))
        keep = read_keep(m)
        if img is None or keep is None or keep.shape != img.shape[:2]:
            continue
        img = hide(img, keep)
        edge = cv2.morphologyEx((~keep).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
        img[edge] = (0, 0, 255)
        h, w = img.shape[:2]
        k = thumb / max(h, w)
        t = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        cv2.putText(t, m.name[:-8], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return None
    th, tw = max(t.shape[0] for t in tiles), max(t.shape[1] for t in tiles)
    cols = min(6, len(tiles))
    sheet = np.zeros((th * ((len(tiles) + cols - 1) // cols), tw * cols, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:r * th + t.shape[0], c * tw:c * tw + t.shape[1]] = t
    dest = out / "people-check.jpg"
    cv2.imwrite(str(dest), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return dest
