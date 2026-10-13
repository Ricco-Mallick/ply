"""frames: video (or a folder of photos) -> sharp, evenly spaced, resized JPEGs in <out>/frames.

Video path: ffmpeg oversamples at `fps * oversample` (auto-rotated, long side capped), then
for every window of `oversample` consecutive frames the sharpest one (variance of the
Laplacian) is kept. Frames far blurrier than their neighbours are dropped as long as that
does not open a gap of more than `max_gap` windows (gaps break SfM more than blur does).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np

from plypipe.util import Ctx, need, run

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian on a 640-px greyscale copy (resolution independent enough
    to compare frames of one video)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    s = 640.0 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def select_sharpest(scores: list[float], window: int, rel_min: float = 0.35, max_gap: int = 2) -> list[int]:
    """Indices to keep: the sharpest per window, minus outliers that are much blurrier than
    the median of kept frames, never dropping more than `max_gap` windows in a row."""
    if window < 1:
        raise ValueError("window must be >= 1")
    picks = []
    for start in range(0, len(scores), window):
        seg = scores[start:start + window]
        picks.append(start + int(np.argmax(seg)))
    if not picks:
        return []
    med = float(np.median([scores[i] for i in picks]))
    keep, gap = [], 0
    for i in picks:
        if scores[i] < rel_min * med and gap < max_gap:
            gap += 1
            continue
        keep.append(i)
        gap = 0
    return keep


def probe_duration(video: Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
                             capture_output=True, text=True, check=True).stdout
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return None


def _scale_filter(long_side: int) -> str:
    # never upscale; keep even dimensions
    return (f"scale='if(gte(iw,ih),min({long_side},iw),-2)':'if(gte(iw,ih),-2,min({long_side},ih))'"
            ":flags=lanczos")


def run_stage(ctx: Ctx, src: Path) -> dict:
    fps = float(ctx.opt("fps", 2.5))
    oversample = int(ctx.opt("oversample", 4))
    long_side = int(ctx.opt("long_side", 1600))
    max_frames = ctx.opt("max_frames")
    out = ctx.frames_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if src.is_dir() and not any(p.suffix.lower() in VIDEO_EXT or p.suffix.lower() == ".tar" for p in src.iterdir()):
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not files:
            raise ValueError(f"no images in {src}")
        kept = 0
        scores = []
        for i, p in enumerate(files):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                ctx.log.emit("frames", "warn", f"unreadable image skipped: {p.name}")
                continue
            h, w = img.shape[:2]
            s = long_side / max(h, w)
            if s < 1:
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            scores.append(sharpness(img))
            cv2.imwrite(str(out / f"{kept:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept += 1
        return {"source": "images", "input": len(files), "kept": kept, "dropped": len(files) - kept,
                "sharpnessMedian": float(np.median(scores)) if scores else 0.0}

    tars = sorted(p for p in src.iterdir() if p.suffix.lower() == ".tar") if src.is_dir() else []
    videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXT) if src.is_dir() else [src]
    if tars:
        return _ar_frames(ctx, src, tars, videos, out, fps, oversample, long_side, max_frames)
    if not videos:
        raise ValueError(f"no videos in {src}")
    # the capture app ships capture.json (quality + sensor log) next to the clips; keep it with the job
    if src.is_dir() and (src / "capture.json").exists():
        shutil.copy2(src / "capture.json", ctx.out / "capture.json")
    clips, kept = [], 0
    for v in videos:
        st = _video_frames(ctx, v, out, kept, fps, oversample, long_side, max_frames / len(videos) if max_frames else None)
        clips.append({"file": v.name, "firstFrame": kept, **st})
        kept += st["kept"]
    if len(clips) == 1:
        return {"source": "video", **{k: v for k, v in clips[0].items() if k not in ("file", "firstFrame")}}
    return {"source": "videos", "clips": clips, "kept": kept, "dropped": sum(c["dropped"] for c in clips),
            "sharpnessMedian": float(np.median([c["sharpnessMedian"] for c in clips]))}


def _video_frames(ctx: Ctx, src: Path, out: Path, start: int, fps: float, oversample: int, long_side: int,
                  max_frames: float | None) -> dict:
    """Sharpest frame per window of one video, written to out/ numbered from `start`."""
    need("ffmpeg", "Install ffmpeg (apt install ffmpeg) or put a static build on PATH.")
    duration = probe_duration(src)
    if max_frames and duration:
        fps = min(fps, float(max_frames) / duration)
    raw = ctx.out / "frames_raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vf", f"fps={fps * oversample:.4f},{_scale_filter(long_side)}", "-q:v", "2", str(raw / "%06d.jpg")],
        ctx, "frames")
    files = sorted(raw.glob("*.jpg"))
    if not files:
        raise RuntimeError("ffmpeg produced no frames (is the input a video?)")
    scores = []
    for i, p in enumerate(files):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        scores.append(sharpness(img))
        if i % 200 == 0:
            ctx.log.emit("frames", "progress", f"{src.name}: scored {i}/{len(files)}", done=i, total=len(files))
    keep = select_sharpest(scores, oversample)
    for n, i in enumerate(keep):
        shutil.move(str(files[i]), out / f"{start + n:06d}.jpg")
    shutil.rmtree(raw)
    return {"source": "video", "durationSec": duration, "fps": fps, "oversample": oversample,
            "extracted": len(files), "windows": -(-len(files) // oversample), "kept": len(keep),
            "dropped": -(-len(files) // oversample) - len(keep),
            "sharpnessMedian": float(np.median([scores[i] for i in keep])) if keep else 0.0}


def _ar_frames(ctx: Ctx, src: Path, tars: list[Path], videos: list[Path], out: Path, fps: float, oversample: int,
               long_side: int, max_frames) -> dict:
    """AR scans from the capture app: tars of keyframes (already chosen on the phone for motion and spacing) plus
    poses.json (camera-to-world in ARCore's 'local-floor' space: metres, +y up, floor at y = 0). Frames are copied
    in order, the softest few percent dropped; poses go to ar_poses.json for align (real scale, level floor)."""
    if (src / "capture.json").exists():
        shutil.copy2(src / "capture.json", ctx.out / "capture.json")
    kept, segments, dropped, scores_all = 0, [], 0, []
    for seg, tp in enumerate(tars):
        with tarfile.open(tp) as tf:
            meta = json.loads(tf.extractfile("poses.json").read())
            frames = []
            for kf in meta["keyframes"]:
                try:
                    data = tf.extractfile(kf["file"]).read()
                except (KeyError, AttributeError):
                    continue
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                frames.append((kf, img, sharpness(img)))
        if not frames:
            ctx.log.emit("frames", "warn", f"{tp.name}: no readable keyframes")
            continue
        sh = np.array([f[2] for f in frames])
        floor = np.percentile(sh, float(ctx.opt("ar_drop_pct", 5)))  # the phone already skips fast motion
        # many frames (video mode records 4/s): keep the sharpest of each run of neighbours, up to the cap
        cap = int((max_frames or int(ctx.opt("ar_max_frames", 450))) / len(tars))
        step = max(1, int(np.ceil(len(frames) / cap)))
        keep = {i for w0 in range(0, len(frames), step) for i in [max(range(w0, min(w0 + step, len(frames))), key=lambda k: frames[k][2])]}
        seg_frames = []
        for i, (kf, img, sc) in enumerate(frames):
            if i not in keep or (step == 1 and sc < floor):
                dropped += 1
                continue
            h, w = img.shape[:2]
            s = long_side / max(h, w)
            if s < 1:
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            name = f"{kept:06d}.jpg"
            cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            seg_frames.append({"frame": name, "m": kf["m"], "t": kf.get("t"), "fx": kf["fx"] * min(1, s), "fy": kf["fy"] * min(1, s),
                               "cx": kf["cx"] * min(1, s), "cy": kf["cy"] * min(1, s)})
            scores_all.append(sc)
            kept += 1
        segments.append({"file": tp.name, "keyframes": len(frames), "kept": len(seg_frames), "frames": seg_frames,
                         "coverage": meta.get("coverage"), "device": meta.get("device")})
    if not kept:
        raise ValueError("the AR scans held no usable frames")
    (ctx.out / "ar_poses.json").write_text(json.dumps({"space": "local-floor", "segments": segments}))
    extra = []
    for v in videos:  # uploaded videos next to AR scans: frames appended after the scans (no poses)
        st = _video_frames(ctx, v, out, kept, fps, oversample, long_side, None)
        extra.append({"file": v.name, "firstFrame": kept, **st})
        kept += st["kept"]
    return {"source": "ar", "segments": [{k: v for k, v in sg.items() if k != "frames"} for sg in segments], "videos": extra,
            "kept": kept, "dropped": dropped, "sharpnessMedian": float(np.median(scores_all)) if scores_all else 0.0}
