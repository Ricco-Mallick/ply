# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""People (and anything else that moved) masked with YOLO, for GPUs SAM 3 does not fit (a 4 GB GTX 1650).

Same job and same output as plypipe.people with SAM 3: masks/<frame>.jpg.png (white = room, black = person), frames
that are mostly person moved aside, masks/people.json. Everything after it (the camera solve's masks, undistortion,
the trainer's loss) cannot tell which model made them.

Models (ultralytics, downloaded on first use, a few hundred MB of VRAM):
  * YOLOE-11 seg with text prompts: the same open vocabulary as SAM 3's prompts ("person,ceiling fan")
  * YOLO11 seg (COCO): the fallback when YOLOE cannot load; it knows 80 classes (person yes, ceiling fan no), and a
    prompt it has no class for is reported, not silently dropped

Runs in the Python that has torch (on a local PC the training venv, PLY_TRAIN_PYTHON), as its own process:
    python -m plypipe.people_yolo --frames <out>/frames --out <out> --prompts "person" [--threshold 0.3]
Prints the report as JSON on its last line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import os

YOLOE_MODEL = "yoloe-11l-seg.pt"
COCO_MODEL = "yolo11m-seg.pt"
# weights land here, not in whatever folder the pipeline was started from (ultralytics' default)
MODEL_DIR = Path(os.environ.get("PLY_MODEL_DIR") or Path.home() / ".cache" / "ply" / "models")


def weights(name: str) -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODEL_DIR / name)
THRESHOLD = 0.3


def yoloe_detector(prompts: list[str], threshold: float):
    from ultralytics import YOLOE
    model = YOLOE(weights(YOLOE_MODEL))
    model.set_classes(prompts, model.get_text_pe(prompts))
    return _detector(model, None, threshold, f"ultralytics/{YOLOE_MODEL}", prompts, [])


def coco_detector(prompts: list[str], threshold: float):
    from ultralytics import YOLO
    model = YOLO(weights(COCO_MODEL))
    names = {v.lower(): k for k, v in model.names.items()}
    keep = [names[p.lower()] for p in prompts if p.lower() in names]
    missing = [p for p in prompts if p.lower() not in names]
    if not keep:
        raise RuntimeError(f"none of the prompts ({', '.join(prompts)}) is a COCO class")
    return _detector(model, keep, threshold, f"ultralytics/{COCO_MODEL}", prompts, missing)


def _detector(model, classes, threshold, name, prompts, missing):
    import cv2

    def detect(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        res = model.predict(rgb[..., ::-1].copy(), conf=threshold, classes=classes, verbose=False, retina_masks=True)[0]
        out = np.zeros((h, w), bool)
        if res.masks is None:
            return out
        for m in res.masks.data.cpu().numpy():
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            out |= m > 0.5
        return out

    detect.prompts = ",".join(prompts)
    detect.model = name
    detect.missing = missing
    return detect


def load(prompts: list[str], threshold: float, log=print):
    try:
        return yoloe_detector(prompts, threshold)
    except Exception as e:  # no CLIP text model download, an older ultralytics
        log(f"[people] YOLOE unavailable ({type(e).__name__}: {str(e)[:120]}); using COCO YOLO11-seg")
        return coco_detector(prompts, threshold)


def main(argv: list[str] | None = None) -> int:
    from plypipe import people
    ap = argparse.ArgumentParser(prog="python -m plypipe.people_yolo")
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", default=people.PROMPT)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--max-share", type=float, default=people.MAX_SHARE)
    a = ap.parse_args(argv)
    prompts = [p.strip() for p in a.prompts.split(",") if p.strip()]
    detect = load(prompts, a.threshold, log=lambda m: print(m, flush=True))
    rep = people.mask_frames(a.frames, a.out, detect, a.max_share,
                             log=lambda msg, **kw: print(json.dumps({"stage": "sfm", "event": "progress", "msg": msg, **kw}), flush=True))
    if detect.missing:
        rep["promptsNotFound"] = detect.missing
        (a.out / people.MASKS / "people.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"plyPeople": rep}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
