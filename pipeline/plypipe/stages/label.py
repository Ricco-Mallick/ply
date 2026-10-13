"""label: find objects with SAM 3 and place them in 3D -> places.json (the walkthrough's pins, search and AI tint).

1. SAM 3 (facebook/sam3 through transformers, text prompts) runs on up to --label-frames undistorted frames, each frame
   encoded once and asked for every prompt.
2. Lifting needs no depth: each mask keeps the camera-solving points that frame actually observed inside it
   (COLMAP's 2D-3D correspondences), so a detection becomes a set of 3D point ids.
3. Detections of the same class that share points (or sit within 40 cm) are one object; an object needs two frames,
   or one very confident one. Its box is the 10th-90th percentile of its points, after dropping outliers.

Needs a GPU, transformers >= 5 and access to the gated facebook/sam3 weights (HF_TOKEN). If any of that is missing the
stage logs a warning and writes an empty places.json: labelling is optional, the twin is not.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from plypipe.util import Ctx

# indoor things worth finding; covers rooms, classrooms, halls and chapels (one prompt per class, singular)
PROMPTS = ["door", "window", "chair", "table", "desk", "bench", "pew", "bed", "sofa", "shelf", "cabinet", "whiteboard",
           "blackboard", "projector screen", "television", "lamp", "potted plant", "altar", "statue", "stained glass window",
           "ceiling fan", "clock", "staircase", "sign", "podium"]
CATEGORY = {"door": "Exits", "staircase": "Exits", "chair": "Furniture", "table": "Furniture", "desk": "Furniture",
            "bench": "Furniture", "pew": "Furniture", "bed": "Furniture", "sofa": "Furniture", "podium": "Furniture",
            "altar": "Furniture", "shelf": "Storage", "cabinet": "Storage"}
# shown as pins straight away; the rest are searchable and appear in the AI views
PINNED = {"door", "staircase", "whiteboard", "blackboard", "projector screen", "altar", "statue", "podium", "stained glass window"}


def lift(dets: list[dict], xyz: dict[int, np.ndarray], min_frames: int = 2, solo_score: float = 0.85) -> list[dict]:
    """dets: {cls, score, frame, ids (set of point3D ids)} -> objects {cls, conf, frames, center, half}."""
    objs: list[dict] = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        pts = np.array([xyz[i] for i in d["ids"] if i in xyz])
        if len(pts) < 4:
            continue
        c = np.median(pts, axis=0)
        best, best_s = None, 0.0
        for o in objs:
            if o["cls"] != d["cls"]:
                continue
            j = len(o["ids"] & d["ids"]) / max(1, min(len(o["ids"]), len(d["ids"])))
            near = np.linalg.norm(o["c"] - c) < 0.4
            s = j if j > 0.15 else (0.16 if near else 0)
            if s > best_s:
                best, best_s = o, s
        if best is None:
            objs.append({"cls": d["cls"], "ids": set(d["ids"]), "scores": [d["score"]], "frames": {d["frame"]}, "c": c})
        else:
            best["ids"] |= d["ids"]
            best["scores"].append(d["score"])
            best["frames"].add(d["frame"])
            best["c"] = np.median(np.array([xyz[i] for i in best["ids"] if i in xyz]), axis=0)
    # one physical thing seen as several classes (a table also called "desk" and "podium"): keep the best-supported
    for o in objs:
        o["support"] = len(o["frames"]) * float(np.mean(sorted(o["scores"])[-5:]))
    objs.sort(key=lambda o: -o["support"])
    kept_objs: list[dict] = []
    for o in objs:
        if any(len(o["ids"] & k["ids"]) / max(1, min(len(o["ids"]), len(k["ids"]))) > 0.5 for k in kept_objs):
            continue
        kept_objs.append(o)
    out = []
    for o in kept_objs:
        if len(o["frames"]) < min_frames and max(o["scores"]) < solo_score:
            continue
        p = np.array([xyz[i] for i in o["ids"] if i in xyz])
        med = np.median(p, axis=0)
        mad = np.median(np.abs(p - med), axis=0) + 1e-3
        p = p[(np.abs(p - med) <= 3 * 1.4826 * mad).all(axis=1)]
        if len(p) < 4:
            continue
        lo, hi = np.percentile(p, 10, axis=0), np.percentile(p, 90, axis=0)
        conf = float(np.mean(sorted(o["scores"])[-5:])) * min(1.0, 0.6 + 0.2 * len(o["frames"]))
        out.append({"cls": o["cls"], "conf": round(conf, 3), "frames": len(o["frames"]), "points": len(p),
                    "center": [round(float(v), 3) for v in (lo + hi) / 2], "half": [round(float(max(0.05, v)), 3) for v in (hi - lo) / 2]})
    return out


def places_from(objects: list[dict]) -> list[dict]:
    """Objects -> the viewer's places (names numbered per class, nearest-first order not needed)."""
    n: dict[str, int] = defaultdict(int)
    places = []
    for o in sorted(objects, key=lambda o: (o["cls"], -o["conf"])):
        n[o["cls"]] += 1
        name = o["cls"].capitalize() + (f" {n[o['cls']]}" if n[o["cls"]] > 1 else "")
        places.append({"id": f"{o['cls'].replace(' ', '-')}-{n[o['cls']]}", "name": name, "cat": CATEGORY.get(o["cls"], "Things"), "p": o["center"],
                       "pinned": o["cls"] in PINNED and o["conf"] >= 0.5, "ai": {"cls": o["cls"], "conf": o["conf"], "half": o["half"]}})
    return places


def run_stage(ctx: Ctx) -> dict:
    out_file = ctx.out / "places.json"
    if ctx.opt("no_labels"):
        out_file.write_text(json.dumps({"version": 1, "source": "none", "places": []}))
        return {"skipped": "--no-labels"}
    try:
        import pycolmap
        import torch
        from PIL import Image
        from transformers import Sam3Model, Sam3Processor
    except Exception as e:  # pragma: no cover - GPU image only
        out_file.write_text(json.dumps({"version": 1, "source": "none", "places": []}))
        ctx.log.emit("label", "warn", f"labelling skipped: {e}")
        return {"skipped": str(e)}
    if not torch.cuda.is_available():
        out_file.write_text(json.dumps({"version": 1, "source": "none", "places": []}))
        ctx.log.emit("label", "warn", "labelling skipped: no GPU")
        return {"skipped": "no GPU"}
    rec = pycolmap.Reconstruction(str(ctx.data_dir / "sparse" / "0"))
    xyz = {pid: np.asarray(p.xyz, float) for pid, p in rec.points3D.items()}
    ims = sorted(rec.images.values(), key=lambda im: im.name)
    k = max(1, int(np.ceil(len(ims) / int(ctx.opt("label_frames", 120)))))
    ims = ims[::k]
    try:
        model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16).to("cuda").eval()
        proc = Sam3Processor.from_pretrained("facebook/sam3")
    except Exception as e:  # no access to the gated weights, no network...
        out_file.write_text(json.dumps({"version": 1, "source": "none", "places": []}))
        ctx.log.emit("label", "warn", f"labelling skipped: could not load SAM 3 ({str(e)[:200]})")
        return {"skipped": "weights"}
    dets: list[dict] = []
    try:
        dets = _detect(ctx, model, proc, ims, torch, Image)
    except Exception as e:  # OOM, an API change, a bad frame...: no labels, but the twin goes on
        ctx.log.emit("label", "warn", f"labelling stopped: {type(e).__name__}: {str(e)[:200]}")
        out_file.write_text(json.dumps({"version": 1, "source": "none", "places": []}))
        return {"skipped": f"{type(e).__name__}"}
    finally:
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()  # training runs next in a separate process: give it the whole GPU
    objects = lift(dets, xyz)
    places = places_from(objects)
    out_file.write_text(json.dumps({"version": 1, "source": "sam3", "places": places}, indent=1))
    counts: dict[str, int] = defaultdict(int)
    for o in objects:
        counts[o["cls"]] += 1
    return {"frames": len(ims), "detections": len(dets), "objects": len(objects), "byClass": dict(counts)}

def _detect(ctx: Ctx, model, proc, ims, torch, Image) -> list[dict]:
    """SAM 3 on each frame (encoded once), every prompt; masks -> the COLMAP point ids observed inside them."""
    texts = {p: proc(text=p, return_tensors="pt").to("cuda") for p in PROMPTS}
    dets = []
    thr = float(ctx.opt("label_threshold", 0.5))
    for n, im in enumerate(ims):
        img = Image.open(ctx.data_dir / "images" / im.name).convert("RGB")
        W, H = img.size
        obs = [(p.xy, p.point3D_id) for p in im.points2D if p.has_point3D()]
        if not obs:
            continue
        uv = np.floor(np.array([o[0] for o in obs])).astype(int)  # COLMAP pixel centres are at +0.5
        ids = np.array([o[1] for o in obs])
        ok = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < W) & (uv[:, 1] < H)
        uv, ids = uv[ok], ids[ok]
        inputs = proc(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ve = model.get_vision_features(pixel_values=inputs.pixel_values)
            for prompt, t in texts.items():
                outp = model(vision_embeds=ve, input_ids=t.input_ids, attention_mask=t.get("attention_mask"))
                res = proc.post_process_instance_segmentation(outp, threshold=thr, mask_threshold=0.5, target_sizes=[[H, W]])[0]
                for m, sc in zip(res["masks"], res["scores"]):
                    mm = m.bool().cpu().numpy()
                    hit = set(ids[mm[uv[:, 1], uv[:, 0]]].tolist())
                    if len(hit) >= 4:
                        dets.append({"cls": prompt, "score": float(sc), "frame": im.name, "ids": hit})
        if n % 5 == 0:
            ctx.log.emit("label", "progress", f"frame {n + 1}/{len(ims)}", done=n + 1, total=len(ims))
    return dets
