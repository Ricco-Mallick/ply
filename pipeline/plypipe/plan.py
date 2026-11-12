# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""plan: the settings this capture needs, measured rather than assumed.

A fixed kept frame rate and a fixed gaussian cap are wrong for both a 30 s scan of one room and an
8 min walk of a floor. This module reads the inputs (clip length and pixels, the phone's own motion
log when the capture app produced one) and the solved model (surface area, views that registered),
and returns the settings that fit:

  * kept frames per second   follows how fast the operator actually moved
  * working resolution       follows the pixels the clips have
  * MCMC gaussian cap        follows the room's surface area and the number of views
  * training steps           follows the views (each image seen about fifty times)
  * export thresholds        probed against the model's own values, never assumed

Everything is written to <out>/plan.json so a finished run can be audited afterwards: what was
decided, from what, and why.
"""

from __future__ import annotations

import json
from pathlib import Path

# kept frames: a normal walk wants about one every half metre; faster or turning wants more
_SECONDS_PER_FRAME = 0.46
_FPS_MIN, _FPS_MAX = 1.5, 3.2
_FRAMES_MIN, _FRAMES_MAX = 700, 3200

# pixels: never upscale, and a phone render gains little past 2k from a 4k clip
_SIDE_MIN, _SIDE_MAX = 1280, 2048
_SIDE_SHARE = 0.45

# the phone file: steps trained on from the model export makes for phones (fewer splats, SH 2), at its own splat
# count (plypipe.finetune). Measured at this length only: +0.80 dB on the chapel for 151 s of an L40S, +0.73 and
# +0.52 dB on Dr Johnson for 65 s (docs/EXPERIMENTS.md, Q6); --lite-finetune 0 turns it off
LITE_FINETUNE_STEPS = 3_000

# gaussians: enough to cover the surfaces once, capped again by what the views can support. The view cap
# counts pixels, not frames: 2400 per view-megapixel is the old 2200 per 1280x720 video frame, while a
# 1.2 MP photo carries more. Both caps scale with the preset; a view cap that ignored it made `high` equal `standard`
_GAUSSIANS_PER_M2 = 4200
_GAUSSIANS_PER_VIEW_MP = 2400
_DEFAULT_VIEW_MP = 0.92  # 1280 x 720, the planned working size of a 1080p clip
_CAP_MIN = 300_000
# ceilings: Dr Johnson trained cleanly at 1.47M splats (31.0 dB as the viewer shows it) and went unstable at 3M in
# both runs that tried (loss climbing from 3.6k steps, 10.7 dB). Until a larger cap is proven, stay under it
_PRESET_CAP_MAX = {"quick": 1_000_000, "standard": 2_000_000, "high": 2_000_000}

# steps: one view every ~50 steps is where gsplat stops adding detail. Small captures still need gsplat's
# own schedules (7k / 30k), which is what the per-preset floors are; the preset scales the rest
_STEPS_PER_VIEW = 50
_PRESET_STEP = {"quick": 0.35, "standard": 1.0, "high": 1.25}
_PRESET_CAP = {"quick": 0.5, "standard": 1.0, "high": 2.0}
_PRESET_FLOOR = {"quick": 7_000, "standard": 30_000, "high": 40_000}
_PRESET_CEIL = {"quick": 24_000, "standard": 55_000, "high": 60_000}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def motion(capture: dict | None) -> dict:
    """How the operator moved, from the capture app's own samples (flow %/s and turn deg/s).

    Without capture.json (an upload, an older capture) there is nothing to measure, so the plan
    falls back to a normal walk rather than guessing a number."""
    flows: list[float] = []
    turns: list[float] = []
    for c in (capture or {}).get("clips") or []:
        for q in c.get("quality") or []:
            if isinstance(q, (list, tuple)) and len(q) >= 6:  # tMs, grade, sharp, luma, flow, turn
                flows.append(abs(float(q[4])))
                turns.append(abs(float(q[5])))
    f, t = _pct(flows, 0.8), _pct(turns, 0.8)
    if f is None or t is None:
        return {"source": "none", "kind": "normal", "factor": 1.0}
    factor = 1.0 + _clamp((f - 18) / 45, -0.25, 0.55) + _clamp((t - 25) / 70, -0.15, 0.3)
    factor = round(_clamp(factor, 0.7, 1.6), 2)
    kind = "fast" if factor > 1.2 else "slow" if factor < 0.9 else "normal"
    return {"source": "sensors", "kind": kind, "factor": factor, "flowP80": round(f, 1), "turnP80": round(t, 1)}


def input_plan(duration_s: float, clips: int, source_long: int, capture: dict | None) -> dict:
    """Frames and pixels for the clips, before anything has been solved."""
    mo = motion(capture)
    budget = int(_clamp(2.0 * duration_s + 400, _FRAMES_MIN, _FRAMES_MAX))
    kept = _clamp((1 / _SECONDS_PER_FRAME) * mo["factor"], _FPS_MIN, _FPS_MAX)
    capped = duration_s > 0 and kept * duration_s > budget
    if capped:
        kept = budget / duration_s
    side = int(_clamp(round(source_long * _SIDE_SHARE / 16) * 16, _SIDE_MIN, _SIDE_MAX))
    return {
        "keptFps": round(kept, 2),
        "longSide": side,
        "frameBudget": budget,
        "expectedFrames": int(round(kept * duration_s)),
        "clips": clips,
        "durationSec": round(duration_s, 1),
        "sourceLongSide": source_long,
        "motion": mo,
        "why": (f"{mo['kind']} walk ({mo['source']}) → {round(kept, 2)} frames/s, "
                f"{side} px from {source_long} px source" + (", capped by the frame budget" if capped else "")),
    }


def surface_area(bounds: dict | None, ceiling: float | None = None) -> dict:
    """Wall + floor + ceiling area of the solved box, with the ceiling taken from the shell when it has one."""
    if not bounds:
        return {"floorM2": None, "surfaceM2": None, "heightM": None}
    (x0, y0, z0), (x1, y1, z1) = bounds["min"], bounds["max"]
    dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    h = _clamp(ceiling or dy, 2.2, 6.0)
    floor = dx * dz
    return {"floorM2": round(floor, 1), "surfaceM2": round(2 * floor + 2 * (dx + dz) * h, 1),
            "heightM": round(h, 2), "sizeM": [round(dx, 2), round(h, 2), round(dz, 2)]}


def model_plan(images: int, bounds: dict | None, preset: str = "standard", ceiling: float | None = None,
               *, iterations: int | None = None, cap_max: int | None = None, megapixels: float | None = None) -> dict:
    """Training budget once the model is solved. `megapixels` is the size of one training view."""
    area = surface_area(bounds, ceiling)
    mult = _PRESET_CAP.get(preset, 1.0)
    mp = megapixels or _DEFAULT_VIEW_MP
    notes: list[str] = []
    if area["surfaceM2"]:
        cap = int(area["surfaceM2"] * _GAUSSIANS_PER_M2 * mult)
        notes.append(f"{area['surfaceM2']} m² of surface × {_GAUSSIANS_PER_M2}/m² × {mult}")
    else:  # no bounds yet: fall back to what the views can carry
        cap = int(images * mp * _GAUSSIANS_PER_VIEW_MP * mult) if images else _CAP_MIN
        notes.append("no bounds: budget from the views")
    cap_hi = _PRESET_CAP_MAX.get(preset, _PRESET_CAP_MAX["standard"])
    cap = int(_clamp(cap, _CAP_MIN, cap_hi))
    if images:
        per_view = int(images * mp * _GAUSSIANS_PER_VIEW_MP * mult)
        if cap > per_view:
            cap = max(_CAP_MIN, per_view)
            notes.append(f"capped at {_GAUSSIANS_PER_VIEW_MP}/view-MP × {images} views × {mp:.2f} MP × {mult}")
    floor = _PRESET_FLOOR.get(preset, _PRESET_FLOOR["standard"])
    steps = int(_clamp(images * _STEPS_PER_VIEW * _PRESET_STEP.get(preset, 1.0),
                       floor, _PRESET_CEIL.get(preset, _PRESET_CEIL["standard"])))
    notes.append(f"{images} views → {steps} steps ({steps / max(images, 1):.0f}/view)")
    return {
        "preset": preset,
        "iterations": iterations or steps,
        "capMax": cap_max or cap,
        "planIterations": steps,
        "planCapMax": cap,
        "views": images,
        "viewMegapixels": round(mp, 3),
        **area,
        "why": "; ".join(notes),
    }


def write(out: Path, section: str, data: dict) -> dict:
    """Merge one section into <out>/plan.json and return the whole plan."""
    path = Path(out) / "plan.json"
    try:
        plan = json.loads(path.read_text())
    except (OSError, ValueError):
        plan = {}
    plan[section] = data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1))
    return plan


def read(out: Path) -> dict:
    try:
        return json.loads((Path(out) / "plan.json").read_text())
    except (OSError, ValueError):
        return {}
