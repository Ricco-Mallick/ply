# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""review: what the reviewer thinks of a finished run, in sentences a human can agree or disagree with.

This is the reasoning layer in front of the human gate. It reads the same report the gate reads, weighs
what went right against what is still risky, and recommends publish, retrain or recapture with a
confidence. It is deterministic on purpose: no model key is required for a job to be reviewable, and the
same report always gets the same opinion. When a model is available, its note can be attached to the job
alongside this one, but the numbers stay the source of truth.
"""

from __future__ import annotations

RECOMMEND = {"publish": "publish", "retrain": "retrain with refined settings", "recapture": "recapture before retraining"}


def _vps(report: dict) -> float | None:
    f = report.get("frames") or {}
    clips = f.get("clips") or []
    secs = sum(float(c.get("durationSec") or 0) for c in clips) if clips else float(f.get("durationSec") or 0)
    return round(int(f.get("kept") or 0) / secs, 2) if secs else None


def opinion(report: dict) -> dict:
    """{verdict, confidence, recommend, good[], risk[], note} from a finished run's report."""
    frames = report.get("frames") or {}
    sfm = report.get("sfm") or {}
    align = report.get("align") or {}
    train = report.get("train") or {}
    export = report.get("export") or {}
    quality = report.get("quality") or {}

    views = _vps(report)
    registered = sfm.get("registeredRatio")
    reproj = sfm.get("meanReprojErrorPx")
    residual = align.get("floorResidualMedianM")
    kept = export.get("keptFraction")
    clip = export.get("clippedFraction")
    floater = (export.get("floaterClean") or {}).get("keptFraction")

    good: list[str] = []
    risk: list[str] = []

    if registered is not None and registered >= 0.95:
        good.append(f"camera solving held: {registered:.0%} of frames registered" + (f" at {reproj:.2f} px" if isinstance(reproj, (int, float)) else ""))
    if isinstance(residual, (int, float)) and residual < 0.01:
        good.append(f"the floor sits within {residual * 1000:.0f} mm, so the twin is metric before anything is walked")
    if views is not None and views >= 2.1:
        good.append(f"{views:.1f} frames per second of footage: enough overlap for surfaces to be fitted, not invented")
    if floater is not None and floater >= 0.9:
        good.append(f"the floater pass kept {floater:.0%} of splats, so it removed space, not surfaces")

    if views is not None and views < 1.8:
        risk.append(f"{views:.1f} frames per second is under what the fit needs; expect softness wherever the camera passed quickly")
    if isinstance(kept, (int, float)) and kept < 0.85:
        risk.append(f"export kept {kept:.0%} of the trained splats, so the training cap is above what this room carries")
    if isinstance(clip, (int, float)) and clip > 0.02:
        risk.append(f"{clip:.1%} of splats sit at the top of the colour range: the capture clipped, the twin will glare")
    if train.get("antialiased") is False:
        risk.append("trained without antialiasing, which reads as sparkle at distance")
    if export.get("floaterClean") is None:
        risk.append("the floater pass did not record a result")

    fails = [i for i in quality.get("issues", []) if i.get("level") == "fail"]
    if fails:
        verdict, recommend, confidence = "reject", "recapture" if (views or 0) < 1.8 else "retrain", 0.8 if len(fails) > 1 else 0.65
    elif risk:
        verdict, recommend, confidence = "publish with reservations", "retrain", 0.55
    else:
        verdict, recommend, confidence = "publish", "publish", 0.75

    note = (f"{len(good)} things went right and {len(risk)} are still visible in the numbers. "
            f"Recommendation: {RECOMMEND[recommend]}. ")
    if recommend == "publish":
        note += "Nothing here would stop me showing this to a judge."
    elif recommend == "retrain":
        note += "The capture is usable; the run above it is not yet tuned to it."
    else:
        note += "No amount of training fixes a walk that did not see enough of the room."

    return {"verdict": verdict, "confidence": confidence, "recommend": recommend,
            "good": good, "risk": risk, "note": note, "reviewer": "rules v1"}
