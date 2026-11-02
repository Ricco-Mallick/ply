# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""verify: the gate a finished run has to pass before anyone looks at the twin.

Every failure that reached a viewer so far was visible in the run's own numbers and nothing read them:
the chapel trained from 0.53 frames per second, the floater pass never ran, and a third of the splats
were deleted at export. This module reads report.json, says what is wrong in words that point at the
cause, and lets the worker fail the job instead of publishing a twin nobody would accept.

Run it by hand: python -m plypipe.verify out/report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# what a capture has to carry for the appearance fit to have enough constraints
MIN_VIEWS_PER_SECOND = 1.8
WARN_VIEWS_PER_SECOND = 2.1
# with a plan, the floor follows it: a slow walk is planned at ~1.5/s and has as many views per metre as a
# normal walk at 2.2/s, so failing it against 1.8 would fail a run for doing what it was told
PLANNED_FAIL_SHARE = 0.8
PLANNED_WARN_SHARE = 0.9
ABS_MIN_VIEWS_PER_SECOND = 1.2
MIN_REGISTERED_RATIO = 0.9
MIN_SPZ_MB = 3.0
MIN_KEPT_FRACTION = 0.5
WARN_KEPT_FRACTION = 0.85
MAX_CLIPPED_FRACTION = 0.02
# held-out views (train --holdout): below this the twin does not reproduce photos it was not trained on
WARN_HELD_OUT_PSNR = 20.0
# the exported twin rendered without any appearance model, at the held-out photos (export.viewerEval)
BRIGHTNESS_TOLERANCE = 0.08
WARN_HOLES = 0.03


def _views_per_second(frames: dict) -> float | None:
    clips = frames.get("clips") or []
    seconds = sum(float(c.get("durationSec") or 0) for c in clips) if clips else float(frames.get("durationSec") or 0)
    kept = int(frames.get("kept") or 0)
    return round(kept / seconds, 3) if seconds else None


def view_floors(plan: dict | None) -> tuple[float, float, float | None]:
    """(fail, warn, planned rate): absolute without a plan, relative to the planned rate with one."""
    inp = (plan or {}).get("input") or {}
    planned = inp.get("fpsUsed") or inp.get("keptFps")
    if not isinstance(planned, (int, float)) or planned <= 0:
        return MIN_VIEWS_PER_SECOND, WARN_VIEWS_PER_SECOND, None
    fail = max(ABS_MIN_VIEWS_PER_SECOND, min(MIN_VIEWS_PER_SECOND, PLANNED_FAIL_SHARE * planned))
    warn = max(fail, min(WARN_VIEWS_PER_SECOND, PLANNED_WARN_SHARE * planned))
    return round(fail, 2), round(warn, 2), float(planned)


def check(report: dict, plan: dict | None = None) -> dict:
    """Return {"ok", "issues", "metrics"}; issues carry level fail|warn, a code, and a sentence.

    `plan` is <out>/plan.json; with it, the views floor follows the rate the plan chose for this walk."""
    issues: list[dict] = []

    def add(level: str, code: str, message: str) -> None:
        issues.append({"level": level, "code": code, "message": message})

    frames = report.get("frames") or {}
    views = _views_per_second(frames)
    fail_vps, warn_vps, planned = view_floors(plan)
    of_plan = f" (planned {planned:.2f}/s)" if planned else ""
    if views is None:
        add("warn", "views-unknown", "the frame report has no duration: views per second could not be checked")
    elif views < fail_vps:
        add("fail", "views", f"{views} frames per second of footage{of_plan}, under {fail_vps}. "
            "The appearance fit is inventing surface between views. Re-run the clips with the planned frame rate "
            "before training, and check nothing passed --max-frames.")
    elif views < warn_vps:
        add("warn", "views-thin", f"{views} frames per second{of_plan}, thin for a room: expect softness on large surfaces.")

    sfm = report.get("sfm") or {}
    ratio = sfm.get("registeredRatio")
    if isinstance(ratio, (int, float)) and ratio < MIN_REGISTERED_RATIO:
        add("warn", "registered", f"only {ratio:.0%} of the frames registered; the walk probably has a gap or a blur stretch.")

    train = report.get("train") or {}
    if train and train.get("antialiased") is False:
        add("warn", "antialias", "trained without antialiasing: distant gaussians will sparkle and read harsh.")

    if train.get("partial"):
        add("warn", "partial-train", f"training was stopped at step {train.get('fromStep')} of {train.get('plannedSteps')} "
            "(time limit) and this twin is that snapshot: usable, but softer than a finished run.")

    held = train.get("eval") or {}
    psnr = held.get("ccPsnr") or held.get("psnr")
    if isinstance(psnr, (int, float)) and psnr < WARN_HELD_OUT_PSNR:
        add("warn", "held-out", f"held-out views score {psnr:.1f} dB PSNR (colour-corrected): the twin does not "
            "reproduce the photos it was not trained on. Expect blur or doubled surfaces; check the camera solve first.")

    export = report.get("export") or {}
    clean = export.get("floaterClean")
    if export and not clean:
        add("fail", "floaters", "the floater pass did not run, so the twin ships with whatever blobs the fit produced.")
    elif clean and not clean.get("kept"):
        add("fail", "floaters-empty", "the floater pass removed everything: the model is not usable.")

    kept = export.get("keptFraction")
    if isinstance(kept, (int, float)):
        if kept < MIN_KEPT_FRACTION:
            add("fail", "pruned", f"export kept only {kept:.0%} of the trained splats: the training budget was spent on "
                "splats that were then deleted. Train to the size you ship (the plan does this).")
        elif kept < WARN_KEPT_FRACTION:
            add("warn", "pruned-some", f"export kept {kept:.0%} of the trained splats; the cap is above what the room holds.")

    viewer = export.get("viewerEval") or {}
    gain = viewer.get("brightness")
    if isinstance(gain, (int, float)) and abs(gain - 1) > BRIGHTNESS_TOLERANCE:
        add("warn", "brightness", f"the twin renders ×{gain:.2f} the photos' brightness in the viewer. The appearance model "
            "(PPISP) corrected exposure during training but is not part of the file; train with --bilateral-grid.")
    holes = viewer.get("holes")
    if isinstance(holes, (int, float)) and holes > WARN_HOLES:
        add("warn", "holes", f"{holes:.1%} of the held-out views is see-through (splat coverage under 90 %): expect dark gaps. "
            "More gaussians (--cap-max) or --random-bkgd close them.")

    spz = export.get("spzMB")
    if isinstance(spz, (int, float)) and spz < MIN_SPZ_MB:
        add("fail", "tiny", f"{spz} MB twin: that is not a room, something collapsed during export.")

    clipped = export.get("clippedFraction")
    if isinstance(clipped, (int, float)) and clipped > MAX_CLIPPED_FRACTION:
        add("warn", "clipping", f"{clipped:.1%} of splats sit at the top of the colour range: highlights will glare. "
            "Use the viewer's exposure control, and fix the capture exposure next time.")

    return {
        "ok": not any(i["level"] == "fail" for i in issues),
        "issues": issues,
        "metrics": {"viewsPerSecond": views, "viewsFloor": fail_vps, "plannedViewsPerSecond": planned,
                    "registeredRatio": ratio, "antialiased": train.get("antialiased"),
                    "keptFraction": kept, "clippedFraction": clipped, "spzMB": spz,
                    "heldOut": {k: held.get(k) for k in ("psnr", "ssim", "lpips", "ccPsnr", "views")} if held else None,
                    "viewer": viewer or None},
    }


def main(argv: list[str] | None = None) -> int:
    args = (argv or sys.argv)[1:]
    if not args:
        print("usage: python -m plypipe.verify <report.json>", file=sys.stderr)
        return 2
    report = json.loads(Path(args[0]).read_text())
    from plypipe import plan  # plan.json sits next to report.json in a job folder
    result = check(report, plan.read(Path(args[0]).parent))
    print(json.dumps(result, indent=1))
    for i in result["issues"]:
        print(f"[{i['level']}] {i['code']}: {i['message']}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
