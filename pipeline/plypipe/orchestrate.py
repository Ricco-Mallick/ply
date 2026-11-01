# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""orchestrate: judge a finished run and decide what the next attempt should change.

The gates in `verify` say whether a twin is worth publishing. This module answers the next question:
if it is not, what should change? It returns the flags for another attempt, each with a reason, so a
job can retry itself once instead of waiting for someone to notice and re-queue it by hand.

Nothing here invents quality. It only changes inputs that provably caused the failure: the kept frame
rate (views), the gaussian cap (pruning), the matcher and mapper (registration), antialiasing.
"""

from __future__ import annotations

MAX_ATTEMPTS = 2

_FLAGS_WITH_VALUE = {"--fps", "--max-frames", "--cap-max", "--iterations", "--features", "--mapper", "--sh-degree"}


def _drop(args: list[str], *names: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] in names:
            i += 2 if args[i] in _FLAGS_WITH_VALUE else 1
            continue
        out.append(args[i])
        i += 1
    return out


def _set(args: list[str], flag: str, value: str) -> list[str]:
    return [*_drop(args, flag), flag, value]


def refine(report: dict, args: list[str], plan: dict | None = None) -> tuple[list[str], list[str]]:
    """Next attempt's flags and the reasons, from a report that failed the gate."""
    out = list(args)
    notes: list[str] = []
    frames = report.get("frames") or {}
    clips = frames.get("clips") or []
    seconds = sum(float(c.get("durationSec") or 0) for c in clips) if clips else float(frames.get("durationSec") or 0)
    kept = int(frames.get("kept") or 0)
    views = kept / seconds if seconds else None
    planned = ((plan or {}).get("input") or {}).get("keptFps")

    # 1. not enough views: the frame cap is almost always the cause, and it is explicit in the args
    if views is not None and views < 1.8:
        target = float(planned or 2.5)
        out = _drop(out, "--max-frames")
        out = _set(out, "--fps", f"{min(3.2, max(target, views * 2)):.2f}")
        notes.append(f"views were {views:.2f}/s: dropped any frame cap, kept rate set to {min(3.2, max(target, views * 2)):.2f}/s")

    # 2. export deleted most of what training fitted: cap above what the room can carry
    export = report.get("export") or {}
    kept_fraction, gaussians_out = export.get("keptFraction"), export.get("gaussiansOut")
    if isinstance(kept_fraction, (int, float)) and kept_fraction < 0.85 and gaussians_out:
        cap = int(min(4_000_000, max(400_000, gaussians_out * 1.25)))
        out = _set(out, "--cap-max", str(cap))
        notes.append(f"export kept {kept_fraction:.0%}: training cap set to {cap:,} so the budget matches what ships")

    # 3. antialiasing was off for some reason (old preset, explicit flag)
    if (report.get("train") or {}).get("antialiased") is False:
        out = _drop(out, "--no-antialias")
        notes.append("antialiasing was off: turned back on")

    # 4. weak registration: the learned features or the global mapper on an unusual capture
    ratio = (report.get("sfm") or {}).get("registeredRatio")
    if isinstance(ratio, (int, float)) and ratio < 0.9:
        out = _set(out, "--features", "sift")
        out = _set(out, "--mapper", "incremental")
        notes.append(f"only {ratio:.0%} registered: SIFT features and the incremental mapper")

    return out, notes


def decide(input_plan: dict | None, attempt: int) -> dict:
    """What the run was told before it started, for the record the operator reads."""
    plan = input_plan or {}
    return {"attempt": attempt,
            "keptFps": plan.get("keptFps"), "longSide": plan.get("longSide"),
            "why": plan.get("why"), "motion": (plan.get("motion") or {}).get("kind")}
