# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""What a worker job may spend, in one pure module shared by the worker, the Modal deploy and the tests.

Stage limits stop one stuck stage; the job cap stops the job as a whole. The cap counts GPU time since the job was
claimed and is carried across the CPU camera solve (plypipe.handoff): the GPU is not held there, the CPU function has its
own Modal timeout, and the cap has to fit inside one GPU container. Time spent queued does not count either.

Every default is at least 1.5x the slowest healthy time seen on an L4 with the chapel (1,236 frames of 4K, IMPROVEMENT.md):

  frames     15   9 min decoding on the CPU, 2 min with NVDEC
  sfm_match  45   15 min matching on an L4 (all the GPU does when the solve goes to the CPU box)
  sfm       180   117 min when one box does both: 15 matching + 102 solving (global mapper rejected first, then incremental)
  align      10   not timed on its own; the chapel passed under the old 5 min, doubled
  occupancy  10   same
  label      30   not timed on its own; the chapel passed under the old 15 min, doubled
  train     120   14 min at Quick on an L4, about 58 at High (29 on an L40S); stretches to the projected finish within the cap
  export     40   2 to 18 min
  mesh       70   not measured (opt-in stage)
  cap       240   about 135 min of GPU time for the chapel at High: 25 frames and matching, 100 for the rest

PLY_STAGE_LIMITS="sfm=120,train=90" overrides single limits and PLY_JOB_MAX_MIN the cap. The sum of the stage limits is
longer than the cap on purpose: the cap ends a job whose stages are each slow but not stuck.
"""

from __future__ import annotations

import os

STAGE_LIMITS_MIN = {"frames": 15, "sfm_match": 45, "sfm": 180, "align": 10, "occupancy": 10, "label": 30,
                    "train": 120, "export": 40, "mesh": 70}
JOB_MAX_MIN = 240.0
MAX_INPUT_MB = 12288.0  # three times the chapel's 4.1 GB of clips
EXPORT_RESERVE_MIN = 10  # publishing, after export's own limit: training stops this long before the cap leaves export room
START_SLACK_MIN = 5      # a fresh container needs a moment between setting its deadline and its first claim
CPU_HANDBACK_MIN = 30    # the solve box copies the solved folder back and commits the Volume after the camera solve


def stage_limits(overrides: str | None = None) -> dict[str, float]:
    text = os.environ.get("PLY_STAGE_LIMITS", "") if overrides is None else overrides
    lim = dict(STAGE_LIMITS_MIN)
    for part in filter(None, (p.strip() for p in text.split(","))):
        k, _, v = part.partition("=")
        lim[k.strip()] = float(v)
    return lim


def _positive(name: str, value: str | None, default: float) -> float:
    raw = os.environ.get(name, "") if value is None else value
    out = float(raw) if raw.strip() else default
    if out <= 0:
        raise ValueError(f"{name} must be above 0, got {raw!r}")
    return out


def job_cap_min(value: str | None = None) -> float:
    return _positive("PLY_JOB_MAX_MIN", value, JOB_MAX_MIN)


def max_input_mb(value: str | None = None) -> float:
    return _positive("PLY_MAX_INPUT_MB", value, MAX_INPUT_MB)


def worst_case_min(limits: dict[str, float], cap_min: float) -> float:
    """The longest a job can hold the GPU: every stage at its limit, never past the cap. A worker checks this against
    its machine's deadline before it claims a job (sfm_match is a tighter limit on sfm, not another stage)."""
    stages = [v for k, v in limits.items() if k != "sfm_match"]
    return cap_min if any(v <= 0 for v in stages) else min(cap_min, sum(stages))


def fit_problems(limits: dict[str, float], cap_min: float, container_s: float, margin_s: float, cpu_timeout_s: float) -> list[str]:
    """Why these limits cannot run in a GPU container of `container_s` (whose deadline sits `margin_s` before its end)
    and a CPU solve function of `cpu_timeout_s`. Empty when they fit."""
    room = (container_s - margin_s) / 60 - START_SLACK_MIN
    out = []
    if cap_min > room:
        out.append(f"the {cap_min:g}-minute job cap does not fit the container: {room:g} min usable "
                   f"({container_s / 60:g} min timeout, {margin_s / 60:g} min deadline margin, {START_SLACK_MIN} min to start)")
    out += [f"the {k} limit of {v:g} min is longer than the {cap_min:g}-minute job cap, so it could never fire"
            for k, v in limits.items() if v > cap_min]
    if limits["export"] + EXPORT_RESERVE_MIN >= cap_min:
        out.append(f"export ({limits['export']:g} min) and its {EXPORT_RESERVE_MIN} min reserve leave no time for training in the {cap_min:g}-minute cap")
    if limits.get("sfm_match", limits["sfm"]) > limits["sfm"]:
        out.append("the sfm_match limit is longer than the sfm limit")
    if limits["sfm"] + CPU_HANDBACK_MIN > cpu_timeout_s / 60:
        out.append(f"the sfm limit of {limits['sfm']:g} min plus {CPU_HANDBACK_MIN} min to hand the solve back does not fit "
                   f"the CPU function's {cpu_timeout_s / 60:g}-minute timeout")
    return out
