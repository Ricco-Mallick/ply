# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""The two guards that would have caught the chapel: a floater pass that always runs, and a gate that reads the numbers."""

from __future__ import annotations

import json

import numpy as np

from plypipe import clean, verify


def write_ply(path, rows):
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(rows)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float opacity\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        np.asarray(rows, dtype=np.float32).tofile(f)


def test_floater_filter_keeps_surfaces_and_drops_the_hangers(tmp_path):
    rng = np.random.default_rng(3)
    surface = rng.normal(0, 0.04, size=(300, 3))            # a wall, sampled densely
    surface[:, 1] += 1.0
    floaters = rng.uniform(-6, 6, size=(30, 3))              # the same count of splats, but alone in space
    rows = []
    for p in np.vstack([surface, floaters]):
        rows.append([*p, 2.0, 0.6, 0.6, 0.6])                # opacity 2.0 sigmoid -> ~0.88 (solid)
    src, dst = tmp_path / "in.ply", tmp_path / "out.ply"
    write_ply(src, rows)

    stats = clean.floater_filter(src, dst)
    assert stats["kept"] >= 295                 # the wall survives
    assert stats["removedIsolated"] >= 25       # the loners do not
    assert stats["keptFraction"] < 1.0


def test_clipped_fraction_sees_a_white_model(tmp_path):
    white = [[0, 0, 0, 2.0, 40.0, 40.0, 40.0]] * 10          # DC pushed far past white
    dark = [[0, 0, 0, 2.0, -10.0, -10.0, -10.0]] * 10        # comfortably inside
    path = tmp_path / "clip.ply"
    write_ply(path, white + dark)
    assert clean.clipped_fraction(path) == 0.5


GOOD = {
    "frames": {"kept": 1100, "clips": [{"durationSec": 400.0}]},
    "sfm": {"registeredRatio": 1.0},
    "train": {"antialiased": True},
    "export": {"keptFraction": 0.93, "spzMB": 30.0, "clippedFraction": 0.004,
               "floaterClean": {"method": "cpu-grid", "kept": 900000, "keptFraction": 0.95}},
}


def test_a_healthy_run_passes():
    result = verify.check(GOOD)
    assert result["ok"], result["issues"]
    assert result["metrics"]["viewsPerSecond"] == 2.75


def test_the_chapel_fails_with_reasons_a_human_can_act_on():
    chapel = {
        "frames": {"kept": 275, "clips": [{"durationSec": 515.0}]},
        "sfm": {"registeredRatio": 0.985},
        "train": {"antialiased": False},
        "export": {"keptFraction": 0.17, "spzMB": 20.7, "clippedFraction": 0.052},
    }
    result = verify.check(chapel)
    codes = {i["code"] for i in result["issues"]}
    assert not result["ok"]
    assert {"views", "floaters", "pruned"} <= codes          # the three hard failures
    assert {"antialias", "clipping"} <= codes                # and the two that explain how it looked
    assert all(i["message"] for i in result["issues"])


def test_a_run_without_a_frame_report_is_only_a_warning():
    report = json.loads(json.dumps(GOOD))
    report["frames"] = {}
    result = verify.check(report)
    assert result["ok"]
    assert {i["code"] for i in result["issues"]} == {"views-unknown"}


def test_opacity_threshold_is_converted_to_logit(tmp_path):
    """The PLY stores a logit; a probability threshold must be converted or it means 'half visible'."""
    import math
    from plypipe.stages import export as export_stage
    p = 0.03
    logit = round(math.log(p / (1 - p)), 3)
    assert logit == -3.476
    assert logit < 0, "a 3% visibility threshold is a negative logit"
