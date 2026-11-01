# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Refining a failed run: only the settings that caused the failure change."""

from plypipe import orchestrate


def test_a_frame_cap_is_dropped_and_the_rate_raised():
    report = {"frames": {"kept": 275, "clips": [{"durationSec": 515.0}]}}
    args = ["--preset", "high", "--max-frames", "760"]
    out, notes = orchestrate.refine(report, args, {"input": {"keptFps": 2.5}})
    assert "--max-frames" not in out
    assert out[out.index("--fps") + 1] == "2.50"
    assert any("views were" in n for n in notes)


def test_a_pruned_export_lowers_the_training_cap():
    report = {"frames": {"kept": 1000, "clips": [{"durationSec": 400.0}]},
              "export": {"keptFraction": 0.17, "gaussiansOut": 679517}}
    out, notes = orchestrate.refine(report, ["--preset", "high"])
    cap = int(out[out.index("--cap-max") + 1])
    assert 700_000 < cap < 1_000_000
    assert any("export kept" in n for n in notes)


def test_antialiasing_comes_back_and_weak_registration_switches_matcher():
    report = {"frames": {"kept": 1000, "clips": [{"durationSec": 400.0}]},
              "train": {"antialiased": False}, "sfm": {"registeredRatio": 0.72}}
    out, notes = orchestrate.refine(report, ["--preset", "high", "--no-antialias"])
    assert "--no-antialias" not in out
    assert out[out.index("--features") + 1] == "sift"
    assert out[out.index("--mapper") + 1] == "incremental"
    assert len(notes) == 2


def test_a_healthy_run_changes_nothing():
    report = {"frames": {"kept": 1100, "clips": [{"durationSec": 400.0}]}, "train": {"antialiased": True},
              "sfm": {"registeredRatio": 1.0}, "export": {"keptFraction": 0.95, "gaussiansOut": 900000}}
    out, notes = orchestrate.refine(report, ["--preset", "high"])
    assert out == ["--preset", "high"] and notes == []
