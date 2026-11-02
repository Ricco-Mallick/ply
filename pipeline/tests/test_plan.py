# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""The plan is measured, not assumed: these tests pin the behaviour that matters."""

from __future__ import annotations

from plypipe import plan

ROOM = {"min": [-5.3, 0.0, -3.4], "max": [5.2, 4.4, 3.3]}


def capture(flow: float, turn: float) -> dict:
    return {"clips": [{"quality": [[i, "good", 600, 130, flow, turn] for i in range(20)]}]}


def test_sensor_motion_scales_the_kept_rate():
    slow = plan.input_plan(120.0, 1, 3840, capture(6, 5))
    fast = plan.input_plan(120.0, 1, 3840, capture(48, 70))
    assert plan.motion(capture(48, 70))["kind"] == "fast"
    assert plan.motion(capture(6, 5))["kind"] in ("slow", "normal")
    assert fast["keptFps"] > slow["keptFps"]
    assert slow["keptFps"] >= plan._FPS_MIN and fast["keptFps"] <= plan._FPS_MAX


def test_without_capture_json_it_is_a_normal_walk():
    p = plan.input_plan(120.0, 1, 3840, None)
    assert p["motion"]["kind"] == "normal"
    assert 2.0 <= p["keptFps"] <= 2.4


def test_long_footage_is_capped_by_the_frame_budget():
    p = plan.input_plan(1800.0, 4, 3840, None)
    assert p["expectedFrames"] <= p["frameBudget"]


def test_resolution_follows_the_source_pixels():
    assert plan.input_plan(60, 1, 3840, None)["longSide"] == 1728
    assert plan.input_plan(60, 1, 1920, None)["longSide"] == plan._SIDE_MIN  # never upscale past the source
    assert plan.input_plan(60, 1, 8000, None)["longSide"] == plan._SIDE_MAX


def test_gaussian_budget_follows_surface_area():
    big = plan.model_plan(2000, {"min": [-12, 0, -9], "max": [12, 4.2, 9]}, "standard", 3.4)
    small = plan.model_plan(2000, {"min": [-2, 0, -2], "max": [2, 3, 2]}, "standard", 3.0)
    assert big["capMax"] > small["capMax"]
    assert big["iterations"] == small["iterations"]  # steps follow the views, area follows the cap


def test_a_short_capture_cannot_claim_the_area_budget():
    few = plan.model_plan(200, {"min": [-12, 0, -9], "max": [12, 4.2, 9]}, "standard", 3.4)
    assert few["capMax"] <= 200 * plan._DEFAULT_VIEW_MP * plan._GAUSSIANS_PER_VIEW_MP


def test_preset_scales_but_never_past_the_ceiling():
    light = plan.model_plan(900, ROOM, "quick", 3.1)
    best = plan.model_plan(900, ROOM, "high", 3.1)
    assert light["capMax"] < best["capMax"]
    assert best["iterations"] <= plan._PRESET_CEIL["high"]
    assert best["capMax"] <= plan._PRESET_CAP_MAX["high"]


def test_steps_follow_the_view_count():
    few = plan.model_plan(300, ROOM, "standard", 3.1)
    many = plan.model_plan(900, ROOM, "standard", 3.1)
    assert few["iterations"] < many["iterations"]
    assert few["iterations"] >= plan._PRESET_FLOOR["standard"]


def test_explicit_overrides_win():
    p = plan.model_plan(900, ROOM, "standard", 3.1, iterations=12345, cap_max=777_000)
    assert p["iterations"] == 12345 and p["capMax"] == 777_000
    assert p["planIterations"] != 12345  # the plan still records what it would have chosen


def test_presets_differ_on_a_small_capture():
    """263 photos (Dr Johnson): floors and the view cap used to make all three presets the same run."""
    house = {"min": [-6.0, 0.0, -4.0], "max": [6.0, 3.0, 5.0]}
    q, s, h = (plan.model_plan(263, house, p, 3.0, megapixels=1.17) for p in ("quick", "standard", "high"))
    assert q["iterations"] < s["iterations"] < h["iterations"]
    assert q["capMax"] < s["capMax"] < h["capMax"]
    assert (q["iterations"], s["iterations"], h["iterations"]) == (7_000, 30_000, 40_000)


def test_the_view_cap_counts_pixels():
    house = {"min": [-6.0, 0.0, -4.0], "max": [6.0, 3.0, 5.0]}
    video = plan.model_plan(263, house, "standard", 3.0)
    photos = plan.model_plan(263, house, "standard", 3.0, megapixels=1.17)
    assert photos["capMax"] > video["capMax"]
    assert video["capMax"] == int(263 * 0.92 * 2400)  # unchanged for a 1280x720 video frame: 2200/view as before
