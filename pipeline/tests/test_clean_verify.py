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


def test_value_filters_use_splat_transforms_units():
    """splat-transform converts opacity/scale to the PLY's raw space itself and rejects an opacity outside (0, 1);
    handing it a logit (-3.476 for 3%) crashed every cleaned export."""
    from plypipe.stages import export as export_stage
    flags = export_stage.value_filter_flags(0.03, 1.2)
    assert flags[:2] == ["-V", "opacity,gt,0.0300"]
    assert "scale_0,lt,1.20" in flags and "scale_2,lt,1.20" in flags
    assert export_stage.value_filter_flags(0.002, None) == ["-V", "opacity,gt,0.0020"]
    for p in (0.0, 1.0, -3.476):
        v = float(export_stage.value_filter_flags(p, None)[1].split(",")[-1])
        assert 0 < v < 1


def _choose(good: set[tuple[str, str]], unscored: bool = False):
    """choose_cleanup over three filters and three steps, where only the (filter, cleanup) pairs in `good` pass."""
    from plypipe.stages import export as export_stage
    filters = [{"why": n} for n in ("strict", "looser", "loosest")]
    steps, calls = [(True, True), (False, True), (False, False)], []

    def measure(f, step):
        calls.append((f["why"], export_stage.cleanup_name(step)))
        return None if unscored else {"ok": calls[-1] in good}

    f, step, score = export_stage.choose_cleanup(filters, steps, measure, lambda s: bool(s) and s["ok"])
    return (f["why"], export_stage.cleanup_name(step)), score, calls


def test_a_cleanup_the_viewer_does_not_pay_for_is_taken_at_the_first_try():
    assert _choose({("strict", "crop+floaters")}) == (("strict", "crop+floaters"), {"ok": True}, [("strict", "crop+floaters")])
    picked, _, calls = _choose({("strict", "values only"), ("looser", "crop+floaters")})
    assert picked == ("strict", "values only") and calls == [("strict", "crop+floaters"), ("strict", "floaters"), ("strict", "values only")]


def test_a_value_filter_that_costs_the_picture_even_alone_is_loosened():
    """Dr Johnson with a weak opacity regulariser: the strictest filter kept 88 % of the splats, so the count let it
    through, and it cost 1.8 dB and 7 % of the pixels under every cleanup. It used to ship like that."""
    picked, score, calls = _choose({("loosest", "values only"), ("loosest", "floaters")})
    assert picked == ("loosest", "floaters") and score == {"ok": True}
    assert calls == [("strict", "crop+floaters"), ("strict", "floaters"), ("strict", "values only"),
                     ("looser", "values only"),                              # failed alone: no heavier step can save it
                     ("loosest", "values only"), ("loosest", "crop+floaters"), ("loosest", "floaters")]
    picked, score, calls = _choose({("looser", "values only")})
    assert picked == ("looser", "values only") and score == {"ok": True}       # the heavier steps were measured and left out
    assert calls[-3:] == [("looser", "values only"), ("looser", "crop+floaters"), ("looser", "floaters")]


def test_the_last_try_stands_when_nothing_passes_and_nothing_is_loosened_without_a_score():
    picked, score, calls = _choose(set())
    assert picked == ("loosest", "values only") and score == {"ok": False} and len(calls) == 5
    picked, score, calls = _choose(set(), unscored=True)
    assert picked == ("strict", "values only") and score is None and len(calls) == 3


def test_floater_filter_keeps_a_plain_wall_of_big_splats(tmp_path):
    """MCMC covers a plain wall with few large splats; one per cell, straddling cell edges, is still a wall."""
    zs, ys = np.meshgrid(np.arange(0, 3.0, 0.15), np.arange(0, 2.7, 0.15))
    wall = np.c_[np.full(zs.size, 2.06), ys.ravel(), zs.ravel()]       # x on a cell edge, 15 cm spacing
    lone = np.array([[0.5, 1.2, 1.5], [4.0, 0.4, -2.0], [-3.0, 2.0, 5.0]])
    rows = [[*p, 2.0, 0.5, 0.5, 0.5] for p in np.vstack([wall, lone])]
    src, dst = tmp_path / "in.ply", tmp_path / "out.ply"
    write_ply(src, rows)
    stats = clean.floater_filter(src, dst)
    assert stats["kept"] == len(wall)
    assert stats["removedIsolated"] == 3 and stats["removedFaint"] == 0


def test_neighbour_counts_see_the_adjacent_cells():
    xyz = np.array([[0.11, 0, 0], [0.13, 0, 0], [0.35, 0, 0], [5.0, 5.0, 5.0]])
    assert clean.neighbour_counts(xyz, 0.12).tolist() == [2, 3, 2, 1]


SLOW_WALK_PLAN = {"input": {"keptFps": 1.52, "fpsUsed": 1.52, "motion": {"kind": "slow"}}}


def test_a_slow_walk_at_its_planned_rate_passes():
    """The plan lowers the rate for a slow walk; the gate must not fail the run for following it."""
    report = json.loads(json.dumps(GOOD))
    report["frames"] = {"kept": 580, "clips": [{"durationSec": 400.0}]}      # 1.45/s
    assert not verify.check(report)["ok"]                                    # the absolute floor would fail it
    result = verify.check(report, SLOW_WALK_PLAN)
    assert result["ok"], result["issues"]
    assert result["metrics"]["viewsFloor"] == 1.22


def test_a_capped_run_still_fails_against_its_plan():
    report = json.loads(json.dumps(GOOD))
    report["frames"] = {"kept": 542, "clips": [{"durationSec": 484.0}]}      # the classroom: 1.12/s
    result = verify.check(report, {"input": {"keptFps": 2.5, "fpsUsed": 2.5}})
    assert not result["ok"] and "views" in {i["code"] for i in result["issues"]}
    assert "planned 2.50/s" in next(i["message"] for i in result["issues"] if i["code"] == "views")


def test_the_floor_never_drops_below_the_absolute_minimum():
    fail, warn, planned = verify.view_floors({"input": {"fpsUsed": 0.5}})
    assert fail == verify.ABS_MIN_VIEWS_PER_SECOND and warn >= fail and planned == 0.5


def test_held_out_scores_are_reported_and_a_poor_twin_is_flagged():
    report = json.loads(json.dumps(GOOD))
    report["train"]["eval"] = {"psnr": 27.2, "ssim": 0.87, "lpips": 0.25, "ccPsnr": 27.9, "views": 9}
    result = verify.check(report)
    assert result["ok"] and result["metrics"]["heldOut"]["ccPsnr"] == 27.9
    report["train"]["eval"] = {"psnr": 16.0, "ccPsnr": 17.5, "views": 9}
    result = verify.check(report)
    assert result["ok"] and "held-out" in {i["code"] for i in result["issues"]}


def test_viewer_brightness_and_holes_are_flagged():
    report = json.loads(json.dumps(GOOD))
    report["export"]["viewerEval"] = {"psnr": 27.0, "brightness": 1.21, "holes": 0.06, "views": 9}
    codes = {i["code"] for i in verify.check(report)["issues"]}
    assert {"brightness", "holes"} <= codes and verify.check(report)["ok"]
    report["export"]["viewerEval"] = {"psnr": 28.0, "brightness": 1.02, "holes": 0.01, "views": 9}
    assert not {"brightness", "holes"} & {i["code"] for i in verify.check(report)["issues"]}


def test_held_out_split_matches_gsplat():
    from plypipe import viewer_eval
    assert viewer_eval.held_out([f"{i:06d}.jpg" for i in range(70)], 30) == [0, 30, 60]
    assert viewer_eval.held_out(["a"], 0) == []


def test_neighbour_counts_survive_far_outliers():
    """A trained model can carry a few splats kilometres out; the grid must not grow with them."""
    rng = np.random.default_rng(2)
    wall = np.c_[rng.uniform(0, 4, 5000), rng.uniform(0, 2.5, 5000), np.zeros(5000)]
    far = np.array([[1e7, 0, 0], [-3e6, 5e6, 1e7]])
    counts = clean.neighbour_counts(np.vstack([wall, far]), 0.12)
    assert (counts[:5000] >= 3).all() and counts[-2:].tolist() == [1, 1]


def test_big_lone_splats_are_surfaces_not_floaters(tmp_path):
    """A far wall seen through a door is a few big splats, each alone in its 12 cm cell."""
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
              "property float x\nproperty float y\nproperty float z\nproperty float opacity\n"
              "property float scale_0\nproperty float scale_1\nproperty float scale_2\nend_header\n")
    rows = [[0, 1, 0, 2.0, np.log(0.4), np.log(0.4), np.log(0.01)],       # 40 cm splat, alone: a far wall
            [5, 1, 0, 2.0, np.log(0.02), np.log(0.02), np.log(0.02)],     # 2 cm splat, alone: a floater
            [-5, 1, 0, 2.0, np.log(0.2), np.log(0.05), np.log(0.05)]]     # 20 cm along one axis: kept
    with open(tmp_path / "in.ply", "wb") as f:
        f.write(header.encode()); np.asarray(rows, np.float32).tofile(f)
    stats = clean.floater_filter(tmp_path / "in.ply", tmp_path / "out.ply")
    assert stats["kept"] == 2 and stats["removedIsolated"] == 1


def test_the_run_parser_accepts_no_clean():
    """The export has always read `no_clean`; the run parser never offered it, so cleanup could not be turned off."""
    from plypipe.cli import build_parser
    args = build_parser().parse_args(["run", "in", "--out", "work/x", "--no-clean"])
    assert args.no_clean is True, "plypipe run --no-clean must reach the export stage"
    assert build_parser().parse_args(["run", "in", "--out", "work/x"]).no_clean is False
