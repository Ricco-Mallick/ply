# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Which mapper the camera solve starts with (ordered video of 400 frames or more skips the global mapper), the two
opt-in fast solves (hierarchical, keyframes): keyframe choice, the fallback, and every other frame registered."""

from __future__ import annotations

import json
import types

import pytest

from plypipe import cli
from plypipe.log import Log
from plypipe.stages import sfm
from plypipe.util import Ctx


@pytest.mark.parametrize("source, n, expected", [
    ("video", 399, "global"),
    ("video", 400, "incremental"),
    ("videos", 1236, "incremental"),
    ("ar", 400, "incremental"),
    ("images", 5000, "global"),    # an unordered photo set has no sequence to anchor the incremental mapper
    (None, 5000, "global"),        # a job without a frames report
])
def test_mapper_by_source_and_frame_count(source, n, expected):
    assert sfm.choose_mapper(None, source, n)[0] == expected


def test_an_explicit_mapper_always_wins():
    assert sfm.choose_mapper("global", "videos", 1236) == ("global", "requested")
    assert sfm.choose_mapper("incremental", "images", 12) == ("incremental", "requested")


def test_the_threshold_is_an_option():
    assert sfm.choose_mapper(None, "video", 150, incremental_min=100)[0] == "incremental"
    assert sfm.choose_mapper(None, "video", 150, incremental_min=151)[0] == "global"


def test_the_reason_says_why():
    assert "1236 frames" in sfm.choose_mapper(None, "videos", 1236)[1]
    assert sfm.choose_mapper(None, "videos", 20)[1] == "default"


def test_the_cli_does_not_make_the_default_look_explicit():
    opts = vars(cli.build_parser().parse_args(["run", "in", "--out", "out"]))
    assert opts["mapper"] is None and opts["incremental_min"] == 400
    assert vars(cli.build_parser().parse_args(["run", "in", "--out", "out", "--mapper", "global"]))["mapper"] == "global"


def _solve(tmp_path, source, n, **opts):
    """Run _map against a pycolmap stub whose mappers return nothing: it stops right after the choice."""
    calls: list = []

    def mapping(name):
        def run(db, images, out, options):
            calls.append((name, options))
            return {}
        return run

    pycolmap = types.SimpleNamespace(
        incremental_mapping=mapping("incremental"), global_mapping=mapping("global"),
        IncrementalPipelineOptions=lambda: types.SimpleNamespace(mapper=types.SimpleNamespace(), max_runtime_seconds=-1),
        GlobalPipelineOptions=lambda: "global-options")
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts=opts)
    with pytest.raises(RuntimeError) as e:
        sfm._map(ctx, pycolmap, tmp_path / "colmap", tmp_path / "db", tmp_path / "images", n, source, "sift", "sift",
                 "sequential", True, None)
    lines = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    return calls, str(e.value), [line["msg"] for line in lines if "mapping (" in line.get("msg", "")]


def test_a_long_video_solve_starts_incremental_and_logs_why(tmp_path):
    calls, error, logged = _solve(tmp_path, "videos", 1236)
    assert [c[0] for c in calls] == ["incremental"] and error.startswith("incremental mapping produced no model")
    assert logged == ["incremental mapping (videos of 1236 frames, at least 400)"]
    assert calls[0][1].max_runtime_seconds == -1      # no other model to fall back on: no time cap by default


def test_a_short_video_still_starts_global(tmp_path):
    calls, _, logged = _solve(tmp_path, "video", 399)
    assert [c[0] for c in calls] == ["global"] and calls[0][1] == "global-options"
    assert logged == ["global mapping (default)"]


def test_an_explicit_global_on_a_long_video_is_respected(tmp_path):
    calls, _, logged = _solve(tmp_path, "video", 1236, mapper="global")
    assert [c[0] for c in calls] == ["global"] and logged == ["global mapping (requested)"]


def test_a_lower_threshold_and_an_explicit_time_cap_apply_to_the_direct_route(tmp_path):
    calls, _, _ = _solve(tmp_path, "video", 150, incremental_min=100, incremental_max_s=3600)
    assert [c[0] for c in calls] == ["incremental"] and calls[0][1].max_runtime_seconds == 3600


@pytest.mark.parametrize("source, n", [("videos", 1236), ("video", 399), ("images", 5000), (None, 12)])
def test_the_fast_mappers_are_never_chosen_unasked_and_always_honoured_when_asked(source, n):
    assert sfm.choose_mapper(None, source, n)[0] in ("global", "incremental")
    for name in sfm.FAST_MAPPERS:
        assert sfm.choose_mapper(name, source, n) == (name, "requested")


def test_the_cli_takes_the_two_new_mappers_and_their_options():
    parse = lambda *a: vars(cli.build_parser().parse_args(["run", "in", "--out", "out", *a]))
    assert parse()["hier_leaf_max"] == 500 and parse()["keyframe_step"] == 4 and parse()["keyframe_max_gap"] == 8
    got = parse("--mapper", "keyframes", "--keyframe-step", "2", "--hier-leaf-max", "300")
    assert got["mapper"] == "keyframes" and got["keyframe_step"] == 2 and got["hier_leaf_max"] == 300
    assert parse("--mapper", "hierarchical")["mapper"] == "hierarchical"


def _names(n, start=0):
    return [f"{i:04d}.jpg" for i in range(start, start + n)]


def test_keyframes_are_every_nth_frame_with_the_first_and_last():
    assert sfm.select_keyframes(_names(11), step=4, max_gap=8) == _names(11)[0:9:4] + ["0010.jpg"]
    assert sfm.select_keyframes(_names(9), step=4) == ["0000.jpg", "0004.jpg", "0008.jpg"]
    assert sfm.select_keyframes(_names(1)) == ["0000.jpg"] and sfm.select_keyframes([]) == []


def test_keyframes_keep_every_clip_boundary():
    names = _names(10) + _names(7, 10) + _names(1, 17)
    clips = {n: ("a" if i < 10 else "b" if i < 17 else "c") for i, n in enumerate(sorted(names))}
    keys = sfm.select_keyframes(names, clips, step=4)
    for first, last in (("0000.jpg", "0009.jpg"), ("0010.jpg", "0016.jpg"), ("0017.jpg", "0017.jpg")):
        assert first in keys and last in keys
    assert keys == sorted(keys) and len(set(keys)) == len(keys)


def test_a_gap_between_keyframes_is_capped():
    keys = sfm.select_keyframes(_names(100), step=40, max_gap=10)
    gaps = [int(b[:4]) - int(a[:4]) for a, b in zip(keys, keys[1:])]
    assert max(gaps) <= 10 and keys[0] == "0000.jpg" and keys[-1] == "0099.jpg"


def test_keyframes_without_a_direct_match_are_split_until_they_have_one():
    # the database pairs frames at offsets 1, 2, 4, 8 ...: three apart is not among them
    linked = lambda a, b: int(b[:4]) - int(a[:4]) in (1, 2, 4, 8, 16, 32)
    keys = sfm.select_keyframes(_names(25), step=3, max_gap=8, linked=linked)
    assert all(linked(a, b) for a, b in zip(keys, keys[1:])) and keys[0] == "0000.jpg" and keys[-1] == "0024.jpg"
    assert len(sfm.select_keyframes(_names(25), step=4, max_gap=8, linked=linked)) == 7   # 0, 4 ... 24: nothing to split


class _Rec:
    def __init__(self, registered):
        self.registered = registered
        self.images = {}
        self.cameras = {1: types.SimpleNamespace(params=[500.0])}

    def num_reg_images(self): return self.registered
    def write(self, path): pass
    def write_text(self, path): pass
    def num_points3D(self): return 1000
    def compute_mean_reprojection_error(self): return 0.5


def _fallback_run(tmp_path, mapper, fast_registers, incremental_registers=100, **opts):
    """_map against a pycolmap stub: the fast mapper registers `fast_registers` of 100 frames, the incremental one the rest."""
    (tmp_path / "images").mkdir()
    for n in _names(100):
        (tmp_path / "images" / n).write_bytes(b"")
    calls = []

    def incremental(db, images, out, options, input_path=""):
        first_of_keyframes = mapper == "keyframes" and not calls   # the stub's keyframe solve registers `fast_registers` frames
        calls.append(("incremental", bool(input_path)))
        return {0: _Rec(fast_registers if first_of_keyframes else 100 if input_path else incremental_registers)}

    def hierarchical(db, images, out, options):
        calls.append(("hierarchical", False))
        return {0: _Rec(fast_registers)}

    pairs = [i * 2147483647 + i + k for i in range(1, 100) for k in (1, 2, 4, 8) if i + k <= 100]
    db = types.SimpleNamespace(read_all_images=lambda: [types.SimpleNamespace(name=n, image_id=i + 1) for i, n in enumerate(_names(100))],
                               read_two_view_geometry_num_inliers=lambda: (pairs, [100] * len(pairs)), close=lambda: None)
    pycolmap = types.SimpleNamespace(
        incremental_mapping=incremental, hierarchical_mapping=hierarchical, bundle_adjustment=lambda rec, options=None: None,
        BundleAdjustmentOptions=lambda: None, Database=types.SimpleNamespace(open=lambda path: db),
        IncrementalPipelineOptions=lambda: types.SimpleNamespace(mapper=types.SimpleNamespace(), max_runtime_seconds=-1),
        HierarchicalPipelineOptions=lambda: types.SimpleNamespace(clustering_options=types.SimpleNamespace(), incremental_options=None),
        undistort_images=lambda dense, sparse, images: (dense / "sparse").mkdir(parents=True), Reconstruction=lambda path: _Rec(100))
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={"mapper": mapper, **opts})
    report = sfm._map(ctx, pycolmap, tmp_path / "colmap", tmp_path / "db", tmp_path / "images", 100, "images", "sift", "sift",
                      "sequential", False, None)
    return report, calls


def test_a_hierarchical_solve_under_90_percent_is_solved_again_incrementally_and_says_so(tmp_path):
    report, calls = _fallback_run(tmp_path, "hierarchical", 80)
    assert [c[0] for c in calls] == ["hierarchical", "incremental"]
    fb = report["mapperFallback"]
    assert fb["reason"] == "80% registered" and fb["used"] == "incremental" and fb["hierarchical"]["registered"] == 80
    assert fb["incremental"]["registered"] == 100 and report["mapper"] == "hierarchical"
    assert set(report["mapperSeconds"]) == {"hierarchical", "bundle", "fallback"}


def test_a_hierarchical_solve_at_90_percent_is_kept(tmp_path):
    report, calls = _fallback_run(tmp_path, "hierarchical", 90)
    assert [c[0] for c in calls] == ["hierarchical"] and report["mapperFallback"] is None and report["registered"] == 100


def test_the_fallback_can_be_turned_off_for_a_fast_mapper(tmp_path):
    report, calls = _fallback_run(tmp_path, "hierarchical", 50, no_mapper_fallback=True)
    assert [c[0] for c in calls] == ["hierarchical"] and report["mapperFallback"] is None


def test_a_fast_mapper_that_finds_no_model_falls_back_instead_of_failing(tmp_path):
    report, _ = _fallback_run(tmp_path, "hierarchical", 0)
    assert report["mapperFallback"]["reason"] == "0% registered" and report["mapperFallback"]["used"] == "incremental"


def test_a_keyframe_solve_reports_its_keyframes_and_the_seconds_of_each_part(tmp_path):
    report, calls = _fallback_run(tmp_path, "keyframes", 90)   # 0, 4 ... 96, and 99 split from 96 (three apart: no match)
    assert [c[1] for c in calls] == [False, True] and report["keyframes"] == 27 and report["mapperFallback"] is None
    assert set(report["mapperSeconds"]) == {"keyframes", "register", "bundle"}


@pytest.fixture(scope="module")
def synthetic_scene(tmp_path_factory):
    """60 textured views of the synthetic room as a two-clip video job, SIFT features and sequential matches in its database."""
    pytest.importorskip("pycolmap")
    import cv2
    import test_sfm_e2e as e2e
    from synth import camera_loop
    e2e.W_IMG, e2e.H_IMG, e2e.F = 640, 480, 467.0
    root = tmp_path_factory.mktemp("scene")
    fcs = e2e.faces()
    centers, rots = camera_loop(60, inward=True)
    (root / "frames").mkdir()
    for i, (c, R) in enumerate(zip(centers, rots)):
        cv2.imwrite(str(root / "frames" / f"{i:04d}.jpg"), e2e.render(R, c, fcs), [cv2.IMWRITE_JPEG_QUALITY, 92])
    (root / "frames_index.json").write_text(json.dumps([{"frame": f"{i:04d}.jpg", "clip": "a.mp4" if i < 30 else "b.mp4"} for i in range(60)]))
    (root / ".done").mkdir()
    (root / ".done" / "frames.json").write_text(json.dumps({"source": "video"}))
    code = cli.main(["run", str(root), "--out", str(root), "--only", "sfm", "--features", "sift", "--device", "cpu", "--matcher", "sequential",
                     "--no-loop-detection", "--sfm-part", "match", "--split-min", "0", "--quiet"])
    assert code == 0 and (root / "colmap" / "database.db").exists()
    return root


def _resolve(scene, tmp_path, *args):
    import shutil
    job = tmp_path / "job"
    shutil.copytree(scene, job, ignore=shutil.ignore_patterns("sfm.json", "sparse*"))
    code = cli.main(["run", str(job), "--out", str(job), "--only", "sfm", "--reuse-matches", "--features", "sift", "--device", "cpu", "--quiet", *args])
    assert code == 0, (job / "log.jsonl").read_text()[-1500:]
    return json.loads((job / ".done" / "sfm.json").read_text())


@pytest.mark.slow
def test_registering_the_other_frames_leaves_none_unposed_on_the_synthetic_scene(synthetic_scene, tmp_path):
    rep = _resolve(synthetic_scene, tmp_path, "--mapper", "keyframes")
    assert rep["mapper"] == "keyframes" and rep["mapperFallback"] is None
    assert rep["registered"] == rep["images"] == 60 and 15 <= rep["keyframes"] < 60
    assert set(rep["mapperSeconds"]) == {"keyframes", "register", "bundle"} and rep["meanReprojErrorPx"] < 0.5


@pytest.mark.slow
def test_a_hierarchical_solve_of_the_synthetic_scene_registers_every_frame_from_several_clusters(synthetic_scene, tmp_path):
    rep = _resolve(synthetic_scene, tmp_path, "--mapper", "hierarchical", "--hier-leaf-max", "20")
    assert rep["mapper"] == "hierarchical" and rep["mapperFallback"] is None and rep["registered"] == rep["images"] == 60
    assert set(rep["mapperSeconds"]) == {"hierarchical", "bundle"}
