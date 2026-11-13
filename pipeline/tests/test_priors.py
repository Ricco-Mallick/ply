# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""priors (2b): the scale from a monocular depth model's metric depth against the camera solve, on a CPU with stand-ins.

The room is a 5 x 2.6 x 4 m box seen from a loop of cameras; the solve is the same room in arbitrary units (S_TRUE metres
per unit). The stand-in predictor returns the room's true metric depth, with noise, a wrong scale on some frames, or a
different wrong scale on every frame.
"""
import json
import time
import zlib

import cv2
import numpy as np
import pytest

from plypipe import colmap_txt, geometry
from plypipe.log import Log
from plypipe.stages import align, priors
from plypipe.util import Ctx

S_TRUE = 0.41  # metres per unit of the solve
ROOM = np.array([5.0, 2.6, 4.0])
W, H, F = 320, 240, 280.0
ORIGIN = np.array([1.0, 0.3, -2.0])  # the solve's origin, in room metres


def _camera(i, n):
    a = 2 * np.pi * i / n
    c = np.array([2.5 + 0.6 * np.cos(a), 1.4, 2.0 + 0.5 * np.sin(a)])
    f = np.array([np.cos(a + 0.3), -0.1, np.sin(a + 0.3)])
    f /= np.linalg.norm(f)
    down = np.array([0.0, -1.0, 0.0])
    down = down - (down @ f) * f
    down /= np.linalg.norm(down)
    return np.stack([np.cross(down, f), down, f]), c  # cam_from_world rotation, centre (room metres)


def room_depth(R, c):
    """Metric depth (z along the optical axis) of the inside of the box from a camera at c."""
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = R.T @ np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)]).reshape(3, -1)
    t = np.full(d.shape[1], np.inf)
    for ax in range(3):
        for wall in (0.0, ROOM[ax]):
            with np.errstate(divide="ignore", invalid="ignore"):
                hit = (wall - c[ax]) / d[ax]
            t = np.where(hit > 1e-6, np.minimum(t, hit), t)
    return t.reshape(H, W)  # the ray's z is 1, so its length parameter is the depth


def write_job(out, n_frames=200, pause=None, n_points=3000, seed=0):
    """A job folder with dense/sparse_txt (the solve in arbitrary units), dense/images and the cameras' true poses."""
    rng = np.random.default_rng(seed)
    pts = []
    for ax in range(3):
        for wall in (0.0, ROOM[ax]):
            p = rng.uniform(0, 1, (n_points // 6, 3)) * ROOM
            p[:, ax] = wall
            pts.append(p)
    pts = np.concatenate(pts)
    xyz = (pts - ORIGIN) / S_TRUE
    imgs, poses = [], {}
    (out / "dense" / "images").mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        j = i if pause is None else (min(i, pause[0]) if i < pause[1] else i - (pause[1] - pause[0]))  # a stretch of frames at one spot
        R, c = _camera(j, 120)
        if pause is not None and pause[0] <= i < pause[1]:
            c = c + rng.normal(0, 0.002, 3)
        name = f"{i:06d}.jpg"
        poses[name] = (R, c)
        cm = (c - ORIGIN) / S_TRUE
        Xc = xyz @ R.T - R @ cm
        z = Xc[:, 2]
        uv = F * Xc[:, :2] / np.where(z > 0, z, 1)[:, None] + [W / 2, H / 2]
        seen = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        line = " ".join(f"{uv[k, 0]:.3f} {uv[k, 1]:.3f} {k + 1}" for k in np.nonzero(seen)[0])
        imgs.append(colmap_txt.Image(i + 1, geometry.mat_to_quat(R), -R @ cm, 1, name, line))
        cv2.imwrite(str(out / "dense" / "images" / name), np.full((H, W, 3), 90, np.uint8))
    n = len(xyz)
    model = colmap_txt.Model([f"1 PINHOLE {W} {H} {F} {F} {W / 2} {H / 2}"], imgs, np.arange(1, n + 1), xyz, np.full((n, 3), 128, np.uint8),
                             np.full(n, 0.5), np.full(n, 3, np.int32), ["128 128 128 0.5 1 0 2 0 3 0"] * n)
    colmap_txt.write_model(model, out / "dense" / "sparse_txt")
    return model, poses


def stand_in(poses, sigma=0.02, factor=lambda name: 1.0, normals=False, slow=0.0):
    def predict(frame):
        R, c = poses[frame.name]
        rng = np.random.default_rng(zlib.crc32(frame.name.encode()))
        depth = room_depth(R, c) * factor(frame.name) * np.exp(rng.normal(0, sigma, (H, W)))
        if slow:
            time.sleep(slow)
        return priors.Depth(depth, np.tile([0.0, 0.0, -1.0], (H, W, 1)) if normals else None, np.ones((H, W)) if normals else None)
    return predict


def wrong_on_30_percent(name):
    k = zlib.crc32(name.encode()) % 10
    return {0: 1.6, 1: 0.55, 2: 1.25}.get(k, 1.0)


def disagreeing(name):
    return float(np.exp(np.random.default_rng(zlib.crc32(name.encode())).normal(0, 0.15)))


def ctx_for(out, **opts):
    return Ctx(out=out, log=Log(out, quiet=True), opts=opts)


def test_the_scale_is_recovered_within_one_percent_with_thirty_percent_wrong_frames(tmp_path):
    _, poses = write_job(tmp_path)
    r = priors.run_stage(ctx_for(tmp_path), stand_in(poses, factor=wrong_on_30_percent))
    assert r["status"] == "ok" and r["frames"] >= 100 and r["chosen"] == 150 and r["frame"] == "world"
    assert abs(r["scale"] / S_TRUE - 1) < 0.01, r
    assert r["spread"] < priors.MAX_SPREAD
    assert json.loads((tmp_path / priors.RESULT).read_text())["scale"] == r["scale"]


def test_frames_that_disagree_are_reported_inconsistent_and_align_keeps_the_camera_height_prior_and_says_so(tmp_path):
    _, poses = write_job(tmp_path / "a")
    write_job(tmp_path / "b")
    r = priors.run_stage(ctx_for(tmp_path / "a"), stand_in(poses, factor=disagreeing))
    assert r["status"] == "inconsistent" and r["spread"] > priors.MAX_SPREAD and "disagree" in r["reason"]
    rep = align.run_stage(ctx_for(tmp_path / "a"))
    base = align.run_stage(ctx_for(tmp_path / "b"))
    scene = json.loads((tmp_path / "a" / "scene.json").read_text())
    assert rep["scaleSource"] == "camera-height-prior" and rep["scale"] == pytest.approx(base["scale"])
    assert "disagree" in scene["scaleFallback"] and scene["scaleSource"] == "camera-height-prior" and "scaleSpread" not in scene
    assert set(scene["scaleCandidates"]) == {"camera-height-prior"}


def test_align_takes_the_monocular_scale_when_the_frames_agree_and_records_its_spread_and_the_other_candidate(tmp_path):
    _, poses = write_job(tmp_path)
    r = priors.run_stage(ctx_for(tmp_path), stand_in(poses, factor=wrong_on_30_percent))
    rep = align.run_stage(ctx_for(tmp_path))
    scene = json.loads((tmp_path / "scene.json").read_text())
    assert rep["scaleSource"] == scene["scaleSource"] == "monocular-depth"
    assert scene["scale"] == pytest.approx(r["scale"]) and abs(scene["scale"] / S_TRUE - 1) < 0.01
    assert scene["scaleSpread"] == pytest.approx(r["spread"]) and "scaleFallback" not in scene
    assert scene["scaleCandidates"]["camera-height-prior"] != pytest.approx(scene["scale"], rel=1e-3)  # 1.45 m camera height is not this room's
    # the model really is in metres now: the room is 5 m along its longest axis, floor at y = 0
    aligned = colmap_txt.read_model(tmp_path / "data" / "sparse" / "0")
    assert aligned.xyz[:, 1].min() == pytest.approx(0.0, abs=0.05)
    assert np.ptp(aligned.xyz[:, [0, 2]], axis=0).max() == pytest.approx(5.0, rel=0.03)
    assert priors.readout(tmp_path)["priors"]["twinScale"] == pytest.approx(r["scale"])


def test_align_without_a_priors_result_is_what_it_was(tmp_path):
    model, _ = write_job(tmp_path)
    rep = align.run_stage(ctx_for(tmp_path))
    scene = json.loads((tmp_path / "scene.json").read_text())
    assert set(scene) == {"version", "units", "up", "handedness", "floorY", "scale", "scaleSource", "cameraHeight", "transform", "bounds", "cameraPath",
                          "spawn", "tour", "shell"}
    assert scene["scaleSource"] == "camera-height-prior" and "scaleCandidates" not in rep and "scaleFallback" not in rep
    centers, _, _ = model.cameras()
    heights = geometry.camera_centers_and_ups  # the cameras sit 1.4 m up: the prior scales that to 1.45 m
    assert scene["scale"] == pytest.approx(1.45 / (1.4 / S_TRUE), rel=0.03) and heights is not None and len(centers) == 200


def test_a_reference_measurement_beats_the_monocular_scale_and_a_stale_one_is_ignored(tmp_path):
    _, poses = write_job(tmp_path)
    priors.run_stage(ctx_for(tmp_path), stand_in(poses))
    (tmp_path / "scene.json").write_text(json.dumps({"scaleSource": "reference", "scale": 0.5, "reference": {"scale": 0.5, "factor": 1.2}}))
    assert align.run_stage(ctx_for(tmp_path))["scaleSource"] == "reference"
    scene = json.loads((tmp_path / "scene.json").read_text())
    assert scene["scale"] == pytest.approx(0.5) and scene["scaleSource"] == "reference"
    assert np.cbrt(np.linalg.det(np.array(scene["transform"])[:3, :3])) == pytest.approx(0.5)
    (tmp_path / "scene.json").write_text(json.dumps({"scaleSource": "reference", "scale": 0.5, "reference": {"scale": 0.5, "world": {"points": 5, "images": 5}}}))
    rep = align.run_stage(ctx_for(tmp_path))
    scene = json.loads((tmp_path / "scene.json").read_text())
    assert rep["scaleSource"] == "monocular-depth" and "different camera solve" in scene["scaleFallback"] and scene["reference"] is None


def test_the_chosen_frames_are_spread_along_the_path_not_over_the_time_it_took(tmp_path):
    t = np.linspace(0, 6.0, 300)
    path = np.stack([t, np.zeros(300), np.zeros(300)], axis=1)
    path[100:250] = path[100]  # standing still for half the video
    path[250:] = path[250:] - path[250] + path[100] + [0.01, 0, 0]
    chosen = choose = priors.choose_frames(path, 150)
    assert len(chosen) <= 150 and chosen == sorted(chosen) and len(set(chosen)) == len(chosen)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    gaps = np.diff(s[chosen])
    assert gaps.max() <= 2.5 * gaps.mean()
    assert sum(100 <= i < 250 for i in chosen) <= 6  # a pause is not 150 frames of one view
    assert priors.choose_frames(path[:40], 150) == list(range(40)) and choose[0] == 0
    assert set(priors._interleave(list(range(20)))) == set(range(20)) and priors._interleave(list(range(20)))[:4] == [0, 16, 8, 4]


def test_points_behind_the_camera_at_the_edge_or_without_confidence_do_not_count(tmp_path):
    rng = np.random.default_rng(1)
    pred = np.full((100, 200), 3.0)  # metres everywhere
    uv = np.column_stack([rng.uniform(20, 180, 400), rng.uniform(10, 90, 400)])
    z = np.full(400, 2.0)  # SfM units: ratio 1.5
    good, n = priors.frame_ratio(pred, uv, z)
    assert good == pytest.approx(1.5) and n == 400
    bad_uv = np.vstack([uv, [[2.0, 50.0], [198.0, 50.0], [100.0, 1.0], [100.0, 99.0]] * 10])  # on the 5% border
    behind = np.concatenate([z, np.full(40, 2.0)])
    assert priors.frame_ratio(pred, bad_uv, behind)[1] == 400
    behind[:100] = -2.0  # behind the camera: a negative SfM depth would flip the ratio's sign
    assert priors.frame_ratio(pred, bad_uv, behind) == (pytest.approx(1.5), 300)
    conf = np.ones_like(pred)
    conf[:, :100] = 0.1
    pred2 = pred.copy()
    pred2[:, :100] = 30.0  # the unsure half is wildly wrong
    assert priors.frame_ratio(pred2, uv, z, conf)[0] == pytest.approx(1.5) and priors.frame_ratio(pred2, uv, z)[0] != pytest.approx(1.5, rel=0.2)
    assert priors.frame_ratio(np.zeros((100, 200)), uv, z) is None and priors.frame_ratio(pred, uv[:10], z[:10]) is None


def test_depth_and_normals_are_saved_at_half_resolution_with_the_seconds_and_a_time_budget_stops_the_run(tmp_path):
    _, poses = write_job(tmp_path, n_frames=60)
    r = priors.run_stage(ctx_for(tmp_path, priors_frames=40), stand_in(poses, normals=True))
    z = np.load(tmp_path / priors.DEPTH_FILE)
    assert r["depthFile"] == priors.DEPTH_FILE and len(z["names"]) == 40
    assert z["depth"].shape == (40, H // 2, W // 2) and z["depth"].dtype == np.float16 and z["normals"].shape == (40, H // 2, W // 2, 3)
    assert z["normals"].dtype == np.int8 and z["confidence"].dtype == np.uint8 and z["K"][0][0, 0] == pytest.approx(F / 2)
    assert np.isfinite(z["depth"]).all() and 0.3 < float(z["depth"].astype(np.float32).mean()) < 7.0 and (z["normals"][..., 2] == -127).all()
    assert r["seconds"] > 0 and r["predictSeconds"] > 0 and r["secondsPerFrame"] > 0 and r["predictor"] == "predict"
    _, poses = write_job(tmp_path / "slow", n_frames=60)
    r = priors.run_stage(ctx_for(tmp_path / "slow", priors_frames=40, priors_max_seconds=0.15), stand_in(poses, slow=0.05))
    assert "time budget" in r["stopped"] and priors.MIN_FRAMES <= r["predicted"] < 40
    assert r["status"] in ("ok", "inconsistent")


def test_a_predictor_with_no_usable_depth_reports_too_few_frames_and_without_a_gpu_the_real_one_refuses_and_align_goes_on(tmp_path, monkeypatch):
    write_job(tmp_path / "a")
    r = priors.run_stage(ctx_for(tmp_path / "a"), lambda f: priors.Depth(np.zeros((H, W))))
    assert r["status"] == "too-few-frames" and r["scale"] is None
    rep = align.run_stage(ctx_for(tmp_path / "a"))
    assert rep["scaleSource"] == "camera-height-prior" and "too-few-frames" in json.loads((tmp_path / "a" / "scene.json").read_text())["scaleFallback"]
    with pytest.raises(ValueError):
        priors.real_predictor("moge3")
    with pytest.raises((ImportError, RuntimeError)):  # no moge package or no CUDA here
        priors.real_predictor("moge2-l")
    write_job(tmp_path / "b")
    rep = align.run_stage(ctx_for(tmp_path / "b", priors=True))  # --priors on a machine that cannot run the model: the twin is still made
    scene = json.loads((tmp_path / "b" / "scene.json").read_text())
    assert rep["scaleSource"] == "camera-height-prior" and scene["scaleFallback"].startswith("monocular depth unavailable")


def test_a_rescaled_job_keeps_its_reference_scale_when_align_runs_again_and_only_for_the_same_camera_solve(tmp_path):
    from plypipe import rescale
    write_job(tmp_path)
    first = align.run_stage(ctx_for(tmp_path))["scale"]
    rescale.rescale(tmp_path, 1.1, reference={"knownM": 2.2, "measuredM": 2.0})
    ref = json.loads((tmp_path / "scene.json").read_text())["reference"]
    assert ref["world"] == {"points": 3000, "images": 200} and ref["scale"] == pytest.approx(first * 1.1)
    rep = align.run_stage(ctx_for(tmp_path))  # --from align, with the camera-height prior that gave `first` still the default
    assert rep["scaleSource"] == "reference" and rep["scale"] == pytest.approx(first * 1.1)
    aligned = colmap_txt.read_model(tmp_path / "data" / "sparse" / "0")
    assert np.ptp(aligned.xyz[:, [0, 2]], axis=0).max() == pytest.approx(5.0 * 1.1 * first / S_TRUE, rel=0.05)


def test_the_priors_command_prints_the_scale_the_twin_would_get_and_fails_cleanly_without_a_gpu(tmp_path, monkeypatch, capsys):
    from plypipe import cli
    _, poses = write_job(tmp_path, n_frames=60)
    monkeypatch.setattr(priors, "real_predictor", lambda model="moge2-l": stand_in(poses))
    assert cli.main(["priors", str(tmp_path), "--priors-frames", "40", "--priors-model", "moge2-b"]) == 0
    out = capsys.readouterr().out
    got = json.loads((tmp_path / "priors" / "readout.json").read_text())
    assert got["priors"]["status"] == "ok" and abs(got["priors"]["ratio"] / S_TRUE - 1) < 0.02 and got["priors"]["frame"] == "world" and "twinScale" in out
    monkeypatch.undo()
    assert cli.main(["priors", str(tmp_path / "empty")]) == 1
    a = cli.build_parser().parse_args(["run", "x", "--out", "y", "--priors", "--priors-model", "moge2-b"])
    assert a.priors and a.priors_model == "moge2-b" and a.priors_frames == 150 and not cli.build_parser().parse_args(["run", "x", "--out", "y"]).priors
