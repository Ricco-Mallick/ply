# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""plypipe.scan (route 8c): registration of a video's camera-solve points onto a laser scan, the gate, and the scan's
depth maps, on a synthetic room."""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from plypipe import geometry, scan


def room(rng, step=0.03):
    """An L-shaped office 7.5 x 6 m, 3 m high, with desks: walls, floor, ceiling, furniture (not symmetric)."""
    pts = []
    xs, zs, ys = np.arange(0, 7.5, step), np.arange(0, 6.0, step), np.arange(0, 3.0, step)
    for x in (0.0, 7.5):
        Z, Y = np.meshgrid(zs, ys)
        pts.append(np.c_[np.full(Z.size, x), Y.ravel(), Z.ravel()])
    for z in (0.0, 6.0):
        X, Y = np.meshgrid(xs, ys)
        pts.append(np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)])
    X, Z = np.meshgrid(xs, zs)
    pts.append(np.c_[X.ravel(), np.zeros(X.size), Z.ravel()])
    pts.append(np.c_[X.ravel(), np.full(X.size, 3.0), Z.ravel()])
    for (x0, z0, w, d, h) in ((0.3, 0.3, 2.4, 0.8, 0.75), (4.5, 0.4, 1.2, 2.5, 0.75), (2.0, 3.2, 3.0, 0.7, 0.75), (6.6, 4.6, 0.6, 1.0, 1.9)):
        X, Z = np.meshgrid(np.arange(x0, x0 + w, step), np.arange(z0, z0 + d, step))
        pts.append(np.c_[X.ravel(), np.full(X.size, h), Z.ravel()])
        for xx in (x0, x0 + w):
            Zs, Ys = np.meshgrid(np.arange(z0, z0 + d, step), np.arange(0, h, step))
            pts.append(np.c_[np.full(Zs.size, xx), Ys.ravel(), Zs.ravel()])
    p = np.vstack(pts)
    return p + rng.normal(0, 0.003, p.shape)


@pytest.mark.slow
def test_register_recovers_turn_scale_and_offset_with_an_extra_room():
    rng = np.random.default_rng(0)
    sc = room(rng)
    true = scan._similarity(0.86, scan._yaw(37.0), np.array([1.3, 0.0, -2.1]))   # video -> scan
    inv = np.linalg.inv(true)
    video = geometry.apply(inv, sc[rng.choice(len(sc), 30_000, replace=False)]) + rng.normal(0, 0.012, (30_000, 3))
    other = np.c_[rng.uniform(9, 14, 8000), rng.uniform(0, 3, 8000), rng.uniform(-2, 4, 8000)]   # a room the scanner never saw
    T, rep = scan.register(np.vstack([video, geometry.apply(inv, other)]), sc)
    assert T is not None, rep.get("refused")
    probe = np.array([[1.0, 1.0, 1.0], [6.0, 2.0, 5.0]])
    assert np.abs(geometry.apply(T, geometry.apply(inv, probe)) - probe).max() < 0.03
    assert abs(rep["scale"] - 0.86) < 0.01 and rep["medianM"] < 0.03 and rep["insideShare"] < 0.9


@pytest.mark.slow
def test_register_refuses_a_different_place():
    rng = np.random.default_rng(1)
    sc = room(rng)
    elsewhere = np.c_[rng.uniform(0, 12, 20000), rng.uniform(0, 2.6, 20000), rng.uniform(0, 3, 20000)]
    T, rep = scan.register(elsewhere, sc)
    assert T is None and rep["refused"]


def test_render_depth_of_a_wall_and_no_leak_from_behind():
    rng = np.random.default_rng(2)
    X, Y = np.meshgrid(np.arange(-3, 3, 0.01), np.arange(-2, 2, 0.01))
    wall = np.c_[X.ravel(), Y.ravel(), np.full(X.size, 2.0)]
    behind = np.c_[rng.uniform(-3, 3, 50000), rng.uniform(-2, 2, 50000), np.full(50000, 5.0)]
    cam = (640, 480, 500.0, 500.0, 320.0, 240.0)
    d = scan.render_depth(np.vstack([wall, behind]).astype(np.float32), np.eye(3), np.zeros(3), cam)
    assert d.shape == (round(480 / scan.DEPTH_DOWN), round(640 / scan.DEPTH_DOWN))
    seen = d[d > 0]
    assert seen.size > 0.95 * d.size and np.abs(seen - 2.0).max() < 1e-3


def test_agreement_flags_moved_furniture():
    d = np.full((120, 160), 2.0, np.float32)
    uv = np.c_[np.linspace(10, 630, 60), np.linspace(10, 470, 60)]
    err, n = scan.agreement(d, uv, np.full(60, 2.02), 0.25, 0.25)
    assert n == 60 and err < 0.02
    err, _ = scan.agreement(d, uv, np.full(60, 1.4), 0.25, 0.25)   # something now stands in front of the scanned wall
    assert err > scan.AGREE


def test_pinhole_reads_both_pinhole_models():
    cams = scan.pinhole(["# c", "1 PINHOLE 640 480 500 510 320 240", "2 SIMPLE_PINHOLE 100 50 80 50 25", "3 SIMPLE_RADIAL 1 1 1 1 1 0"])
    assert cams == {1: (640, 480, 500.0, 510.0, 320.0, 240.0), 2: (100, 50, 80.0, 80.0, 50.0, 25.0)}


def test_lidar_depth_reads_the_scan_maps_folder_and_range(tmp_path):
    import json
    from plypipe import lidar_depth
    (tmp_path / "scandepth").mkdir()
    np.save(tmp_path / "scandepth" / "000001.npy", np.full((30, 40), 9.0, np.float16))
    (tmp_path / "scandepth.json").write_text(json.dumps({"dir": "scandepth", "rangeM": [0.2, 15], "frames": {"000001.jpg": {"depth": "000001.npy"}}}))
    idx = lidar_depth.read_index(tmp_path / "scandepth.json")
    d = lidar_depth.load_depth(idx["000001.jpg"][0])
    assert abs(float(d.max()) - 9.0) < 0.01
    assert lidar_depth.sample(d, None, 160, 120) is None                    # 9 m is past the iPhone's 6 m
    got = lidar_depth.sample(d, None, 160, 120, near=0.2, far=15.0)
    assert got is not None and np.allclose(got[1], 9.0, atol=0.01)


@pytest.mark.slow
def test_align_with_a_scan_registers_and_writes_depth_and_start(tmp_path):
    """The align stage end to end with --scan: a synthetic solve, its room scanned densely in another frame (Z-up)."""
    import json
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from synth import room_points, write_synthetic_model
    from plypipe.log import Log
    from plypipe.stages import align
    from plypipe.util import Ctx
    out = tmp_path / "job"
    write_synthetic_model(out / "dense" / "sparse_txt")
    (out / "dense" / "images").mkdir(parents=True)
    rng = np.random.default_rng(5)
    dense = room_points(rng, n_floor=60_000, n_wall=120_000, noise=0.002)
    zup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)            # the scanner's file is Z-up and elsewhere
    file_pts = dense @ zup.T + np.array([12.0, -4.0, 0.3])
    np.savez(tmp_path / "scan.npz", points=file_pts.astype(np.float32), colors=np.full((len(file_pts), 3), 200, np.uint8))
    ctx = Ctx(out=out, log=Log(out, quiet=True), opts={"height_cam": 1.45, "scan": str(tmp_path / "scan.npz")})
    rep = align.run_stage(ctx)
    assert rep["scan"]["used"], rep["scan"]
    assert rep["scaleSource"] == "lidar-scan" and rep["scan"]["medianM"] < 0.03
    j = json.loads((out / "scandepth.json").read_text())
    assert j["dir"] == "scandepth" and len(j["frames"]) > 40
    assert (out / "scan" / "init.npz").is_file() and (out / "scan-fit.png").is_file()
    assert not (out / "scan" / "source").exists()
