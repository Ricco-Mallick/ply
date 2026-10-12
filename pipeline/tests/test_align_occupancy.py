import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from plypipe import colmap_txt, geometry
from plypipe.log import Log
from plypipe.stages import align, occupancy
from plypipe.util import Ctx

from synth import TABLE, write_synthetic_model


@pytest.fixture(scope="module")
def aligned(tmp_path_factory):
    out = tmp_path_factory.mktemp("scene")
    truth = write_synthetic_model(out / "dense" / "sparse_txt")
    (out / "dense" / "images").mkdir()
    ctx = Ctx(out=out, log=Log(out, quiet=True), opts={"height_cam": 1.45})
    rep = align.run_stage(ctx)
    return out, ctx, truth, rep


def test_colmap_txt_roundtrip(tmp_path):
    write_synthetic_model(tmp_path / "a")
    m = colmap_txt.read_model(tmp_path / "a")
    colmap_txt.write_model(m, tmp_path / "b")
    m2 = colmap_txt.read_model(tmp_path / "b")
    assert np.allclose(m.xyz, m2.xyz, atol=1e-6)
    assert [i.name for i in m.images] == [i.name for i in m2.images]
    T = np.eye(4); T[:3, :3] = 1.7 * geometry.rotation_between(np.array([0, 0, 1.]), np.array([1, 1, 0.])); T[:3, 3] = [3, 2, 1]
    back = colmap_txt.transformed(colmap_txt.transformed(m, T), np.linalg.inv(T))
    assert np.allclose(back.xyz, m.xyz, atol=1e-6)
    for a, b in zip(m.images, back.images):
        assert np.allclose(geometry.quat_to_mat(a.qvec), geometry.quat_to_mat(b.qvec), atol=1e-8)
        assert np.allclose(a.tvec, b.tvec, atol=1e-6)


def test_align_recovers_gravity_floor_scale_and_axes(aligned):
    out, ctx, truth, rep = aligned
    scene = json.loads((out / "scene.json").read_text())
    M = np.array(scene["transform"]) @ truth["T_true"]  # room frame -> recovered scene frame
    s = np.cbrt(np.linalg.det(M[:3, :3]))
    R = M[:3, :3] / s
    assert abs(s - 1.0) < 0.02, s                                   # metric scale from 1.45 m prior
    assert np.degrees(np.arccos(np.clip((R @ [0, 1, 0])[1], -1, 1))) < 1.0  # gravity
    floor_room = truth["points_room"][np.abs(truth["points_room"][:, 1]) < 0.03]
    fy = geometry.apply(M, floor_room)[:, 1]
    assert np.median(np.abs(fy)) < 0.015 and np.percentile(np.abs(fy), 95) < 0.05  # floor at y = 0
    yaw = np.degrees(np.arctan2(R[0, 2], R[0, 0]))
    assert abs((yaw + 45) % 90 - 45) < 3.0                         # walls on the X/Z axes
    assert scene["up"] == [0, 1, 0] and scene["floorY"] == 0.0 and scene["units"] == "meters"
    assert len(scene["cameraPath"]) > 10 and abs(scene["spawn"]["position"][1] - 1.6) < 1e-9
    # cameras end up at ~1.45 m
    c = np.array(scene["cameraPath"])
    assert abs(np.median(c[:, 1]) - 1.45) < 0.03
    # the aligned model on disk is what gsplat will read
    assert (out / "data" / "sparse" / "0" / "points3D.txt").exists()
    assert (out / "data" / "images").exists()


def test_occupancy_blocks_furniture_and_walls_frees_path(aligned):
    out, ctx, truth, _ = aligned
    rep = occupancy.run_stage(ctx)
    meta = json.loads((out / "occupancy.json").read_text())
    grid = cv2.imread(str(out / "occupancy.png"), cv2.IMREAD_GRAYSCALE)
    assert grid.shape == (meta["height"], meta["width"])
    scene = json.loads((out / "scene.json").read_text())
    M = np.array(scene["transform"]) @ truth["T_true"]

    def val(x, z):
        p = geometry.apply(M, np.array([[x, 0.0, z]]))[0]
        r, c = occupancy.world_to_cell(meta, p[0], p[2])
        return int(grid[r, c])

    x0, z0, x1, z1, _ = TABLE
    assert val((x0 + x1) / 2, (z0 + z1) / 2) == occupancy.BLOCKED          # table
    assert val(3.0, 0.0) == occupancy.BLOCKED                               # wall
    path = truth["centers_room"]
    on_path = [val(p[0], p[2]) for p in path]
    assert np.mean([v == occupancy.FREE for v in on_path]) > 0.95           # walked route is free
    assert val(3.0 + 1.6, 2.0) == occupancy.FREE
    assert rep["blocked"] > 0 and rep["free"] > 0
    assert scene["occupancy"] == {"image": "occupancy.png", "meta": "occupancy.json"}


def test_binary_model_matches_text(aligned):
    """gsplat 1.5.x reads the classic COLMAP binary (its text reader is broken); counts and the first pose must match."""
    import struct
    out = aligned[0]
    sp = out / "data" / "sparse" / "0"
    m = colmap_txt.read_model(sp)
    with (sp / "images.bin").open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        image_id, *qt = struct.unpack("<I4d3d", f.read(4 + 7 * 8))
    with (sp / "points3D.bin").open("rb") as f:
        npts = struct.unpack("<Q", f.read(8))[0]
    with (sp / "cameras.bin").open("rb") as f:
        ncam, (cid, model_id, w, h) = struct.unpack("<Q", f.read(8))[0], struct.unpack("<iiQQ", f.read(24))
    assert ncam == 1 and model_id == 1 and w > 0 and h > 0          # PINHOLE after undistortion
    assert n > 0 and npts > 0
    assert n == len(m.images) and npts == m.num_points
    first = m.images[0]
    assert image_id == first.image_id and np.allclose(qt[:4], first.qvec) and np.allclose(qt[4:], first.tvec)
