# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""plypipe rescale (2r): a twin changes scale without retraining, atomically, in every file that holds a length."""
import gzip
import json
import math
import os
import struct

import numpy as np
import pytest

from plypipe import cli, colmap_txt, geometry, rescale
from plypipe.stages import occupancy

PROPS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
FRAC = 12


def splats(rng, n=400):
    """A wall at x = 2 (y 0..2.5, z 0..4) and some floor, as positions and log-scales."""
    xyz = np.zeros((n, 3))
    xyz[:, 0] = 2.0
    xyz[:, 1] = rng.uniform(0.0, 2.5, n)
    xyz[:, 2] = rng.uniform(0.0, 4.0, n)
    xyz[:40, 0] = rng.uniform(0.0, 2.0, 40)  # floor splats
    xyz[:40, 1] = 0.0
    return xyz, rng.uniform(np.log(0.02), np.log(0.08), (n, 3))


def write_ply(path, xyz, logs):
    n = len(xyz)
    data = np.zeros((n, len(PROPS)), np.float32)
    data[:, 0:3], data[:, 6:9], data[:, 9] = xyz, 0.5, 2.0
    data[:, 10:13], data[:, 13] = logs, 1.0
    head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join(f"property float {p}\n" for p in PROPS) + "end_header\n"
    path.write_bytes(head.encode() + data.tobytes())


def read_ply(path):
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int(next(ln for ln in raw[:end].decode().splitlines() if ln.startswith("element vertex")).split()[-1])
    return np.frombuffer(raw[end:], np.float32).reshape(n, len(PROPS)).astype(np.float64)


def write_spz(path, xyz, logs):
    n = len(xyz)
    q = np.round(xyz * (1 << FRAC)).astype(np.int64) & 0xFFFFFF
    pos = np.stack([q & 255, (q >> 8) & 255, (q >> 16) & 255], axis=-1).astype(np.uint8)
    sc = np.clip(np.round((logs + 10) * 16), 0, 255).astype(np.uint8)
    body = struct.pack("<IIIBBBB", 0x5053474E, 3, n, 0, FRAC, 0, 0) + pos.tobytes() + np.full(n, 200, np.uint8).tobytes() + \
        np.full(3 * n, 128, np.uint8).tobytes() + sc.tobytes() + np.tile(np.array([1, 2, 3, 4], np.uint8), n).tobytes()
    path.write_bytes(gzip.compress(body, mtime=0))


def read_spz(path):
    raw = gzip.decompress(path.read_bytes())
    _, _, n, _, frac, _, _ = struct.unpack("<IIIBBBB", raw[:16])
    b = np.frombuffer(raw, np.uint8, 9 * n, 16).reshape(n, 3, 3).astype(np.int64)
    v = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    xyz = np.where(v & 0x800000, v - (1 << 24), v) / float(1 << frac)
    sc = np.frombuffer(raw, np.uint8, 3 * n, 16 + 13 * n).reshape(n, 3) / 16.0 - 10.0
    rest = raw[16 + 9 * n:16 + 13 * n] + raw[16 + 16 * n:]
    return xyz, sc, rest


def sparse_model(d):
    """Four cameras around the middle of the room looking at the wall, with five points on it, in metres."""
    imgs = []
    for i in range(4):
        c = np.array([0.5 + 0.2 * i, 1.4, 1.0 + 0.5 * i])
        R = np.array([[0, 0, -1.0], [0, -1, 0], [1.0, 0, 0]])  # cam_from_world: forward = +x, down = -y, right = -z
        imgs.append(colmap_txt.Image(i + 1, geometry.mat_to_quat(R), -R @ c, 1, f"{i:06d}.jpg",
                                  " ".join(f"{10 + 5 * k} 20 {k + 1}" for k in range(5)) if i < 2 else ""))
    pts = np.array([[2.0, 1.0 + 0.2 * k, 1.0 + 0.3 * k] for k in range(5)])
    tails = [f"128 128 128 0.5 1 {k} 2 {k}" for k in range(5)]
    model = colmap_txt.Model(["# Camera list", "1 PINHOLE 640 480 500 500 320 240"], imgs, np.arange(1, 6), pts, np.full((5, 3), 128, np.uint8),
                             np.full(5, 0.5), np.full(5, 2, np.int32), tails)
    colmap_txt.write_model(model, d)
    colmap_txt.write_model_bin(model, d)
    return model


def make_twin(folder, job=False, seed=0):
    rng = np.random.default_rng(seed)
    xyz, logs = splats(rng)
    folder.mkdir(parents=True, exist_ok=True)
    write_spz(folder / "scene.spz", xyz, logs)
    write_spz(folder / "scene_lite.spz", xyz[::2], logs[::2])
    scene = {"version": 1, "units": "meters", "floorY": 0.0, "scale": 0.37, "scaleSource": "camera-height-prior", "cameraHeight": 1.45,
             "transform": [[0.37, 0, 0, 1.0], [0, 0.37, 0, 2.0], [0, 0, 0.37, 3.0], [0, 0, 0, 1]],
             "bounds": {"min": [-0.25, -0.25, -0.25], "max": [2.25, 2.75, 4.25]},
             "cameraPath": [[0.5, 1.4, 1.0], [0.9, 1.4, 2.0]],
             "spawn": {"position": [0.6, 1.6, 1.2], "lookAt": [1.6, 1.6, 1.2]},
             "tour": [{"pos": [0.6, 2.0, 1.2], "target": [1.5, 1.1, 2.0], "fov": 62, "dur": 0}],
             "topview": {"version": 1, "origin": [-0.2, -0.3], "metersPerPixel": 0.02, "width": 130, "height": 230, "cutY": 2.4, "splats": 400,
                         "image": "topview.webp"}}
    (folder / "scene.json").write_text(json.dumps(scene))
    grid = np.full((40, 25), 127, np.uint8)
    cell, origin = 0.1, [-0.25, -0.25]
    meta = {"version": 1, "image": "occupancy.png", "cellSize": cell, "origin": origin, "width": 25, "height": 40, "obstacleBand": [0.1, 2.0],
            "floorBand": [0.0, 0.1], "walkedRadius": 0.4, "pathSegments": 1, "seenFree": {"rays": 5, "below": 0.6, "minRays": 2},
            "walk": {"radius": 0.22, "standableM2": 12.5, "cameras": 1.0, "pass": True},
            "depth": {"views": 3, "maxRange": 8.0, "holeM2": 0.5, "points": {"file": "dense_points.npz", "voxel": 0.02}}}
    (folder / "occupancy.json").write_text(json.dumps(meta))
    (folder / "shell.json").write_text(json.dumps({
        "version": 1, "floorY": 0.0, "ceilingY": 2.5,
        "walls": [{"a": [2.0, 0.0], "b": [2.0, 3.0], "length": 3.0, "normal": [-1.0, 0.0], "support": 900, "angle": 90.0}],
        "openings": [{"kind": "door", "wall": 0, "u": 1.0, "width": 0.9, "sill": 0.0, "height": 2.0, "name": "Door"}],
        "objects": [{"id": "bench-1", "name": "Bench", "kind": "bench", "center": [1.0, 0.4, 2.0], "half": [0.5, 0.4, 0.2]}]}))
    (folder / "places.json").write_text(json.dumps({"version": 1, "source": "sam3", "places": [
        {"id": "door-1", "name": "Door", "p": [2.0, 1.0, 1.0], "pinned": True, "ai": {"cls": "door", "conf": 0.9, "half": [0.45, 1.0, 0.1]}}]}))
    (folder / "topview.webp").write_bytes(b"RIFFfake")
    (folder / "occupancy.png").write_bytes(b"\x89PNGfake")
    if job:
        write_ply(folder / "scene.ply", xyz, logs)
        write_ply(folder / "scene_clean.ply", xyz, logs)
        sparse_model(folder / "data" / "sparse" / "0")
        np.savez_compressed(folder / "dense_points.npz", points=rng.uniform(0, 3, (50, 3)).astype(np.float32),
                            normals=np.zeros((50, 3), np.float16), voxel=np.float32(0.02))
        (folder / "report.json").write_text(json.dumps({"align": {"scale": 0.37, "scaleSource": "camera-height-prior", "yawDeg": 3.0,
                                                                    "bounds": scene["bounds"], "floorResidualMedianM": 0.02}, "sfm": {"images": 4}}))
        (folder / ".done").mkdir()
        (folder / ".done" / "align.json").write_text(json.dumps({"scale": 0.37, "scaleSource": "camera-height-prior", "bounds": scene["bounds"]}))
    return xyz, logs


def snapshot(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def test_a_rescale_by_one_point_one_moves_a_known_wall_by_exactly_ten_percent_in_the_splat_the_grid_the_shell_and_the_places(tmp_path):
    xyz, logs = make_twin(tmp_path / "twin")
    before = {n: read_spz(tmp_path / "twin" / n) for n in ("scene.spz", "scene_lite.spz")}
    old = json.loads((tmp_path / "twin" / "occupancy.json").read_text())
    res = rescale.rescale(tmp_path / "twin", 1.1)
    assert res["scaleSource"] == "reference" and res["previousSource"] == "camera-height-prior" and "scene.spz" in res["files"]
    wall = xyz[:, 0] == 2.0
    for name, n_step in (("scene.spz", 1), ("scene_lite.spz", 2)):
        got, sc, rest = read_spz(tmp_path / "twin" / name)
        assert np.abs(got[wall[::n_step]][:, 0] - 2.2).max() <= 1.0 / (1 << FRAC)  # exactly 10% further out, within the fixed point's step
        assert np.abs(got - before[name][0] * 1.1).max() <= 0.5 / (1 << FRAC) + 1e-9  # from what the file held: one rounding
        # sizes: one byte step is 1/16 of a log unit; the nearest step to ln 1.1 = 1.525 steps is two
        assert np.abs((sc - before[name][1]) - math.log(1.1)).max() <= 1 / 32 + 1e-9
        assert rest == before[name][2]  # alpha, colour and rotation bytes are the original's
    meta = json.loads((tmp_path / "twin" / "occupancy.json").read_text())
    assert meta["cellSize"] == pytest.approx(0.11) and meta["origin"] == pytest.approx([-0.275, -0.275])
    assert meta["obstacleBand"] == pytest.approx([0.11, 2.2]) and meta["walkedRadius"] == pytest.approx(0.44)
    assert meta["walk"]["standableM2"] == pytest.approx(12.5 * 1.21) and meta["walk"]["radius"] == 0.22  # the walker stays 22 cm
    assert meta["depth"]["holeM2"] == pytest.approx(0.605) and meta["depth"]["maxRange"] == pytest.approx(8.8) and meta["depth"]["points"]["voxel"] == pytest.approx(0.022)
    assert (meta["width"], meta["height"]) == (25, 40)
    for x, z in ((2.0, 0.5), (2.0, 3.5), (1.0, 2.0)):  # the same cell holds the same splat
        assert occupancy.world_to_cell(meta, x * 1.1, z * 1.1) == occupancy.world_to_cell(old, x, z)
    shell = json.loads((tmp_path / "twin" / "shell.json").read_text())
    w = shell["walls"][0]
    assert w["a"] == pytest.approx([2.2, 0.0]) and w["b"] == pytest.approx([2.2, 3.3]) and w["length"] == pytest.approx(3.3)
    assert w["normal"] == [-1.0, 0.0] and w["support"] == 900 and w["angle"] == 90.0
    assert shell["ceilingY"] == pytest.approx(2.75) and shell["floorY"] == 0.0
    assert shell["openings"][0]["u"] == pytest.approx(1.1) and shell["openings"][0]["height"] == pytest.approx(2.2)
    assert shell["objects"][0]["center"] == pytest.approx([1.1, 0.44, 2.2]) and shell["objects"][0]["half"] == pytest.approx([0.55, 0.44, 0.22])
    pl = json.loads((tmp_path / "twin" / "places.json").read_text())["places"][0]
    assert pl["p"] == pytest.approx([2.2, 1.1, 1.1]) and pl["ai"]["half"] == pytest.approx([0.495, 1.1, 0.11]) and pl["ai"]["conf"] == 0.9


def test_the_rendered_image_is_unchanged_because_every_projected_centre_and_extent_is(tmp_path):
    """No GPU: a splat's image is fixed by its centre and size seen from the camera, so project both before and after."""
    make_twin(tmp_path / "job", job=True)
    before = colmap_txt.read_model(tmp_path / "job" / "data" / "sparse" / "0")
    ply0 = read_ply(tmp_path / "job" / "scene.ply")
    rescale.rescale(tmp_path / "job", 1.1)
    after = colmap_txt.read_model(tmp_path / "job" / "data" / "sparse" / "0")
    ply1 = read_ply(tmp_path / "job" / "scene.ply")

    def project(model, ply, near, k=500.0):
        out = []
        for im in model.images:
            Rc = geometry.quat_to_mat(im.qvec)
            cam = ply[:, :3] @ Rc.T + im.tvec
            ok = cam[:, 2] > near
            uv = k * cam[ok, :2] / cam[ok, 2:3]
            extent = k * np.exp(ply[ok, 10:13]) / cam[ok, 2:3]  # a splat's pixel size along each of its axes
            out.append((ok, uv, extent))
        return out
    for (ok0, uv0, e0), (ok1, uv1, e1) in zip(project(before, ply0, 0.1), project(after, ply1, 0.11)):
        assert (ok0 == ok1).all() and ok0.sum() > 50
        assert np.abs(uv0 - uv1).max() < 1e-2 and np.abs(e0 - e1).max() / e0.max() < 1e-5


def test_a_rescale_then_its_inverse_returns_every_file_to_within_quantisation(tmp_path):
    make_twin(tmp_path / "job", job=True)
    start = snapshot(tmp_path / "job")
    rescale.rescale(tmp_path / "job", 1.1)
    rescale.rescale(tmp_path / "job", 1 / 1.1)
    end = snapshot(tmp_path / "job")
    assert set(start) == set(end) and not any(k.startswith(".rescale") for k in end)
    for name in ("scene.spz", "scene_lite.spz"):
        a, b = read_spz(tmp_path / "job" / name), None
        orig = start[name]
        (tmp_path / "orig.spz").write_bytes(orig)
        o = read_spz(tmp_path / "orig.spz")
        assert np.abs(a[0] - o[0]).max() <= 1.0 / (1 << FRAC) and (a[1] == o[1]).all() and a[2] == o[2]
    for name in ("scene.ply", "scene_clean.ply"):
        (tmp_path / "orig.ply").write_bytes(start[name])
        assert np.abs(read_ply(tmp_path / "job" / name) - read_ply(tmp_path / "orig.ply")).max() < 1e-5
    for name in ("shell.json", "places.json", "occupancy.json"):
        a, b = json.loads(end[name]), json.loads(start[name])
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) or _close(a, b), name
    s0, s1 = json.loads(start["scene.json"]), json.loads(end["scene.json"])
    for k in ("scale", "transform", "bounds", "cameraPath", "spawn", "tour", "topview"):
        assert _close(s1[k], s0[k]), k
    assert s1["scaleSource"] == "reference" and s1["reference"]["previousSource"] == "camera-height-prior" and s1["reference"]["factor"] == pytest.approx(1.0)
    assert start["topview.webp"] == end["topview.webp"] and start["occupancy.png"] == end["occupancy.png"]
    m0 = colmap_txt.read_model(tmp_path / "job" / "data" / "sparse" / "0")
    assert m0.num_points == 5


def _close(a, b, tol=1e-5):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= tol * max(1.0, abs(b))
    return a == b


def test_a_job_folders_sparse_model_the_dense_points_and_the_align_report_are_rescaled_too(tmp_path):
    make_twin(tmp_path / "job", job=True)
    m0 = colmap_txt.read_model(tmp_path / "job" / "data" / "sparse" / "0")
    c0, _, _ = m0.cameras()
    pts0 = np.load(tmp_path / "job" / "dense_points.npz")["points"]
    rescale.rescale(tmp_path / "job", 1.25)
    d = tmp_path / "job" / "data" / "sparse" / "0"
    m1 = colmap_txt.read_model(d)
    c1, _, f1 = m1.cameras()
    assert np.allclose(m1.xyz, m0.xyz * 1.25) and np.allclose(c1, c0 * 1.25) and np.allclose(f1, m0.cameras()[2])  # directions are kept
    assert m1.cameras_lines == m0.cameras_lines and (m1.error == m0.error).all()  # intrinsics and reprojection errors are pixels
    import pycolmap  # the binary model, which gsplat reads, says the same as the text one
    rec = pycolmap.Reconstruction(str(d))
    assert np.allclose(sorted(p.xyz[0] for p in rec.points3D.values()), sorted(m0.xyz[:, 0] * 1.25))
    z = np.load(tmp_path / "job" / "dense_points.npz")
    assert np.allclose(z["points"], pts0 * 1.25, atol=1e-5) and float(z["voxel"]) == pytest.approx(0.025) and z["normals"].dtype == np.float16
    rep = json.loads((tmp_path / "job" / "report.json").read_text())
    assert rep["align"]["scale"] == pytest.approx(0.37 * 1.25) and rep["align"]["scaleSource"] == "reference" and rep["sfm"] == {"images": 4}
    assert rep["align"]["bounds"]["max"] == pytest.approx([2.25 * 1.25, 2.75 * 1.25, 4.25 * 1.25]) and rep["align"]["yawDeg"] == 3.0
    assert json.loads((tmp_path / "job" / ".done" / "align.json").read_text())["scale"] == pytest.approx(0.37 * 1.25)
    assert not (tmp_path / "job" / rescale.STAGING).exists()


def test_scene_json_records_the_reference_and_moves_every_metric_field_but_not_the_eye_height(tmp_path):
    make_twin(tmp_path / "twin")
    rescale.rescale(tmp_path / "twin", 1.1, reference={"knownM": 2.2, "measuredM": 2.0})
    s = json.loads((tmp_path / "twin" / "scene.json").read_text())
    assert s["scaleSource"] == "reference" and s["scale"] == pytest.approx(0.407) and s["floorY"] == 0.0 and s["cameraHeight"] == 1.45
    assert s["reference"] == {"factor": 1.1, "appliedFactor": 1.1, "previousSource": "camera-height-prior", "previousScale": 0.37, "scale": pytest.approx(0.407),
                              "knownM": 2.2, "measuredM": 2.0}
    assert s["transform"][0] == pytest.approx([0.407, 0, 0, 1.1]) and s["transform"][3] == [0, 0, 0, 1]
    assert s["bounds"]["max"] == pytest.approx([2.475, 3.025, 4.675]) and s["cameraPath"][0] == pytest.approx([0.55, 1.54, 1.1])
    assert s["spawn"]["position"] == pytest.approx([0.66, 1.6, 1.32]) and s["spawn"]["lookAt"] == pytest.approx([1.76, 1.6, 1.32])
    assert s["tour"][0]["pos"] == pytest.approx([0.66, 2.0, 1.32]) and s["tour"][0]["fov"] == 62
    assert s["topview"]["origin"] == pytest.approx([-0.22, -0.33]) and s["topview"]["metersPerPixel"] == pytest.approx(0.022)
    assert s["topview"]["cutY"] == pytest.approx(2.64) and s["topview"]["width"] == 130
    rescale.rescale(tmp_path / "twin", 1.05)  # a second measurement compounds and keeps what the first replaced
    r = json.loads((tmp_path / "twin" / "scene.json").read_text())["reference"]
    assert r["factor"] == pytest.approx(1.155) and r["appliedFactor"] == 1.05 and r["previousSource"] == "camera-height-prior" and r["previousScale"] == 0.37


def test_a_run_that_stops_before_it_is_finished_leaves_the_original_and_one_that_stops_while_committing_is_finished_by_the_next(tmp_path, monkeypatch):
    make_twin(tmp_path / "job", job=True)
    start = snapshot(tmp_path / "job")

    def boom(*a):
        raise OSError("disk full")
    monkeypatch.setitem(rescale.HANDLERS, "shell.json", boom)
    with pytest.raises(OSError):
        rescale.rescale(tmp_path / "job", 1.1)
    assert snapshot(tmp_path / "job") == start  # nothing renamed, no staging left
    monkeypatch.undo()

    real = os.replace
    calls = {"n": 0}

    def flaky(a, b):
        if ".rescale" in str(a) and "new" in str(a):
            calls["n"] += 1
            if calls["n"] == 4:
                raise KeyboardInterrupt  # power cut in the middle of the renames
        return real(a, b)
    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(KeyboardInterrupt):
        rescale.rescale(tmp_path / "job", 1.1)
    monkeypatch.undo()
    half = json.loads((tmp_path / "job" / "scene.json").read_text())
    assert (tmp_path / "job" / rescale.STAGING / "commit.json").exists()  # the commit record says what is still to do
    assert rescale.recover(tmp_path / "job") == "finished the interrupted rescale"
    done = json.loads((tmp_path / "job" / "scene.json").read_text())
    assert done["scale"] == pytest.approx(0.37 * 1.1) and half["scale"] in (0.37, done["scale"])
    assert not (tmp_path / "job" / rescale.STAGING).exists()
    assert json.loads((tmp_path / "job" / "shell.json").read_text())["ceilingY"] == pytest.approx(2.75)
    assert colmap_txt.read_model(tmp_path / "job" / "data" / "sparse" / "0").xyz[0, 0] == pytest.approx(2.2)
    assert read_ply(tmp_path / "job" / "scene.ply")[0, 0] == pytest.approx(start_x(start) * 1.1, abs=1e-5)


def start_x(snap):
    raw = snap["scene.ply"]
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    return float(np.frombuffer(raw[end:end + 4], np.float32)[0])


def test_a_stale_staging_without_a_commit_record_is_dropped_and_a_factor_outside_the_range_needs_force(tmp_path):
    make_twin(tmp_path / "twin")
    stale = tmp_path / "twin" / rescale.STAGING / "new"
    stale.mkdir(parents=True)
    (stale / "scene.json").write_text("{}")
    assert rescale.recover(tmp_path / "twin") == "discarded an unfinished rescale"
    start = snapshot(tmp_path / "twin")
    for bad in (0.19, 5.01, 0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            rescale.rescale(tmp_path / "twin", bad)
    assert snapshot(tmp_path / "twin") == start
    rescale.rescale(tmp_path / "twin", 6.0, force=True)
    assert json.loads((tmp_path / "twin" / "scene.json").read_text())["scale"] == pytest.approx(0.37 * 6)
    with pytest.raises(FileNotFoundError):
        rescale.rescale(tmp_path / "empty", 1.1)


def test_the_command_takes_a_factor_or_a_known_and_a_measured_length_and_a_dry_run_writes_nothing(tmp_path, capsys):
    make_twin(tmp_path / "twin")
    start = snapshot(tmp_path / "twin")
    assert cli.main(["rescale", str(tmp_path / "twin"), "--known", "2.2", "--measured", "2.0", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["factor"] == pytest.approx(1.1) and "scene.spz" in out["files"] and snapshot(tmp_path / "twin") == start
    assert cli.main(["rescale", str(tmp_path / "twin"), "--known", "2.2", "--measured", "2.0"]) == 0
    s = json.loads((tmp_path / "twin" / "scene.json").read_text())
    assert s["reference"]["knownM"] == 2.2 and s["reference"]["measuredM"] == 2.0 and s["scale"] == pytest.approx(0.407)
    for argv in (["--factor", "1.1", "--known", "2"], ["--known", "2"], [], ["--factor", "9"]):
        with pytest.raises(SystemExit):
            cli.main(["rescale", str(tmp_path / "twin"), *argv])
    assert rescale.factor_from(2.2, 2.0) == pytest.approx(1.1) and rescale.measure([0, 0, 0], [3, 4, 0]) == 5.0
    with pytest.raises(ValueError):
        rescale.factor_from(0, 2.0)
