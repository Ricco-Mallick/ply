# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""The top view: a synthetic room with a floor, two boxes of different size and a ceiling, seen from above."""
import gzip
import json
import struct

import cv2
import numpy as np
import pytest

from plypipe import topview
from plypipe.log import Log
from plypipe.stages import export
from plypipe.util import Ctx

FLOOR, WALL, CEILING = (0.8, 0.7, 0.5), (0.5, 0.5, 0.5), (0.1, 0.9, 0.1)  # RGB, 0..1
RED, BLUE, YELLOW = (0.9, 0.1, 0.1), (0.1, 0.1, 0.9), (0.95, 0.9, 0.1)
STEP = 0.05
C0 = 0.28209479177387814


def plane(axis, value, lo, hi, color, normal_scale=0.002, spacing=STEP):
    """Splats of a flat patch: `axis` is the normal's axis (0 x, 1 y, 2 z), `lo`/`hi` the other two axes' ranges."""
    others = [a for a in range(3) if a != axis]
    g = np.meshgrid(np.arange(lo[0] + spacing / 2, hi[0], spacing), np.arange(lo[1] + spacing / 2, hi[1], spacing))
    xyz = np.zeros((g[0].size, 3))
    xyz[:, others[0]], xyz[:, others[1]], xyz[:, axis] = g[0].ravel(), g[1].ravel(), value
    scale = np.full((len(xyz), 3), 0.03)
    scale[:, axis] = normal_scale
    return xyz, scale, np.tile(color, (len(xyz), 1))


def room(ceiling_y=2.6):
    """6 x 4 m, walls to 2.6 m, a red-topped box 1.0 x 0.5 m (x 1..2, z 0.5..1), a smaller blue-topped one 0.4 x 0.4 m
    at x 4.5..4.9, z 2.8..3.2, a yellow-topped one on the first box's other side, a green ceiling."""
    parts = [plane(1, 0.0, (0, 0), (6, 4), FLOOR),
             plane(2, 0.0, (0, 0), (6, 2.6), WALL), plane(2, 4.0, (0, 0), (6, 2.6), WALL),
             plane(0, 0.0, (0, 0), (2.6, 4), WALL), plane(0, 6.0, (0, 0), (2.6, 4), WALL),
             plane(1, 0.9, (1, 0.5), (2, 1), RED), plane(1, 0.5, (4.5, 2.8), (4.9, 3.2), BLUE),
             plane(1, 0.7, (3, 0.5), (3.3, 1.5), YELLOW),
             plane(1, ceiling_y, (0, 0), (6, 4), CEILING)]
    xyz, scale, rgb = (np.concatenate(p) for p in zip(*parts))
    return xyz, scale, rgb


def floaters(n=600, seed=0):
    rng = np.random.default_rng(seed)
    xyz = np.c_[rng.uniform(-60, 70, n), rng.uniform(0.2, 1.7, n), rng.uniform(-60, 70, n)]
    return xyz, np.full((n, 3), 0.03), np.tile((0.2, 0.2, 0.2), (n, 1))


def with_floaters(scene):
    return tuple(np.concatenate(p) for p in zip(scene, floaters()))


def yaw(deg):
    h = np.radians(deg) / 2
    return np.array([np.cos(h), 0, np.sin(h), 0])  # w, x, y, z


def write_ply(path, xyz, scale, rgb, alpha=0.9, quat=None):
    n = len(xyz)
    quat = np.tile([1.0, 0, 0, 0], (n, 1)) if quat is None else quat
    cols = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "f_rest_1", "f_rest_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros((n, len(cols)), np.float32)
    data[:, 0:3] = xyz
    data[:, 6:9] = (np.asarray(rgb) - 0.5) / C0
    data[:, 12] = np.log(alpha / (1 - alpha))
    data[:, 13:16] = np.log(scale)
    data[:, 16:20] = quat
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + \
             "".join(f"property float {c}\n" for c in cols) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def pack_quat(q):
    """(w, x, y, z) -> SPZ v3 'smallest three' (4 bytes)."""
    xyzw = np.array([q[1], q[2], q[3], q[0]], float)
    big = int(np.argmax(np.abs(xyzw)))
    xyzw = xyzw if xyzw[big] >= 0 else -xyzw
    comp = 0
    for j in range(4):
        if j != big:
            comp = (comp << 10) | (int(xyzw[j] < 0) << 9) | int(round(abs(xyzw[j]) / np.sqrt(0.5) * 511))
    return struct.pack("<I", comp | (big << 30))


def write_spz(path, xyz, scale, rgb, alpha=0.9, quat=None, version=3):
    n = len(xyz)
    quat = np.tile([1.0, 0, 0, 0], (n, 1)) if quat is None else quat
    frac = 12
    fixed = np.round(np.asarray(xyz) * (1 << frac)).astype(np.int64) & 0xFFFFFF
    pos = np.stack([(fixed >> 0) & 255, (fixed >> 8) & 255, (fixed >> 16) & 255], 2).astype(np.uint8)
    if version == 3:
        rot = b"".join(pack_quat(q) for q in quat)
    else:
        rot = b"".join(bytes(np.clip(np.round((q if q[0] >= 0 else -q)[1:] * 127.5 + 127.5), 0, 255).astype(np.uint8)) for q in quat)
    body = (pos.tobytes() + np.full(n, round(alpha * 255), np.uint8).tobytes()
            + np.clip(np.round(((np.asarray(rgb) - 0.5) / C0 * 0.15 + 0.5) * 255), 0, 255).astype(np.uint8).tobytes()
            + np.clip(np.round((np.log(scale) + 10) * 16), 0, 255).astype(np.uint8).tobytes() + rot)
    header = struct.pack("<IIIBBBB", topview.SPZ_MAGIC, version, n, 0, frac, 0, 0)
    path.write_bytes(gzip.compress(header + body, mtime=0))


def at(image, meta, x, z):
    """BGRA at a world point, through the metadata alone."""
    c = int((x - meta["origin"][0]) / meta["metersPerPixel"])
    r = int((z - meta["origin"][1]) / meta["metersPerPixel"])
    return image[r, c]


def rgb_of(bgra):
    return tuple(int(v) for v in bgra[[2, 1, 0]])


def near(bgra, expected, tol=14):
    return all(abs(a - round(b * 255)) <= tol for a, b in zip(rgb_of(bgra), expected))


@pytest.fixture(scope="module")
def drawn(tmp_path_factory):
    folder = tmp_path_factory.mktemp("room")
    scene = with_floaters(room())
    write_ply(folder / "scene.ply", *scene)
    image, meta = topview.render(topview.read_splats(folder / "scene.ply"))
    return folder, scene, image, meta


def test_box_is_where_the_metadata_says_and_not_mirrored(drawn):
    _, _, image, meta = drawn
    assert near(at(image, meta, 1.5, 0.75), RED)
    assert near(at(image, meta, 4.7, 3.0), BLUE)
    assert near(at(image, meta, 3.15, 1.0), YELLOW)
    # the same points mirrored left-right, top-bottom or transposed hold floor, not furniture
    for x, z in ((6 - 1.5, 0.75), (1.5, 4 - 0.75), (0.75, 1.5), (6 - 4.7, 3.0), (4.7, 4 - 3.0)):
        assert near(at(image, meta, x, z), FLOOR)


def test_box_has_its_size_and_orientation_in_pixels(drawn):
    _, _, image, meta = drawn
    b, g, r = (image[..., k].astype(int) for k in range(3))
    red = (r > 180) & (g < 80) & (b < 80) & (image[..., 3] == 255)
    rows, cols = np.flatnonzero(red.any(1)), np.flatnonzero(red.any(0))
    mpp = meta["metersPerPixel"]
    assert abs((cols[-1] - cols[0] + 1) * mpp - 1.0) < 0.12  # 1.0 m along x ...
    assert abs((rows[-1] - rows[0] + 1) * mpp - 0.5) < 0.12  # ... 0.5 m along z
    assert abs(meta["origin"][0] + (cols[0] + cols[-1] + 1) / 2 * mpp - 1.5) < 0.05
    assert abs(meta["origin"][1] + (rows[0] + rows[-1] + 1) / 2 * mpp - 0.75) < 0.05


def test_ceiling_is_cut_away_and_the_floor_shows(drawn):
    _, _, image, meta = drawn
    assert meta["cutY"] == pytest.approx(1.8)
    b, g, r = (image[..., k].astype(int) for k in range(3))
    inside = image[..., 3] == 255
    assert not ((g > r + 60) & (g > b + 60) & inside).any()  # nothing of the green ceiling
    assert near(at(image, meta, 3.0, 2.0), FLOOR)


def test_transparent_outside_and_floaters_do_not_stretch_the_image(drawn):
    _, _, image, meta = drawn
    mpp = meta["metersPerPixel"]
    assert image[0, 0, 3] == 0 and image[-1, -1, 3] == 0 and image[0, -1, 3] == 0 and image[-1, 0, 3] == 0
    assert 6 / mpp < meta["width"] < 7 / mpp and 4 / mpp < meta["height"] < 5 / mpp  # the room plus a margin, not 130 m
    assert abs(meta["origin"][0] - 0.0) < 0.4 and abs(meta["origin"][1] - 0.0) < 0.4
    assert meta["metersPerPixel"] == 0.02 and meta["splats"] == len(room()[0]) + 600


def test_a_low_room_is_cut_under_its_ceiling():
    xyz, scale, rgb = room(ceiling_y=2.2)
    s = topview._splats(xyz, scale, np.tile([1.0, 0, 0, 0], (len(xyz), 1)), np.full(len(xyz), 0.9), (rgb - 0.5) / C0)
    _, meta = topview.render(s)
    assert meta["cutY"] == pytest.approx(1.75, abs=0.06)


def test_cut_and_pixel_size_can_be_overridden(drawn):
    folder, _, _, _ = drawn
    image, meta = topview.render(topview.read_splats(folder / "scene.ply"), None, cut=0.3, mpp=0.05)
    assert meta["cutY"] == 0.3 and meta["metersPerPixel"] == 0.05
    assert near(at(image, meta, 1.5, 0.75), FLOOR)  # the box's top is above the cut; nothing hides the floor under it


def test_ply_and_spz_agree(drawn, tmp_path):
    _, scene, image, meta = drawn
    write_spz(tmp_path / "scene.spz", *scene)
    other, other_meta = topview.render(topview.read_splats(tmp_path / "scene.spz"))
    assert other_meta["origin"] == meta["origin"] and other_meta["cutY"] == meta["cutY"]
    assert abs(other_meta["width"] - meta["width"]) <= 1 and abs(other_meta["height"] - meta["height"]) <= 1
    h, w = min(image.shape[0], other.shape[0]), min(image.shape[1], other.shape[1])
    assert np.abs(other[:h, :w].astype(int) - image[:h, :w].astype(int)).mean() < 2.0


@pytest.mark.parametrize("version", [2, 3])
def test_spz_decodes_what_the_ply_holds(tmp_path, version):
    rng = np.random.default_rng(3)
    n = 400
    xyz = rng.uniform(-20, 20, (n, 3))
    scale = np.exp(rng.integers(-60, -25, (n, 3)) / 16.0)  # exact on the SPZ's log-scale grid
    quat = rng.normal(size=(n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    quat *= np.where(quat[:, :1] < 0, -1, 1)
    rgb = rng.uniform(0.1, 0.9, (n, 3))
    write_ply(tmp_path / "a.ply", xyz, scale, rgb, 0.5, quat)
    write_spz(tmp_path / "a.spz", xyz, scale, rgb, 0.5, quat, version)
    a, b = topview.read_ply(tmp_path / "a.ply"), topview.read_spz(tmp_path / "a.spz")
    for name in ("x", "y", "z"):
        assert np.abs(getattr(a, name) - getattr(b, name)).max() < 1 / 4096
    for name in ("sxx", "szz", "sxz", "syy"):
        assert np.abs(getattr(a, name) - getattr(b, name)).max() < 0.05 * max(getattr(a, "sxx").max(), 1e-9)
    assert np.abs(a.rgb - b.rgb).max() < 0.01
    assert np.abs(a.alpha - b.alpha).max() < 0.01


def test_a_rotated_splat_has_the_rotated_footprint(tmp_path):
    """A splat 20 cm long along x, turned 90° about y, lies along z in the top view."""
    write_ply(tmp_path / "r.ply", np.array([[0.0, 0.0, 0.0]]), np.array([[0.2, 0.01, 0.02]]), np.array([[0.5, 0.5, 0.5]]),
              quat=yaw(90)[None])
    s = topview.read_ply(tmp_path / "r.ply")
    assert s.sxx[0] == pytest.approx(0.02 ** 2, rel=1e-3) and s.szz[0] == pytest.approx(0.2 ** 2, rel=1e-3)
    assert abs(s.sxz[0]) < 1e-6


def test_cli_writes_the_image_and_the_scene_block(drawn, tmp_path, capsys):
    folder, scene, _, _ = drawn
    write_spz(tmp_path / "scene.spz", *scene)
    (tmp_path / "scene.json").write_text(json.dumps({"bounds": {"min": [0, -0.3, 0], "max": [6, 2.7, 4]}, "floorY": 0}))
    assert topview.main([str(tmp_path), "--write-scene"]) == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads((tmp_path / "scene.json").read_text())
    assert written["topview"] == printed and written["floorY"] == 0
    assert set(printed) == {"version", "image", "origin", "metersPerPixel", "width", "height", "cutY", "source", "splats"}
    assert printed["source"] == "scene.spz" and printed["version"] == 1
    image = cv2.imread(str(tmp_path / printed["image"]), cv2.IMREAD_UNCHANGED)
    assert image.shape == (printed["height"], printed["width"], 4) and image[0, 0, 3] == 0
    assert near(at(image, printed, 1.5, 0.75), RED, tol=20)


def test_export_records_the_top_view(drawn, tmp_path):
    folder, _, _, _ = drawn
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={})
    (tmp_path / "scene.json").write_text(json.dumps({"floorY": 0}))
    top = export._topview(ctx, folder / "scene.ply")
    assert top and (tmp_path / top["image"]).exists()
    assert json.loads((tmp_path / "scene.json").read_text())["topview"] == top


def test_export_survives_a_failing_top_view(drawn, tmp_path, monkeypatch):
    folder, _, _, _ = drawn
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={})
    (tmp_path / "scene.json").write_text(json.dumps({"floorY": 0}))

    def boom(*a, **k):
        raise RuntimeError("no room found in the splats")
    monkeypatch.setattr(export.topview_mod, "build", boom)
    assert export._topview(ctx, folder / "scene.ply") is None
    assert "topview" not in json.loads((tmp_path / "scene.json").read_text())
    warnings = [json.loads(ln) for ln in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any(w["event"] == "warn" and "top view skipped" in w["msg"] for w in warnings)
    assert export._topview(ctx, tmp_path / "missing.ply") is None  # and so does an unreadable source
