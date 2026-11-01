"""Runs the real splat-transform through npx (skipped without Node or with PLY_SKIP_NPX=1)."""

import os
import shutil
import subprocess

import numpy as np
import pytest

from plypipe import geometry
from plypipe.log import Log
from plypipe.stages import export
from plypipe.util import Ctx, update_scene

pytestmark = pytest.mark.skipif(not shutil.which("npx") or os.environ.get("PLY_SKIP_NPX") == "1",
                                reason="needs Node/npx and network for splat-transform")

PROPS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)] + \
        ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def write_gs_ply(path, xyz, rng):
    n = xyz.shape[0]
    data = np.zeros((n, len(PROPS)), np.float32)
    data[:, 0:3] = xyz
    data[:, 6:9] = rng.normal(0, 0.5, (n, 3))
    data[:, 9:54] = rng.normal(0, 0.05, (n, 45))
    data[:, 54] = 2.0
    data[:, 55:58] = np.log(0.02)
    data[:, 58] = 1.0
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + \
             "".join(f"property float {p}\n" for p in PROPS) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def read_gs_ply_xyz(path):
    raw = open(path, "rb").read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode().splitlines()
    n = int(next(l for l in head if l.startswith("element vertex")).split()[-1])
    props = [l.split()[-1] for l in head if l.startswith("property")]
    types = [l.split()[1] for l in head if l.startswith("property")]
    assert set(types) == {"float"}, types
    arr = np.frombuffer(raw[end:end + n * 4 * len(props)], np.float32).reshape(n, len(props))
    return arr[:, [props.index("x"), props.index("y"), props.index("z")]].astype(np.float64)


def test_export_writes_full_and_lite_spz_and_crops(tmp_path):
    rng = np.random.default_rng(0)
    # the room as a model has it: surfaces (floor + one wall, 10 cm apart), not points scattered through the air,
    # which the floater pass rightly removes
    fx, fz = np.meshgrid(np.arange(0.05, 6, 0.1), np.arange(0.05, 4, 0.1))
    wy, wz = np.meshgrid(np.arange(0.15, 2.8, 0.1), np.arange(0.05, 4, 0.1))
    inside = np.r_[np.c_[fx.ravel(), np.zeros(fx.size), fz.ravel()], np.c_[np.full(wy.size, 0.05), wy.ravel(), wz.ravel()]]
    floaters = np.c_[rng.uniform(20, 30, 200), rng.uniform(5, 9, 200), rng.uniform(20, 30, 200)]
    write_gs_ply(tmp_path / "scene.ply", np.concatenate([inside, floaters]), rng)
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={})
    update_scene(ctx, {"bounds": {"min": [0, 0, 0], "max": [6, 2.8, 4]}})
    rep = export.run_stage(ctx)
    assert (tmp_path / "scene.spz").stat().st_size > 1000
    # the viewer (Spark 2.3.1) only reads SPZ v3, which is gzip-wrapped; v4 starts with the raw "NGSP" header
    import gzip, struct
    for name in ("scene.spz", "scene_lite.spz"):
        raw = (tmp_path / name).read_bytes()
        assert raw[:2] == b"\x1f\x8b", f"{name} is not gzip (SPZ v4?)"
        assert struct.unpack("<I", gzip.decompress(raw)[4:8])[0] == 3
    assert (tmp_path / "scene_lite.spz").stat().st_size < (tmp_path / "scene.spz").stat().st_size
    # round-trip the SPZ back to PLY: floaters cropped, positions unchanged (no axis flip)
    subprocess.run(["npx", "-y", export.SPLAT_TRANSFORM, "-w", str(tmp_path / "scene.spz"), str(tmp_path / "back.ply")],
                   check=True, capture_output=True)
    back = read_gs_ply_xyz(tmp_path / "back.ply")
    assert back.shape[0] == len(inside)
    assert rep["floaterFilter"] and rep["floaterClean"]["removedIsolated"] == 0
    assert np.allclose(np.sort(back[:, 1]), np.sort(inside[:, 1]), atol=0.01)


def test_align_actions_match_splat_transform_convention(tmp_path):
    rng = np.random.default_rng(1)
    xyz = rng.normal(0, 1, (500, 3))
    write_gs_ply(tmp_path / "in.ply", xyz, rng)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    T = np.eye(4); T[:3, :3] = 1.8 * geometry.quat_to_mat(q); T[:3, 3] = [0.5, -1.2, 3.0]
    subprocess.run(["npx", "-y", export.SPLAT_TRANSFORM, "-w", str(tmp_path / "in.ply"), *export.align_actions(T),
                    str(tmp_path / "out.ply")], check=True, capture_output=True)
    out = read_gs_ply_xyz(tmp_path / "out.ply")
    assert np.allclose(out, geometry.apply(T, xyz), atol=1e-3)
