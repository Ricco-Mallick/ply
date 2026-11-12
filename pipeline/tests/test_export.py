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
    assert rep["shRestore"]["full"]["restored"] and rep["shRestore"]["full"]["shDegree"] == 3
    assert rep["shRestore"]["lite"]["restored"] and rep["shRestore"]["lite"]["shDegree"] == export.LITE_SH
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


def test_room_crop_keeps_what_the_training_views_draw(tmp_path):
    """The chapel: the box cut its vault, window views and floor reflections. Outside splats the views draw stay."""
    from plypipe import clean
    rng = np.random.default_rng(3)
    inside = rng.uniform([0, 0, 0], [6, 2.8, 4], (300, 3))
    vault = rng.uniform([1, 4.0, 1], [5, 4.5, 3], (40, 3))           # above the box top (2.5 + 0.5), seen by every view
    junk = rng.uniform([20, 5, 20], [30, 9, 30], (60, 3))             # far outside, seen by none
    write_gs_ply(tmp_path / "scene.ply", np.concatenate([inside, vault, junk]), rng)
    lo, hi = export.room_bounds({"bounds": {"min": [0, 0, 0], "max": [6, 2.5, 4]}})
    weights = np.r_[np.full(300, 50.0), np.full(40, 30.0), np.full(60, 0.2)]
    rep = export.needed_crop(tmp_path / "scene.ply", tmp_path / "crop.ply", lo, hi, weights)
    assert rep == {"method": "needed", "outside": 100, "dropped": 60, "keptOutside": 40, "minPx": export.NEEDED_PX}
    header, c = clean.read_ply(tmp_path / "crop.ply")
    assert len(c["x"]) == 340 and c["y"].max() > 4.0 and c["x"].max() < 7
    assert np.allclose(c["f_rest_7"], clean.read_ply(tmp_path / "scene.ply")[1]["f_rest_7"][:340])
    # splat-transform reads the file the crop writes
    subprocess.run(["npx", "-y", export.SPLAT_TRANSFORM, "-w", str(tmp_path / "crop.ply"), str(tmp_path / "back.ply")],
                   check=True, capture_output=True)
    assert read_gs_ply_xyz(tmp_path / "back.ply").shape[0] == 340
    import pytest
    with pytest.raises(ValueError):
        export.needed_crop(tmp_path / "scene.ply", tmp_path / "x.ply", lo, hi, weights[:10])


def test_training_weights_need_a_gpu(tmp_path):
    from plypipe import viewer_eval
    assert viewer_eval.training_weights(tmp_path / "missing.ply", tmp_path) is None


def test_the_phone_file_is_packed_from_the_fine_tuned_model(tmp_path, monkeypatch):
    """The real splat-transform around stand-ins for the GPU work: the twin is cut to the phone size, the 'trainer'
    lifts those splats 1 m, and scene_lite.spz holds the lifted ones with their SH restored."""
    from plypipe import clean, viewer_eval
    from plypipe.stages import train as train_stage
    rng = np.random.default_rng(0)
    write_gs_ply(tmp_path / "scene.ply", rng.normal(size=(3000, 3)) * .5 + [0, 1, 0], rng)
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={"no_clean": True, "lite_max": 1000, "lite_finetune": 300})
    update_scene(ctx, {"bounds": {"min": [-3, -1, -3], "max": [3, 3, 3]}})
    scores = {"scene_lite.ply": 24.0, "point_cloud_299.ply": 25.5}
    monkeypatch.setattr(viewer_eval, "evaluate", lambda ply, *a, **k: {"views": 4, "psnr": scores[ply.name], "ssim": 0.8, "brightness": 1.0,
                                                                       "holes": 0.01, "worstHoles": 0.02} if ply.name in scores else None)
    seen = {}

    def trained(ctx_, ply, steps, sh_degree):
        header, c = clean.read_ply(ply)
        seen.update(steps=steps, sh=sh_degree, splats=len(c["x"]), coefficients=sum(k.startswith("f_rest_") for k in c) // 3)
        c["y"] += 1.0
        out = ctx_.out / "train-lite" / "ply" / "point_cloud_299.ply"
        out.parent.mkdir(parents=True)
        out.write_bytes(header.encode("latin-1") + np.stack(list(c.values()), 1).astype("<f4").tobytes())
        return {"ply": out, "eval": {"psnr": 26.0, "views": 33}}

    monkeypatch.setattr(train_stage, "finetune", trained)
    rep = export.run_stage(ctx)
    assert seen == {"steps": 300, "sh": export.LITE_SH, "splats": 1000, "coefficients": 8}       # it trains the phone model: 1000 splats, SH 2
    tuned = rep["liteFinetune"]
    assert tuned["shipped"] == "finetuned" and (tuned["before"]["psnr"], tuned["after"]["psnr"]) == (24.0, 25.5) and tuned["eval"]["views"] == 33
    assert rep["gaussiansOut"] > 1000 and rep["liteGaussians"] == 1000 and rep["liteMB"] < rep["spzMB"]
    assert rep["shRestore"]["lite"]["restored"] and rep["shRestore"]["lite"]["shDegree"] == export.LITE_SH
    assert not (tmp_path / "train-lite").exists() and not (tmp_path / "scene_lite.ply").exists()
    back = {}
    for name in ("scene.spz", "scene_lite.spz"):
        subprocess.run(["npx", "-y", export.SPLAT_TRANSFORM, "-w", str(tmp_path / name), str(tmp_path / "back.ply")], check=True, capture_output=True)
        back[name] = read_gs_ply_xyz(tmp_path / "back.ply")
    assert len(back["scene_lite.spz"]) == 1000 and back["scene_lite.spz"][:, 1].mean() == pytest.approx(back["scene.spz"][:, 1].mean() + 1.0, abs=0.15)
