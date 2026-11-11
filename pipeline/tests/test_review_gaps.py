"""Gaps the code review found: preview destinations, coverage-check live behaviour, export's phone decimation, tour
keyframes inside the room, per-image cameras when clip sizes differ."""
import json
import shutil

import cv2
import numpy as np
import pytest

from plypipe import tour
from plypipe.live import Live


def test_tour_stays_inside_the_room_and_below_lamps():
    cams = np.c_[np.linspace(-2, 2, 40), np.full(40, 1.45), np.sin(np.linspace(0, 6, 40))]
    b = {"min": [-3, -0.2, -2], "max": [3, 2.8, 2]}
    t = tour.build_tour(cams, b)
    assert len(t) == 8 and t[0]["dur"] == 0 and all(k["dur"] > 0 for k in t[1:])
    for k in t:
        x, y, z = k["pos"]
        assert -3 < x < 3 and -2 < z < 2 and 0.5 < y <= 2.3, k   # in the room, never up at the lamps


def test_coverage_check_jobs_skip_the_live_preview(tmp_path):
    class Api:
        puts = []
        def upload(self, key, path, ctype): self.puts.append(key)
    lv = Live(Api(), "j", tmp_path)
    lv.preview_only = True
    lv.stage_done("sfm"); lv.stage_done("align")
    import time; time.sleep(.5)
    assert not any("live-sparse" in k for k in Api.puts)


# splat-transform's decimation (-d) runs on WebGPU: a GPU, or on a GPU-less Linux box Mesa's software Vulkan
# (apt mesa-vulkan-drivers, lavapipe). Without an adapter it fails with "Unable to retrieve a WebGPU adapter".
@pytest.mark.skipif(not shutil.which("npx"), reason="needs node")
def test_export_decimates_the_phone_file(tmp_path, monkeypatch):
    from plypipe.log import Log
    from plypipe.stages import export
    from plypipe.util import Ctx, update_scene
    from test_export import write_gs_ply
    rng = np.random.default_rng(0)
    write_gs_ply(tmp_path / "scene.ply", rng.normal(size=(3000, 3)) * .5 + [0, 1, 0], rng)
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={"no_clean": True})
    update_scene(ctx, {"bounds": {"min": [-3, -1, -3], "max": [3, 3, 3]}})
    monkeypatch.setattr(export, "LITE_MAX", 1000)
    rep = export.run_stage(ctx)
    assert rep["gaussiansOut"] > 1000 and rep["liteGaussians"] == 1000 and rep["liteMB"] < rep["spzMB"]


def test_per_image_cameras_when_frame_sizes_differ(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    from plypipe.stages.sfm import _camera_mode
    for i, shape in enumerate([(90, 160, 3)] * 3):
        cv2.imwrite(str(tmp_path / f"{i:06d}.jpg"), np.zeros(shape, np.uint8))
    assert _camera_mode(pycolmap, tmp_path) == pycolmap.CameraMode.SINGLE
    cv2.imwrite(str(tmp_path / "000009.jpg"), np.zeros((120, 160, 3), np.uint8))
    assert _camera_mode(pycolmap, tmp_path) == pycolmap.CameraMode.PER_IMAGE
