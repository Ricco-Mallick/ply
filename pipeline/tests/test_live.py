"""Live view publisher: contact sheet of frames, training snapshots through the real splat-transform, loss curve."""
import json
import shutil
import time

import cv2
import numpy as np
import pytest

from plypipe.live import Live
from test_export import write_gs_ply  # synthetic Gaussian PLY


class Api:
    def __init__(self, root):
        self.root, self.puts = root, {}

    def upload(self, key, path, ctype):
        self.puts[key.rsplit("/", 1)[1]] = path.read_bytes()


def wait(cond, s=20):
    t = time.time()
    while time.time() - t < s and not cond():
        time.sleep(.1)
    return cond()


def test_frames_sheet_and_loss_curve(tmp_path):
    (tmp_path / "frames").mkdir()
    rng = np.random.default_rng(0)
    for i in range(37):
        cv2.imwrite(str(tmp_path / "frames" / f"{i:06d}.jpg"), rng.integers(0, 255, (300, 400, 3), np.uint8))
    api = Api(tmp_path)
    lv = Live(api, "j1", tmp_path)
    lv.stage_done("frames")
    assert wait(lambda: "live-frames.jpg" in api.puts and json.loads(api.puts.get("live-index.json", b"{}")).get("frames"))
    sheet = cv2.imdecode(np.frombuffer(api.puts["live-frames.jpg"], np.uint8), cv2.IMREAD_COLOR)
    idx = json.loads(api.puts["live-index.json"])
    assert idx["frames"]["count"] == 37 and sheet.shape[1] == 16 * 128 and idx["stages"]["frames"]["status"] == "done"
    for k in range(200):  # one point kept per 1 % of training
        lv.train_line(k / 200, f"loss=0.{900 - k:03d}| sh degree=3| {k}/200")
    assert 80 <= len(lv.index["loss"]) <= 101 and lv.index["loss"][0] == [0.0, 0.9]


@pytest.mark.skipif(not shutil.which("npx"), reason="needs node")
def test_snapshot_becomes_small_spz(tmp_path):
    ply = tmp_path / "train" / "ply" / "point_cloud_2999.ply"
    ply.parent.mkdir(parents=True)
    write_gs_ply(ply, np.random.default_rng(1).normal(size=(5000, 3)), np.random.default_rng(2))
    api = Api(tmp_path)
    lv = Live(api, "j1", tmp_path)
    lv.snapshot(ply)
    assert "live-train-2999.spz" in api.puts and api.puts["live-train-2999.spz"][:2] == b"\x1f\x8b"  # SPZ v3 = gzip
    assert json.loads(api.puts["live-index.json"])["snapshots"][0]["step"] == 2999
