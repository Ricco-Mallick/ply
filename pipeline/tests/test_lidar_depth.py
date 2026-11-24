"""LiDAR depth as training supervision: sampling the maps (pure numpy) and the Dataset wrap, with a stand-in trainer."""
import json
import types

import cv2
import numpy as np
import pytest

from plypipe import lidar_depth


def test_sample_maps_pixel_centres_and_filters_confidence_and_range():
    d = np.full((192, 256), 2.0, np.float32)
    d[:, :10] = 0.0          # no return
    d[:10, :] = 9.0          # past the useful range
    conf = np.full((192, 256), 2, np.uint8)
    conf[100:, :] = 1        # medium confidence: not used
    pts, dep = lidar_depth.sample(d, conf, 1920, 1440, n=10_000, rng=np.random.default_rng(0))
    assert np.all(dep == 2.0)
    assert pts[:, 0].min() >= 10 * 7.5 - 0.5 and pts[:, 1].max() < 100 * 7.5   # columns 0-9 and rows 100+ are out
    assert pts[:, 1].min() >= 10 * 7.5 - 0.5
    one = lidar_depth.sample(np.full((4, 4), 1.0, np.float32), None, 8, 8, n=1)  # too few pixels: nothing
    assert one is None
    full = lidar_depth.sample(np.full((192, 256), 1.0, np.float32), None, 256, 192, n=100_000)
    assert np.allclose(sorted(set(full[0][:, 0].tolist()))[:2], [0.0, 1.0])     # the centre of pixel 0 is 0 at the same size


def test_sample_refuses_a_cropped_picture():
    assert lidar_depth.sample(np.full((192, 256), 1.0, np.float32), None, 1920, 1080) is None   # 16:9 vs 4:3


def test_install_swaps_sparse_points_for_lidar_samples(tmp_path):
    torch = pytest.importorskip("torch")
    (tmp_path / "lidar").mkdir()
    cv2.imwrite(str(tmp_path / "lidar" / "000007.png"), np.full((192, 256), 1500, np.uint16))
    cv2.imwrite(str(tmp_path / "lidar" / "000007.conf.png"), np.full((192, 256), 2, np.uint8))
    (tmp_path / "lidar.json").write_text(json.dumps({"frames": {"000007.jpg": {"depth": "000007.png", "confidence": "000007.conf.png"}}}))

    class Dataset:
        def __init__(self):
            self.parser = types.SimpleNamespace(image_names=["000007.jpg", "000008.jpg"]); self.indices = [0, 1]
        def __getitem__(self, i):
            return {"image": torch.zeros(480, 640, 3), "points": torch.zeros(5, 2), "depths": torch.ones(5)}
    st = types.SimpleNamespace(Dataset=Dataset)
    stats = lidar_depth.install(st, tmp_path / "lidar.json", log=lambda *_: None)
    ds = st.Dataset()
    a, b = ds[0], ds[1]
    assert a["points"].shape == (lidar_depth.SAMPLES, 2) and torch.allclose(a["depths"], torch.tensor(1.5))
    assert float(a["points"][:, 0].max()) < 640 and float(a["points"][:, 1].max()) < 480
    assert b["points"].shape == (5, 2)                     # no map for this frame: the sparse points stay
    assert stats["used"] == 1 and stats["fallback"] == 0
