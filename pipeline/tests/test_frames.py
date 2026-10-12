import shutil

import cv2
import numpy as np
import pytest

from plypipe.log import Log
from plypipe.stages import frames
from plypipe.util import Ctx


def checker(seed: int, blur: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.random((60, 80)) > 0.5).astype(np.uint8) * 255
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def test_sharpness_orders_by_blur():
    s = [frames.sharpness(checker(0, b)) for b in (0, 1, 3, 6)]
    assert s == sorted(s, reverse=True)


def test_select_sharpest_per_window_and_gap_limit():
    scores = [1, 9, 2, 3,  5, 1, 1, 1,  0.1, 0.2, 0.1, 0.1,  0.1, 0.1, 0.2, 0.1,  0.1, 0.1, 0.1, 0.3,  7, 1, 1, 1]
    keep = frames.select_sharpest(scores, 4, rel_min=0.35, max_gap=2)
    assert keep[0] == 1 and keep[1] == 4
    # three blurry windows in a row: at most 2 may be dropped
    assert 19 in keep and 9 not in keep and 14 not in keep
    assert keep[-1] == 20


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_video_stage_keeps_sharp_frames(tmp_path):
    vid = tmp_path / "walk.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    sharp_ids = []
    for i in range(80):
        sharp = i % 4 == 2
        if sharp:
            sharp_ids.append(i)
        w.write(checker(i, 0 if sharp else 4))
    w.release()
    ctx = Ctx(out=tmp_path / "out", log=Log(tmp_path / "out", quiet=True), opts={"fps": 2.5, "oversample": 4, "long_side": 400})
    rep = frames.run_stage(ctx, vid)
    kept = sorted((tmp_path / "out" / "frames").glob("*.jpg"))
    assert 18 <= rep["kept"] == len(kept) <= 21
    img = cv2.imread(str(kept[0]))
    assert max(img.shape[:2]) == 400                       # long side capped
    s_kept = np.median([frames.sharpness(cv2.imread(str(p))) for p in kept])
    assert s_kept > 3 * frames.sharpness(checker(0, 4))    # sharp ones were chosen
