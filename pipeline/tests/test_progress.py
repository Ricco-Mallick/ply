"""SfM progress from COLMAP log lines (shapes copied from a real pycolmap 4.2.1 run)."""

from plypipe.progress import SfmProgress


def L(src: str, msg: str) -> str:
    return f"I20261013 07:20:21.222608 135433265709184 {src}.cc:208] {msg}\n"


def test_sfm_progress_walks_forward_through_the_steps():
    sp = SfmProgress(n_images=180)
    seen = [sp.feed(L("feature_extraction", "Processed file [90/180]")),
            sp.feed(L("pairing", "Processing block [1/4, 1/4]")),
            sp.feed(L("pairing", "Processing block [3/4, 2/4]")),
            sp.feed(L("global_mapper", "=== Running rotation averaging ===")),
            sp.feed(L("global_mapper", "Global bundle adjustment iteration 2 / 3 finished")),
            sp.feed(L("undistorters", "Undistorting image [180/180]"))]
    fr = [r[0] for r in seen]
    assert fr == sorted(fr) and fr[0] == 0.05 and fr[-1] == 1.0
    assert abs(fr[2] - (0.10 + 0.5 * 9 / 16)) < 1e-9  # block 3,2 of 4x4: 9 blocks done
    assert seen[2][1] == "matching frames · 56%" and seen[3][1] == "placing cameras · averaging rotations"


def test_unrelated_lines_and_json_are_ignored():
    sp = SfmProgress()
    assert sp.feed('{"stage": "sfm", "event": "progress"}') is None
    assert sp.feed(L("feature_extraction", "  Features:        6122 (SIFT)")) is None
    assert sp.feed("Creating SIFT GPU feature matcher") is None
    assert sp.feed(L("incremental_mapper", "Registering image #5 (num_reg_frames=40)")) is None  # n_images unknown
