import json
import shutil

import json
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


def test_folder_of_clips_numbers_frames_continuously(tmp_path):
    """The capture app uploads several clips of one space plus capture.json; frames run on straight through."""
    src = tmp_path / "capture"
    src.mkdir()
    for c, n in (("clip-01.mp4", 40), ("clip-02.mp4", 24)):
        w = cv2.VideoWriter(str(src / c), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
        for i in range(n):
            w.write(checker(i + n, 0 if i % 4 == 2 else 4))
        w.release()
    (src / "capture.json").write_text('{"version": 1}')
    ctx = Ctx(out=tmp_path / "out", log=Log(tmp_path / "out", quiet=True), opts={"fps": 2.5, "oversample": 4, "long_side": 400})
    rep = frames.run_stage(ctx, src)
    kept = sorted((tmp_path / "out" / "frames").glob("*.jpg"))
    assert rep["source"] == "videos" and [c["file"] for c in rep["clips"]] == ["clip-01.mp4", "clip-02.mp4"]
    assert rep["kept"] == len(kept) == sum(c["kept"] for c in rep["clips"])
    assert [p.stem for p in kept] == [f"{i:06d}" for i in range(len(kept))]    # no gaps, no overwrites
    assert rep["clips"][1]["firstFrame"] == rep["clips"][0]["kept"]
    assert (tmp_path / "out" / "capture.json").exists()


def test_spans_marked_for_replacing_are_skipped_and_frames_are_indexed(tmp_path):
    """The app can mark a stretch of a clip as replaced by a new clip (capture.json clips[].exclude, seconds)."""
    src = tmp_path / "capture"
    src.mkdir()
    w = cv2.VideoWriter(str(src / "clip-01.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for i in range(80):  # 8 s
        w.write(checker(i, 0 if i % 4 == 2 else 4))
    w.release()
    (src / "capture.json").write_text('{"version": 1, "clips": [{"file": "clip-01.mp4", "exclude": [[2.0, 5.0]]}]}')
    ctx = Ctx(out=tmp_path / "out", log=Log(tmp_path / "out", quiet=True), opts={"fps": 2.5, "oversample": 4, "long_side": 400})
    rep = frames.run_stage(ctx, src)
    idx = json.loads((tmp_path / "out" / "frames_index.json").read_text())
    assert rep["excluded"] >= 6 and len(idx) == rep["kept"]
    assert all(not (2.0 <= e["t"] <= 5.0) for e in idx) and all(e["clip"] == "clip-01.mp4" for e in idx)
    assert [e["frame"] for e in idx] == [f"{i:06d}.jpg" for i in range(len(idx))]


def test_gpu_decoding_only_where_nvdec_decodes_the_clip(tmp_path, monkeypatch):
    import subprocess
    import types
    from plypipe.stages import frames as fr
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"")
    monkeypatch.setattr(fr, "_HWDEC", {})
    monkeypatch.setattr(fr.shutil, "which", lambda name: None)
    assert fr.decode_args(clip) == []                       # no NVIDIA driver: software, nothing probed
    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd[0])
        if cmd[0] == "ffprobe":
            return types.SimpleNamespace(stdout="hevc\n", returncode=0)
        return types.SimpleNamespace(returncode=0 if probe_ok else 1)
    monkeypatch.setattr(fr.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(subprocess, "run", fake_run)
    probe_ok = True
    assert fr.decode_args(clip) == ["-hwaccel", "cuda"] and fr.decode_args(clip) == ["-hwaccel", "cuda"]
    assert calls.count("ffmpeg") == 1                       # probed once per codec
    monkeypatch.setattr(fr, "_HWDEC", {})
    probe_ok = False
    assert fr.decode_args(clip) == []                       # NVDEC refused the codec: software
    monkeypatch.setenv("PLY_NO_HWDEC", "1")
    assert fr.decode_args(clip) == []


def test_a_clip_soft_throughout_is_reported():
    """The chapel's 0053: 27% of the sharpest clip's sharpness, nearly every frame kept against its own median."""
    from plypipe.stages import frames as fr
    clips = [{"file": "0051.MP4", "sharpnessMedian": 1957.0}, {"file": "0053.MP4", "sharpnessMedian": 526.0},
             {"file": "0064.MP4", "sharpnessMedian": 1450.0}]
    assert fr.blurry_clips(clips) == [{"file": "0053.MP4", "ratio": 0.27}]
    assert fr.blurry_clips([{"file": "a", "sharpnessMedian": 0.0}]) == []


def tagged_clip(path, trc=None, primaries=None, space=None, pix_fmt="yuv420p", seconds=3):
    """A tiny clip with ffmpeg's test pattern, tagged the way a phone tags its HDR or SDR recordings."""
    import subprocess
    tags = [x for flag, v in (("-color_trc", trc), ("-color_primaries", primaries), ("-colorspace", space)) if v for x in (flag, v)]
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={seconds}",
                    "-c:v", "libx264", "-pix_fmt", pix_fmt, *tags, str(path)], check=True)
    return path


def warnings(out) -> list[str]:
    lines = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    return [e["msg"] for e in lines if e["stage"] == "frames" and e["event"] == "warn"]


needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed")


@needs_ffmpeg
def test_probe_reads_each_clips_colour_tags(tmp_path):
    hlg = tagged_clip(tmp_path / "hlg.mp4", "arib-std-b67", "bt2020", "bt2020nc", "yuv420p10le")
    sdr = tagged_clip(tmp_path / "sdr.mp4", "bt709", "bt709", "bt709")
    assert frames.probe_video(hlg)["color"] == {"transfer": "arib-std-b67", "primaries": "bt2020", "space": "bt2020nc",
                                                "pixFmt": "yuv420p10le", "bitDepth": 10, "hdr": "HLG"}
    assert frames.probe_video(sdr)["color"] == {"transfer": "bt709", "primaries": "bt709", "space": "bt709",
                                                "pixFmt": "yuv420p", "bitDepth": 8, "hdr": None}
    assert frames.probe_video(tagged_clip(tmp_path / "pq.mp4", "smpte2084", "bt2020", "bt2020nc"))["color"]["hdr"] == "PQ"
    assert frames.probe_video(sdr)["durationSec"] and frames.probe_video(sdr)["longSide"] == 160   # the plan's probe is unchanged


@needs_ffmpeg
def test_hdr_clip_is_recorded_and_warned_about_without_changing_the_frames(tmp_path):
    src = tmp_path / "capture"
    src.mkdir()
    tagged_clip(src / "clip-01.mp4", "bt709", "bt709", "bt709")
    tagged_clip(src / "clip-02.mp4", "arib-std-b67", "bt2020", "bt2020nc", "yuv420p10le")
    ctx = Ctx(out=tmp_path / "out", log=Log(tmp_path / "out", quiet=True), opts={"fps": 2.5, "oversample": 2, "long_side": 160})
    rep = frames.run_stage(ctx, src)
    colors = {c["file"]: c["color"] for c in rep["clips"]}
    assert colors["clip-01.mp4"]["transfer"] == "bt709" and colors["clip-01.mp4"]["hdr"] is None
    assert colors["clip-02.mp4"] == {"transfer": "arib-std-b67", "primaries": "bt2020", "space": "bt2020nc",
                                     "pixFmt": "yuv420p10le", "bitDepth": 10, "hdr": "HLG"}
    assert rep["hdrClips"] == ["clip-02.mp4"] and rep["colorTransfers"] == {"bt709": 1, "arib-std-b67": 1}
    warned = warnings(tmp_path / "out")
    assert len(warned) == 1 and "clip-02.mp4 (HLG)" in warned[0] and "clip-01.mp4" not in warned[0]
    assert "washed out" in warned[0] and "not applied yet" in warned[0]
    assert rep["kept"] == len(list((tmp_path / "out" / "frames").glob("*.jpg")))      # extraction itself is untouched


@needs_ffmpeg
def test_sdr_and_untagged_clips_do_not_warn_and_a_single_clip_reports_its_colours(tmp_path):
    sdr = tagged_clip(tmp_path / "sdr.mp4", "bt709", "bt709", "bt709")
    ctx = Ctx(out=tmp_path / "o1", log=Log(tmp_path / "o1", quiet=True), opts={"fps": 2.5, "oversample": 2, "long_side": 160})
    rep = frames.run_stage(ctx, sdr)
    assert rep["source"] == "video" and rep["color"]["transfer"] == "bt709" and rep["hdrClips"] == []
    assert warnings(tmp_path / "o1") == []
    bare = tagged_clip(tmp_path / "bare.mp4")                   # no colour tags at all: the stage must still run
    ctx = Ctx(out=tmp_path / "o2", log=Log(tmp_path / "o2", quiet=True), opts={"fps": 2.5, "oversample": 2, "long_side": 160})
    rep = frames.run_stage(ctx, bare)
    assert rep["color"]["hdr"] is None and rep["kept"] > 0 and rep["hdrClips"] == []
    assert rep["colorTransfers"] == {(rep["color"]["transfer"] or "unknown"): 1}
    assert warnings(tmp_path / "o2") == []


def test_colour_summary_and_missing_probe_tags():
    assert frames._color_of({}) == {"transfer": None, "primaries": None, "space": None, "pixFmt": None, "bitDepth": None, "hdr": None}
    assert frames._color_of({"color_transfer": "unknown", "pix_fmt": "yuv420p", "bits_per_raw_sample": "10"})["bitDepth"] == 10
    clips = [{"file": "a", "color": frames._color_of({"color_transfer": "smpte2084"})}, {"file": "b", "color": frames._color_of({})},
             {"file": "c", "color": frames._color_of({"color_transfer": "unknown"})}]
    assert frames.color_summary(clips) == {"colorTransfers": {"smpte2084": 1, "unknown": 2}, "hdrClips": ["a"]}
