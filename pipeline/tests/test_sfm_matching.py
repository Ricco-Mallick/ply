# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Pair selection without a GPU: which matcher each capture gets, and the retrieval tree's download guard."""

from __future__ import annotations

import hashlib
import types

from plypipe.stages import sfm


class Opts:
    def __init__(self, *a):
        self.args = a


def fake_pycolmap(calls: list):
    rec = lambda name: (lambda db, **kw: calls.append((name, kw)))
    return types.SimpleNamespace(
        FeatureMatcherType=types.SimpleNamespace(ALIKED_LIGHTGLUE="lg", SIFT_BRUTEFORCE="bf"),
        FeatureMatchingOptions=Opts, SequentialPairingOptions=Opts, VocabTreePairingOptions=Opts,
        match_exhaustive=rec("exhaustive"), match_sequential=rec("sequential"), match_vocabtree=rec("vocabtree"))


def test_video_beyond_the_exhaustive_limit_closes_loops(tmp_path):
    calls: list = []
    m = sfm._match(fake_pycolmap(calls), tmp_path / "db", "aliked", 1100, "auto", 700, None, tmp_path / "tree.bin")
    assert m == "sequential"
    pairing = calls[0][1]["pairing_options"]
    assert pairing.loop_detection and pairing.vocab_tree_path == str(tmp_path / "tree.bin") and pairing.overlap == 20


def test_video_without_a_tree_still_matches_sequentially(tmp_path):
    calls: list = []
    assert sfm._match(fake_pycolmap(calls), tmp_path / "db", "sift", 1100, "auto", 700, None, None) == "sequential"
    assert not hasattr(calls[0][1]["pairing_options"], "loop_detection")


def test_photos_are_exhaustive_then_retrieved(tmp_path):
    calls: list = []
    pc = fake_pycolmap(calls)
    assert sfm._match(pc, tmp_path / "db", "aliked", 263, "auto", 500, None, tmp_path / "t", unordered=True) == "exhaustive"
    assert sfm._match(pc, tmp_path / "db", "aliked", 900, "auto", 500, None, tmp_path / "t", unordered=True) == "vocabtree"
    assert [c[0] for c in calls] == ["exhaustive", "vocabtree"]


def test_vocab_tree_is_checked_and_cached(tmp_path, monkeypatch):
    payload = b"tree bytes"
    monkeypatch.setitem(sfm.VOCAB_TREES, "sift", ("https://example.invalid/t.bin", hashlib.sha256(payload).hexdigest()))
    fetched = []
    monkeypatch.setattr(sfm.urllib.request, "urlopen",
                        lambda url, timeout=0: fetched.append(url) or types.SimpleNamespace(read=lambda: payload))
    path = sfm.vocab_tree("sift", tmp_path)
    assert path.read_bytes() == payload and sfm.vocab_tree("sift", tmp_path) == path and len(fetched) == 1
    monkeypatch.setitem(sfm.VOCAB_TREES, "aliked", ("https://example.invalid/a.bin", "0" * 64))  # wrong checksum
    assert sfm.vocab_tree("aliked", tmp_path) is None and not (tmp_path / "a.bin").exists()


def test_a_failed_download_disables_loops_instead_of_failing(tmp_path, monkeypatch):
    def boom(url, timeout=0):
        raise OSError("TLS handshake failed")
    monkeypatch.setattr(sfm.urllib.request, "urlopen", boom)
    assert sfm.vocab_tree("sift", tmp_path) is None


def test_aliked_is_probed_in_a_throwaway_process(tmp_path):
    """The CPU and the cuda12 4.2.1 wheels both lack ONNX: the probe must say no without taking this process down."""
    import cv2
    import numpy as np
    (tmp_path / "f").mkdir()
    cv2.imwrite(str(tmp_path / "f" / "000000.jpg"), (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8))
    import pytest
    pytest.importorskip("pycolmap")
    assert sfm.aliked_available(tmp_path / "f", "cpu") is False   # and we are still running
    assert sfm.aliked_available(tmp_path / "empty", "cpu") is False


class FakeImage:
    def __init__(self, name, centre, points, frame_id):
        self.name, self._c, self.num_points3D, self.frame_id, self.has_pose = name, centre, points, frame_id, True

    def projection_center(self):
        return self._c


class FakeRec:
    def __init__(self, images):
        self.images = {i: im for i, im in enumerate(images)}

    def deregister_frame(self, frame_id):
        for im in self.images.values():
            if im.frame_id == frame_id:
                im.has_pose = False


def test_views_without_an_anchor_are_dropped():
    """The demo's global solve kept a 0-point view 4 million median distances away; gsplat diverged on it."""
    import numpy as np
    rng = np.random.default_rng(1)
    ims = [FakeImage(f"{i:06d}.jpg", rng.normal(0, 2, 3), 400, i) for i in range(30)]
    ims.append(FakeImage("000030.jpg", np.array([4e6, 0, 0]), 0, 30))          # unanchored and far away
    ims.append(FakeImage("000031.jpg", np.array([0.5, 0, 0.2]), 5, 31))       # in place, but pinned by 5 points
    ims.append(FakeImage("000032.jpg", np.array([300.0, 0, 0]), 350, 32))     # well observed, yet 100x out
    rec = FakeRec(ims)
    assert sorted(sfm.drop_unanchored(rec)) == ["000030.jpg", "000031.jpg", "000032.jpg"]
    assert sum(im.has_pose for im in rec.images.values()) == 30


def _walk(n=60, clip="a.mp4", start=0, offset=(0.0, 0.0, 0.0)):
    import numpy as np
    return [FakeImage(f"{start + i:06d}.jpg", np.array([0.1 * i, 0.0, 0.0]) + offset, 400, start + i) for i in range(n)]


def test_a_pose_out_and_back_between_neighbours_is_dropped():
    """The chapel's pew frames: one view 50 median steps off the walk, its two neighbours one step apart."""
    import numpy as np
    ims = _walk()
    ims[30]._c = np.array([3.0, 5.0, 0.0])
    rec = FakeRec(ims)
    assert sfm.trajectory_jumps(rec) == 2
    assert sfm.drop_spikes(rec) == ["000030.jpg"]
    assert not rec.images[30].has_pose and sfm.trajectory_jumps(rec) == 0


def test_a_real_jump_in_the_walk_is_not_a_spike():
    """A path that breaks and carries on elsewhere (a folded model) is a jump to report, not one frame to drop."""
    import numpy as np
    ims = _walk()
    for im in ims[30:]:
        im._c = im._c + np.array([0.0, 6.0, 0.0])
    rec = FakeRec(ims)
    assert sfm.drop_spikes(rec) == [] and sfm.trajectory_jumps(rec) == 1


def test_steps_between_clips_are_not_jumps():
    """The classroom's global solve reported 2 jumps that were clip boundaries; any jump sends it to the slow mapper."""
    ims = _walk(40, "a.mp4") + _walk(40, "b.mp4", start=40, offset=(-20.0, 3.0, 0.0))
    clips = {im.name: ("a.mp4" if int(im.name[:6]) < 40 else "b.mp4") for im in ims}
    rec = FakeRec(ims)
    assert sfm.trajectory_jumps(rec) == 1
    assert sfm.trajectory_jumps(rec, clip_of=clips) == 0
    assert sfm.drop_spikes(rec, clip_of=clips) == []


def test_clip_index_reads_the_frames_stage(tmp_path):
    import json
    (tmp_path / "frames_index.json").write_text(json.dumps([{"frame": "000000.jpg", "clip": "a.mp4", "t": 0.1}]))
    assert sfm.clip_index(tmp_path) == {"000000.jpg": "a.mp4"}
    assert sfm.clip_index(tmp_path / "missing") is None


def test_long_captures_get_the_lighter_incremental_schedule():
    import pytest
    pycolmap = pytest.importorskip("pycolmap")
    small, big = sfm.incremental_options(pycolmap, 300), sfm.incremental_options(pycolmap, 1236, 5400)
    assert small.ba_global_frames_ratio == pytest.approx(1.1) and not small.mapper.ba_global_ignore_redundant_points3D
    assert big.ba_global_frames_ratio == pytest.approx(1.25) and big.ba_global_points_ratio == pytest.approx(1.25)
    assert big.mapper.ba_global_ignore_redundant_points3D and big.max_runtime_seconds == 5400


def test_reuse_matches_skips_extraction_and_matching(tmp_path, monkeypatch):
    import json
    from plypipe.log import Log
    from plypipe.util import Ctx
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={"reuse_matches": True})
    ctx.frames_dir.mkdir(parents=True)
    for i in range(12):
        (ctx.frames_dir / f"{i:06d}.jpg").write_bytes(b"x")
    ctx.colmap_dir.mkdir(parents=True)
    (ctx.colmap_dir / "database.db").write_bytes(b"db")
    (ctx.colmap_dir / "sparse_all").mkdir()
    (ctx.colmap_dir / "matching.json").write_text(json.dumps({"features": "sift", "matcher": "sequential", "loopDetection": True}))
    seen = {}
    monkeypatch.setattr(sfm, "_pycolmap", lambda: types.SimpleNamespace(Device=lambda d: d))
    monkeypatch.setattr(sfm, "_extract_and_match", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-matched")))
    monkeypatch.setattr(sfm, "_map", lambda ctx, pc, cdir, db, images, n, source, features, requested, matcher, loops, cam:
                        seen.update(features=features, matcher=matcher, loops=loops, n=n) or {"ok": True})
    assert sfm.run_stage(ctx) == {"ok": True}
    assert seen == {"features": "sift", "matcher": "sequential", "loops": True, "n": 12}
    assert (ctx.colmap_dir / "database.db").exists() and not (ctx.colmap_dir / "sparse_all").exists()


def test_sfm_part_match_stops_before_mapping(tmp_path, monkeypatch):
    """The GPU half of a split solve: matches written, nothing mapped, the stage left open for the CPU half."""
    from plypipe import cli
    from plypipe.util import Ctx
    from plypipe.log import Log
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={})
    ctx.frames_dir.mkdir(parents=True)
    for i in range(12):
        (ctx.frames_dir / f"{i:06d}.jpg").write_bytes(b"x")
    (tmp_path / ".done").mkdir()
    (tmp_path / ".done" / "frames.json").write_text('{"source": "videos"}')
    monkeypatch.setattr(sfm, "_pycolmap", lambda: types.SimpleNamespace(Device=lambda d: d))
    monkeypatch.setattr(sfm, "_extract_and_match", lambda *a, **k: ("aliked", "lightglue", True))
    mapped = []
    monkeypatch.setattr(sfm, "_map", lambda *a, **k: mapped.append(1) or {"registered": 12, "images": 12})
    run = lambda *extra: cli.main(["run", str(tmp_path / "in"), "--out", str(tmp_path), "--only", "sfm", "--sfm-part", "match", "--quiet", *extra])
    assert run("--split-min", "10") == 0
    assert not mapped and not (tmp_path / ".done" / "sfm.json").exists()
    assert '"paused"' in (tmp_path / "log.jsonl").read_text()
    assert run() == 0                                   # 12 frames is under the default 400: solved on this box
    assert mapped and (tmp_path / ".done" / "sfm.json").exists()


def test_pairs_cover_clips_by_sequence_and_retrieval():
    import numpy as np
    from plypipe.stages import sfm_torch
    names = [f"{i:06d}.jpg" for i in range(60)]
    clips = {n: ("a" if i < 30 else "b") for i, n in enumerate(names)}
    seq = sfm_torch.make_pairs(names, clips, None, overlap=5, retrieve=0)
    assert (0, 5) in seq and (0, 6) not in seq and (0, 8) in seq and (0, 16) in seq   # next 5, then 2^k ahead
    assert all(clips[names[i]] == clips[names[j]] for i, j in seq)                      # never across clips
    desc = np.zeros((60, 4)); desc[:, 0] = 1; desc[5, :] = [0, 1, 0, 0]; desc[45, :] = [0, 1, 0, 0]
    both = sfm_torch.make_pairs(names, clips, desc, overlap=5, retrieve=1)
    assert (5, 45) in both                                                               # retrieval links the clips
    photos = sfm_torch.make_pairs(names[:10], None, None, ordered=False)
    assert len(photos) == 45                                                             # small photo set: exhaustive
