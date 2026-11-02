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
    import pycolmap
    assert sfm.aliked_available(tmp_path / "f", "cpu") is False   # and we are still running
    assert sfm.aliked_available(tmp_path / "empty", "cpu") is False
