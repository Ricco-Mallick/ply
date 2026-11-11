# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Which mapper the camera solve starts with: ordered video of 400 frames or more skips the global mapper."""

from __future__ import annotations

import json
import types

import pytest

from plypipe import cli
from plypipe.log import Log
from plypipe.stages import sfm
from plypipe.util import Ctx


@pytest.mark.parametrize("source, n, expected", [
    ("video", 399, "global"),
    ("video", 400, "incremental"),
    ("videos", 1236, "incremental"),
    ("ar", 400, "incremental"),
    ("images", 5000, "global"),    # an unordered photo set has no sequence to anchor the incremental mapper
    (None, 5000, "global"),        # a job without a frames report
])
def test_mapper_by_source_and_frame_count(source, n, expected):
    assert sfm.choose_mapper(None, source, n)[0] == expected


def test_an_explicit_mapper_always_wins():
    assert sfm.choose_mapper("global", "videos", 1236) == ("global", "requested")
    assert sfm.choose_mapper("incremental", "images", 12) == ("incremental", "requested")


def test_the_threshold_is_an_option():
    assert sfm.choose_mapper(None, "video", 150, incremental_min=100)[0] == "incremental"
    assert sfm.choose_mapper(None, "video", 150, incremental_min=151)[0] == "global"


def test_the_reason_says_why():
    assert "1236 frames" in sfm.choose_mapper(None, "videos", 1236)[1]
    assert sfm.choose_mapper(None, "videos", 20)[1] == "default"


def test_the_cli_does_not_make_the_default_look_explicit():
    opts = vars(cli.build_parser().parse_args(["run", "in", "--out", "out"]))
    assert opts["mapper"] is None and opts["incremental_min"] == 400
    assert vars(cli.build_parser().parse_args(["run", "in", "--out", "out", "--mapper", "global"]))["mapper"] == "global"


def _solve(tmp_path, source, n, **opts):
    """Run _map against a pycolmap stub whose mappers return nothing: it stops right after the choice."""
    calls: list = []

    def mapping(name):
        def run(db, images, out, options):
            calls.append((name, options))
            return {}
        return run

    pycolmap = types.SimpleNamespace(
        incremental_mapping=mapping("incremental"), global_mapping=mapping("global"),
        IncrementalPipelineOptions=lambda: types.SimpleNamespace(mapper=types.SimpleNamespace(), max_runtime_seconds=-1),
        GlobalPipelineOptions=lambda: "global-options")
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts=opts)
    with pytest.raises(RuntimeError) as e:
        sfm._map(ctx, pycolmap, tmp_path / "colmap", tmp_path / "db", tmp_path / "images", n, source, "sift", "sift",
                 "sequential", True, None)
    lines = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    return calls, str(e.value), [line["msg"] for line in lines if "mapping (" in line.get("msg", "")]


def test_a_long_video_solve_starts_incremental_and_logs_why(tmp_path):
    calls, error, logged = _solve(tmp_path, "videos", 1236)
    assert [c[0] for c in calls] == ["incremental"] and error.startswith("incremental mapping produced no model")
    assert logged == ["incremental mapping (videos of 1236 frames, at least 400)"]
    assert calls[0][1].max_runtime_seconds == -1      # no other model to fall back on: no time cap by default


def test_a_short_video_still_starts_global(tmp_path):
    calls, _, logged = _solve(tmp_path, "video", 399)
    assert [c[0] for c in calls] == ["global"] and calls[0][1] == "global-options"
    assert logged == ["global mapping (default)"]


def test_an_explicit_global_on_a_long_video_is_respected(tmp_path):
    calls, _, logged = _solve(tmp_path, "video", 1236, mapper="global")
    assert [c[0] for c in calls] == ["global"] and logged == ["global mapping (requested)"]


def test_a_lower_threshold_and_an_explicit_time_cap_apply_to_the_direct_route(tmp_path):
    calls, _, _ = _solve(tmp_path, "video", 150, incremental_min=100, incremental_max_s=3600)
    assert [c[0] for c in calls] == ["incremental"] and calls[0][1].max_runtime_seconds == 3600
