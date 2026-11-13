# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Measuring a camera-solve variant (modal/sfm_cpu.py) without Modal: budget and stop-loss, the result record and the
table, and the Modal function itself run locally on the synthetic scene, with a folder standing in for the Volume."""

import importlib.util
import json
import shutil
import types
from pathlib import Path

import pytest

from plypipe import evalrun, solvebench
from test_sfm_plan import synthetic_scene  # noqa: F401  (the fixture)

pytest.importorskip("modal")
pytest.importorskip("modal.volume", reason="no modal client: `import modal` finds only the pipeline/modal folder")

SCRIPT = Path(__file__).resolve().parents[1] / "modal" / "sfm_cpu.py"


def load(monkeypatch, **env):
    for k in ("PLY_SFM_CPU", "PLY_SFM_VARIANT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location("sfm_cpu_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_budget_is_the_plans_rule_at_modals_cpu_price():
    assert solvebench.hourly(8) == pytest.approx(0.632) and solvebench.hourly(16) == pytest.approx(1.264)
    for v in solvebench.VARIANTS:
        for c in (8, 16):
            assert solvebench.solve_deadline_min(v, c) == pytest.approx(1.5 * solvebench.estimate_min(v, c), abs=1)
            assert solvebench.timeout_min(v, c) == solvebench.solve_deadline_min(v, c) + solvebench.GRACE_MIN
    assert solvebench.cost_usd(58.8, 8) == 0.62  # the incremental solve's measured 58.8 min (docs/IMPROVEMENT.md: $0.68 billed)
    assert "stopped at 68 min" in solvebench.describe("keyframes", 8)


def test_an_unknown_variant_or_core_count_is_refused_before_anything_is_launched():
    for v, c in (("global", 8), ("keyframes", 12)):
        with pytest.raises(ValueError, match="unknown variant/cores"):
            solvebench.estimate_min(v, c)


def test_the_modal_function_is_sized_and_stopped_for_its_variant(monkeypatch):
    for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODAL_CONFIG_PATH", "/nonexistent/modal.toml")
    m = load(monkeypatch, PLY_SFM_VARIANT="hierarchical", PLY_SFM_CPU="16")
    assert m.app.name == "ply-sfm-cpu" and callable(m.main) and callable(m.measure.local) and callable(m.solve.local)
    original = next(v for k, v in vars(m.measure).items() if k.startswith("_sync_original"))
    info = getattr(original, "_function_info", None)
    if info is not None:  # modal 1.6.x's layout
        assert info.cpu == 16 and info.memory_mib == 64 * 1024 and info.timeout == 55 * 60
    with pytest.raises(ValueError):
        load(monkeypatch, PLY_SFM_CPU="12")


def test_the_table_has_a_row_per_variant_and_the_time_against_the_incremental_solve():
    base = {"cores": 8, "mapper": "incremental", "variant": "incremental", "images": 1236, "registered": 1223, "meanReprojErrorPx": 0.582,
            "tracks": 431000, "costUsd": 0.63, "seconds": {"solve": 3600.0}, "ateCm": {"rmse": 0.0, "median": 0.0}}
    fast = {**base, "variant": "keyframes", "mapper": "keyframes", "registered": 1220, "costUsd": 0.4, "seconds": {"solve": 1800.0},
            "ateCm": {"rmse": 1.4, "median": 1.1}, "mapperFallback": {"used": "incremental"}}
    lines = solvebench.table({"incremental-8": base, "keyframes-8": fast}).splitlines()
    assert lines[0].split()[:4] == ["variant", "cores", "solve", "min"] and len(lines) == 3
    assert "keyframes" in lines[1] and "0.50" in lines[1] and "1.40" in lines[1] and lines[1].rstrip().endswith("incremental")
    assert "incremental" in lines[2] and "1.00" in lines[2] and "1223/1236" in lines[2]


def test_collect_with_nothing_saved_yet_is_an_empty_table_not_a_crash(tmp_path):
    for name in ("FileNotFoundError", "NotFoundError"):
        def listdir(root, name=name):
            raise type(name, (Exception,), {})(f"{root}: not found")
        assert solvebench.collect(types.SimpleNamespace(listdir=listdir), tmp_path) == {}
    def broken(root):
        raise ConnectionError("the Volume is unreachable")
    with pytest.raises(ConnectionError):
        solvebench.collect(types.SimpleNamespace(listdir=broken), tmp_path)


def test_collect_copies_each_variants_result_and_skips_the_unfinished(tmp_path):
    files = {"chapel/solve/keyframes-8/result.json": json.dumps({"variant": "keyframes", "cores": 8})}

    def read_file(path):
        if path not in files:
            raise FileNotFoundError(path)
        yield files[path].encode()

    volume = types.SimpleNamespace(listdir=lambda root: [types.SimpleNamespace(path=f"{root}/{n}") for n in ("keyframes-8", "hierarchical-8")],
                                   read_file=read_file)
    got = solvebench.collect(volume, tmp_path)
    assert list(got) == ["keyframes-8"] and (tmp_path / "keyframes-8" / "result.json").exists()


@pytest.mark.slow
def test_a_variant_is_solved_scored_saved_and_usable_by_the_eval_runner(monkeypatch, tmp_path, synthetic_scene):
    m = load(monkeypatch)
    vol = tmp_path / "vol"
    shutil.copytree(synthetic_scene, vol / "scene" / "job", ignore=shutil.ignore_patterns("sfm.json", "sparse*", "dense", "log.jsonl"))
    monkeypatch.setattr(m, "VOL", vol)
    monkeypatch.setattr(m, "JOB", tmp_path / "work" / "job")
    monkeypatch.setattr(m, "vol", types.SimpleNamespace(commit=lambda: None))

    def run(variant, **kw):
        job = solvebench.make_job(variant, 8, "scene/job", reference=kw.get("reference", ""), extra_args="--features sift --matcher sequential")
        return m.measure.local(job)

    first = run("incremental", reference="scene/job/colmap/none")      # no reference yet: the solve is kept, the error says what is missing
    assert first["registered"] == first["images"] == 60 and first["ateCm"] is None and first["error"].startswith("ATE not scored")
    reference = f"{solvebench.run_dir('incremental', 8)}/job/data/sparse/0"
    second = run("keyframes", reference=reference)
    assert second["mapper"] == "keyframes" and second["registered"] == 60 and second["keyframes"] < 60 and second["error"] is None
    assert second["ateCm"]["matched"] == 60 and second["ateCm"]["rmse"] < 5            # centimetres, against the aligned reference
    assert set(second["seconds"]["mapperParts"]) == {"keyframes", "register", "bundle"} and second["seconds"]["wall"] > 0
    saved = vol / solvebench.run_dir("keyframes", 8)
    assert json.loads((saved / "result.json").read_text())["variant"] == "keyframes"
    assert evalrun.check_job(saved / "job") == []                                      # train and export can start from it as it is
    assert not (saved / "job" / "colmap" / "database.db").exists() and not (saved / "job" / "frames").exists()


BA_SCRIPT = Path(__file__).resolve().parents[1] / "modal" / "sfm_gpu_ba_check.py"


def load_ba_check():
    spec = importlib.util.spec_from_file_location("sfm_gpu_ba_check_under_test", BA_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gpu_bundle_adjustment_check_loads_and_probes_the_options_on_the_cpu_build(monkeypatch):
    pycolmap = pytest.importorskip("pycolmap")
    for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODAL_CONFIG_PATH", "/nonexistent/modal.toml")
    m = load_ba_check()
    assert m.app.name == "ply-sfm-ba-check" and callable(m.run.local) and callable(m.main)
    info = m.probe(pycolmap)
    assert info["ceresUseGpu"] and info["baUseGpu"] and info["caspar"] and info["backends"] == ["CASPAR", "CERES"] and info["hasCuda"] is False
    assert m.run_solver("ceres-cpu", Path("/nonexistent/model"))["ran"] is False   # a solver that dies is a row, not an exception


@pytest.mark.slow
def test_the_check_times_each_solver_from_the_same_start_and_reports_the_one_that_cannot_run(tmp_path, synthetic_scene):
    from test_sfm_plan import _resolve
    pytest.importorskip("pycolmap")
    _resolve(synthetic_scene, tmp_path, "--mapper", "incremental")
    m = load_ba_check()
    got = m.check(tmp_path / "job" / "colmap" / "sparse" / "0", tmp_path / "work", frames=30)
    cpu = got["solvers"]["ceres-cpu"]
    assert got["start"]["frames"] == 30 and cpu["ran"] and cpu["finalCost"] < 0.01 * cpu["initialCost"]
    assert cpu["meanErrorPxAfter"] < got["start"]["meanErrorPx"] / 5
    assert got["solvers"]["ceres-gpu"]["initialCost"] == pytest.approx(cpu["initialCost"])          # one start for every solver
    assert got["solvers"]["caspar"]["ran"] is False and "CASPAR" in got["solvers"]["caspar"]["why"]   # this wheel has no CASPAR
    assert "caspar" not in got["ran"] and json.dumps(got)


MATCH_SCRIPT = Path(__file__).resolve().parents[1] / "modal" / "sfm_match.py"
SAVED_JOB = {"opts": {"features": "aliked", "matcher": "auto", "camera_model": "SIMPLE_RADIAL", "focal_35mm": None, "exhaustive_max": 250,
                      "no_loop_detection": False, "mapper": None, "device": "auto"}}
CHAPEL_MATCHING = {"features": "sift", "matcher": "sequential", "loopDetection": True}   # the 68 bytes the live account has


def load_match(monkeypatch):
    for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODAL_CONFIG_PATH", "/nonexistent/modal.toml")
    spec = importlib.util.spec_from_file_location("sfm_match_under_test", MATCH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_match_step_is_estimated_and_stopped_like_every_paid_run():
    assert solvebench.match_hourly() == pytest.approx(1.432) and solvebench.match_deadline_min() == 30   # 1.5 times the 20 min estimate
    assert solvebench.match_cost_usd(solvebench.MATCH_ESTIMATE_MIN) == 0.48 and solvebench.match_cost_usd(30) == 0.72
    lines = solvebench.plan().splitlines()
    assert lines[0].startswith("match on an L4") and "estimate 20 min = $0.48" in lines[0] and "timeout 30 min ($0.72)" in lines[0]
    assert len(lines) == 8 and lines[-1] == solvebench.SHIP_RULE and "30% faster" in lines[-1] and "ATE under 2 cm" in lines[-1]


def test_the_match_arguments_are_the_saved_jobs_options_with_the_features_its_matching_really_used():
    args = solvebench.match_args(SAVED_JOB, CHAPEL_MATCHING)
    assert args == ["--features", "sift", "--matcher", "sequential", "--camera-model", "SIMPLE_RADIAL", "--exhaustive-max", "250",
                    "--device", "cuda", "--only", "sfm", "--sfm-part", "match", "--split-min", "1"]   # aliked requested, SIFT used; no mapper
    assert "--mapper" not in args and "--reuse-matches" not in args
    prior = {"opts": {**SAVED_JOB["opts"], "focal_35mm": 26.0, "camera_model": "OPENCV", "no_loop_detection": True}}
    args = solvebench.match_args(prior, None, "--device cpu")
    assert args[:4] == ["--features", "aliked", "--matcher", "auto"] and ["--focal-35mm", "26.0"] == args[args.index("--focal-35mm"):][:2]
    assert "--no-loop-detection" in args and args[-2:] == ["--device", "cpu"]   # the caller's arguments come last and win
    torch = solvebench.match_args(SAVED_JOB, {"features": "aliked", "matcher": "lightglue", "loopDetection": True})
    assert torch[:2] == ["--features", "aliked-lg"] and "--matcher" not in torch
    assert solvebench.match_args(None, None)[:2] == ["--device", "cuda"]          # nothing saved: the pipeline's own defaults


def test_the_match_step_refuses_missing_frames_and_an_existing_database_without_force():
    on_volume = lambda *names: (lambda rel: rel in names)
    why = solvebench.match_preflight(on_volume("colmap/matching.json"), force=False)
    assert why.startswith("the source job lacks frames, frames_index.json, .done/frames.json") and "modal volume ls ply-bench" in why
    assert solvebench.match_preflight(on_volume("frames", "frames_index.json"), force=True).startswith("the source job lacks .done/frames.json")
    ready = ("frames", "frames_index.json", ".done/frames.json")
    assert solvebench.match_preflight(on_volume(*ready), force=False) is None
    assert "not replaced without --force" in solvebench.match_preflight(on_volume(*ready, "colmap/database.db"), force=False)
    assert solvebench.match_preflight(on_volume(*ready, "colmap/database.db"), force=True) is None
    seen = types.SimpleNamespace(listdir=lambda d: [types.SimpleNamespace(path=f"{d}/{n}") for n in ("frames", "job.json")] if d == "chapel/high/job" else [])
    have = solvebench.volume_have(seen, "chapel/high/job")
    assert have("frames") and not have("frames_index.json") and not have("colmap/database.db")


def _volume(tmp_path, with_frames=True, database=False):
    src = tmp_path / "vol" / "chapel" / "high" / "job"
    if with_frames:
        (src / "frames").mkdir(parents=True)
        (src / "frames" / "0000.jpg").write_bytes(b"jpg")
        (src / ".done").mkdir()
        (src / ".done" / "frames.json").write_text(json.dumps({"source": "videos"}))
        (src / "frames_index.json").write_text("[]")
    src.mkdir(parents=True, exist_ok=True)
    (src / "job.json").write_text(json.dumps(SAVED_JOB))
    (src / "colmap").mkdir(exist_ok=True)
    (src / "colmap" / "matching.json").write_text(json.dumps(CHAPEL_MATCHING))
    if database:
        (src / "colmap" / "database.db").write_bytes(b"the first database")
    return src


def _fake_plypipe(calls, work, fail=None, model=False):
    """Stands in for the pipeline: writes what `--sfm-part match` writes (a database and matching.json), or fails."""
    import sqlite3

    def run(args, deadline):
        calls.append(args)
        if fail:
            return fail
        out = work / "colmap"
        out.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(out / "database.db")
        con.executescript("create table images(image_id); create table keypoints(image_id, rows); create table matches(pair_id, rows);"
                          "create table two_view_geometries(pair_id, rows);"
                          "insert into images values (1),(2); insert into keypoints values (1, 100),(2, 80);"
                          "insert into matches values (3, 40),(4, 0); insert into two_view_geometries values (3, 30),(4, 0);")
        con.commit()
        con.close()
        (out / "matching.json").write_text(json.dumps(CHAPEL_MATCHING))
        if model:
            (out / "sparse" / "0").mkdir(parents=True)
        return None
    return run



def test_the_match_step_saves_the_database_with_what_it_did_and_runs_no_mapper(monkeypatch, tmp_path):
    m = load_match(monkeypatch)
    src = _volume(tmp_path)
    work = tmp_path / "work"
    calls = []
    monkeypatch.setattr(m, "JOB", work)
    monkeypatch.setattr(m, "_plypipe", _fake_plypipe(calls, work))
    monkeypatch.setattr(m, "VOL", tmp_path / "vol")
    assert m.app.name == "ply-sfm-match" and callable(m.match.local) and callable(m.main)
    original = next(v for k, v in vars(m.match).items() if k.startswith("_sync_original"))
    info = getattr(original, "_function_info", None)
    if info is not None:  # modal 1.6.x's layout: the function timeout is 1.5 times the 20 min estimate, on 8 cores and 32 GiB
        assert info.cpu == 8 and info.memory_mib == 32 * 1024 and info.timeout == 30 * 60
    rec = m.run_match(src, False, {"name": "NVIDIA L4", "memoryMiB": 23034, "priced": "L4"})
    assert calls == [solvebench.match_args(SAVED_JOB, CHAPEL_MATCHING)] and "--mapper" not in calls[0]
    assert (src / "colmap" / "database.db").stat().st_size > 0 and json.loads((src / "colmap" / "matching.json").read_text()) == CHAPEL_MATCHING
    saved = json.loads((src / "colmap" / "match-run.json").read_text())
    assert saved == rec and rec["error"] is None and rec["features"] == "sift" and rec["matcher"] == "sequential" and rec["loopDetection"] is True
    assert (rec["images"], rec["keypoints"], rec["pairsMatched"], rec["pairsVerified"]) == (2, 180, 1, 1) and rec["gpu"]["name"] == "NVIDIA L4"
    assert set(rec["seconds"]) == {"restore", "match", "save", "wall"} and rec["estimateMin"] == 20 and not (src / "colmap" / "database.db.part").exists()
    assert not (src / "colmap" / "sparse").exists()


def test_the_match_step_does_not_replace_a_database_without_force_and_stops_at_missing_frames(monkeypatch, tmp_path):
    m = load_match(monkeypatch)
    work = tmp_path / "work"
    calls = []
    monkeypatch.setattr(m, "JOB", work)
    monkeypatch.setattr(m, "_plypipe", _fake_plypipe(calls, work))
    monkeypatch.setattr(m, "VOL", tmp_path / "vol")
    src = _volume(tmp_path, database=True)
    with pytest.raises(FileNotFoundError, match="not replaced without --force"):
        m.run_match(src, False, {})
    assert (src / "colmap" / "database.db").read_bytes() == b"the first database" and calls == []
    m.run_match(src, True, {})
    assert (src / "colmap" / "database.db").read_bytes() != b"the first database" and len(calls) == 1
    empty = _volume(tmp_path / "other", with_frames=False)
    with pytest.raises(FileNotFoundError, match="lacks frames, frames_index.json, .done/frames.json"):
        m.run_match(empty, False, {})
    assert len(calls) == 1 and not (empty / "colmap" / "database.db").exists()


def test_a_failed_match_step_saves_its_record_and_no_database(monkeypatch, tmp_path):
    m = load_match(monkeypatch)
    work = tmp_path / "work"
    monkeypatch.setattr(m, "JOB", work)
    monkeypatch.setattr(m, "VOL", tmp_path / "vol")
    src = _volume(tmp_path)
    monkeypatch.setattr(m, "_plypipe", _fake_plypipe([], work, fail="stop-loss: the matching was still running at its deadline"))
    rec = m.run_match(src, False, {})
    assert rec["error"].startswith("stop-loss") and not (src / "colmap" / "database.db").exists()
    assert json.loads((src / "colmap" / "match-run.json").read_text())["error"] == rec["error"]
    monkeypatch.setattr(m, "_plypipe", _fake_plypipe([], work, model=True))
    rec = m.run_match(src, False, {})
    assert "mapper must not run" in rec["error"] and not (src / "colmap" / "database.db").exists()


@pytest.mark.slow
def test_the_match_step_on_the_synthetic_scene_leaves_a_database_and_no_model_that_every_mapper_then_maps(monkeypatch, tmp_path, synthetic_scene):
    from test_sfm_plan import _resolve
    m = load_match(monkeypatch)
    src = tmp_path / "vol" / "scene" / "job"
    shutil.copytree(synthetic_scene, src, ignore=shutil.ignore_patterns("colmap", "sfm.json", "sparse*", "dense", "log.jsonl"))
    (src / "job.json").write_text(json.dumps({"opts": {"features": "aliked", "matcher": "sequential", "no_loop_detection": True}}))
    (src / "colmap").mkdir()
    (src / "colmap" / "matching.json").write_text(json.dumps({"features": "sift", "matcher": "sequential", "loopDetection": False}))
    monkeypatch.setattr(m, "VOL", tmp_path / "vol")
    monkeypatch.setattr(m, "JOB", tmp_path / "work")
    rec = m.run_match(src, False, {"name": "cpu"}, "--device cpu")
    assert rec["error"] is None and rec["features"] == "sift" and rec["images"] == 60 and rec["pairsVerified"] > 0
    assert not (tmp_path / "work" / "colmap" / "sparse").exists() and not (tmp_path / "work" / ".done" / "sfm.json").exists()
    saved = tmp_path / "saved"
    shutil.copytree(src, saved, ignore=shutil.ignore_patterns("sfm.json", "sparse*", "match-run.json"))
    for mapper in ("incremental", "hierarchical", "keyframes"):
        rep = _resolve(saved, tmp_path / mapper, "--mapper", mapper)
        assert rep["mapper"] == mapper and rep["registered"] == 60 and rep["mapperFallback"] is None
