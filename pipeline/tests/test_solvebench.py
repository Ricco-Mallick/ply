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
