"""pipeline/modal/ply_eval.py: that it loads and builds its Modal objects without reaching Modal, and that its function
body does what the docstring says, run in-process against local folders with the plypipe stages replaced by stand-ins
(the real stages need a GPU). Skipped where `modal` is not installed."""

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from plypipe import evalrun
from test_evalrun import finished_report, saved_job

pytest.importorskip("modal")
pytest.importorskip("modal.volume", reason="no modal client: `import modal` finds only the pipeline/modal folder")
SCRIPT = Path(__file__).resolve().parents[1] / "modal" / "ply_eval.py"


def load(monkeypatch, name="ply_eval_under_test", **env):
    for k in ("PLY_EVAL_GPU", "PLY_EVAL_TIMEOUT_MIN", "PLY_EVAL_SCENARIO"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_loads_and_builds_its_modal_objects_without_a_connection(monkeypatch):
    for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODAL_CONFIG_PATH", "/nonexistent/modal.toml")
    m = load(monkeypatch)
    assert m.app.name == "ply-eval" and callable(m.main) and callable(m.run_variant.local)
    assert m.GPUS == ["L40S"] and m.CEILING_MIN == 32                             # q1: 15k steps with the grid, 2k after the fold, the phone file's 3k
    original = next(v for k, v in vars(m.run_variant).items() if k.startswith("_sync_original"))
    info = getattr(original, "_function_info", None)
    if info is not None:                                                          # modal 1.6.3's layout; the GPU, size and timeout it will ask for
        assert info.gpus == [("L40S", 1)] and info.cpu == 8 and info.memory_mib == 32768 and info.timeout == 34 * 60


def test_a_gpu_fallback_is_opt_in_and_lengthens_the_timeout(monkeypatch):
    m = load(monkeypatch, "ply_eval_fallback", PLY_EVAL_GPU="L40S,L4")
    assert m.GPUS == ["L40S", "L4"] and m.CEILING_MIN == 51
    assert load(monkeypatch, "ply_eval_longer", PLY_EVAL_TIMEOUT_MIN="45").CEILING_MIN == 45


def test_the_scenario_sizes_the_container_and_must_match_the_launch(monkeypatch):
    m = load(monkeypatch, "ply_eval_small", PLY_EVAL_SCENARIO="drjohnson", PLY_EVAL_GPU="L4,L40S")
    assert m.SCENARIO == "drjohnson" and m.CEILING_MIN == 29
    original = next(v for k, v in vars(m.run_variant).items() if k.startswith("_sync_original"))
    info = getattr(original, "_function_info", None)
    if info is not None:
        assert info.gpus == [("L4", 1), ("L40S", 1)] and info.cpu == 4 and info.memory_mib == 16384 and info.timeout == 31 * 60
    with pytest.raises(SystemExit, match="PLY_EVAL_SCENARIO=chapel"):          # sized for Dr Johnson: a chapel run would get its container
        m.main.info.raw_f(scenario="chapel", variant="q0a")
    with pytest.raises(SystemExit, match="not one of"):
        load(monkeypatch, "ply_eval_unknown", PLY_EVAL_SCENARIO="garden")


def test_the_stop_loss_kills_a_stage_that_outlives_its_deadline(monkeypatch):
    m = load(monkeypatch)
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    monkeypatch.setattr(evalrun, "plypipe_command", lambda stage, job_dir, args, python="python": sleeper)
    start = time.time()
    assert m._run("train", [], time.time() + 1).startswith("stop-loss: train")
    assert time.time() - start < 20
    monkeypatch.setattr(evalrun, "plypipe_command", lambda *a, **k: [sys.executable, "-c", "raise SystemExit(3)"])
    assert m._run("export", [], time.time() + 30) == "export exited with 3"
    monkeypatch.setattr(evalrun, "plypipe_command", lambda *a, **k: [sys.executable, "-c", "pass"])
    assert m._run("export", [], time.time() + 30) is None


class Rig:
    """The function body against local folders: a seeded Volume, a job folder, stand-in stages that leave what the real ones leave."""

    def __init__(self, tmp_path, monkeypatch, fail_export=False):
        self.m = load(monkeypatch)
        self.vol, self.commits, self.stages = tmp_path / "vol", [], []
        saved_job(self.vol / "chapel" / "high" / "job")
        monkeypatch.setattr(self.m, "VOL", self.vol)
        monkeypatch.setattr(self.m, "JOB", tmp_path / "job")
        monkeypatch.setattr(self.m, "_gpu", lambda: {"name": "NVIDIA L40S", "memoryMiB": 46068, "priced": "L40S"})
        monkeypatch.setattr(self.m, "vol", SimpleNamespace(commit=lambda: self.commits.append(len(self.stages))))
        monkeypatch.setattr(self.m, "_run", self.stage)
        self.fail_export, self.job_dir = fail_export, tmp_path / "job"

    def stage(self, stage, args, deadline):
        job = self.job_dir
        self.stages.append((stage, args, (job / ".done" / "sfm.json").exists(), (job / "data" / "images").is_symlink()))
        if stage == "export" and self.fail_export:
            return "export exited with 1"
        report = json.loads((job / "report.json").read_text())
        assert "train" not in report or stage == "export"                       # the old run's sections were dropped on restore
        report[stage] = finished_report()[stage]
        (job / "report.json").write_text(json.dumps(report))
        (job / ".done").mkdir(exist_ok=True)
        (job / ".done" / f"{stage}.json").write_text("{}")
        for name in {"train": ["scene.ply"], "export": ["scene.spz", "scene_lite.spz"]}[stage]:
            (job / name).write_bytes(b"x" * 10)
        (job / "eval").mkdir(exist_ok=True)
        (job / "eval" / f"{stage}.png").write_bytes(b"png")
        return None

    def go(self, variant="q0a", **kw):
        return self.m.run_variant.local(evalrun.make_job("chapel", "chapel/high/job", variant, revision={"revision": "abc", "codeHash": "1"}, **kw))

    def saved(self, variant="q0a"):
        return self.vol / "chapel" / "eval" / variant


def test_a_run_restores_trains_exports_saves_and_writes_metrics(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    result = rig.go()
    assert [s[0] for s in rig.stages] == ["train", "export"] and all(s[1] == evalrun.VARIANTS["q0a"]["args"] for s in rig.stages)
    assert all(s[2] and s[3] for s in rig.stages)                               # the solve and the images link were there before each stage
    assert result["complete"] and result["variant"] == "q0a" and result["gpu"]["name"] == "NVIDIA L40S" and result["gate"] is not None
    assert result["files"] == {"scene.ply": 10, "scene.spz": 10, "scene_lite.spz": 10} and result["revision"] == {"revision": "abc", "codeHash": "1"}
    saved = rig.saved()
    assert json.loads((saved / "metrics.json").read_text())["complete"] is True
    assert {".done/train.json", ".done/export.json", "scene.ply", "scene.spz", "report.json", "eval/train.png"} <= {
        p.relative_to(saved).as_posix() for p in saved.rglob("*") if p.is_file()}
    assert not (saved / "data").exists() and not (saved / "dense").exists() and (saved / "frames_index.json").exists()
    assert "quality" in json.loads((saved / "report.json").read_text())
    assert len(rig.commits) >= 3 and rig.commits[0] == 1                           # committed after the train stage, before export started
    old = rig.vol / "chapel" / "high" / "job"
    assert (old / "scene.ply").exists() and (old / "report.json").read_text().count("quality") == 1   # the saved solve is never written to


def test_a_complete_variant_is_not_run_again_unless_forced(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    first = rig.go()
    assert rig.go() == first and len(rig.stages) == 2
    again = rig.go(force=True)
    assert len(rig.stages) == 4 and again["complete"] and len(again["invocations"]) == 1
    with pytest.raises(ValueError, match="other arguments"):
        rig.m.run_variant.local(evalrun.make_job("chapel", "chapel/high/job", "q0a") | {"args": evalrun.BASE + ["--pose-opt"]})


def test_a_failed_export_leaves_train_saved_and_a_relaunch_runs_only_export(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, fail_export=True)
    with pytest.raises(RuntimeError, match="export exited with 1"):
        rig.go()
    partial = json.loads((rig.saved() / "metrics.json").read_text())
    assert not partial["complete"] and "export" in partial["missing"] and partial["invocations"][0]["error"] == "export exited with 1"
    assert (rig.saved() / ".done" / "train.json").exists() and (rig.saved() / "scene.ply").exists() and not (rig.saved() / ".done" / "export.json").exists()
    rig.stages.clear()
    rig.fail_export = False
    result = rig.go()
    assert [s[0] for s in rig.stages] == ["export"]                              # train was not repeated
    assert result["complete"] and [i["ran"] for i in result["invocations"]] == [["train"], ["export"]]
    assert result["invocations"][0]["error"] == "export exited with 1" and result["invocations"][1]["resumedAfter"] == ["train"]
    assert result["costUsd"] == round(sum(i["costUsd"] for i in result["invocations"]), 2)


def test_a_photo_set_without_a_frames_index_runs_on_the_small_scenario(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    (saved_job(rig.vol / "drjohnson" / "prep") / "frames_index.json").unlink()
    result = rig.m.run_variant.local(evalrun.make_job("drjohnson", "", "q0a", gpus=("L4", "L40S")))
    assert result["complete"] and [s[1] for s in rig.stages] == [evalrun.SMALL, evalrun.SMALL]
    assert (rig.vol / "drjohnson" / "eval" / "q0a" / "metrics.json").exists() and not (rig.vol / "chapel" / "eval").exists()
    assert result["invocations"][0]["pricePerHour"] == pytest.approx(1.95 + 4 * 0.047 + 16 * 0.008)   # the rig's L40S at this scenario's size
    (rig.vol / "chapel" / "high" / "job" / "frames_index.json").unlink()
    with pytest.raises(RuntimeError, match="frames_index.json"):                # a video's block comes from its clips: still required
        rig.go()


def test_a_volume_without_the_saved_job_fails_before_any_gpu_work(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="lacks"):
        rig.m.run_variant.local(evalrun.make_job("chapel", "chapel/missing/job", "q0a"))
    assert rig.stages == []
