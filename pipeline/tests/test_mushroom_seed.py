"""pipeline/modal/mushroom_seed.py: that it loads without reaching Modal, what `plan` says, and the solve function's body run in-process with
the pipeline replaced by a stand-in (the real solve needs a GPU). Skipped where `modal` is not installed."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_mushroom import walk, write_text_model

from plypipe import mushroom, mushroom_fetch

pytest.importorskip("modal")
pytest.importorskip("modal.volume", reason="no modal client: `import modal` finds only the pipeline/modal folder")
SCRIPT = Path(__file__).resolve().parents[1] / "modal" / "mushroom_seed.py"


def load(monkeypatch):
    for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODAL_CONFIG_PATH", "/nonexistent/modal.toml")
    spec = importlib.util.spec_from_file_location("mushroom_seed_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_loads_without_a_connection_and_its_timeouts_are_one_and_a_half_times_the_estimates(monkeypatch):
    m = load(monkeypatch)
    assert m.app.name == "ply-mushroom-seed" and callable(m.main) and callable(m.fetch.local) and callable(m.solve.local)
    slowest = max(mushroom_fetch.fetch_estimate(r)["seconds"] for r in mushroom.ROOMS)
    assert m.FETCH_TIMEOUT_S >= 1.5 * slowest and m.FETCH_TIMEOUT_S < 1.5 * slowest + 120
    assert m.SOLVE_TIMEOUT_S == (23 + 2) * 60
    original = next(v for k, v in vars(m.solve).items() if k.startswith("_sync_original"))
    info = getattr(original, "_function_info", None)
    if info is not None:
        assert info.gpus == [("L4", 1), ("L40S", 1)] and info.cpu == 8 and info.memory_mib == 16384 and info.timeout == 25 * 60


def test_the_plan_states_the_bytes_and_the_costs_before_anything_is_started(monkeypatch):
    m = load(monkeypatch)
    fetch = m.describe("fetch", ["honka", "coffee_room"])
    assert "honka: fetches 410 MB" in fetch and "coffee_room: fetches 471 MB" in fetch and "total 882 MB" in fetch and "stop-loss" in fetch
    solve = m.describe("solve", ["honka"])
    assert "L4: 15 min per room = $0.33" in solve and "L40S" in solve and "needs fetch done first" in solve
    with pytest.raises(SystemExit, match="--step fetch|solve"):
        m.main.info.raw_f(step="download", rooms="honka")
    with pytest.raises(SystemExit):
        m.main.info.raw_f(step="fetch", rooms="kitchen")


def seeded(tmp_path, n=40):
    root = tmp_path / "vol" / "mushroom"
    names = [f"frame_{i + 1:05d}" for i in range(n)]
    (root / "honka" / "in").mkdir(parents=True)
    for k in names:
        (root / "honka" / "in" / f"{k}.jpg").write_bytes(b"jpg")
    ref = root / "honka" / "reference"
    ref.mkdir()
    centres = walk(n)[:, :3, 3]
    (ref / "meta.json").write_text(json.dumps({"names": names, "inputNames": names, "centres": centres.tolist(), "poses": "transformations.json",
                                               "direction": {"direction": "as-stored", "confident": True}}))
    np.save(ref / "ref_points.npy", np.random.default_rng(0).uniform(0, 1, (500, 3)).astype(np.float32) * [6, 4, 2.5])
    return root, centres


def test_the_solve_runs_the_pipeline_scores_the_cameras_and_saves_the_job(tmp_path, monkeypatch):
    m = load(monkeypatch)
    root, centres = seeded(tmp_path)
    commits, calls = [], []

    def pipeline(cmd, deadline):
        """What the solve leaves: the twin's cameras (its metres 6% longer than the room's), the images, the markers and the database."""
        calls.append(cmd)
        job = Path(cmd[cmd.index("--out") + 1])
        poses = walk(len(centres)).copy()
        poses[:, :3, 3] = centres * 1.06
        write_text_model(job / "data" / "sparse" / "0", [f"{i:06d}.jpg" for i in range(len(centres))], poses)
        for rel in ("colmap/database.db", "frames/000000.jpg", "dense/images/000000.jpg", ".done/occupancy.json", "scene.json"):
            (job / rel).parent.mkdir(parents=True, exist_ok=True)
            (job / rel).write_bytes(b"x")
        (job / "report.json").write_text(json.dumps({"frames": {"kept": 40}, "sfm": {"registered": 40}, "align": {"scale": 1.0}}))
        return None

    monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(m, "WORK", tmp_path / "work")
    monkeypatch.setattr(m, "vol", SimpleNamespace(commit=lambda: commits.append(1)))
    monkeypatch.setattr(m, "_run", pipeline)
    result = m.solve.local("honka")
    cmd = calls[0]
    assert cmd[1:4] == ["-m", "plypipe", "run"] and cmd[cmd.index("--until") + 1] == "occupancy" and "--no-labels" in cmd
    assert result["error"] is None and result["score"]["scaleErrorPct"] == pytest.approx(6.0, abs=0.2) and result["score"]["cameras"]["matched"] == 40
    saved = root / "honka" / "job"
    assert (saved / ".done" / "occupancy.json").exists() and (saved / "dense" / "images" / "000000.jpg").exists() and (saved / "data" / "sparse" / "0" / "images.txt").exists()
    assert not (saved / "colmap" / "database.db").exists() and not (saved / "frames").exists() and (root / "honka" / "solve.json").exists() and commits
    calls.clear()
    assert m.solve.local("honka")["score"]["scaleErrorPct"] == result["score"]["scaleErrorPct"] and calls == []   # solved already: nothing runs again


def test_a_solve_without_the_fetched_input_stops_before_any_gpu_work(tmp_path, monkeypatch):
    m = load(monkeypatch)
    monkeypatch.setattr(m, "ROOT", tmp_path / "vol" / "mushroom")
    with pytest.raises(FileNotFoundError, match="run --step fetch first"):
        m.solve.local("honka")


def test_a_stopped_solve_saves_what_it_has_and_fails_loudly(tmp_path, monkeypatch):
    m = load(monkeypatch)
    root, _ = seeded(tmp_path)
    monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(m, "WORK", tmp_path / "work")
    monkeypatch.setattr(m, "vol", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(m, "_run", lambda cmd, deadline: "stop-loss: the solve was still running at its deadline")
    with pytest.raises(RuntimeError, match="stop-loss"):
        m.solve.local("honka")
    assert json.loads((root / "honka" / "solve.json").read_text())["error"].startswith("stop-loss")
    assert not (root / "honka" / "job" / ".done" / "occupancy.json").exists()      # a relaunch solves again
