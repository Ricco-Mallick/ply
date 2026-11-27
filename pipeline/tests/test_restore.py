"""plypipe.restore without a GPU: the loop's save/exit/resume decisions, the key, atomic metadata, signals."""

import json
import signal
import sys
import types

import pytest

from plypipe import restore
from plypipe.restore import Loop, _S


@pytest.fixture(autouse=True)
def fresh(tmp_path):
    _S.runner = types.SimpleNamespace(splats={"means": [0] * 5})
    _S.stop, _S.dir, _S.resume, _S.key, _S.every_s, _S.max_steps = None, tmp_path, False, "k1", 0, 100
    _S.log = lambda m: None
    yield
    _S.dir = None


def run_loop(stop_at=None, saver=None, loader=None, clock=None, n=100):
    seen, lines = [], []
    loop = Loop(range(0, n), saver=saver or (lambda r, step, why: seen.append(("save", step, why))),
                loader=loader, clock=clock or (lambda: 0.0), out=lines.append)
    steps = []
    for step in loop:
        steps.append(step)
        if step == stop_at:
            _S.stop = "SIGTERM"
    return steps, seen, lines


def test_a_signal_saves_the_last_finished_step_and_exits_paused():
    with pytest.raises(SystemExit) as e:
        run_loop(stop_at=41)
    assert e.value.code == restore.PAUSED == 75


def test_the_saved_step_is_the_one_before_the_signal_was_seen():
    saved = []
    with pytest.raises(SystemExit):
        run_loop(stop_at=41, saver=lambda r, step, why: saved.append((step, why)))
    assert saved == [(41, "SIGTERM")]  # step 41 ran to the end; 42 never started


def test_resume_continues_after_the_saved_step(tmp_path):
    (tmp_path / restore.FILE).write_bytes(b"x")
    (tmp_path / restore.META).write_text(json.dumps({"step": 41, "key": "k1", "splats": 5}))
    _S.resume = True
    steps, _, _ = run_loop(loader=lambda runner, path: 41)
    assert steps[0] == 42 and steps[-1] == 99


def test_a_restore_point_from_other_settings_is_not_used(tmp_path):
    (tmp_path / restore.FILE).write_bytes(b"x")
    (tmp_path / restore.META).write_text(json.dumps({"step": 41, "key": "other"}))
    _S.resume = True
    steps, _, _ = run_loop(loader=lambda *a: pytest.fail("must not load"))
    assert steps[0] == 0


def test_autosave_on_the_clock():
    t = iter(range(0, 10_000, 30))  # 30 s per step
    _S.every_s = 600
    _, seen, _ = run_loop(clock=lambda: next(t), n=60)
    assert [s for s in seen if s[2] == "autosave"] and all(b[1] - a[1] >= 9 for a, b in zip(seen, seen[1:]))


def test_progress_lines_are_json_with_step_and_splats():
    _, _, lines = run_loop(n=3)
    rec = json.loads(lines[0])["plyTrain"]
    assert rec["step"] == 0 and rec["of"] == 3 and rec["splats"] == 5


def test_the_key_ignores_where_files_live_and_the_restore_flags():
    a = restore.restore_key(["py", "--data", "/tmp/work/j1/out/data", "--cap-max", "900000", "--restore", "/x/r", "--resume"])
    b = restore.restore_key(["py", "--data", "/kaggle/working/j1/out/data", "--cap-max", "900000", "--restore", "/y/r"])
    c = restore.restore_key(["py", "--data", "/tmp/work/j1/out/data", "--cap-max", "1200000"])
    assert a == b != c


def test_only_the_training_bar_is_replaced():
    real = types.SimpleNamespace(tqdm=lambda it=None, *a, **k: ("real", it), trange=lambda *a: "trange")
    shim = restore._TqdmShim(real)
    assert shim.tqdm(range(0, 10))[0] == "real"  # not the training range
    assert isinstance(shim.tqdm(range(0, 100)), Loop)
    assert shim.tqdm(range(0, 100))[0] == "real"  # only once
    assert shim.trange(5) == "trange"


def test_first_signal_asks_second_one_forces(monkeypatch):
    exits = []
    monkeypatch.setattr(restore.os, "_exit", exits.append)
    restore._on_signal(signal.SIGTERM, None)
    assert _S.stop == "SIGTERM" and not exits
    restore._on_signal(signal.SIGINT, None)
    assert exits == [130]


def test_util_run_reports_a_pause_and_forwards_structured_lines(tmp_path):
    from plypipe.log import Log
    from plypipe.util import Ctx, Paused, run
    child = tmp_path / "child.py"
    child.write_text("import json,sys\nprint(json.dumps({'plyTrain': {'step': 5, 'of': 10}}), flush=True)\n"
                     "print(json.dumps({'plyRestore': {'step': 5}}), flush=True)\nsys.exit(75)\n")
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={})
    with pytest.raises(Paused):
        run([sys.executable, str(child)], ctx, "train", own_session=True)
    events = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any(e.get("train", {}).get("step") == 5 for e in events)
    assert any(e["event"] == "restore" for e in events)


def test_the_key_does_not_depend_on_the_machine():
    a = restore.restore_key(["py", "--cap-max", "9", "--packed", "--loader-workers", "2"])
    assert a == restore.restore_key(["py", "--cap-max", "9"])


def test_worker_reads_structured_training_progress_and_rounds_continue_the_bar():
    from plypipe import worker
    f0 = worker.frac_from({"train": {"step": 15000, "of": 30000}})
    f1 = worker.frac_from({"train": {"step": 30000, "of": 30000}})
    r1 = worker.frac_from({"train": {"step": 0, "of": 7500, "round": "1"}})
    r1e = worker.frac_from({"train": {"step": 7500, "of": 7500, "round": "1"}})
    r2 = worker.frac_from({"train": {"step": 100, "of": 7500, "round": "2"}})
    assert 0 < f0 < f1 <= r1 < r1e <= r2 < 1
