# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""The split camera solve without Modal: GPU worker matches and hands over, the CPU solver maps and re-queues,
the GPU worker resumes from the solved cameras. Stand-ins for the pipeline and the API, as in test_worker_resume."""

import json
import sys

from plypipe import handoff, worker
from test_worker_resume import Store, job

FAKE = r'''
import json, sys, pathlib
a = sys.argv; out = pathlib.Path(a[a.index("--out") + 1]); (out / ".done").mkdir(parents=True, exist_ok=True)
only = a[a.index("--only") + 1] if "--only" in a else None
for st in ["frames", "sfm", "align", "occupancy", "train", "export"]:
    if only and st != only:
        continue
    if (out / ".done" / f"{st}.json").exists():
        print(json.dumps({"stage": st, "event": "skip", "msg": ""}), flush=True); continue
    print(json.dumps({"stage": st, "event": "start", "msg": ""}), flush=True)
    if st == "frames":
        (out / "frames").mkdir(exist_ok=True); (out / "frames" / "000000.jpg").write_bytes(b"jpg")
    if st == "sfm" and "--sfm-part" in a:
        (out / "colmap").mkdir(exist_ok=True); (out / "colmap" / "database.db").write_bytes(b"db")
        (out / "colmap" / "matching.json").write_text("{}")
        print(json.dumps({"stage": "sfm", "event": "paused", "msg": "matched"}), flush=True); sys.exit(0)
    if st == "sfm":
        if "--reuse-matches" in a:  # the CPU half maps the database the GPU half matched
            assert (out / "colmap" / "database.db").exists(), a
        (out / "dense" / "sparse_txt").mkdir(parents=True, exist_ok=True)  # undistorted model, as the real stage
        (out / "dense" / "sparse_txt" / "cameras.txt").write_text("1 PINHOLE")
        (out / ".done" / "sfm.json").write_text(json.dumps({"registered": 9, "images": 10}))
        print(json.dumps({"stage": "sfm", "event": "done", "msg": ""}), flush=True); continue
    if st == "align":  # the real align reads the undistorted text model
        assert (out / "dense" / "sparse_txt" / "cameras.txt").exists(), "dense/sparse_txt/cameras.txt missing"
    (out / ".done" / f"{st}.json").write_text("{}")
    print(json.dumps({"stage": st, "event": "done", "msg": ""}), flush=True)
if not only:
    (out / "scene.spz").write_bytes(b"spz"); (out / "report.json").write_text("{}")
'''


def test_a_long_capture_is_solved_on_a_cpu_box_and_resumed(tmp_path, monkeypatch):
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    hd = tmp_path / "handoff"
    hd.mkdir()
    monkeypatch.setenv("PLY_PIPELINE_CMD", f"{sys.executable} {fake}")
    monkeypatch.setenv("PLY_HANDOFF_DIR", str(hd))
    monkeypatch.setenv("PLY_CPU_SFM", "1")
    spawned = []
    monkeypatch.setattr(handoff, "spawn", lambda fn, *a: spawned.append((fn, *a)))
    store = Store()

    # 1. GPU: frames + matching, then hand over; the job is neither done nor failed
    worker.run_job(store, job(), tmp_path / "work", [])
    assert spawned == [("solve", "j1")] and handoff.state("j1", hd) == "matched"
    assert (hd / "j1" / "colmap" / "database.db").exists()
    assert not any(p and p.get("status") in ("failed", "review", "done") for p in store.posts)
    assert any(p and "CPU box" in str(p.get("message")) for p in store.posts)
    assert store.clip_downloads == 1

    # 2. CPU: map the matches, heartbeat, park the solved folder (without the database) and re-queue
    cpu = Store()
    assert handoff.solve("j1", hd, cpu, work=tmp_path / "solve", runner=[sys.executable, str(fake)])
    assert handoff.state("j1", hd) == "solved" and not (hd / "j1" / "colmap" / "database.db").exists()
    rep = json.loads((hd / "j1" / ".done" / "sfm.json").read_text())
    assert rep["solvedOn"] == "cpu" and rep["registered"] == 9
    assert cpu.posts[-1]["status"] == "queued" and "9/10 frames" in cpu.posts[-1]["message"]

    # 3. GPU again: resumes after sfm, no clip download, publishes for review, clears the hand-over folder
    worker.run_job(store, job(), tmp_path / "work2", [])
    assert store.clip_downloads == 1
    assert store.posts[-1]["status"] == "review"
    assert not (hd / "j1").exists()


def test_a_failed_cpu_solve_fails_the_job_and_the_gpu_solves_on_retry(tmp_path, monkeypatch):
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    hd = tmp_path / "handoff"
    hd.mkdir()
    monkeypatch.setenv("PLY_PIPELINE_CMD", f"{sys.executable} {fake}")
    monkeypatch.setenv("PLY_HANDOFF_DIR", str(hd))
    monkeypatch.setenv("PLY_CPU_SFM", "1")
    monkeypatch.setattr(handoff, "spawn", lambda fn, *a: None)
    store = Store()
    worker.run_job(store, job(), tmp_path / "work", [])
    broken = tmp_path / "broken.py"
    broken.write_text("import sys; print('mapper crashed'); sys.exit(2)")
    cpu = Store()
    assert not handoff.solve("j1", hd, cpu, work=tmp_path / "solve", runner=[sys.executable, str(broken)])
    assert cpu.posts[-1]["status"] == "failed" and "mapper crashed" in cpu.posts[-1]["message"]
    # a requeue (by hand or the stale rule) solves on the GPU box instead of handing over again
    worker.run_job(store, job(), tmp_path / "work2", [])
    assert store.posts[-1]["status"] == "review" and not (hd / "j1").exists()


def test_short_captures_and_coverage_checks_are_not_split():
    assert handoff.solver_args(["--preset", "high", "--sfm-part", "match", "--split-min", "400", "--until", "sfm"]) == ["--preset", "high"]


def test_no_handoff_dir_means_no_split(tmp_path, monkeypatch):
    monkeypatch.delenv("PLY_HANDOFF_DIR", raising=False)
    assert handoff.root() is None
