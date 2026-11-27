"""Checkpoints and stop-loss, with a stand-in pipeline that honours .done markers like plypipe does."""

import sys
from pathlib import Path

from plypipe import worker

FAKE = r'''
import json, os, sys, time, pathlib
a = sys.argv; out = pathlib.Path(a[a.index("--out") + 1]); (out / ".done").mkdir(parents=True, exist_ok=True)
crash_at = os.environ.get("FAKE_CRASH_AT"); slow_at = os.environ.get("FAKE_SLOW_AT")
for st in ["frames", "sfm", "align", "occupancy", "train", "export"]:
    if (out / ".done" / f"{st}.json").exists():
        print(json.dumps({"stage": st, "event": "skip", "msg": ""}), flush=True); continue
    print(json.dumps({"stage": st, "event": "start", "msg": ""}), flush=True)
    if st == slow_at:
        if st == "train" and os.environ.get("FAKE_SNAPSHOT"):  # gsplat's part-way PLY (PLY_LIVE_SNAPSHOTS)
            (out / "train" / "ply").mkdir(parents=True, exist_ok=True)
            (out / "train" / "ply" / "point_cloud_1200.ply").write_bytes(b"ply snapshot")
        time.sleep(60)
    if st == crash_at:
        print("boom", flush=True); sys.exit(3)
    if st == "sfm":
        (out / "dense").mkdir(exist_ok=True); (out / "dense" / "model.txt").write_text("cams")
    (out / ".done" / f"{st}.json").write_text("{}")
    print(json.dumps({"stage": st, "event": "done", "msg": ""}), flush=True)
(out / "scene.spz").write_bytes(b"spz"); (out / "report.json").write_text("{}")
'''


class Store:
    """A fake Ply API + bucket: uploads land in a dict, downloads come from it (404 otherwise)."""
    def __init__(self):
        self.objects, self.posts, self.clip_downloads = {}, [], 0

    def req(self, method, path, body=None, ctype=None, tries=5):
        self.posts.append(body)
        return 200, {"ok": True}

    def download(self, key, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if key.startswith("captures/"):
            self.clip_downloads += 1
            dest.write_bytes(b"video")
            return
        if key not in self.objects:
            raise OSError("404")
        dest.write_bytes(self.objects[key])

    def upload(self, key, path, ctype):
        self.objects[key] = Path(path).read_bytes()


def job(args=("--preset", "quick")):
    return {"id": "j1", "files": ["captures/c/clip-01.mp4"], "args": list(args)}


def setup(tmp_path, monkeypatch):
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    monkeypatch.setenv("PLY_PIPELINE_CMD", f"{sys.executable} {fake}")


def test_retry_resumes_after_the_last_checkpoint_without_redownloading(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    store = Store()
    monkeypatch.setenv("FAKE_CRASH_AT", "train")
    worker.run_job(store, job(), tmp_path / "w1", [])               # a different machine each time
    assert store.posts[-1]["status"] == "failed" and "scenes/j1/checkpoint.tar" in store.objects
    assert store.clip_downloads == 1
    monkeypatch.delenv("FAKE_CRASH_AT")
    worker.run_job(store, job(), tmp_path / "w2", [])
    assert store.posts[-1]["status"] == "review"
    assert store.clip_downloads == 1                                  # resumed past SfM: no second 179 MB download
    assert any("resuming after occupancy" in (p.get("message") or "") for p in store.posts if isinstance(p, dict))


def test_checkpoint_from_other_settings_is_ignored(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    store = Store()
    monkeypatch.setenv("FAKE_CRASH_AT", "train")
    worker.run_job(store, job(), tmp_path / "w1", [])
    monkeypatch.delenv("FAKE_CRASH_AT")
    worker.run_job(store, job(("--preset", "quick", "--features", "sift")), tmp_path / "w2", [])
    assert store.posts[-1]["status"] == "review" and store.clip_downloads == 2   # started over


def test_stage_over_its_limit_is_stopped(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    store = Store()
    monkeypatch.setenv("FAKE_SLOW_AT", "sfm")
    monkeypatch.setenv("PLY_STAGE_LIMITS", "sfm=0.05")              # 3 seconds
    worker.run_job(store, job(), tmp_path / "w1", [])
    final = store.posts[-1]
    assert final["status"] == "failed" and "sfm exceeded its 0.05-minute limit" in final["message"]
    assert not any("retrying" in (p.get("message") or "") for p in store.posts if isinstance(p, dict))


def test_rehearsal_dry_run_finishes_without_results(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    store = Store()
    worker.run_job(store, job(("--preset", "quick", "--train-dry-run", "--until", "train")), tmp_path / "w1", [])
    assert store.posts[-1]["status"] == "done" and store.posts[-1]["result"]["rehearsal"] is True
    assert "scenes/j1/checkpoint.tar" not in store.objects


def test_a_slow_checkpoint_upload_does_not_trip_the_stage_watchdog(tmp_path, monkeypatch):
    """Regression (classroom demo, 2026-10-14): a 600 MB checkpoint uploaded inside the output-reading loop stalled it;
    the watchdog then saw a 2-second stage 'run' past its limit and killed a healthy job."""
    setup(tmp_path, monkeypatch)
    import time as _t

    class SlowStore(Store):
        def upload(self, key, path, ctype):
            if key.endswith("checkpoint.tar"):
                _t.sleep(4)
            super().upload(key, path, ctype)
    store = SlowStore()
    monkeypatch.setenv("PLY_STAGE_LIMITS", "occupancy=0.03,align=0.03,sfm=0.03")  # ~2 s each
    worker.run_job(store, job(), tmp_path / "w", [])
    assert store.posts[-1]["status"] == "review", store.posts[-1]
    assert "scenes/j1/checkpoint.tar" in store.objects  # and the checkpoint still made it


def test_big_checkpoints_are_skipped(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "CKPT_MAX_MB", 0)
    store = Store()
    worker.run_job(store, job(), tmp_path / "w", [])
    assert store.posts[-1]["status"] == "review" and "scenes/j1/checkpoint.tar" not in store.objects


def test_training_over_its_limit_keeps_the_newest_snapshot(tmp_path, monkeypatch):
    """A training run that hits its limit used to fail the job and throw the GPU hours away."""
    setup(tmp_path, monkeypatch)
    store = Store()
    monkeypatch.setenv("FAKE_SLOW_AT", "train")
    monkeypatch.setenv("FAKE_SNAPSHOT", "1")
    monkeypatch.setenv("PLY_STAGE_LIMITS", "train=0.05")            # 3 seconds
    worker.run_job(store, job(), tmp_path / "w1", [])
    final = store.posts[-1]
    assert final["status"] == "review", final
    assert any("exporting that snapshot" in (p.get("message") or "") for p in store.posts if isinstance(p, dict))
    out = tmp_path / "w1" / "j1" / "out"
    assert (out / "scene.ply").read_bytes() == b"ply snapshot"
    assert store.objects["scenes/j1/scene.ply"] == b"ply snapshot"      # the model is kept with the results
    assert "scenes/j1/checkpoint.tar" in store.objects


def test_training_stopped_before_any_snapshot_still_fails_clearly(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    store = Store()
    monkeypatch.setenv("FAKE_SLOW_AT", "train")
    monkeypatch.setenv("PLY_STAGE_LIMITS", "train=0.05")
    worker.run_job(store, job(), tmp_path / "w1", [])
    final = store.posts[-1]
    assert final["status"] == "failed" and "train exceeded its 0.05-minute limit" in final["message"]


def test_train_limit_stretches_to_the_projected_finish_within_the_deadline():
    base = 120 * 60
    # 40 % done after 100 min: finishing needs ~287 min with margin; a deadline 5 h away allows it
    assert abs(worker.train_limit_s(base, 6000, 0.4, 10_000 + 5 * 3600, 70 * 60, 10_000) - 6000 / 0.4 * 1.15) < 1
    # the deadline leaves only 30 min after export's reserve: stop there (and salvage), never past the machine
    assert worker.train_limit_s(base, 7200, 0.4, 10_000 + 100 * 60, 70 * 60, 10_000) == 7200 + 30 * 60
    # too early to project, or no deadline: the fixed limit, at most tripled
    assert worker.train_limit_s(base, 600, 0.02, 0, 0, 0) == base
    assert worker.train_limit_s(base, 7000, 0.1, 0, 0, 0) == 3 * base


PAUSING = r'''
import json, os, signal, sys, time, pathlib
a = sys.argv; out = pathlib.Path(a[a.index("--out") + 1]); (out / ".done").mkdir(parents=True, exist_ok=True)
r = out / "train" / "restore"
resumed = (r / "restore.json").exists()
for st in ["frames", "sfm", "align", "occupancy"]:
    print(json.dumps({"stage": st, "event": "start", "msg": ""}), flush=True)
    (out / ".done" / f"{st}.json").write_text("{}")
    print(json.dumps({"stage": st, "event": "done", "msg": ""}), flush=True)
print(json.dumps({"stage": "train", "event": "start", "msg": "resumed" if resumed else ""}), flush=True)
if resumed:  # the second machine: carries on from the restore point and finishes
    (out / "resumed-from.txt").write_text((r / "restore.json").read_text())
    print(json.dumps({"stage": "train", "event": "done", "msg": ""}), flush=True)
    (out / "scene.spz").write_bytes(b"spz"); (out / "report.json").write_text("{}")
    sys.exit(0)
def stop(*_):  # what plypipe + the trainer do on SIGTERM: save at the end of the step, exit 75
    r.mkdir(parents=True, exist_ok=True)
    (r / "restore.pt").write_bytes(b"state")
    (r / "restore.json").write_text(json.dumps({"step": 1234, "of": 30000, "key": "k"}))
    sys.exit(75)
signal.signal(signal.SIGTERM, stop)
open(os.environ["FAKE_READY"], "w").write("1")
time.sleep(60)
'''


def test_a_stopped_worker_saves_uploads_and_requeues_paused_and_another_machine_continues(tmp_path, monkeypatch):
    import threading
    import time as t
    fake = tmp_path / "pausing.py"
    fake.write_text(PAUSING)
    monkeypatch.setenv("PLY_PIPELINE_CMD", f"{sys.executable} {fake}")
    monkeypatch.setenv("FAKE_READY", str(tmp_path / "ready"))
    monkeypatch.setitem(worker.STOP, "signal", None)
    store = Store()
    th = threading.Thread(target=worker.run_job, args=(store, job(), tmp_path / "w1", []))
    th.start()
    for _ in range(200):
        if (tmp_path / "ready").exists():
            break
        t.sleep(0.05)
    worker._stop_requested(15, None)  # SIGTERM to the worker (a cloud stop, the PC's window closing)
    th.join(30)
    assert not th.is_alive()
    last = store.posts[-1]
    assert last["status"] == "queued" and last["paused"] is True and "step 1234" in last["message"]
    assert store.objects["scenes/j1/restore/restore.pt"] == b"state"
    worker.STOP.update(signal=None, proc=None)
    worker.run_job(store, job(), tmp_path / "w2", [])  # another machine
    assert (tmp_path / "w2" / "j1" / "out" / "resumed-from.txt").exists()
    assert store.posts[-1]["status"] == "review"
