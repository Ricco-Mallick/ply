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
    assert store.posts[-1]["status"] == "done"
    assert store.clip_downloads == 1                                  # resumed past SfM: no second 179 MB download
    assert any("resuming after occupancy" in (p.get("message") or "") for p in store.posts if isinstance(p, dict))


def test_checkpoint_from_other_settings_is_ignored(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    store = Store()
    monkeypatch.setenv("FAKE_CRASH_AT", "train")
    worker.run_job(store, job(), tmp_path / "w1", [])
    monkeypatch.delenv("FAKE_CRASH_AT")
    worker.run_job(store, job(("--preset", "quick", "--features", "sift")), tmp_path / "w2", [])
    assert store.posts[-1]["status"] == "done" and store.clip_downloads == 2   # started over


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
