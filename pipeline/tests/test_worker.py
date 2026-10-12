"""Worker plumbing without a GPU or the API: a stand-in pipeline that aborts in sfm, then succeeds on the SIFT retry."""

import json
import sys
from pathlib import Path

from plypipe import worker

FAKE = r'''
import json, os, sys, pathlib
a = sys.argv; out = pathlib.Path(a[a.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
print(json.dumps({"stage": "frames", "event": "done", "msg": ""}), flush=True)
print(json.dumps({"stage": "sfm", "event": "start", "msg": ""}), flush=True)
if "--features" not in a:
    print("F1013 01:23:45 feature_extraction.cc:123] Check failed: onnx session", flush=True)
    os.abort()
for st in ["align", "occupancy", "train", "export"]:
    print(json.dumps({"stage": st, "event": "done", "msg": ""}), flush=True)
(out / "scene.spz").write_bytes(b"spz"); (out / "report.json").write_text("{}")
print(json.dumps({"stage": "export", "event": "done", "msg": "args=" + " ".join(a[4:])}), flush=True)
'''


class FakeApi:
    def __init__(self):
        self.posts, self.uploads = [], []

    def req(self, method, path, body=None, ctype=None, tries=5):
        self.posts.append(body)
        return 200, {"ok": True}

    def download(self, key, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"video")

    def upload(self, key, path, ctype):
        self.uploads.append(key)


def run(tmp_path, monkeypatch, script):
    fake = tmp_path / "fake.py"
    fake.write_text(script)
    monkeypatch.setenv("PLY_PIPELINE_CMD", f"{sys.executable} {fake}")
    api = FakeApi()
    worker.run_job(api, {"id": "j1", "files": ["captures/c/clip-01.mp4"], "args": ["--preset", "quick"]}, tmp_path / "work", [])
    return api


def test_sfm_crash_is_retried_with_sift_and_succeeds(tmp_path, monkeypatch):
    api = run(tmp_path, monkeypatch, FAKE)
    final = api.posts[-1]
    assert final["status"] == "done"
    assert any("retrying with SIFT" in (p.get("message") or "") for p in api.posts if isinstance(p, dict))
    assert "scenes/j1/scene.spz" in api.uploads


def test_failure_message_carries_the_crash_output(tmp_path, monkeypatch):
    api = run(tmp_path, monkeypatch, FAKE.replace('if "--features" not in a:', 'if True:'))
    final = api.posts[-1]
    assert final["status"] == "failed"
    assert "Check failed: onnx session" in final["message"] and "exited with -6" in final["message"]
    assert "scenes/j1/pipeline.log" in api.uploads
