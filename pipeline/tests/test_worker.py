"""Worker plumbing without a GPU or the API: a stand-in pipeline that aborts in sfm, then succeeds on the SIFT retry."""

import json
import sys
import urllib.parse
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
print("I20261013 07:20:21.222608 1354 pairing.cc:208] Processing block [2/2, 1/2]", flush=True)
import time; time.sleep(float(os.environ.get("FAKE_SFM_SLEEP", "0")))
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

    def upload_many(self, items):
        self.uploads.extend(key for key, _, _ in items)


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
    assert final["status"] == "review"
    assert any("retrying with SIFT" in (p.get("message") or "") for p in api.posts if isinstance(p, dict))
    assert "scenes/j1/scene.spz" in api.uploads


def test_failure_message_carries_the_crash_output(tmp_path, monkeypatch):
    api = run(tmp_path, monkeypatch, FAKE.replace('if "--features" not in a:', 'if True:'))
    final = api.posts[-1]
    assert final["status"] == "failed"
    assert "Check failed: onnx session" in final["message"] and "exited with -6" in final["message"]
    assert "scenes/j1/pipeline.log" in api.uploads


def test_sfm_progress_reaches_the_status_page(tmp_path, monkeypatch):
    """COLMAP's own matching counter moves the job's progress and message while sfm runs."""
    monkeypatch.setenv("FAKE_SFM_SLEEP", "22")  # long enough for one 20 s heartbeat inside sfm
    api = run(tmp_path, monkeypatch, FAKE)
    beat = [p for p in api.posts if isinstance(p, dict) and (p.get("message") or "").startswith("sfm: matching frames")]
    assert beat and beat[-1]["message"] == "sfm: matching frames · 50%" and beat[-1]["progress"] > 0.05 + 0.2 * 0.3


# publishing the evidence frames: one Hugging Face commit for the folder, one bucket request per file otherwise
EVIDENCE = FAKE.replace('(out / "scene.spz")', """ev = out / "evidence"; (ev / "frames").mkdir(parents=True, exist_ok=True)
for i in range(60): (ev / "frames" / f"{i:03d}.jpg").write_bytes(b"jpg")
(ev / "poses.json").write_text("[]")
(out / "scene.spz")""", 1)
EVIDENCE_KEY = "scenes/j1/evidence/"


class BucketApi(worker.Api):
    """The real routing between Hugging Face and the bucket, with the network replaced by a record of what was sent."""

    def __init__(self):
        super().__init__("http://api.invalid", "token")
        self.posts, self.puts, self.fail_puts = [], [], False

    def req(self, method, path, body=None, ctype=None, tries=5):
        if method == "PUT":
            if self.fail_puts:
                raise RuntimeError("bucket down")
            self.puts.append((urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)["key"][0], ctype))
        else:
            self.posts.append(body)
        return 200, {"ok": True}

    def download(self, key, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"video")


class FakeHf:
    def __init__(self, fail_batch=False):
        self.batches, self.singles, self.fail_batch = [], [], fail_batch

    def install(self, monkeypatch, configured=True):
        monkeypatch.setattr(worker.hf, "configured", lambda: configured)
        monkeypatch.setattr(worker.hf, "upload", lambda key, path: self.singles.append(key))
        monkeypatch.setattr(worker.hf, "upload_many", self.upload_many)
        return self

    def upload_many(self, files):
        if self.fail_batch:
            raise RuntimeError("commit rejected")
        self.batches.append([key for key, _ in files])


def publish(tmp_path, monkeypatch, api):
    fake = tmp_path / "fake.py"
    fake.write_text(EVIDENCE)
    monkeypatch.setenv("PLY_PIPELINE_CMD", f"{sys.executable} {fake}")
    worker.run_job(api, {"id": "j1", "files": ["captures/c/clip-01.mp4"], "args": ["--preset", "quick"]}, tmp_path / "work", [])
    return api


def test_evidence_goes_to_hugging_face_in_one_commit(tmp_path, monkeypatch):
    hf = FakeHf().install(monkeypatch)
    api = publish(tmp_path, monkeypatch, BucketApi())
    batch = [b for b in hf.batches if any(k.startswith(EVIDENCE_KEY) for k in b)]
    assert len(batch) == 1
    frames = [k for k in batch[0] if k.startswith(EVIDENCE_KEY + "frames/")]
    assert len(frames) == 48 and frames[0].endswith("/000.jpg") and frames[-1].endswith("/059.jpg")   # the spread of 60
    assert batch[0][-1] == EVIDENCE_KEY + "poses.json" and len(batch[0]) == 49
    assert not [k for k in hf.singles if k.startswith(EVIDENCE_KEY)] and not api.puts
    assert api.posts[-1]["status"] == "review"


def test_evidence_goes_file_by_file_to_the_bucket_without_hugging_face(tmp_path, monkeypatch):
    hf = FakeHf().install(monkeypatch, configured=False)
    api = publish(tmp_path, monkeypatch, BucketApi())
    sent = [(k, c) for k, c in api.puts if k.startswith(EVIDENCE_KEY)]
    assert len(sent) == 49 and not hf.batches and not hf.singles
    assert sent[0] == (EVIDENCE_KEY + "frames/000.jpg", "image/jpeg")
    assert sent[-1] == (EVIDENCE_KEY + "poses.json", "application/json")
    assert api.posts[-1]["status"] == "review"


def test_a_rejected_commit_sends_the_evidence_to_the_bucket_not_to_per_file_commits(tmp_path, monkeypatch):
    hf = FakeHf(fail_batch=True).install(monkeypatch)
    api = publish(tmp_path, monkeypatch, BucketApi())
    assert len([k for k, _ in api.puts if k.startswith(EVIDENCE_KEY)]) == 49
    assert not [k for k in hf.singles if k.startswith(EVIDENCE_KEY)]
    assert api.posts[-1]["status"] == "review"


def test_a_failing_evidence_upload_never_fails_the_job(tmp_path, monkeypatch, capsys):
    FakeHf(fail_batch=True).install(monkeypatch)
    api = BucketApi()
    real = api.req
    api.req = lambda method, path, *a, **k: (_ for _ in ()).throw(RuntimeError("bucket down")) if "evidence" in path else real(method, path, *a, **k)
    publish(tmp_path, monkeypatch, api)
    assert "evidence upload failed: bucket down" in capsys.readouterr().out
    assert api.posts[-1]["status"] == "review"
