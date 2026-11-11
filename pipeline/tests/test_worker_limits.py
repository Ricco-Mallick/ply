"""What one job may spend: the GPU-time cap (also across the CPU camera solve), the GPU's share of a split solve, and the
size of the clips it may download. Stand-in pipeline and API, as in test_worker_resume."""

import io
import json
import sys
import time
import urllib.request

import pytest

from plypipe import handoff, limits, orchestrate, verify, worker
from test_handoff import FAKE as HANDOFF_FAKE
from test_worker_resume import Store, job, setup


def messages(store):
    return [p.get("message") or "" for p in store.posts if isinstance(p, dict)]


def test_a_normal_job_is_unchanged_by_the_cap(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    for name in ("PLY_STAGE_LIMITS", "PLY_JOB_MAX_MIN", "PLY_MAX_INPUT_MB"):
        monkeypatch.delenv(name, raising=False)
    assert worker.stage_limits() == limits.STAGE_LIMITS_MIN and worker.job_cap_min() == 240
    store = Store()
    worker.run_job(store, job(), tmp_path / "w", [])
    assert store.posts[-1]["status"] == "review" and "scenes/j1/checkpoint.tar" in store.objects
    assert not any("time cap" in m or "input limit" in m for m in messages(store))


def test_the_job_cap_stops_a_long_stage_and_names_itself(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    store = Store()
    monkeypatch.setenv("FAKE_SLOW_AT", "sfm")
    monkeypatch.setenv("PLY_JOB_MAX_MIN", "0.1")                    # 6 seconds, far inside sfm's own limit
    t0 = time.time()
    worker.run_job(store, job(), tmp_path / "w", [])
    final = store.posts[-1]
    assert final["status"] == "failed"
    assert "sfm was stopped: the job reached its 0.1-minute time cap (GPU time since it was claimed)" in final["message"]
    assert time.time() - t0 < 40 and not any("retrying" in m for m in messages(store))


def test_training_stopped_by_the_job_cap_still_exports_its_snapshot(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    store = Store()
    monkeypatch.setenv("FAKE_SLOW_AT", "train")
    monkeypatch.setenv("FAKE_SNAPSHOT", "1")
    monkeypatch.setenv("PLY_JOB_MAX_MIN", "0.2")                    # shorter than export's reserve: training stops at once
    worker.run_job(store, job(), tmp_path / "w1", [])
    assert store.posts[-1]["status"] == "review", store.posts[-1]
    assert any("exporting that snapshot" in m and "the job reached its 0.2-minute time cap" in m for m in messages(store))
    assert (tmp_path / "w1" / "j1" / "out" / "scene.ply").read_bytes() == b"ply snapshot"
    assert store.objects["scenes/j1/scene.ply"] == b"ply snapshot" and "scenes/j1/checkpoint.tar" in store.objects


def test_the_clip_download_counts_toward_the_cap(tmp_path, monkeypatch):
    class Slow(Store):
        def download(self, key, dest):
            super().download(key, dest)
            time.sleep(4)
    store = Slow()
    monkeypatch.setenv("PLY_JOB_MAX_MIN", "0.05")                   # 3 seconds
    setup(tmp_path, monkeypatch)
    worker.run_job(store, {**job(), "files": ["captures/c/clip-01.mp4", "captures/c/clip-02.mp4"]}, tmp_path / "w", [])
    assert store.posts[-1]["status"] == "failed" and "downloading clips was stopped: the job reached its 0.05-minute time cap" in store.posts[-1]["message"]
    assert store.clip_downloads == 1


# the split solve: the stand-in from test_handoff, with optional sleeps in matching and align
SPLIT = HANDOFF_FAKE.replace("import json, sys, pathlib", "import json, os, sys, pathlib, time", 1).replace(
    '    if st == "align":  # the real align',
    '    if st == "align" and os.environ.get("FAKE_ALIGN_SLEEP"):\n        time.sleep(float(os.environ["FAKE_ALIGN_SLEEP"]))\n    if st == "align":  # the real align', 1).replace(
    '    if st == "sfm" and "--sfm-part" in a:\n',
    '    if st == "sfm" and "--sfm-part" in a:\n        time.sleep(float(os.environ.get("FAKE_MATCH_SLEEP", "0")))\n', 1)
assert "FAKE_ALIGN_SLEEP" in SPLIT and "FAKE_MATCH_SLEEP" in SPLIT


def split_setup(tmp_path, monkeypatch):
    fake = tmp_path / "fake.py"
    fake.write_text(SPLIT)
    hd = tmp_path / "handoff"
    hd.mkdir()
    monkeypatch.setenv("PLY_PIPELINE_CMD", f"{sys.executable} {fake}")
    monkeypatch.setenv("PLY_HANDOFF_DIR", str(hd))
    monkeypatch.setenv("PLY_CPU_SFM", "1")
    monkeypatch.setattr(handoff, "spawn", lambda fn, *a: None)
    return fake, hd


def set_used(hd, seconds):
    f = hd / "j1" / handoff.MARK
    mark = json.loads(f.read_text())
    mark["job"]["capUsedS"] = seconds
    f.write_text(json.dumps(mark))


def test_the_cap_is_carried_across_the_cpu_solve(tmp_path, monkeypatch):
    fake, hd = split_setup(tmp_path, monkeypatch)
    monkeypatch.setenv("PLY_JOB_MAX_MIN", "1")
    store = Store()
    worker.run_job(store, job(), tmp_path / "work", [])               # GPU: frames and matching, then handed over
    mark = json.loads((hd / "j1" / handoff.MARK).read_text())
    assert mark["state"] == "matched" and mark["job"]["id"] == "j1" and 0 <= mark["job"]["capUsedS"] < 30
    set_used(hd, 57)                                                  # as if matching had taken 57 s of the minute
    assert handoff.solve("j1", hd, Store(), work=tmp_path / "solve", runner=[sys.executable, str(fake)])
    assert json.loads((hd / "j1" / handoff.MARK).read_text())["job"]["capUsedS"] == 57   # the solve box passes it on
    monkeypatch.setenv("FAKE_ALIGN_SLEEP", "60")                      # alone it would fit the cap; with the 57 s it cannot
    t0 = time.time()
    worker.run_job(store, job(), tmp_path / "work2", [])
    final = store.posts[-1]
    assert final["status"] == "failed" and "align was stopped: the job reached its 1-minute time cap" in final["message"]
    assert time.time() - t0 < 30 and store.clip_downloads == 1


def test_a_job_whose_solver_died_keeps_counting_on_the_gpu(tmp_path, monkeypatch):
    fake, hd = split_setup(tmp_path, monkeypatch)
    monkeypatch.setenv("PLY_JOB_MAX_MIN", "1")
    store = Store()
    worker.run_job(store, job(), tmp_path / "work", [])
    set_used(hd, 60)                                                  # still "matched": the solver never reported back
    worker.run_job(store, job(), tmp_path / "work2", [])
    final = store.posts[-1]
    assert final["status"] == "failed" and "had already used its 1-minute time cap" in final["message"]
    assert not (tmp_path / "work2" / "j1" / "pipeline.log").exists() and not (hd / "j1").exists()


def test_the_gpus_share_of_a_split_solve_has_its_own_limit(tmp_path, monkeypatch):
    fake, hd = split_setup(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MATCH_SLEEP", "4")
    monkeypatch.setenv("PLY_STAGE_LIMITS", "sfm=0.05")                # 3 s: the whole-solve limit does not govern matching alone
    store = Store()
    worker.run_job(store, job(), tmp_path / "work", [])
    assert handoff.state("j1", hd) == "matched" and not any(p.get("status") == "failed" for p in store.posts if isinstance(p, dict))
    monkeypatch.setenv("PLY_STAGE_LIMITS", "sfm_match=0.05")
    second = Store()
    worker.run_job(second, {**job(), "id": "j2"}, tmp_path / "work2", [])
    assert second.posts[-1]["status"] == "failed" and "sfm exceeded its 0.05-minute limit" in second.posts[-1]["message"]


# a stand-in that finishes every stage, optionally slowly in one, for the quality-gate retry
GATED = r'''
import json, os, sys, time, pathlib
a = sys.argv; out = pathlib.Path(a[a.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
for st in ["frames", "sfm", "align", "occupancy", "train", "export"]:
    print(json.dumps({"stage": st, "event": "start", "msg": ""}), flush=True)
    if st == os.environ.get("FAKE_SLEEP_AT"):
        time.sleep(float(os.environ["FAKE_SLEEP"]))
    print(json.dumps({"stage": st, "event": "done", "msg": ""}), flush=True)
(out / "scene.spz").write_bytes(b"spz"); (out / "report.json").write_text("{}")
'''


def gated_job(tmp_path, monkeypatch, cap, sleep):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake = tmp_path / "gated.py"
    fake.write_text(GATED)
    monkeypatch.setenv("PLY_PIPELINE_CMD", f"{sys.executable} {fake}")
    monkeypatch.setenv("PLY_JOB_MAX_MIN", cap)
    monkeypatch.setenv("FAKE_SLEEP_AT", "occupancy")
    monkeypatch.setenv("FAKE_SLEEP", sleep)
    monkeypatch.setattr(verify, "check", lambda report, plan: {"ok": False, "issues": []})
    monkeypatch.setattr(orchestrate, "refine", lambda report, args, plan: (["--preset", "high"], ["too soft"], "frames"))
    store = Store()
    worker.run_job(store, job(), tmp_path / "w", [])
    return store


def test_a_retry_the_cap_cannot_pay_for_is_skipped_and_the_finished_attempt_is_kept(tmp_path, monkeypatch):
    store = gated_job(tmp_path, monkeypatch, "240", "1")
    assert store.posts[-1]["status"] == "review" and len(store.posts[-1]["result"]["attempts"]) == 2   # room: the retry runs
    store = gated_job(tmp_path / "capped", monkeypatch, "0.1", "4")                   # 6 s cap, a 4 s attempt: 2 s left
    assert store.posts[-1]["status"] == "review" and len(store.posts[-1]["result"]["attempts"]) == 1


class Sized(Store):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def input_size(self, key):
        return self.sizes.get(key)


def test_inputs_over_the_size_cap_are_refused_before_any_download(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    monkeypatch.setenv("PLY_MAX_INPUT_MB", "100")
    store = Sized({"captures/c/clip-01.mp4": 80 * 2**20, "captures/c/clip-02.mp4": 40 * 2**20})
    worker.run_job(store, {**job(), "files": list(store.sizes)}, tmp_path / "w", [])
    final = store.posts[-1]
    assert final["status"] == "failed" and "the clips total 0.1 GB, over this worker's 0.0976562 GB input limit (PLY_MAX_INPUT_MB)" in final["message"]
    assert store.clip_downloads == 0 and not (tmp_path / "w" / "j1" / "pipeline.log").exists()


def test_the_chapel_sized_capture_is_accepted(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    monkeypatch.delenv("PLY_MAX_INPUT_MB", raising=False)
    keys = [f"captures/c/clip-0{i}.mp4" for i in range(1, 9)]
    store = Sized({k: int(4.1e9 / 8) for k in keys})                  # eight 4K clips, 4.1 GB
    worker.run_job(store, {**job(), "files": keys}, tmp_path / "w", [])
    assert store.posts[-1]["status"] == "review" and store.clip_downloads == 8


def test_a_size_the_server_would_not_say_is_caught_once_the_file_is_down(tmp_path, monkeypatch):
    class Big(Store):
        def download(self, key, dest):
            super().download(key, dest)
            dest.write_bytes(b"x" * 2000)
    setup(tmp_path, monkeypatch)
    monkeypatch.setenv("PLY_MAX_INPUT_MB", "0.001")                   # 1,048 bytes
    store = Big()
    worker.run_job(store, job(), tmp_path / "w", [])
    assert store.posts[-1]["status"] == "failed" and "input limit (PLY_MAX_INPUT_MB)" in store.posts[-1]["message"]
    assert not (tmp_path / "w" / "j1" / "pipeline.log").exists()


class Resp(io.BytesIO):
    def __init__(self, body, headers, status=200):
        super().__init__(body)
        self.headers, self.status = headers, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class LinkApi(worker.Api):
    def __init__(self):
        super().__init__("http://api.invalid", "token")

    def req(self, method, path, body=None, ctype=None, tries=5):
        return 200, {"url": "http://bucket.invalid/clip"}


def test_the_size_is_read_from_the_headers_and_a_download_stops_at_the_cap(tmp_path, monkeypatch):
    api, seen, answer = LinkApi(), [], []

    def urlopen(request, timeout=0):
        seen.append(request.get_header("Range"))
        if isinstance(answer[0], Exception):
            raise answer[0]
        return answer[0]
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    answer[:] = [Resp(b"x", {"content-range": "bytes 0-0/4400000000"}, 206)]
    assert api.input_size("captures/c/a.mp4") == 4_400_000_000 and seen == ["bytes=0-0"]
    answer[:] = [Resp(b"xyz", {"content-length": "3"}, 200)]                        # a server that ignores Range
    assert api.input_size("captures/c/a.mp4") == 3
    answer[:] = [Resp(b"", {"content-type": "text/html", "content-length": "90"}, 200)]
    assert api.input_size("gdrive:" + "a" * 25 + ":clip.mp4") is None
    answer[:] = [OSError("offline")]
    assert api.input_size("captures/c/a.mp4") is None

    dest = tmp_path / "in" / "a.mp4"
    monkeypatch.setenv("PLY_MAX_INPUT_MB", "0.001")
    answer[:] = [Resp(b"x" * 3000, {})]
    with pytest.raises(RuntimeError, match="larger than this worker's .* input limit"):
        api.download("captures/c/a.mp4", dest)
    assert not dest.exists()
    monkeypatch.delenv("PLY_MAX_INPUT_MB")
    answer[:] = [Resp(b"x" * 3000, {})]
    api.download("captures/c/a.mp4", dest)
    assert dest.read_bytes() == b"x" * 3000
