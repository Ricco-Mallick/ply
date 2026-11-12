"""scripts/plyctl.py: the smoke test's upload, wait and checks against a stand-in API. Nothing here reaches the network."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "plyctl.py"
spec = importlib.util.spec_from_file_location("plyctl", SCRIPT)
plyctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plyctl)

BASE = "https://api.test"
FILES = {n: 10 for n in (*plyctl.REQUIRED, "topview.webp", "scene.ply")}


def finished(status="review", **change):
    report = {s: {"seconds": 1} for s in plyctl.STAGES}
    report["sfm"] = {"registered": 64, "images": 66, "mapper": "global", "mapperReason": "66 frames: below 400"}
    report["train"] = {"maxSteps": 2000, "seconds": 61.0}
    report["export"] = {"seconds": 1, "liteFinetune": {"steps": 3000, "shipped": "finetuned", "before": {"psnr": 18.9}, "after": {"psnr": 20.4},
                                                       "seconds": 48.0}}
    result = {"report": report, "files": dict(FILES), "quality": {"ok": True, "issues": []}, "attempts": [{"attempt": 1}]}
    return {"id": "job-1", "status": status, "stage": "export", "message": "ready for review", "result": {**result, **change}}


def test_a_file_is_cut_into_the_apps_parts():
    mb = 1024 * 1024
    assert plyctl.parts(40 * mb) == [(1, 0, 16 * mb), (2, 16 * mb, 16 * mb), (3, 32 * mb, 8 * mb)]
    assert plyctl.parts(16 * mb) == [(1, 0, 16 * mb)] and plyctl.parts(1) == [(1, 0, 1)]
    with pytest.raises(ValueError, match="empty"):
        plyctl.parts(0)


def test_a_finished_job_with_every_stage_and_file_passes():
    problems, notes = plyctl.check(finished())
    assert problems == [] and notes == ["64 of 66 views solved, global mapper (66 frames: below 400); 2000 steps in 61.0 s",
                                        "phone file: 18.9 dB, 20.4 dB after 3000 steps (48.0 s); shipped the finetuned one"]
    assert plyctl.check(finished("done"))[0] == []


def test_a_failed_gate_is_said_and_does_not_fail_the_smoke_test():
    gate = {"ok": False, "issues": [{"level": "fail", "code": "tiny"}, {"level": "warn", "code": "held-out"}]}
    problems, notes = plyctl.check(finished(quality=gate, attempts=[{}, {}]))
    assert problems == [] and notes[0] == "gate: fail tiny, warn held-out" and notes[1] == "2 attempts: the gate asked for a retry"


def test_what_fails_the_smoke_test():
    assert plyctl.check({"id": "j", "status": "failed", "message": "sfm: no model"})[0] == ["the job ended as failed: sfm: no model"]
    report = finished()["result"]["report"]
    del report["label"]
    assert plyctl.check(finished(report=report))[0] == ["no report from the label stage"]
    assert plyctl.check(finished(quality=None))[0] == ["the quality gate did not run"]
    files = {k: v for k, v in FILES.items() if k != "scene_lite.spz"}
    assert plyctl.check(finished(files=files))[0] == ["scene_lite.spz was not published"]
    problems, notes = plyctl.check(finished(), served=lambda path: not path.endswith(("occupancy.png", "topview.webp")))
    assert problems == ["occupancy.png is listed, and the API does not serve it"]
    assert notes[0] == "topview.webp is not there"                                 # a short walk may have no room outline: said, not failed


def test_the_phone_files_fine_tune_must_have_run_to_the_end():
    def with_tune(tune):
        report = finished()["result"]["report"]
        report["export"] = {"seconds": 1, **({"liteFinetune": tune} if tune else {})}
        return plyctl.check(finished(report=report))

    # measured and not better: the export kept the untuned file, as it should
    problems, notes = with_tune({"steps": 3000, "shipped": "untuned", "why": "measured: the fine-tuned model was not the better one",
                                 "before": {"psnr": 20.0}, "after": {"psnr": 19.8}, "seconds": 50.0})
    assert problems == [] and notes[-1] == "phone file: 20.0 dB, 19.8 dB after 3000 steps (50.0 s); shipped the untuned one"
    # the export falls back without failing the job, so the smoke test is where a broken fine-tune shows
    assert with_tune({"steps": 3000, "shipped": "untuned", "why": "CUDA out of memory"})[0] == ["the phone file's fine-tune did not finish: CUDA out of memory"]
    assert with_tune({"steps": 3000, "shipped": "untuned", "why": "not measured: needs a GPU and held-out views"})[0][0].startswith("the phone file's fine-tune did not finish")
    assert with_tune(None)[0][0].startswith("the export did not fine-tune the phone file")


class FakeService:
    """The API and the storage behind it: answers what plyctl asks and keeps every request."""

    def __init__(self, statuses=("queued", "running", "review")):
        self.requests, self.statuses, self.stored = [], list(statuses), {}

    def send(self, method, url, data, headers):
        self.requests.append((method, url, headers, data))
        if url.startswith("https://storage.test/"):
            self.stored[url] = data
            return 200, {"ETag": f'"etag-{len(self.stored)}"'}, b""
        path = url.removeprefix(BASE)
        body = json.loads(data) if data else None
        if path == "/v1/whoami":
            return (200, {}, b'{"role":"invite","label":"team"}') if headers.get("x-ply-key") == "the-code" else (401, {}, b'{"error":"no invite"}')
        if path == "/v1/uploads":
            self.key = f"captures/{body['captureId']}/{body['name']}"
            return 200, {}, json.dumps({"key": self.key, "uploadId": "up-1"}).encode()
        if path == "/v1/uploads/sign":
            return 200, {}, json.dumps({"urls": {str(n): f"https://storage.test/{n}" for n in body["parts"]}}).encode()
        if path == "/v1/uploads/complete":
            self.completed = body
            return 200, {}, b"{}"
        if path == "/v1/jobs" and method == "POST":
            self.job = body
            return 201, {}, b'{"jobId":"job-1"}'
        if path == "/v1/jobs/job-1":
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return 200, {}, json.dumps(finished(status) if status in plyctl.ENDED else {"id": "job-1", "status": status, "stage": "sfm"}).encode()
        if method == "HEAD":
            return 200, {}, b""
        return 404, {}, b'{"error":"not found"}'


def run_smoke(tmp_path, monkeypatch, service, size=20 * 1024 * 1024, minutes=45):
    clip = tmp_path / "walk.mp4"
    clip.write_bytes(b"v" * size)
    monkeypatch.setattr(plyctl, "PART", 16 * 1024 * 1024)
    said, clock = [], iter(range(0, 10**6, 60))
    api = plyctl.Api(BASE, "the-code", service.send)
    code = plyctl.cmd_smoke(api, SimpleNamespace(clip=clip, minutes=minutes), said.append, lambda s: None, lambda: next(clock))
    return code, said


def test_smoke_uploads_in_parts_queues_a_short_job_and_passes(tmp_path, monkeypatch):
    service = FakeService()
    code, said = run_smoke(tmp_path, monkeypatch, service)
    assert code == 0 and said[-1] == "smoke passed: job job-1"
    assert [len(d) for d in service.stored.values()] == [16 * 1024 * 1024, 4 * 1024 * 1024]
    assert service.completed["parts"] == [{"partNumber": 1, "etag": '"etag-1"'}, {"partNumber": 2, "etag": '"etag-2"'}]
    job = service.job
    assert job["args"] == ["--preset", "quick", "--iterations", "2000", "--sh-degree", "3"] and job["files"] == [service.key]
    assert job["captureId"].startswith("smoke-") and service.key == f"captures/{job['captureId']}/walk.mp4"
    assert "--until" not in job["args"]                                             # a full run: the worker judges and publishes it
    # the team code goes to the API and never to the storage links
    assert all(h.get("x-ply-key") == "the-code" for m, u, h, d in service.requests if u.startswith(BASE) and m != "HEAD")
    assert not any("x-ply-key" in h for m, u, h, d in service.requests if u.startswith("https://storage.test/"))


def test_smoke_fails_on_a_failed_job_and_gives_up_on_one_that_never_ends(tmp_path, monkeypatch):
    code, said = run_smoke(tmp_path, monkeypatch, FakeService(("running", "failed")))
    assert code == 1 and said[-1] == "smoke failed: job job-1" and any(s.startswith("FAILED: the job ended as failed") for s in said)
    code, said = run_smoke(tmp_path, monkeypatch, FakeService(("running",)), minutes=3)
    assert code == 2 and "smoke --job job-1" in said[-1]


def test_a_queued_job_is_picked_up_again_without_uploading_or_a_team_code():
    service, said, clock = FakeService(("running", "review")), [], iter(range(0, 10**6, 60))
    args = SimpleNamespace(clip=None, job="job-1", minutes=45)
    code = plyctl.cmd_smoke(plyctl.Api(BASE, "", service.send), args, said.append, lambda s: None, lambda: next(clock))
    assert code == 0 and said[-1] == "smoke passed: job job-1"
    assert not any(m in ("POST", "PUT") for m, u, h, d in service.requests) and not service.stored


def test_a_refused_team_code_stops_before_anything_is_uploaded(tmp_path, monkeypatch):
    service = FakeService()
    clip = tmp_path / "walk.mp4"
    clip.write_bytes(b"v")
    with pytest.raises(RuntimeError, match="401"):
        plyctl.cmd_smoke(plyctl.Api(BASE, "wrong", service.send), SimpleNamespace(clip=clip, minutes=1), lambda s: None)
    assert [u for m, u, h, d in service.requests] == [BASE + "/v1/whoami"]
    assert plyctl.main(["--api", BASE, "smoke", "--clip", str(clip)]) == 2          # no code at all: refused before any request
