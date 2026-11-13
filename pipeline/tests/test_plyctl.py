"""scripts/plyctl.py: the smoke test's upload, wait and checks, and the logs/stop/cost commands, against stand-ins.
Nothing here reaches the network or Modal."""

import importlib.util
import json
import urllib.error
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


# ---- logs, stop and cost: what phase 0.7 adds, still against stand-ins and never the network ----

JOB = {"id": "job-1", "status": "running", "stage": "sfm", "message": "sfm: matching"}
NOT_HERE = (404, {}, b'{"error":"not found"}')


class FakeJobs:
    """The API for logs and stop: one job, its log, the job list, and whether a cancel route answers."""

    def __init__(self, job=None, log=None, cancel=NOT_HERE, jobs=None):
        self.job, self.log, self.cancel, self.jobs = dict(job or JOB), log, cancel, jobs
        self.requests = []

    def send(self, method, url, data, headers):
        path = url.removeprefix(BASE)
        self.requests.append((method, path, headers))
        if path == "/v1/jobs":
            if self.jobs is None:
                return 401, {}, b'{"error":"missing or wrong key"}'
            return 200, {}, json.dumps({"jobs": self.jobs}).encode()
        if path == "/v1/jobs/job-1":
            return 200, {}, json.dumps(self.job).encode()
        if path == "/v1/jobs/job-1/cancel" and method == "POST":
            return self.cancel
        if path == "/v1/scenes/job-1/pipeline.log":
            return (200, {}, self.log) if self.log is not None else NOT_HERE
        return NOT_HERE


def logs_call(fake, tail=None, follow=False, sleep=lambda _s: None):
    said = []
    a = SimpleNamespace(job="job-1", tail=tail, follow=follow)
    return plyctl.cmd_logs(plyctl.Api(BASE, "", fake.send), a, said.append, sleep), said


def test_logs_prints_the_pipeline_log_and_its_last_lines_with_tail():
    fake = FakeJobs(log=b"frames: 64\nsfm: matching\n")
    code, said = logs_call(fake)
    assert code == 0 and said == ["frames: 64\nsfm: matching"]
    assert all("x-ply-key" not in h for _m, _p, h in fake.requests)     # a job read, and its log, are public
    assert logs_call(FakeJobs(log=b"frames: 64\nsfm: matching\n"), tail=1) == (0, ["sfm: matching"])
    assert logs_call(FakeJobs(log=b"a\nb\nc"), tail=2) == (0, ["b\nc"])  # an unfinished last line is not lost
    with pytest.raises(ValueError, match="--tail"):
        logs_call(FakeJobs(log=b"a\n"), tail=0)


def test_logs_falls_back_to_the_job_message_when_the_api_has_no_log():
    code, said = logs_call(FakeJobs(log=None))
    assert code == 0                                                    # not an error: a running job has no log yet
    assert said[0].startswith("no pipeline.log for job job-1 (running)")
    assert said[1].startswith("the running container's own log")
    assert said[-1] == "the job's message: sfm: matching"


def test_logs_follow_prints_only_what_is_new_and_stops_when_the_job_ends():
    fake, steps = FakeJobs(log=b"one\n"), []

    def sleep(_seconds):
        steps.append(1)
        if len(steps) == 1:
            fake.log += b"two\n"
        else:
            fake.job["status"] = "review"                               # the last lines land with the final push
            fake.log += b"three\n"

    code, said = logs_call(fake, follow=True, sleep=sleep)
    assert code == 0 and said == ["one", "two", "three"]


def test_logs_follow_holds_an_unfinished_line_until_it_is_complete():
    fake, steps = FakeJobs(log=b"par"), []

    def sleep(_seconds):
        steps.append(1)
        if len(steps) == 1:
            fake.log = b"partial line\n"
        else:
            fake.job["status"] = "done"

    assert logs_call(fake, follow=True, sleep=sleep) == (0, ["partial line"])


def test_logs_follow_waits_for_a_log_published_when_the_run_fails():
    fake = FakeJobs(log=None)

    def sleep(_seconds):
        fake.job["status"] = "failed"
        fake.log = b"training: CUDA out of memory\n"

    code, said = logs_call(fake, follow=True, sleep=sleep)
    assert code == 0 and said[0].startswith("no pipeline.log") and said[-1] == "training: CUDA out of memory"


def test_logs_follow_stops_when_a_job_ends_without_ever_publishing_a_log():
    fake = FakeJobs(log=None)
    code, said = logs_call(fake, follow=True, sleep=lambda _s: fake.job.update(status="review"))
    assert code == 0 and said[0].startswith("no pipeline.log")


class FakeModal:
    """The Modal CLI: answers by the exact argument list, and keeps every call. Nothing here reaches Modal."""

    def __init__(self, answers=None):
        self.calls, self.answers = [], answers or {}

    def call(self, args, timeout=120):
        self.calls.append(args)
        return self.answers.get(tuple(args), (0, "", ""))


def stop_call(fake, modal, key="the-code", profile="ricco-mallick"):
    said = []
    code = plyctl.cmd_stop(plyctl.Api(BASE, key, fake.send), SimpleNamespace(job="job-1", profile=profile),
                           said.append, modal)
    return code, said


def test_stop_stops_the_only_running_job_and_says_the_api_cannot_cancel_it():
    fake = FakeJobs(jobs=[{"id": "job-1", "status": "running"}])
    modal = FakeModal()
    code, said = stop_call(fake, modal)
    assert modal.calls == [["--profile", "ricco-mallick", "app", "stop", "ply", "-y"]]
    assert any(s.startswith("the API has no cancel route") for s in said)
    assert any(s.startswith("stopped the Modal app ply on profile ricco-mallick") for s in said)
    assert any("re-queued after 20 minutes" in s for s in said)
    assert code == 1                                                    # it will be picked up again: not cancelled


def test_stop_reports_a_cancel_route_when_the_api_has_one():
    fake = FakeJobs(cancel=(200, {}, b'{"ok":true}'), jobs=[{"id": "job-1", "status": "running"}])
    modal = FakeModal()
    code, said = stop_call(fake, modal)
    assert code == 0 and "the API cancelled job job-1" in said
    assert modal.calls == [["--profile", "ricco-mallick", "app", "stop", "ply", "-y"]]


def test_stop_leaves_the_app_alone_while_another_job_runs():
    fake = FakeJobs(jobs=[{"id": "job-1", "status": "running"}, {"id": "job-2", "status": "running"}])
    modal = FakeModal()
    code, said = stop_call(fake, modal)
    assert code == 1 and modal.calls == []
    assert any("other job(s) are running (job-2)" in s for s in said)


def test_stop_says_a_queued_job_cannot_be_cancelled_and_leaves_the_app_alone():
    fake = FakeJobs(job={"id": "job-1", "status": "queued", "message": "queued"})
    modal = FakeModal()
    code, said = stop_call(fake, modal)
    assert code == 1 and modal.calls == []
    assert any("the API has no cancel route" in s for s in said)
    assert any("still queued" in s for s in said)


def test_stop_needs_the_team_code_to_know_that_no_other_job_is_running():
    modal = FakeModal()
    code, said = stop_call(FakeJobs(), modal, key="")
    assert code == 1 and modal.calls == []
    assert any("needs the team code" in s for s in said)


def test_stop_does_nothing_to_a_job_waiting_for_review():
    fake = FakeJobs(job={"id": "job-1", "status": "review", "stage": "export", "message": "ready"})
    modal = FakeModal()
    code, said = stop_call(fake, modal)
    assert code == 0 and modal.calls == [] and not any(m == "POST" for m, _p, _h in fake.requests)
    assert any("waits for a decision" in s for s in said)


def test_a_failed_modal_stop_is_one_clean_line():
    fake = FakeJobs(jobs=[{"id": "job-1", "status": "running"}])
    box = ("╭─ Error ─────╮\n│ Modal profile 'ricco-mallick' was not found in /home/ricco/.modal.toml.     │\n"
           "│ Run `modal profile list` to see available profiles.                        │\n╰─────────────╯\n")
    modal = FakeModal({("--profile", "ricco-mallick", "app", "stop", "ply", "-y"): (1, "", box)})
    code, said = stop_call(fake, modal)
    assert code == 1
    assert any("could not stop the Modal app ply" in s and "no such modal profile" in s for s in said)
    assert "Traceback" not in " ".join(said)


PROFILES = [{"name": "ricco-mallick", "workspace": "ricco-mallick", "active": True},
            {"name": "riccomarcelino", "workspace": "riccomarcelino", "active": False}]


def summary(credits, metered, billed):
    return json.dumps({"metered_cost": metered, "billed_cost": billed,
                       "adjustments": {"credits": credits, "free_storage": "-0.10"},
                       "metered_cost_breakdown": {"deployed_apps": metered}})


def cost_call(modal, profile="all"):
    said = []
    return plyctl.cmd_cost(None, SimpleNamespace(profile=profile), said.append, modal), said


def test_cost_prints_one_table_of_credits_and_metered_cost_per_profile():
    modal = FakeModal({
        ("profile", "list", "--json"): (0, json.dumps(PROFILES), ""),
        ("--profile", "ricco-mallick", "billing", "summary", "--json"): (0, summary("-28.29", "28.67723303", "0E-8"), ""),
        ("--profile", "riccomarcelino", "billing", "summary", "--json"): (0, summary("-25.07", "25.98", "0.40"), ""),
    })
    code, said = cost_call(modal)
    assert code == 0
    assert said[0].startswith("billing cycle 20")
    assert said[1].split() == ["profile", "credits", "metered", "billed"]
    assert said[2].split() == ["ricco-mallick", "$28.29", "$28.68", "$0.00"]
    assert said[3].split() == ["riccomarcelino", "$25.07", "$25.98", "$0.40"]


def test_cost_takes_one_profile_without_listing_them_all():
    modal = FakeModal({("--profile", "riccomarcelino", "billing", "summary", "--json"): (0, summary("-1.5", "2", "0.5"), "")})
    code, said = cost_call(modal, "riccomarcelino")
    assert code == 0 and modal.calls == [["--profile", "riccomarcelino", "billing", "summary", "--json"]]
    assert said[-1].split() == ["riccomarcelino", "$1.50", "$2.00", "$0.50"]


def test_cost_says_no_such_profile_and_a_logged_out_cli_without_a_traceback():
    box = ("╭─ Error ─────╮\n│ Modal profile 'nope' was not found in /home/ricco/.modal.toml. Run `modal    │\n"
           "│ profile list` to see available profiles.                                    │\n╰─────────────╯\n")
    code, said = cost_call(FakeModal({("--profile", "nope", "billing", "summary", "--json"): (1, "", box)}), "nope")
    assert code == 1 and said[-1] == "nope: no such modal profile 'nope' (see `modal profile list`)"

    out = ("╭─ Error ─────╮\n│ Token missing. Could not authenticate client. If you have token credentials, │\n"
           "│ see modal.com/docs/sdk/py/latest/config for setup help.                      │\n╰─────────────╯\n")
    ans = {("--profile", "riccomarcelino", "billing", "summary", "--json"): (1, "", out)}
    code, said = cost_call(FakeModal(ans), "riccomarcelino")
    assert code == 1 and said[-1] == ("riccomarcelino: the modal CLI is logged out for profile 'riccomarcelino': "
                                      "run `modal token new --profile riccomarcelino`")


def test_a_missing_modal_cli_is_one_clear_error(monkeypatch, capsys):
    monkeypatch.setattr(plyctl, "modal_exe", lambda: "/nonexistent/modal")
    assert plyctl.main(["cost", "--profile", "ricco-mallick"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: the modal CLI is not installed at /nonexistent/modal")
    assert "Traceback" not in err


def test_a_dead_network_is_one_clear_error(monkeypatch, capsys):
    def offline(*_a, **_k):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(plyctl, "_send", offline)
    assert plyctl.main(["--api", BASE, "logs", "job-1"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "no route to host" in err and "Traceback" not in err
