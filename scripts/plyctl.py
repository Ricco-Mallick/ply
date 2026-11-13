#!/usr/bin/env python3
# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Operate the live Ply service from a shell: what is running, a job's log, stopping work, the cycle's cost, and a
smoke test for after every worker deploy.

    python3 scripts/plyctl.py status                  # workers, queue, and with the team code the newest jobs
    python3 scripts/plyctl.py status --job <id>       # one job
    python3 scripts/plyctl.py logs <id>               # the job's pipeline log, or its message when there is no log
    python3 scripts/plyctl.py stop <id>               # cancel the job if the API can, and end the GPU app if it is alone
    python3 scripts/plyctl.py cost [--profile all|name]   # credits and metered cost this cycle, per Modal account
    python3 scripts/plyctl.py smoke --clip clip.mp4   # a short clip through the real API, then every result is checked
    python3 scripts/plyctl.py smoke --job <id>        # the same wait and checks for a job that is already queued

`smoke` uploads the clip in parts as the capture app does, queues a Quick job of 2,000 training steps at SH 3, waits
for it and checks that every stage reported, that the gate ran, that the export's fine-tune of the phone file ran to
the end, and that the API serves each result file. It exits 0 when
all of that holds, 1 when a check fails and 2 when the job is still running at the time limit. The job stays in
review: rejecting it would queue it again. A capture under 400 frames is solved on the GPU box, so the hand-off to
the CPU solver is not covered.

`logs` reads scenes/<job>/pipeline.log through the API. A worker keeps that file next to the results when a run
fails, so a job that never failed has none: `logs` then prints the job's message and says the log is not there.
`--follow` keeps reading while the job is queued or running. `stop` asks the API's cancel route (there is none yet,
and it says so), then stops the `ply` Modal app on its profile only when this is the only running job, because the
app is shared. `cost` runs `modal billing summary --json` for each profile and prints credits, metered and billed
cost for the cycle; the summary holds no secret. The Modal CLI is `modal` on PATH, else pipeline/.venv/bin/modal, or
PLY_MODAL.

The API is PLY_API (default: the deployed one). The team code is read from PLY_TEAM_KEY or from the first line of
--key-file; it is sent to the API only, never to the storage links. The clip is not in the repository: a 30-second
walk is kept as smoke/smoke-chapel-30s.mp4 on the `ply-bench` Volume (`modal volume get ply-bench smoke/... .`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

API = "https://ply-api.ricco-ply.workers.dev"
PART = 16 * 1024 * 1024  # the capture app's part size; storage wants at least 5 MB for every part but the last
# Quick, but at SH 3: the phone file then has one SH band less than the twin, so the export fine-tunes it as it does
# for a Standard or High job
SMOKE_ARGS = ["--preset", "quick", "--iterations", "2000", "--sh-degree", "3"]
STAGES = ("frames", "sfm", "align", "occupancy", "label", "train", "export")
REQUIRED = ("scene.spz", "scene_lite.spz", "scene.json", "report.json", "occupancy.json", "occupancy.png", "places.json")
EXPECTED = ("topview.webp", "evidence/poses.json")  # a short walk may have no wall ring to draw; evidence never fails a job
ENDED = ("review", "done", "failed")
ACTIVE = ("queued", "running")  # still going: stop can act on it, and logs --follow keeps reading
FOLLOW_EVERY = 3.0  # seconds between reads of a followed log
MODAL_APP = "ply"  # pipeline/modal/ply_modal.py: modal.App("ply")
DEFAULT_PROFILE = "ricco-mallick"  # where the GPU app is expected: --profile or MODAL_PROFILE says otherwise
MODAL_TIMEOUT = 120.0  # a Modal CLI call that has not answered in two minutes is not going to


def say(line: str) -> None:
    print(line, flush=True)  # a log file or a pipe must show progress while the job runs


class Api:
    """The Ply API over urllib. `send(method, url, body, headers)` returns (status, headers, bytes): tests replace it."""

    def __init__(self, base: str = API, key: str = "", send=None):
        self.base, self.key, self.send = base.rstrip("/"), key.strip(), send or _send

    def request(self, method: str, path: str, body=None, content_type: str = "application/json"):
        """One request: (status, headers, bytes). `call` raises on an error status; this one does not."""
        data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
        headers = {"content-type": content_type} if data is not None else {}
        if self.key:
            headers["x-ply-key"] = self.key
        return self.send(method, self.base + path, data, headers)

    def call(self, method: str, path: str, body=None, content_type: str = "application/json"):
        status, _, raw = self.request(method, path, body, content_type)
        if status >= 400:
            raise RuntimeError(f"{method} {path}: {status} {raw[:200].decode('utf-8', 'replace')}")
        return json.loads(raw) if raw else None

    def served(self, path: str) -> bool:
        return self.send("HEAD", self.base + path, None, {})[0] == 200

    def put_part(self, url: str, data: bytes) -> str:
        status, headers, raw = self.send("PUT", url, data, {})  # a signed storage link: the team code stays out of it
        etag = {k.lower(): v for k, v in headers.items()}.get("etag")
        if status >= 400 or not etag:
            raise RuntimeError(f"storage refused a part: {status} {raw[:200].decode('utf-8', 'replace')}")
        return etag


def _send(method: str, url: str, data: bytes | None, headers: dict):
    req = urllib.request.Request(url, data=data, method=method, headers={"user-agent": "plyctl", **headers})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def parts(size: int, part: int = PART) -> list[tuple[int, int, int]]:
    """(part number, offset, length) for a file of `size` bytes."""
    if size <= 0:
        raise ValueError("the clip is empty")
    return [(i + 1, off, min(part, size - off)) for i, off in enumerate(range(0, size, part))]


def upload(api: Api, capture_id: str, clip: Path, say=say) -> str:
    """One clip to captures/<capture_id>/, in parts, the way the capture app sends it. Returns its storage key."""
    made = api.call("POST", "/v1/uploads", {"captureId": capture_id, "name": clip.name, "type": "video/mp4"})
    key, upload_id, done = made["key"], made["uploadId"], []
    plan = parts(clip.stat().st_size)
    urls = api.call("POST", "/v1/uploads/sign", {"key": key, "uploadId": upload_id, "parts": [n for n, _, _ in plan]})["urls"]
    with clip.open("rb") as f:
        for n, off, length in plan:
            f.seek(off)
            done.append({"partNumber": n, "etag": api.put_part(urls[str(n)], f.read(length))})
            say(f"uploaded part {n} of {len(plan)}")
    api.call("POST", "/v1/uploads/complete", {"key": key, "uploadId": upload_id, "parts": done})
    return key


def wait(api: Api, job_id: str, minutes: float, say=say, sleep=time.sleep, clock=time.time, every: float = 15) -> dict:
    """Poll a job until it ends or the time is up; says each change of stage or status once."""
    end, seen = clock() + minutes * 60, None
    while True:
        job = api.call("GET", f"/v1/jobs/{job_id}")
        now = (job.get("status"), job.get("stage"))
        if now != seen:
            seen = now
            say(f"{datetime.now(timezone.utc):%H:%M:%S} {job.get('status')} {job.get('stage') or ''} {job.get('message') or ''}".rstrip())
        if job.get("status") in ENDED or clock() >= end:
            return job
        sleep(every)


def check(job: dict, served=lambda path: True) -> tuple[list[str], list[str]]:
    """What is wrong with a finished smoke job, and what is only worth saying. `served(path)`: does the API serve it."""
    problems, notes = [], []
    if job.get("status") not in ("review", "done"):
        return [f"the job ended as {job.get('status')}: {str(job.get('message') or '')[-400:]}"], notes
    result = job.get("result") or {}
    report, files = result.get("report") or {}, result.get("files") or {}
    problems += [f"no report from the {s} stage" for s in STAGES if not report.get(s)]
    quality = result.get("quality")
    if not isinstance(quality, dict) or "ok" not in quality:
        problems.append("the quality gate did not run")
    elif not quality["ok"]:  # 2,000 steps on 30 seconds of video is not a twin: the gate is expected to say so
        notes.append("gate: " + ", ".join(f"{i.get('level')} {i.get('code')}" for i in quality.get("issues") or []))
    base = f"/v1/scenes/{job.get('id')}/"
    for name in REQUIRED:
        if name not in files:
            problems.append(f"{name} was not published")
        elif not served(base + name):
            problems.append(f"{name} is listed, and the API does not serve it")
    notes += [f"{name} is not there" for name in EXPECTED if not served(base + name)]
    if len(result.get("attempts") or []) > 1:
        notes.append(f"{len(result['attempts'])} attempts: the gate asked for a retry")
    sfm, train = report.get("sfm") or {}, report.get("train") or {}
    notes.append(f"{sfm.get('registered')} of {sfm.get('images')} views solved, {sfm.get('mapper')} mapper ({sfm.get('mapperReason')}); "
                 f"{train.get('maxSteps')} steps in {train.get('seconds')} s")
    # the phone file's fine-tune falls back to the untuned file by design, so only this check sees it fail
    tune = (report.get("export") or {}).get("liteFinetune")
    if not tune:
        problems.append("the export did not fine-tune the phone file (a job without --sh-degree 3, or a worker from before the fine-tune)")
    elif tune.get("shipped") != "finetuned" and not str(tune.get("why") or "").startswith("measured"):
        problems.append(f"the phone file's fine-tune did not finish: {str(tune.get('why') or '')[-300:]}")
    else:
        notes.append(f"phone file: {(tune.get('before') or {}).get('psnr')} dB, {(tune.get('after') or {}).get('psnr')} dB after "
                     f"{tune.get('steps')} steps ({tune.get('seconds')} s); shipped the {tune.get('shipped')} one")
    return problems, notes


def read_key(a) -> str:
    if a.key_file:
        return Path(a.key_file).read_text().splitlines()[0].strip()
    return os.environ.get("PLY_TEAM_KEY", "").strip()


def modal_exe() -> str:
    """The `modal` CLI: PLY_MODAL, else one on PATH, else the repo's venv, the way the watch scripts call it."""
    if os.environ.get("PLY_MODAL"):
        return os.environ["PLY_MODAL"]
    found = shutil.which("modal")
    if found:
        return found
    local = Path(__file__).resolve().parents[1] / "pipeline" / ".venv" / "bin" / "modal"
    return str(local) if local.exists() else "modal"


def _run(cmd: list[str], timeout: float) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError(f"the modal CLI is not installed at {cmd[0]}: install it, or set PLY_MODAL")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"`{' '.join(cmd)}` did not answer in {timeout:g} s")
    return p.returncode, p.stdout, p.stderr


class Modal:
    """The `modal` CLI. `call(args, timeout)` returns (exit code, stdout, stderr): tests replace it."""

    def __init__(self, exe: str = "", run=None):
        self.exe, self.run = exe or modal_exe(), run or _run

    def call(self, args: list[str], timeout: float = MODAL_TIMEOUT) -> tuple[int, str, str]:
        return self.run([self.exe, *args], timeout)


BOX = "╭╮╰╯│─┏┓┗┛━┃ "
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def cli_error(stderr: str) -> str:
    """The CLI's error box as one line: no frame, no colour, no newlines."""
    lines = [ANSI.sub("", line).strip(BOX) for line in stderr.splitlines()]
    return " ".join(" ".join(line.split()) for line in lines if line.split()) or "no message"


def modal_failure(profile: str, code: int, stderr: str) -> str:
    """What went wrong with the Modal CLI, said the way it can be acted on."""
    msg = cli_error(stderr)
    if "Token missing" in msg or "Token not found" in msg or "Could not authenticate" in msg:
        return f"the modal CLI is logged out for profile '{profile}': run `modal token new --profile {profile}`"
    if "Modal profile" in msg and "was not found" in msg:
        return f"no such modal profile '{profile}' (see `modal profile list`)"
    return f"`modal` exited {code}: {msg}"


def profiles(modal: Modal) -> list[str]:
    """Every Modal profile the CLI knows; `--profile all` asks each of them for its billing summary."""
    code, out, err = modal.call(["profile", "list", "--json"])
    if code != 0:
        raise RuntimeError(f"cannot list the Modal profiles: {cli_error(err)}")
    try:
        rows = json.loads(out)
    except ValueError:
        raise RuntimeError("`modal profile list --json` did not print JSON")
    if not isinstance(rows, list):
        return []
    return [str(r["name"]) for r in rows if isinstance(r, dict) and r.get("name")]


def money(value) -> str:
    return f"${Decimal(str(value)):,.2f}"


def money(value) -> str:
    return f"${Decimal(str(value)):,.2f}"


def cmd_status(api: Api, a, say=say) -> int:
    if a.job:
        j = api.call("GET", f"/v1/jobs/{a.job}")
        say(f"{j['id']}  {j.get('status')}  {j.get('stage') or '-'}  {round((j.get('progress') or 0) * 100)}%  {j.get('message') or ''}")
        return 0
    h = api.call("GET", "/v1/health")
    say(f"workers online {h.get('workersOnline')}, starting {h.get('workersStarting') or 0}, jobs queued {h.get('queued')}")
    for w in h.get("workers") or []:
        say(f"  {w.get('name')}: {'busy' if w.get('busy') else 'idle'}, seen {w.get('seenSecondsAgo')} s ago")
    if api.key:
        for j in api.call("GET", "/v1/jobs")["jobs"][:a.jobs]:
            say(f"  {j['id']}  {j.get('status'):7} {j.get('stage') or '-':9} {datetime.fromtimestamp(j['createdAt'] / 1000, timezone.utc):%Y-%m-%d %H:%M}  {j.get('name')}")
    return 0


def cmd_smoke(api: Api, a, say=say, sleep=time.sleep, clock=time.time) -> int:
    job_id = getattr(a, "job", None)
    if not job_id:
        who = api.call("GET", "/v1/whoami")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        capture = f"smoke-{stamp}"  # not `rehearsal-`: those are never sent to the GPU worker
        say(f"team code accepted ({who.get('label')}); uploading {a.clip.name} ({a.clip.stat().st_size / 2**20:.1f} MB) as {capture}")
        key = upload(api, capture, a.clip, say)
        job_id = api.call("POST", "/v1/jobs", {"captureId": capture, "name": f"Smoke test {stamp}", "settings": {"smoke": True},
                                               "args": SMOKE_ARGS, "files": [key]})["jobId"]
        say(f"queued job {job_id}: {' '.join(SMOKE_ARGS)}")
    job = wait(api, job_id, a.minutes, say, sleep, clock)
    if job.get("status") not in ENDED:
        say(f"still {job.get('status')} after {a.minutes:g} min: `plyctl.py smoke --job {job_id}` picks it up again")
        return 2
    problems, notes = check(job, api.served)
    for n in notes:
        say("note: " + n)
    for p in problems:
        say("FAILED: " + p)
    say(f"smoke {'failed' if problems else 'passed'}: job {job_id}")
    return 1 if problems else 0


def _lines(text: str, say) -> str:
    """Print the complete lines of `text` and return the unfinished last one, for a followed log."""
    lines = text.split("\n")
    rest = lines.pop()
    for line in lines:
        say(line)
    return rest


def no_log(api: Api, job_id: str, say=say) -> dict:
    """The job's message, when the API serves no pipeline.log for it; returns the job."""
    job = api.call("GET", f"/v1/jobs/{job_id}")
    say(f"no pipeline.log for job {job_id} ({job.get('status')}): a worker publishes one at "
        f"scenes/{job_id}/pipeline.log only when a run fails")
    if job.get("status") == "running":
        say("the running container's own log is `modal app logs ply`, with MODAL_PROFILE set to its account")
    say(f"the job's message: {str(job.get('message') or '').strip() or '(none)'}")
    return job


def cmd_logs(api: Api, a, say=say, sleep=time.sleep) -> int:
    if a.tail is not None and a.tail < 1:
        raise ValueError("--tail must be 1 or more")
    path = f"/v1/scenes/{a.job}/pipeline.log"
    seen, first, pending, told = 0, True, "", False
    while True:
        status, _, raw = api.request("GET", path)
        if status == 404:
            if told:
                job = api.call("GET", f"/v1/jobs/{a.job}")
            else:
                job = no_log(api, a.job, say)
                told = True
            if not a.follow or job.get("status") not in ACTIVE:
                return 0
            sleep(FOLLOW_EVERY)
            continue
        if status >= 400:
            raise RuntimeError(f"GET {path}: {status} {raw[:200].decode('utf-8', 'replace')}")
        if len(raw) < seen:  # the file was replaced: read it again rather than print the wrong bytes
            seen, first = 0, True
        text = raw.decode("utf-8", "replace")
        if first:
            if a.tail is not None:
                text = "".join(f"{line}\n" for line in text.splitlines()[-a.tail:])
            if not a.follow:
                if text:
                    say(text.rstrip("\n"))
                return 0
            pending = _lines(text, say)
            first = False
        elif len(raw) > seen:
            pending = _lines(pending + raw[seen:].decode("utf-8", "replace"), say)
        seen = len(raw)
        if api.call("GET", f"/v1/jobs/{a.job}").get("status") not in ACTIVE:
            status, _, raw = api.request("GET", path)  # the last lines are written after the final heartbeat
            if status == 200 and len(raw) > seen:
                pending = _lines(pending + raw[seen:].decode("utf-8", "replace"), say)
            if pending:
                say(pending)
            return 0
        sleep(FOLLOW_EVERY)


def other_running(api: Api, job_id: str, say=say) -> list[str] | None:
    """The ids of the other running jobs, or None when that cannot be known (no team code)."""
    if not api.key:
        say("not stopping the Modal app: checking whether another job is running needs the team code "
            "(PLY_TEAM_KEY or --key-file)")
        return None
    running = [j["id"] for j in api.call("GET", "/v1/jobs").get("jobs") or [] if j.get("status") == "running"]
    if job_id not in running:
        say("not stopping the Modal app: the job is not among the newest jobs the API lists, so the other running "
            "ones cannot be counted")
        return None
    return [i for i in running if i != job_id]


def stop_app(modal: Modal, profile: str, say=say) -> bool:
    """End the deployed `ply` app on that profile; True when the CLI said it stopped it."""
    code, _, err = modal.call(["--profile", profile, "app", "stop", MODAL_APP, "-y"])
    if code == 0:
        say(f"stopped the Modal app {MODAL_APP} on profile {profile} (`modal app stop {MODAL_APP} -y`)")
        return True
    say(f"could not stop the Modal app {MODAL_APP} on profile {profile}: {modal_failure(profile, code, err)}")
    return False


def cmd_stop(api: Api, a, say=say, modal=None) -> int:
    modal = modal or Modal()
    job = api.call("GET", f"/v1/jobs/{a.job}")
    status = job.get("status") or "unknown"
    say(f"job {a.job}: {status}{' ' + job['stage'] if job.get('stage') else ''}")
    if status not in ACTIVE:
        say(f"nothing to stop: the run is over ({status})")
        if status == "review":
            say("it waits for a decision (approve or reject); stop does not decide")
        return 0
    others = other_running(api, a.job, say) if status == "running" else []
    code, _, raw = api.request("POST", f"/v1/jobs/{a.job}/cancel", {})
    cancelled = 200 <= code < 300
    detail = raw[:200].decode("utf-8", "replace")
    if cancelled:
        say(f"the API cancelled job {a.job}")
    elif code == 404:
        say(f"the API has no cancel route (POST /v1/jobs/{a.job}/cancel answered 404): job {a.job} is still {status}")
    elif code in (401, 403):
        say(f"the API refused the cancel ({code}): a cancel route is there and needs the team code")
    elif code == 409:
        say(f"the API would not cancel job {a.job} ({code}): {detail}")
    else:
        raise RuntimeError(f"POST /v1/jobs/{a.job}/cancel: {code} {detail}")
    if status == "queued":
        if not cancelled:
            say("nothing was stopped: the job is still queued, and stopping the GPU app would not take it off the queue")
        return 0 if cancelled else 1
    if others is None:
        return 1
    if others:
        say(f"left the Modal app running: {len(others)} other job(s) are running ({', '.join(others[:5])})")
        if cancelled:
            say(f"job {a.job} is cancelled, but its container, if any, is still going")
        return 1
    stopped = stop_app(modal, a.profile, say)
    if cancelled and stopped:
        return 0
    if stopped and not cancelled:
        say(f"the job still shows {status} in the API: with no cancel route and no heartbeat it is re-queued after "
            "20 minutes and will be picked up again")
    return 1


def cmd_cost(_api: Api, a, say=say, modal=None) -> int:
    modal = modal or Modal()
    wanted = profiles(modal) if a.profile == "all" else [a.profile]
    if not wanted:
        raise RuntimeError("no Modal profiles to ask: `modal profile list` shows none")
    say(f"billing cycle {datetime.now(timezone.utc):%Y-%m}")
    rows, failed = [("profile", "credits", "metered", "billed")], 0
    for profile in wanted:
        code, out, err = modal.call(["--profile", profile, "billing", "summary", "--json"])
        if code != 0:
            say(f"{profile}: {modal_failure(profile, code, err)}")
            failed += 1
            continue
        try:
            summary = json.loads(out)
            metered, billed = Decimal(summary["metered_cost"]), Decimal(summary["billed_cost"])
            credits = (summary.get("adjustments") or {}).get("credits")
            credits = Decimal(str(credits)) * -1 if credits is not None else None
            if credits is not None and not credits:
                credits = Decimal(0)  # Decimal("-0") would print as $-0.00
        except (ValueError, KeyError, TypeError, InvalidOperation):
            say(f"{profile}: `modal billing summary --json` did not print a summary this command reads")
            failed += 1
            continue
        rows.append((profile, money(credits) if credits is not None else "-", money(metered), money(billed)))
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    for row in rows if len(rows) > 1 else []:
        say(f"{row[0]:<{widths[0]}}  {row[1]:>{widths[1]}}  {row[2]:>{widths[2]}}  {row[3]:>{widths[3]}}".rstrip())
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--api", default=os.environ.get("PLY_API", API))
    ap.add_argument("--key-file", help="a file with the team code on its first line (default: PLY_TEAM_KEY)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--job")
    s.add_argument("--jobs", type=int, default=8, help="how many of the newest jobs to list")
    k = sub.add_parser("smoke")
    which = k.add_mutually_exclusive_group(required=True)
    which.add_argument("--clip", type=Path, help="upload this clip and queue a job")
    which.add_argument("--job", help="wait for and check a smoke job that is already queued (reading needs no team code)")
    k.add_argument("--minutes", type=float, default=45, help="how long to wait for the job")
    g = sub.add_parser("logs")
    g.add_argument("job", help="the job whose pipeline log to read")
    g.add_argument("--tail", type=int, help="only the last N lines (the first read when following)")
    g.add_argument("--follow", action="store_true", help="keep reading while the job is queued or running")
    q = sub.add_parser("stop")
    q.add_argument("job", help="the job to stop")
    q.add_argument("--profile", default=os.environ.get("MODAL_PROFILE") or DEFAULT_PROFILE,
                   help=f"the Modal profile that runs the {MODAL_APP} app (default: {DEFAULT_PROFILE})")
    c = sub.add_parser("cost")
    c.add_argument("--profile", default="all", help="'all' or one Modal profile name")
    a = ap.parse_args(argv)
    api = Api(a.api, read_key(a))
    if a.cmd == "smoke" and a.clip and not api.key:
        print("smoke needs the team code: PLY_TEAM_KEY or --key-file", file=sys.stderr)
        return 2
    try:
        if a.cmd == "smoke":
            return cmd_smoke(api, a)
        if a.cmd == "logs":
            return cmd_logs(api, a)
        if a.cmd == "stop":
            return cmd_stop(api, a)
        if a.cmd == "cost":
            return cmd_cost(api, a)
        return cmd_status(api, a)
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, OSError, ValueError, ArithmeticError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
