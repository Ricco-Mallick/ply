#!/usr/bin/env python3
# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Operate the live Ply service from a shell: what is running, and a smoke test for after every worker deploy.

    python3 scripts/plyctl.py status                  # workers, queue, and with the team code the newest jobs
    python3 scripts/plyctl.py status --job <id>       # one job
    python3 scripts/plyctl.py smoke --clip clip.mp4   # a short clip through the real API, then every result is checked
    python3 scripts/plyctl.py smoke --job <id>        # the same wait and checks for a job that is already queued

`smoke` uploads the clip in parts as the capture app does, queues a Quick job of 2,000 training steps at SH 3, waits
for it and checks that every stage reported, that the gate ran, that the export's fine-tune of the phone file ran to
the end, and that the API serves each result file. It exits 0 when
all of that holds, 1 when a check fails and 2 when the job is still running at the time limit. The job stays in
review: rejecting it would queue it again. A capture under 400 frames is solved on the GPU box, so the hand-off to
the CPU solver is not covered.

The API is PLY_API (default: the deployed one). The team code is read from PLY_TEAM_KEY or from the first line of
--key-file; it is sent to the API only, never to the storage links. The clip is not in the repository: a 30-second
walk is kept as smoke/smoke-chapel-30s.mp4 on the `ply-bench` Volume (`modal volume get ply-bench smoke/... .`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
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


def say(line: str) -> None:
    print(line, flush=True)  # a log file or a pipe must show progress while the job runs


class Api:
    """The Ply API over urllib. `send(method, url, body, headers)` returns (status, headers, bytes): tests replace it."""

    def __init__(self, base: str = API, key: str = "", send=None):
        self.base, self.key, self.send = base.rstrip("/"), key.strip(), send or _send

    def call(self, method: str, path: str, body=None, content_type: str = "application/json"):
        data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
        headers = {"content-type": content_type} if data is not None else {}
        if self.key:
            headers["x-ply-key"] = self.key
        status, _, raw = self.send(method, self.base + path, data, headers)
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
    a = ap.parse_args(argv)
    api = Api(a.api, read_key(a))
    if a.cmd == "smoke" and a.clip and not api.key:
        print("smoke needs the team code: PLY_TEAM_KEY or --key-file", file=sys.stderr)
        return 2
    try:
        return cmd_smoke(api, a) if a.cmd == "smoke" else cmd_status(api, a)
    except (RuntimeError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
