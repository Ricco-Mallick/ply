"""GPU worker: pull jobs from the Ply API (cloud/src/index.ts), run the pipeline, publish the results.

    PLY_API=https://ply-api.<you>.workers.dev PLY_WORKER_TOKEN=... python -m plypipe.worker

Loop: claim the oldest queued job -> download its clips + capture.json -> `python -m plypipe run <folder> <args>`
-> tail log.jsonl and post stage/progress (also a heartbeat, so a dead worker's job is re-queued after 20 min)
-> upload scene.spz, scene_lite.spz, scene.json, occupancy.png/.json, report.json to scenes/<jobId>/ -> done.
Exits after --idle-exit minutes with no work (Kaggle GPU hours are rationed), 0 = never.
Standard library only, so it runs anywhere the pipeline does.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# rough share of wall time per stage on a T4, for one overall progress number
WEIGHTS = {"frames": .05, "sfm": .20, "align": .02, "occupancy": .02, "train": .66, "export": .05}
ORDER = list(WEIGHTS)
RESULTS = ["scene.spz", "scene_lite.spz", "scene.json", "occupancy.png", "occupancy.json", "report.json"]
PART = 32 * 1024 * 1024
# Cloudflare rejects the default "Python-urllib" user agent with 403 before the request reaches the API
UA = {"user-agent": "ply-worker/1.0 (+https://ricco.codes/ply)"}


class Api:
    def __init__(self, base: str, token: str):
        self.base, self.token = base.rstrip("/"), token

    def req(self, method: str, path: str, body: bytes | dict | None = None, ctype: str | None = None, tries: int = 5):
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        headers = {"authorization": f"Bearer {self.token}", **UA}
        if isinstance(body, dict):
            headers["content-type"] = "application/json"
        elif ctype:
            headers["content-type"] = ctype
        for i in range(tries + 1):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.base + path, data=data, method=method, headers=headers), timeout=300) as r:
                    raw = r.read()
                    return (r.status, json.loads(raw) if raw and r.headers.get("content-type", "").startswith("application/json") else raw)
            except urllib.error.HTTPError as e:
                if e.code < 500 or i == tries:
                    raise RuntimeError(f"{method} {path}: {e.code} {e.read()[:300]!r}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if i == tries:
                    raise RuntimeError(f"{method} {path}: {e}") from None
            time.sleep([1, 3, 8, 20, 40][min(i, 4)])

    def download(self, key: str, dest: Path):
        """The API hands out a signed link; the bytes come straight from the bucket (no auth header on that request)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        _, info = self.req("GET", "/v1/worker/file?" + urllib.parse.urlencode({"key": key}))
        for i in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(info["url"], headers=UA), timeout=600) as resp, dest.open("wb") as f:
                    shutil.copyfileobj(resp, f, 8 * 1024 * 1024)
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if i == 3:
                    raise
                time.sleep(3 * (i + 1))

    def upload(self, key: str, path: Path, ctype: str):
        size = path.stat().st_size
        if size <= 40e6:
            self.req("PUT", "/v1/files?" + urllib.parse.urlencode({"key": key}), path.read_bytes(), ctype)
            return
        _, up = self.req("POST", "/v1/uploads", {"key": key, "type": ctype})
        n = -(-size // PART)
        _, signed = self.req("POST", "/v1/uploads/sign", {"key": up["key"], "uploadId": up["uploadId"], "parts": list(range(1, n + 1))})
        parts = []
        with path.open("rb") as f:
            for i in range(1, n + 1):
                chunk = f.read(PART)
                for t in range(5):  # straight to the bucket; keep its ETag for the complete call
                    try:
                        with urllib.request.urlopen(urllib.request.Request(signed["urls"][str(i)], data=chunk, method="PUT", headers=UA), timeout=600) as r:
                            parts.append({"partNumber": i, "etag": r.headers["ETag"]})
                        break
                    except (urllib.error.URLError, TimeoutError, ConnectionError):
                        if t == 4:
                            raise
                        time.sleep(3 * (t + 1))
        self.req("POST", "/v1/uploads/complete", {"key": up["key"], "uploadId": up["uploadId"], "parts": parts})


CTYPE = {".spz": "application/octet-stream", ".json": "application/json", ".png": "image/png"}


def overall(stage: str | None, frac: float) -> float:
    if stage not in WEIGHTS:
        return 0.0
    i = ORDER.index(stage)
    return round(sum(WEIGHTS[s] for s in ORDER[:i]) + WEIGHTS[stage] * max(0.0, min(1.0, frac)), 4)


def frac_from(ev: dict) -> float | None:
    if "done" in ev and "total" in ev and ev["total"]:
        return ev["done"] / ev["total"]
    m = re.search(r"(\d+)\s*/\s*(\d+)", ev.get("msg", ""))  # gsplat's "step 2800/7000"-style lines
    if m and int(m.group(2)) > 0:
        return int(m.group(1)) / int(m.group(2))
    return None


def run_job(api: Api, job: dict, work: Path, extra: list[str]) -> None:
    jid = job["id"]
    root = work / jid
    inp, out = root / "input", root / "out"
    state = {"stage": None, "frac": 0.0, "msg": "downloading clips", "alive": True}
    post = lambda **kw: api.req("POST", f"/v1/worker/jobs/{jid}", kw)

    def beat():  # heartbeat + progress every 20 s, independent of log activity
        while state["alive"]:
            try:
                post(stage=state["stage"], progress=overall(state["stage"], state["frac"]), message=state["msg"])
            except Exception as e:  # a missed beat is fine; the next one retries
                print(f"[worker] heartbeat failed: {e}", file=sys.stderr)
            for _ in range(20):
                if not state["alive"]:
                    return
                time.sleep(1)

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    try:
        for k in job["files"]:
            api.download(k, inp / Path(k).name)
        # PLY_PIPELINE_CMD swaps the pipeline for a stand-in (tests: the worker plumbing without a GPU)
        runner = os.environ.get("PLY_PIPELINE_CMD", f"{sys.executable} -m plypipe").split()
        def pipeline(args: list[str]):
            """Run the pipeline once; returns (exit code, last error event, last lines of non-JSON output)."""
            cmd = [*runner, "run", str(inp), "--out", str(out), *args, *extra]
            print("[worker] " + " ".join(cmd), flush=True)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            last_err, tail = "", collections.deque(maxlen=40)
            with (root / "pipeline.log").open("a") as logf:
                for line in p.stdout:  # the pipeline prints the same JSON lines it writes to log.jsonl, plus library output
                    sys.stdout.write(line)
                    logf.write(line)
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        if line.strip():
                            tail.append(line.rstrip()[:300])
                        continue
                    st = ev.get("stage")
                    if st in WEIGHTS:
                        if st != state["stage"]:
                            state["stage"], state["frac"] = st, 0.0
                            try:  # tell the status page straight away; the heartbeat covers the rest
                                post(stage=st, progress=overall(st, 0.0), message=f"{st}: started")
                            except Exception:
                                pass
                        if ev.get("event") == "done":
                            state["frac"] = 1.0
                        f = frac_from(ev) if ev.get("event") == "progress" else None
                        if f is not None:
                            state["frac"] = max(state["frac"], f)
                        elif ev.get("event") == "progress" and ev.get("msg"):  # named steps inside a stage ("matching", "incremental mapping")
                            state["msg"] = f"{st}: {ev['msg']}"[:200]
                        if ev.get("event") in ("start", "done", "warn"):
                            state["msg"] = f"{st}: {ev.get('msg') or ev.get('event')}"[:200]
                    if ev.get("event") == "error":
                        last_err = f"{st}: {ev.get('msg')}"
            return p.wait(), last_err, list(tail)

        state["msg"] = "starting pipeline"
        code, last_err, tail = pipeline(job["args"])
        # a native crash in camera solving (abort/segfault: negative exit code) is usually the learned features or the
        # global mapper on an unusual capture; retry once from the sfm stage with classic SIFT + incremental mapping
        if code < 0 and state["stage"] == "sfm" and "incremental" not in job["args"]:
            base = [a for i, a in enumerate(job["args"]) if a not in ("--features", "--mapper", "--matcher")
                    and (i == 0 or job["args"][i - 1] not in ("--features", "--mapper", "--matcher"))]
            retry = [*base, "--features", "sift", "--mapper", "incremental", "--matcher", "sequential", "--force", "sfm"]
            post(stage="sfm", progress=overall("sfm", 0.0), message=f"sfm crashed (exit {code}); retrying with SIFT + incremental mapper")
            state["frac"] = 0.0
            code, last_err, tail = pipeline(retry)
        if code != 0:
            try:  # keep the full log next to the results for debugging
                api.upload(f"scenes/{jid}/pipeline.log", root / "pipeline.log", "text/plain")
            except Exception:
                pass
            detail = " | ".join(tail[-6:])
            raise RuntimeError(f"{last_err or f'pipeline exited with {code}'}{' :: ' + detail if detail else ''}")
        state["msg"] = "publishing results"
        files = {}
        for name in RESULTS:
            f = out / name
            if f.exists():
                api.upload(f"scenes/{jid}/{name}", f, CTYPE.get(f.suffix, "application/octet-stream"))
                files[name] = f.stat().st_size
        if "scene.spz" not in files:
            raise RuntimeError("pipeline finished without scene.spz")
        report = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
        state["alive"] = False
        post(status="done", stage="export", progress=1, message="done", result={"base": f"/v1/scenes/{jid}/", "files": files, "report": report})
    except Exception as e:
        state["alive"] = False
        post(status="failed", message=str(e)[:480])
        print(f"[worker] job {jid} failed: {e}", file=sys.stderr, flush=True)
    finally:
        state["alive"] = False
        t.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default=os.environ.get("PLY_API"))
    ap.add_argument("--token", default=os.environ.get("PLY_WORKER_TOKEN"))
    ap.add_argument("--name", default=os.environ.get("PLY_WORKER_NAME", socket.gethostname()))
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("PLY_WORK_DIR", "work")))
    ap.add_argument("--idle-exit", type=float, default=float(os.environ.get("PLY_IDLE_EXIT_MIN", 30)), help="minutes")
    ap.add_argument("--once", action="store_true", help="process at most one job")
    ap.add_argument("--keep", action="store_true", help="keep job folders after upload")
    ap.add_argument("extra", nargs="*", help="extra plypipe flags for every job (e.g. --device cuda)")
    a = ap.parse_args(argv)
    if not a.api or not a.token:
        ap.error("set PLY_API and PLY_WORKER_TOKEN (or --api/--token)")
    api, idle_since = Api(a.api, a.token), time.time()
    print(f"[worker] {a.name} polling {a.api}", flush=True)
    while True:
        status, job = api.req("POST", "/v1/worker/claim", {"worker": a.name})
        if status == 200 and job:
            print(f"[worker] claimed {job['id']} ({job['name']})", flush=True)
            run_job(api, job, a.work, a.extra)
            if not a.keep:
                shutil.rmtree(a.work / job["id"], ignore_errors=True)
            idle_since = time.time()
            if a.once:
                return 0
            continue
        if a.idle_exit and time.time() - idle_since > a.idle_exit * 60:
            print("[worker] idle, exiting", flush=True)
            return 0
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
