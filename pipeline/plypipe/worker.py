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
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from plypipe.live import Live
from plypipe.progress import SfmProgress

# rough share of wall time per stage on a T4, for one overall progress number
WEIGHTS = {"frames": .05, "sfm": .20, "align": .02, "occupancy": .02, "label": .04, "train": .62, "export": .05}
ORDER = list(WEIGHTS)
RESULTS = ["scene.spz", "scene_lite.spz", "scene.json", "occupancy.png", "occupancy.json", "places.json", "report.json"]
PART = 32 * 1024 * 1024
# stop-loss: a stage that runs past its limit is killed and the job fails with a clear message (minutes;
# PLY_STAGE_LIMITS="sfm=30,train=60" overrides). Generous for a T4; they exist to cap a stuck run, not to race.
STAGE_LIMITS_MIN = {"frames": 15, "sfm": 45, "align": 5, "occupancy": 5, "label": 15, "train": 120, "export": 15}
# the most a job can take (frames..export limits): a worker with a hard deadline (Modal timeout) never starts a job
# it could not finish
JOB_MAX_MIN = sum(STAGE_LIMITS_MIN.values())
PRESET_STEPS = {"quick": 7_000, "standard": 30_000, "high": 40_000}
# checkpoints: after these stages the out/ folder (minus re-creatable bulk) goes to B2, so a retry with the same
# settings resumes there instead of re-downloading the clips and redoing frames + SfM
CHECKPOINT_AFTER = ("sfm", "occupancy")
CKPT_SKIP = {"frames", "frames_raw", "colmap", "train"}
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
        """The API hands out a signed link; the bytes come straight from the bucket (no auth header on that request).
        'gdrive:<id>:<name>' inputs (large shared clips, e.g. demo footage) come straight from Google Drive instead,
        which keeps multi-GB captures off the bucket's daily download allowance."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if key.startswith("gdrive:"):
            url = f"https://drive.usercontent.google.com/download?id={key.split(':')[1]}&export=download&confirm=t"
        else:
            _, info = self.req("GET", "/v1/worker/file?" + urllib.parse.urlencode({"key": key}))
            url = info["url"]
        for i in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as resp:
                    # Drive answers a private / over-quota file with an HTML page and status 200: not a clip
                    if "text/html" in (resp.headers.get("content-type") or ""):
                        raise RuntimeError(f"{key}: the link returned a web page, not the file (is it shared as 'Anyone with the link'?)")
                    with dest.open("wb") as f:
                        shutil.copyfileobj(resp, f, 8 * 1024 * 1024)
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                dest.unlink(missing_ok=True)
                if i == 3:
                    raise
                time.sleep(3 * (i + 1))
            except Exception:
                dest.unlink(missing_ok=True)
                raise

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


def eta(t0: float, f: float) -> str:
    """The " · ~N min left" suffix, from the pace so far, once there is enough of it to mean something."""
    el = time.time() - t0
    if f < 0.03 or f >= 0.99 or el < 45:
        return ""
    left = el * (1 - f) / f
    return f" · ~{max(1, round(left / 60))} min left" if left >= 60 else " · under a minute left"


def stage_limits() -> dict:
    lim = dict(STAGE_LIMITS_MIN)
    for part in filter(None, os.environ.get("PLY_STAGE_LIMITS", "").split(",")):
        k, _, v = part.partition("=")
        lim[k.strip()] = float(v)
    return lim


def save_checkpoint(api, jid: str, out: Path, root: Path, args: list[str], after: str) -> None:
    tar = root / "checkpoint.tar"
    with tarfile.open(tar, "w") as tf:
        meta = json.dumps({"args": args, "after": after, "time": time.time()}).encode()
        info = tarfile.TarInfo("ckpt.json"); info.size = len(meta)
        tf.addfile(info, __import__("io").BytesIO(meta))
        for p in sorted(out.iterdir()):
            if p.name not in CKPT_SKIP:
                tf.add(p, arcname=p.name)
    api.upload(f"scenes/{jid}/checkpoint.tar", tar, "application/x-tar")
    tar.unlink(missing_ok=True)


def load_checkpoint(api, jid: str, out: Path, root: Path, args: list[str]) -> str | None:
    """Restore the last checkpoint if it was made with exactly these settings; returns the stage it covers."""
    tar = root / "checkpoint.tar"
    try:
        api.download(f"scenes/{jid}/checkpoint.tar", tar)
        with tarfile.open(tar) as tf:
            meta = json.load(tf.extractfile("ckpt.json"))
            if meta.get("args") != args:
                return None
            out.mkdir(parents=True, exist_ok=True)
            tf.extractall(out, filter="data")
        (out / "ckpt.json").unlink(missing_ok=True)
        return meta.get("after")
    except Exception:  # no checkpoint yet (404) or unreadable: start from scratch
        return None
    finally:
        tar.unlink(missing_ok=True)


def run_job(api: Api, job: dict, work: Path, extra: list[str]) -> None:
    jid = job["id"]
    root = work / jid
    inp, out = root / "input", root / "out"
    root.mkdir(parents=True, exist_ok=True)
    state = {"stage": None, "frac": 0.0, "msg": "downloading clips", "alive": True, "stage_t0": time.time(), "timeout": None}
    limits = stage_limits()
    rehearsal = "--train-dry-run" in job["args"]
    post = lambda **kw: api.req("POST", f"/v1/worker/jobs/{jid}", kw)
    live = Live(api, jid, out, log=lambda m: print(m, file=sys.stderr, flush=True))  # what the status page shows in 3D
    live.preview_only = "--until" in job["args"]  # coverage checks publish their own preview
    a_ = job["args"]
    final_step = int(a_[a_.index("--iterations") + 1]) if "--iterations" in a_ else PRESET_STEPS.get(a_[a_.index("--preset") + 1] if "--preset" in a_ else "standard")

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
        # inputs are cached per storage key for the whole session: retries and requeues don't download the clips
        # again (B2's free tier caps downloads at 1 GB/day)
        cache = work / "_inputs"
        resumed = load_checkpoint(api, jid, out, root, job["args"])
        if resumed:
            post(message=f"resuming after {resumed} (checkpoint)")
            print(f"[worker] resuming {jid} after {resumed}", flush=True)
        inp.mkdir(parents=True, exist_ok=True)
        for k in ([] if resumed else job["files"]):  # a checkpoint past SfM needs no clips at all
            cached = cache / k.replace(':', '_')
            if k.endswith(".json"):  # capture.json is re-uploaded (e.g. new spans to skip): always fresh
                cached.unlink(missing_ok=True)
            if not cached.exists() or cached.stat().st_size == 0:
                api.download(k, cached.with_suffix(cached.suffix + ".part"))
                cached.with_suffix(cached.suffix + ".part").rename(cached)
            inp.mkdir(parents=True, exist_ok=True)
            dest = inp / (k.split(':', 2)[2] if k.startswith('gdrive:') else Path(k).name)
            n = 2
            while dest.exists() and k.startswith("gdrive:"):  # two shared files with one name: keep both
                dest = dest.with_name(f"{Path(k.split(':', 2)[2]).stem}-{n}{Path(k).suffix}")
                n += 1
            if not dest.exists():
                os.link(cached, dest) if hasattr(os, "link") else shutil.copy2(cached, dest)
        # PLY_PIPELINE_CMD swaps the pipeline for a stand-in (tests: the worker plumbing without a GPU)
        runner = os.environ.get("PLY_PIPELINE_CMD", f"{sys.executable} -m plypipe").split()
        def pipeline(args: list[str]):
            """Run the pipeline once; returns (exit code, last error event, last lines of non-JSON output)."""
            cmd = [*runner, "run", str(inp), "--out", str(out), *args, *extra]
            print("[worker] " + " ".join(cmd), flush=True)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            last_err, tail = "", collections.deque(maxlen=40)
            sfm = SfmProgress()  # real progress inside sfm, from COLMAP's own counters
            step = {"key": None, "t0": time.time()}

            def watchdog():  # stop-loss per stage
                while p.poll() is None:
                    st, lim = state["stage"], limits.get(state["stage"] or "", 0)
                    if st and lim and time.time() - state["stage_t0"] > lim * 60:
                        state["timeout"] = f"{st} exceeded its {lim:g}-minute limit and was stopped"
                        p.kill()
                        return
                    time.sleep(2)
            threading.Thread(target=watchdog, daemon=True).start()
            with (root / "pipeline.log").open("a") as logf:
                for line in p.stdout:  # the pipeline prints the same JSON lines it writes to log.jsonl, plus library output
                    sys.stdout.write(line)
                    logf.write(line)
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        if line.strip():
                            tail.append(line.rstrip()[:300])
                        r = sfm.feed(line) if state["stage"] == "sfm" else None
                        if r:
                            frac, msg, key, sf = r
                            if key != step["key"]:
                                step.update(key=key, t0=time.time())
                            state["frac"] = max(state["frac"], frac)
                            state["msg"] = f"sfm: {msg}{'' if key == 'map' else eta(step['t0'], sf)}"[:200]
                        continue
                    st = ev.get("stage")
                    if st in WEIGHTS:
                        if st != state["stage"]:
                            state["stage"], state["frac"], state["stage_t0"] = st, 0.0, time.time()
                            live.stage(st, "running")
                            if st == "train":  # intermediate PLYs -> small SPZ snapshots for the live view
                                threading.Thread(target=live.poll_snapshots, args=(lambda: p.poll() is None and state["stage"] == "train", final_step), daemon=True).start()
                            try:  # tell the status page straight away; the heartbeat covers the rest
                                post(stage=st, progress=overall(st, 0.0), message=f"{st}: started")
                            except Exception:
                                pass
                        if ev.get("event") == "done":
                            state["frac"] = 1.0
                            live.stage_done(st)
                            if st in CHECKPOINT_AFTER and not rehearsal:
                                try:
                                    state["msg"] = f"{st}: saving checkpoint"
                                    save_checkpoint(api, jid, out, root, job["args"], st)
                                except Exception as e:  # a missed checkpoint only costs time on a retry
                                    print(f"[worker] checkpoint after {st} failed: {e}", flush=True)
                        if st == "sfm" and ev.get("images"):
                            sfm.n_images = int(ev["images"])
                        f = frac_from(ev) if ev.get("event") == "progress" else None
                        if f is not None:
                            state["frac"] = max(state["frac"], f)
                            if st == "train":
                                state["msg"] = f"train: {round(f * 100)}%{eta(state['stage_t0'], f)}"
                                live.train_line(f, ev.get("msg", ""))
                        elif ev.get("event") == "progress" and ev.get("msg"):  # named steps inside a stage ("matching", "incremental mapping")
                            state["msg"] = f"{st}: {ev['msg']}"[:200]
                        if ev.get("event") in ("start", "done", "warn"):
                            state["msg"] = f"{st}: {ev.get('msg') or ev.get('event')}"[:200]
                    if ev.get("event") == "error":
                        last_err = f"{st}: {ev.get('msg')}"
            code = p.wait()
            if state["timeout"]:
                last_err = state["timeout"]
            return code, last_err, list(tail)

        state["msg"] = "starting pipeline"
        code, last_err, tail = pipeline(job["args"])
        # a native crash in camera solving (abort/segfault: negative exit code) is usually the learned features or the
        # global mapper on an unusual capture; retry once from the sfm stage with classic SIFT + incremental mapping
        if code < 0 and state["stage"] == "sfm" and "incremental" not in job["args"] and not state["timeout"]:
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
            detail = "\n".join(tail[-14:])
            raise RuntimeError(f"{last_err or f'pipeline exited with {code}'}{' :: ' + detail if detail else ''}")
        if rehearsal:  # dry run: nothing to publish, the report is the result
            report = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
            state["alive"] = False
            post(status="done", stage="train", progress=1, message="rehearsal passed", result={"rehearsal": True, "report": report})
            return
        until = job["args"][job["args"].index("--until") + 1] if "--until" in job["args"] else None
        if until in ("frames", "sfm", "align", "occupancy"):  # a coverage check from the app: publish the sparse preview
            state["msg"] = "publishing coverage preview"
            from plypipe import preview
            files = {}
            for f in preview.build(out):
                api.upload(f"scenes/{jid}/{f.name}", f, CTYPE.get(f.suffix, "application/octet-stream"))
                files[f.name] = f.stat().st_size
            report = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
            state["alive"] = False
            post(status="done", stage=until, progress=1, message="coverage preview ready", result={"preview": True, "base": f"/v1/scenes/{jid}/", "files": files, "report": report})
            return
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
        post(status="failed", message=str(e)[-3800:])  # keep the end: that is where the real error line is
        print(f"[worker] job {jid} failed: {e}", file=sys.stderr, flush=True)
    finally:
        state["alive"] = False
        t.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    os.environ.setdefault("PLY_LIVE_SNAPSHOTS", "0.1,0.3,0.6")  # live view: the twin at 10, 30 and 60 % of training
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
    deadline = float(os.environ.get("PLY_DEADLINE", 0))  # epoch seconds when this machine will be killed (Modal timeout)
    while True:
        if deadline and deadline - time.time() < JOB_MAX_MIN * 60:
            print(f"[worker] less than {JOB_MAX_MIN} min left before this machine's deadline: not taking new jobs", flush=True)
            return 0
        status, job = api.req("POST", "/v1/worker/claim", {"worker": a.name, "prefix": os.environ.get("PLY_CLAIM_PREFIX", "")})
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
