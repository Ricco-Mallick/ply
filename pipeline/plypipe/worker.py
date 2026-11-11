# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""GPU worker: pull jobs from the Ply API (cloud/src/index.ts), run the pipeline, publish the results.

    PLY_API=https://ply-api.<you>.workers.dev PLY_WORKER_TOKEN=... python -m plypipe.worker

Loop: claim the oldest queued job -> download its clips + capture.json -> `python -m plypipe run <folder> <args>`
-> tail log.jsonl and post stage/progress (also a heartbeat, so a dead worker's job is re-queued after 20 min)
-> upload scene.spz, scene_lite.spz, scene.json, occupancy.png/.json, report.json to scenes/<jobId>/ -> done.
Exits after --idle-exit minutes with no work (Kaggle GPU hours are rationed), 0 = never.
Spend is bounded three ways (plypipe.limits): a limit per stage, one cap on the job's GPU time that survives the CPU
camera solve (PLY_JOB_MAX_MIN), and a cap on the bytes of clips it will download (PLY_MAX_INPUT_MB).
Standard library only, so it runs anywhere the pipeline does.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
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

import numpy as np

from plypipe.limits import EXPORT_RESERVE_MIN, job_cap_min, max_input_mb, stage_limits, worst_case_min
from plypipe.live import Live
from plypipe import hf
from plypipe.progress import SfmProgress

# rough share of wall time per stage on a T4, for one overall progress number
WEIGHTS = {"frames": .05, "sfm": .20, "align": .02, "occupancy": .02, "label": .04, "train": .62, "export": .05}
ORDER = list(WEIGHTS)
RESULTS = ["scene.spz", "scene_lite.spz", "scene.json", "occupancy.png", "occupancy.json", "places.json", "report.json", "mesh.ply"]
# the trained model is what the GPU hours bought. Uploading it means a bad export filter is a re-export,
# not a rebuild: 400 MB of transfer against three hours of L40S
KEEP_MODEL = ["scene.ply", "scene_clean.ply"]
PART = 32 * 1024 * 1024
# stop-loss: a stage that runs past its limit is killed and the job fails with a clear message (minutes, see
# plypipe.limits; PLY_STAGE_LIMITS="sfm=30,train=60" overrides). They exist to cap a stuck run, not to race.
PRESET_STEPS = {"quick": 7_000, "standard": 30_000, "high": 40_000}
# checkpoints: after these stages the out/ folder (minus re-creatable bulk) goes to B2, so a retry with the same
# settings resumes there instead of re-downloading the clips and redoing frames + SfM
# label and train too: a job that dies in training resumes at training, and one that dies in export resumes with the
# trained model instead of buying the GPU hours again
CHECKPOINT_AFTER = ("sfm", "occupancy", "label", "train")
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
        if key.startswith("scenes/") and hf.configured():  # results: 100 GB, no daily download cap
            try:
                hf.download(key, dest)
                return
            except hf.Missing:
                pass  # not there (older job, or HF hiccup): fall through to the bucket
        """The API hands out a signed link; the bytes come straight from the bucket (no auth header on that request).
        'gdrive:<id>:<name>' inputs (large shared clips, e.g. demo footage) come straight from Google Drive instead,
        which keeps multi-GB captures off the bucket's daily download allowance."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = self._input_url(key)
        ceiling = int(max_input_mb() * 2**20)
        for i in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as resp:
                    # Drive answers a private / over-quota file with an HTML page and status 200: not a clip
                    if "text/html" in (resp.headers.get("content-type") or ""):
                        raise RuntimeError(f"{key}: the link returned a web page, not the file (is it shared as 'Anyone with the link'?)")
                    with dest.open("wb") as f:
                        # a file whose size the server did not say still cannot fill the disk: stop at the whole job's cap
                        got = 0
                        while chunk := resp.read(8 * 1024 * 1024):
                            got += len(chunk)
                            if got > ceiling:
                                raise RuntimeError(f"{key}: larger than this worker's {ceiling / 2**30:g} GB input limit (PLY_MAX_INPUT_MB)")
                            f.write(chunk)
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                dest.unlink(missing_ok=True)
                if i == 3:
                    raise
                time.sleep(3 * (i + 1))
            except Exception:
                dest.unlink(missing_ok=True)
                raise

    def _input_url(self, key: str) -> str:
        if key.startswith("gdrive:"):
            return f"https://drive.usercontent.google.com/download?id={key.split(':')[1]}&export=download&confirm=t"
        _, info = self.req("GET", "/v1/worker/file?" + urllib.parse.urlencode({"key": key}))
        return info["url"]

    def input_size(self, key: str) -> int | None:
        """Bytes of an input from the headers of a one-byte ranged request, before any of it is downloaded; None when
        the server does not say."""
        try:
            req = urllib.request.Request(self._input_url(key), headers={**UA, "range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = re.match(r"bytes \d+-\d+/(\d+)$", resp.headers.get("content-range") or "")
                if total:
                    return int(total.group(1))
                length = resp.headers.get("content-length") or ""
                return int(length) if resp.status == 200 and length.isdigit() and "text/html" not in (resp.headers.get("content-type") or "") else None
        except Exception:
            return None

    def upload(self, key: str, path: Path, ctype: str):
        if key.startswith("scenes/") and hf.configured():
            try:
                hf.upload(key, path)
                return
            except Exception as e:  # HF down or the token lost write access: the bucket still has it
                print(f"[worker] HF upload of {key} failed ({e}); using the bucket", flush=True)
        self._bucket_put(key, path, ctype)

    def upload_many(self, items: list[tuple[str, Path, str]]):
        """(key, path, content type) per file. Each Hugging Face upload is its own commit (48 evidence frames billed
        the GPU ~18 minutes), so there the files go up in one; the bucket takes one request per file."""
        put = self.upload
        if items and hf.configured() and all(key.startswith("scenes/") for key, _, _ in items):
            try:
                hf.upload_many([(key, path) for key, path, _ in items])
                return
            except Exception as e:
                print(f"[worker] HF batch upload of {len(items)} files failed ({e}); using the bucket", flush=True)
                put = self._bucket_put  # per-file HF commits would be the slow path again
        for key, path, ctype in items:
            put(key, path, ctype)

    def _bucket_put(self, key: str, path: Path, ctype: str):
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


def _write_evidence(out: Path, long_edge: int = 1280, count: int = 48) -> None:
    """A spread of posed frames a modelling agent can look at: downscaled JPEGs plus their camera positions.

    The pipeline keeps every frame until the container dies; the bundle needs a handful that a person (or an agent)
    can actually open, with enough coverage to judge every wall. Positions come from scene.json's camera path.
    """
    from PIL import Image

    frames = sorted((out / "frames").glob("*.jpg"))
    if not frames:
        return
    ev = out / "evidence" / "frames"
    ev.mkdir(parents=True, exist_ok=True)
    pick = frames if len(frames) <= count else [frames[int(i)] for i in np.linspace(0, len(frames) - 1, count)]
    poses = []
    try:
        scene = json.loads((out / "scene.json").read_text())
    except Exception:
        scene = {}
    path = scene.get("cameraPath") or []
    for index, f in enumerate(pick):
        dest = ev / f"{index:03d}.jpg"
        try:
            img = Image.open(f)
            img.thumbnail((long_edge, long_edge))
            img.convert("RGB").save(dest, quality=82)
        except Exception:
            continue
        pos = None
        if path:
            pos = path[min(len(path) - 1, int(round(index * (len(path) - 1) / max(1, len(pick) - 1))))] if len(path) > 1 else path[0]
        yaw = 0.0
        if pos is not None and path:
            nxt = path[min(len(path) - 1, int(round(index * (len(path) - 1) / max(1, len(pick) - 1))) + 1)]
            yaw = math.atan2(-(nxt[0] - pos[0]), -(nxt[2] - pos[2]))
        poses.append({"index": index, "file": f"evidence/frames/{dest.name}", "position": pos or [0, 1.6, 0], "yaw": round(yaw, 4), "fov": 60})
    (out / "evidence" / "poses.json").write_text(json.dumps({"frames": poses}, indent=1))


CTYPE = {".spz": "application/octet-stream", ".ply": "application/octet-stream", ".json": "application/json", ".png": "image/png"}


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


def train_limit_s(base_s: float, elapsed_s: float, frac: float, deadline: float, reserve_s: float, now: float) -> float:
    """Training's stop-loss. The fixed limit is a guess; a plan with more views or gaussians trains longer. Once 5 %
    is done the projected finish is known: allow it (plus 15 %) as long as the machine's deadline leaves room for
    export after it, and never more than 3x the fixed limit on a machine without a deadline."""
    if not base_s or frac < 0.05:
        return base_s
    projected = elapsed_s / frac * 1.15
    room = elapsed_s + max(0.0, deadline - now - reserve_s) if deadline else 3 * base_s
    return max(base_s, min(projected, room))


def salvage_training(out: Path, planned_steps: int | None) -> dict | None:
    """Training was stopped: keep the newest snapshot as the model instead of throwing the hours away.

    Marks `train` done (partial) so the next pipeline run starts at export. None when no snapshot exists yet."""
    plys = sorted((out / "train" / "ply").glob("point_cloud_*.ply"), key=lambda q: int(q.stem.rsplit("_", 1)[1]))
    if not plys:
        return None
    shutil.copy2(plys[-1], out / "scene.ply")
    preset = out / "train" / "ply_preset.json"
    step = int(plys[-1].stem.rsplit("_", 1)[1])
    rep = {"ply": "scene.ply", "sizeMB": round((out / "scene.ply").stat().st_size / 2**20, 1), "fromStep": step,
           **(json.loads(preset.read_text()) if preset.exists() else {}), "partial": True, "plannedSteps": planned_steps}
    (out / ".done").mkdir(parents=True, exist_ok=True)
    (out / ".done" / "train.json").write_text(json.dumps(rep, indent=2))
    report_path = out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["train"] = rep
    report_path.write_text(json.dumps(report, indent=2))
    return rep


def _restart_at(args: list[str], stage: str) -> list[str]:
    out, i = [], 0
    while i < len(args):
        if args[i] in ("--force", "--from", "--only"):
            i += 2
            continue
        out.append(args[i])
        i += 1
    return [*out, "--from", stage]


def carried_s(jid: str, base: Path) -> float:
    """GPU seconds a job used before it was handed to the CPU solver, from the marker the hand-over left in the shared
    folder (any state: a solver that died leaves the job to be solved on the GPU, and the cap keeps counting)."""
    from plypipe import handoff
    try:
        return max(0.0, float(json.loads((base / jid / handoff.MARK).read_text())["job"].get("capUsedS") or 0))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return 0.0


def input_bytes(api, keys: list[str], cache: Path) -> int:
    """Size of a job's inputs: what is cached counts at its size on disk, the rest at what the server says (an unknown
    size counts as 0 here and is caught once the file is down)."""
    size_of, total = getattr(api, "input_size", None), 0
    for k in keys:
        cached = cache / k.replace(':', '_')
        if cached.exists():
            total += cached.stat().st_size
        elif size_of and not k.endswith(".json"):
            total += size_of(k) or 0
    return total


def check_input_size(total: int) -> None:
    cap_mb = max_input_mb()
    if total > cap_mb * 2**20:
        raise RuntimeError(f"the clips total {total / 2**30:.1f} GB, over this worker's {cap_mb / 1024:g} GB input limit "
                           "(PLY_MAX_INPUT_MB); nothing was processed")


# checkpoints carry the undistorted frames; above this they cost more (upload time, the bucket's 1 GB/day download
# allowance on resume) than redoing the stages on a GPU that is not pre-empted
CKPT_MAX_MB = int(os.environ.get("PLY_CKPT_MAX_MB", 1500))  # results now live on HF (100 GB), not the bucket


def _ckpt_files(out: Path):
    for p in sorted(out.iterdir()):
        # the live view's files come and go on other threads (and are republished anyway): never checkpointed
        if p.name in CKPT_SKIP or p.name.startswith(("live-", "_live_")) or p.name.endswith(".tmp"):
            continue
        yield p


def checkpoint_size(out: Path) -> int:
    total = 0
    for p in _ckpt_files(out):
        if p.is_file():
            total += p.stat().st_size
        elif p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
    return total


def save_checkpoint(api, jid: str, out: Path, root: Path, args: list[str], after: str) -> None:
    tar = root / f"checkpoint-{after}.tar"  # one file per stage: two checkpoints never share a temp file
    with tarfile.open(tar, "w") as tf:
        meta = json.dumps({"args": args, "after": after, "time": time.time()}).encode()
        info = tarfile.TarInfo("ckpt.json"); info.size = len(meta)
        tf.addfile(info, __import__("io").BytesIO(meta))
        for p in _ckpt_files(out):
            tf.add(p, arcname=p.name)
    api.upload(f"scenes/{jid}/checkpoint.tar", tar, "application/x-tar")
    tar.unlink(missing_ok=True)


def _ckpt_key(args: list[str]) -> list[str]:
    """Settings that must match for a checkpoint to be reusable. --force/--from only decide where to start, so a
    resumed run can force a later stage (e.g. redo align) and still reuse an earlier checkpoint."""
    skip = {"--force", "--from", "--only"}
    out, i = [], 0
    while i < len(args):
        if args[i] in skip:
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def load_checkpoint(api, jid: str, out: Path, root: Path, args: list[str]) -> str | None:
    """Restore the last checkpoint if it was made with exactly these settings; returns the stage it covers."""
    tar = root / "checkpoint.tar"
    try:
        api.download(f"scenes/{jid}/checkpoint.tar", tar)
        with tarfile.open(tar) as tf:
            meta = json.load(tf.extractfile("ckpt.json"))
            if _ckpt_key(meta.get("args") or []) != _ckpt_key(args):
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
    t_claim = time.time()
    cap_min = job_cap_min()
    state = {"stage": None, "frac": 0.0, "msg": "downloading clips", "alive": True, "stage_t0": t_claim, "timeout": None,
             "cap_end": t_claim + cap_min * 60}
    limits = stage_limits()
    export_reserve_s = (limits["export"] + EXPORT_RESERVE_MIN) * 60
    cap_msg = lambda st: f"{st or 'startup'} was stopped: the job reached its {cap_min:g}-minute time cap (GPU time since it was claimed)"
    rehearsal = "--train-dry-run" in job["args"]
    post = lambda **kw: api.req("POST", f"/v1/worker/jobs/{jid}", kw)
    live = Live(api, jid, out, log=lambda m: print(m, file=sys.stderr, flush=True))  # what the status page shows in 3D
    live.preview_only = "--until" in job["args"]  # coverage checks publish their own preview
    a_ = job["args"]
    def planned_steps() -> int:
        """Steps this job will actually train for: the plan the frames/train stages wrote, else the preset."""
        try:
            return int(json.loads((out / "plan.json").read_text())["model"]["iterations"])
        except Exception:
            return int(a_[a_.index("--iterations") + 1]) if "--iterations" in a_ else PRESET_STEPS.get(
                a_[a_.index("--preset") + 1] if "--preset" in a_ else "standard")

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
    hd = None
    try:
        # inputs are cached per storage key for the whole session: retries and requeues don't download the clips
        # again (B2's free tier caps downloads at 1 GB/day)
        cache = work / "_inputs"
        # a long capture's camera solve runs on a CPU box (plypipe.handoff): a job it handed back resumes from there
        from plypipe import handoff
        hd = handoff.root()
        split = []
        if hd is not None and os.environ.get("PLY_CPU_SFM", "0") == "1":
            back = handoff.take_back(jid, out, hd)
            state["cap_end"] -= carried_s(jid, hd)
            if state["cap_end"] <= time.time():
                raise RuntimeError(f"the job had already used its {cap_min:g}-minute time cap (GPU time) before the CPU camera solve")
            if back:
                resumed = "sfm"
                post(message="resuming after the CPU camera solve")
                print(f"[worker] {jid}: cameras were solved on a CPU box, resuming", flush=True)
            elif handoff.state(jid, hd) is None and "--until" not in job["args"]:  # long full runs only
                split = ["--sfm-part", "match", "--split-min", os.environ.get("PLY_CPU_SFM_MIN", "400")]
            # matched but not solved (the solver died) or failed there: solve on this box rather than loop
        else:
            back = False
        resumed = "sfm" if back else load_checkpoint(api, jid, out, root, job["args"])
        if resumed:
            post(message=f"resuming after {resumed} (checkpoint)")
            print(f"[worker] resuming {jid} after {resumed}", flush=True)
        inp.mkdir(parents=True, exist_ok=True)
        if not resumed:
            check_input_size(input_bytes(api, job["files"], cache))  # known sizes: refuse before downloading any of it
        for k in ([] if resumed else job["files"]):  # a checkpoint past SfM needs no clips at all
            if time.time() >= state["cap_end"]:
                raise RuntimeError(cap_msg("downloading clips"))
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
        if not resumed:
            check_input_size(input_bytes(api, job["files"], cache))  # sizes the server would not say
        # PLY_PIPELINE_CMD swaps the pipeline for a stand-in (tests: the worker plumbing without a GPU)
        runner = os.environ.get("PLY_PIPELINE_CMD", f"{sys.executable} -m plypipe").split()
        def pipeline(args: list[str]):
            """Run the pipeline once; returns (exit code, last error event, last lines of non-JSON output)."""
            cmd = [*runner, "run", str(inp), "--out", str(out), *args, *split, *extra]
            print("[worker] " + " ".join(cmd), flush=True)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            last_err, tail = "", collections.deque(maxlen=40)
            sfm = SfmProgress()  # real progress inside sfm, from COLMAP's own counters
            ckpts: list[threading.Thread] = []
            ckpt_lock = threading.Lock()  # checkpoints upload in stage order (the later one replaces the earlier)
            step = {"key": None, "t0": time.time()}

            deadline = float(os.environ.get("PLY_DEADLINE") or 0)
            matching_only = "--sfm-part" in cmd  # the CPU box solves, so the GPU's sfm is matching alone

            def watchdog():  # stop-loss per stage, and the job's own cap
                while p.poll() is None:
                    st, lim = state["stage"], limits.get(state["stage"] or "", 0)
                    if st == "sfm" and matching_only:
                        lim = limits.get("sfm_match", lim)
                    now = time.time()
                    lim_s = lim * 60
                    if st == "train":
                        lim_s = train_limit_s(lim_s, now - state["stage_t0"], state["frac"], deadline, export_reserve_s, now)
                    # training stops early enough for export and publishing to finish inside the cap
                    if now >= state["cap_end"] - (export_reserve_s if st == "train" else 0):
                        state["timeout"] = cap_msg(st)
                        state["salvage"] = st == "train"
                        p.kill()
                        return
                    if st and lim_s and now - state["stage_t0"] > lim_s:
                        state["timeout"] = f"{st} exceeded its {lim_s / 60:g}-minute limit and was stopped"
                        state["salvage"] = st == "train"
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
                                threading.Thread(target=live.poll_snapshots, args=(lambda: p.poll() is None and state["stage"] == "train", planned_steps()), daemon=True).start()
                            try:  # tell the status page straight away; the heartbeat covers the rest
                                post(stage=st, progress=overall(st, 0.0), message=f"{st}: started")
                            except Exception:
                                pass
                        if ev.get("event") == "done":
                            state["frac"] = 1.0
                            live.stage_done(st)
                            if st in CHECKPOINT_AFTER and not rehearsal:
                                # in the background: a slow upload must never stall reading the pipeline's output
                                # (a 600 MB checkpoint once blocked this loop and the stage watchdog killed the job)
                                def ckpt(stage=st):
                                    with ckpt_lock:
                                        _ckpt(stage)

                                def _ckpt(stage):
                                    try:
                                        size = checkpoint_size(out)
                                        if size > CKPT_MAX_MB * 2**20:
                                            print(f"[worker] checkpoint after {stage} skipped: {size / 2**20:.0f} MB > {CKPT_MAX_MB} MB", flush=True)
                                            return
                                        save_checkpoint(api, jid, out, root, job["args"], stage)
                                    except Exception as e:  # a missed checkpoint only costs time on a retry
                                        print(f"[worker] checkpoint after {stage} failed: {e}", flush=True)
                                th = threading.Thread(target=ckpt, daemon=True); th.start(); ckpts.append(th)
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
            for th in ckpts:  # let checkpoints finish uploading (outside the stage watchdog, which only runs while the pipeline does)
                th.join(timeout=900)
            if state["timeout"]:
                last_err = state["timeout"]
            return code, last_err, list(tail)

        state["msg"] = "starting pipeline"
        # one attempt is judged, not trusted: the plan sets the run, the gate reads the result, and a
        # failure that settings can fix is retried once with those settings changed. Each attempt is
        # recorded, so the operator sees what was tried and why.
        from plypipe import orchestrate, plan as plan_mod, verify as verify_mod
        args = list(job["args"])
        attempts: list[dict] = []
        # a coverage preview and a rehearsal are not twins: only a full run is judged and retried
        judged = "--until" not in job["args"] and not rehearsal
        for attempt in range(1, (orchestrate.MAX_ATTEMPTS if judged else 1) + 1):
            state["msg"] = f"attempt {attempt}: starting pipeline"
            t_attempt = time.time()
            code, last_err, tail = pipeline(args)
            if code == 0 and split and handoff.paused(out):
                # matched: the mapping goes to a CPU box, and this GPU is free for the next job (or to stop billing)
                used = cap_min * 60 - (state["cap_end"] - time.time())
                handoff.hand_over(jid, {**job, "capUsedS": round(used)}, out, hd)
                handoff.spawn("solve", jid)
                state["alive"] = False
                post(stage="sfm", progress=overall("sfm", 0.3), message="matched on the GPU; solving the cameras on a CPU box")
                print(f"[worker] {jid}: matched, camera solve handed to a CPU box", flush=True)
                return
            split = []  # a retry or a later attempt solves on this box
            # a native crash in camera solving (abort/segfault: negative exit code) is usually the learned features
            # or the global mapper on an unusual capture; retry from sfm with classic SIFT + incremental mapping
            if code < 0 and state["stage"] == "sfm" and "incremental" not in args and not state["timeout"]:
                base = [a for i, a in enumerate(args) if a not in ("--features", "--mapper", "--matcher")
                        and (i == 0 or args[i - 1] not in ("--features", "--mapper", "--matcher"))]
                retry = [*base, "--features", "sift", "--mapper", "incremental", "--matcher", "sequential", "--force", "sfm"]
                post(stage="sfm", progress=overall("sfm", 0.0), message=f"sfm crashed (exit {code}); retrying with SIFT + incremental mapper")
                state["frac"] = 0.0
                code, last_err, tail = pipeline(retry)
                args = retry
            if state.get("salvage"):  # training ran out of time: export the newest snapshot rather than fail the job
                state["salvage"] = False
                kept = salvage_training(out, planned_steps())
                if kept:
                    msg = (f"training stopped at step {kept['fromStep']} of {kept['plannedSteps']} ({state['timeout']}); "
                           "exporting that snapshot")
                    print(f"[worker] {msg}", flush=True)
                    post(stage="export", progress=overall("export", 0.0), message=msg[:200])
                    if not rehearsal:
                        try:
                            if checkpoint_size(out) <= CKPT_MAX_MB * 2**20:
                                save_checkpoint(api, jid, out, root, job["args"], "train")
                        except Exception as e:
                            print(f"[worker] checkpoint after salvaged train failed: {e}", flush=True)
                    state.update(timeout=None, stage=None, frac=0.0, stage_t0=time.time())  # a fresh clock for export
                    args = _restart_at(args, "export")
                    code, last_err, tail = pipeline(args)
            if code != 0:
                break
            report = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
            run_plan = plan_mod.read(out)
            quality = verify_mod.check(report, run_plan)
            attempts.append({"attempt": attempt, "args": args,
                             "settings": run_plan, "quality": quality})
            if not judged or quality["ok"] or attempt == orchestrate.MAX_ATTEMPTS:
                break
            refined, notes, force = orchestrate.refine(report, args, run_plan)
            if refined == args or not notes:
                break  # nothing settings can do; the failure is the capture itself
            if state["cap_end"] - time.time() < (export_reserve_s if force == "export" else time.time() - t_attempt):
                print("[worker] attempt failed the gate; too little of the job's time cap is left to retry: " + "; ".join(notes), flush=True)
                break  # a retry the cap would stop half-way costs the finished attempt's result
            args = [*refined, "--force", force]
            state["frac"] = 0.0
            post(stage="frames", progress=0, message="quality gate: " + "; ".join(notes)[:200])
            print(f"[worker] attempt {attempt} failed the gate, retrying: " + "; ".join(notes), flush=True)
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
        # keep the trained model next to the results: an export-only re-run then costs a minute instead of a retrain
        ply_path = out / "scene.ply"
        if ply_path.exists() and ply_path.stat().st_size < 1.2e9:
            try:
                api.upload(f"scenes/{jid}/scene.ply", ply_path, "application/octet-stream")
            except Exception as e:
                print(f"[worker] keeping scene.ply failed: {e}", flush=True)
        # evidence for a modelling agent: a spread of the posed frames plus their camera positions, so a room can be
        # rebuilt as editable geometry later without re-running the capture
        try:
            _write_evidence(out)
        except Exception as e:
            print(f"[worker] evidence frames failed: {e}", flush=True)
        try:
            ev = out / "evidence"
            frames = ev / "frames"
            have = sorted(frames.glob("*.jpg")) if frames.exists() else []
            picked = have if len(have) <= 48 else [have[int(i)] for i in np.linspace(0, len(have) - 1, 48)]
            batch = [(f"scenes/{jid}/evidence/frames/{f.name}", f, "image/jpeg") for f in picked]
            poses = ev / "poses.json"
            if poses.exists():
                batch.append((f"scenes/{jid}/evidence/poses.json", poses, "application/json"))
            api.upload_many(batch)
        except Exception as e:
            print(f"[worker] evidence upload failed: {e}", flush=True)
        report = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
        quality = verify_mod.check(report, plan_mod.read(out))
        report["quality"] = quality
        report["attempts"] = attempts
        from plypipe import review as review_mod
        opinion = review_mod.opinion(report)
        report["review"] = opinion
        (out / "report.json").write_text(json.dumps(report, indent=1))
        for issue in quality["issues"]:
            print(f"[worker] quality {issue['level']}: {issue['message']}", flush=True)
        print(f"[worker] reviewer: {opinion['verdict']} ({opinion['confidence']:.0%}) — {opinion['note']}", flush=True)
        files = {}
        for name in RESULTS:
            f = out / name
            if f.exists():
                api.upload(f"scenes/{jid}/{name}", f, CTYPE.get(f.suffix, "application/octet-stream"))
                files[name] = f.stat().st_size
        for name in KEEP_MODEL:  # best effort: never fail a good run because the archive upload timed out
            f = out / name
            if f.exists() and f.stat().st_size < 1_500_000_000:
                try:
                    api.upload(f"scenes/{jid}/{name}", f, CTYPE.get(f.suffix, "application/octet-stream"))
                    files[name] = f.stat().st_size
                except Exception as e:
                    print(f"[worker] could not archive {name}: {e}", flush=True)
        if "scene.spz" not in files:
            raise RuntimeError("pipeline finished without scene.spz")
        state["alive"] = False
        # waiting for a human: approve publishes, reject re-queues with the refined settings
        post(status="review", stage="export", progress=1,
             message=("quality gate failed: " + opinion["recommend"]) if not quality["ok"] else "ready for review",
             result={"base": f"/v1/scenes/{jid}/", "files": files, "report": report,
                     "quality": quality, "review": opinion, "attempts": attempts})
        if hd is not None:
            handoff.discard(jid, hd)
    except Exception as e:
        state["alive"] = False
        post(status="failed", message=str(e)[-3800:])  # keep the end: that is where the real error line is
        try:
            if hd is not None:
                handoff.discard(jid, hd)
        except Exception:
            pass
        print(f"[worker] job {jid} failed: {e}", file=sys.stderr, flush=True)
    finally:
        state["alive"] = False
        t.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # live view: the twin at 10, 30, 60 and 85 % of training; the newest is also what a stopped training run keeps
    os.environ.setdefault("PLY_LIVE_SNAPSHOTS", "0.1,0.3,0.6,0.85")
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
    need_min, took_job = worst_case_min(stage_limits(), job_cap_min()), False
    while True:
        if deadline and deadline - time.time() < need_min * 60:
            if not took_job:  # nothing has used the window yet: these limits can never fit this machine
                print(f"[worker] ERROR: a job can take up to {need_min:g} min (stage limits and PLY_JOB_MAX_MIN) but this machine's "
                      "deadline leaves less: lower the limits or raise the container timeout", flush=True)
                return 2
            print(f"[worker] less than {need_min:g} min left before this machine's deadline: not taking new jobs", flush=True)
            return 0
        status, job = api.req("POST", "/v1/worker/claim", {"worker": a.name, "prefix": os.environ.get("PLY_CLAIM_PREFIX", "")})
        if status == 200 and job:
            print(f"[worker] claimed {job['id']} ({job['name']})", flush=True)
            took_job = True
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
