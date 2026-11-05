# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""The camera solve on a CPU box: a job's sfm stage split across two machines.

COLMAP's mapper (global positioning, bundle adjustment) runs on the CPU even on a GPU box: the pycolmap wheels'
Ceres has no CUDA. On the chapel (1,236 frames) mapping was 93 of a 119-minute run with the GPU idle and billed.
So a long capture is split:

  GPU worker   frames, features, matching (`--sfm-part match`), then hands the job folder over and moves on
  CPU solver   maps the matches (`--only sfm --reuse-matches`), heartbeats meanwhile, then re-queues the job
  GPU worker   claims it again, resumes from the solved cameras: align, occupancy, labels, training, export

The job folder travels through a shared directory (a Modal Volume mounted on both functions, PLY_HANDOFF_DIR).
The platform calls (committing the volume, starting the solver or a GPU worker) are hooks set from the environment,
so this module and the worker stay platform-free and testable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# what the solver needs, and what the resumed GPU run needs back; bulk the stages re-create is left behind
SKIP = {"frames_raw", "dense", "train", "evidence"}
MARK = "handoff.json"


def root() -> Path | None:
    d = os.environ.get("PLY_HANDOFF_DIR")
    return Path(d) if d and Path(d).is_dir() else None


def _volume(op: str) -> None:
    """commit / reload the shared Modal Volume (a no-op off Modal)."""
    name = os.environ.get("PLY_HANDOFF_VOLUME")
    if not name:
        return
    import modal
    getattr(modal.Volume.from_name(name), op)()


def commit() -> None:
    _volume("commit")


def reload() -> None:
    try:
        _volume("reload")
    except Exception:  # a reload with open files can fail; the next read retries
        pass


def spawn(fn: str, *args) -> None:
    """Start a function of the deployed app ('solve' or 'work'); PLY_HANDOFF_APP names the app (default 'ply')."""
    import modal
    modal.Function.from_name(os.environ.get("PLY_HANDOFF_APP", "ply"), fn).spawn(*args)


def paused(out: Path) -> bool:
    """The pipeline stopped after matching (--sfm-part match) rather than finishing the solve."""
    return (out / "colmap" / "matching.json").exists() and not (out / ".done" / "sfm.json").exists()


def _copy(src: Path, dst: Path, drop_db: bool = False) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ignore = lambda d, names: [n for n in names if n in SKIP or n.startswith(("live-", "_live_"))
                               or n.endswith((".tmp", ".part")) or (drop_db and n in ("database.db", "sparse_all", "sparse_inc"))]
    shutil.copytree(src, dst, symlinks=True, ignore=ignore)


def hand_over(jid: str, job: dict, out: Path, base: Path) -> Path:
    """GPU side: park the matched job folder in the shared directory for the solver."""
    dst = base / jid
    _copy(out, dst)
    (dst / MARK).write_text(json.dumps({"job": job, "matchedAt": time.time(), "state": "matched"}))
    commit()
    return dst


def state(jid: str, base: Path) -> str | None:
    """'matched' (waiting for or in the solver), 'solved' (ready for the GPU), 'failed', or None."""
    try:
        return json.loads((base / jid / MARK).read_text()).get("state")
    except (OSError, ValueError):
        return None


def take_back(jid: str, out: Path, base: Path) -> bool:
    """GPU side, on a re-claimed job: restore the solved folder into out. False if there is nothing solved."""
    reload()
    if state(jid, base) != "solved" or not (base / jid / ".done" / "sfm.json").exists():
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    _copy(base / jid, out)
    (out / MARK).unlink(missing_ok=True)
    return True


def discard(jid: str, base: Path) -> None:
    shutil.rmtree(base / jid, ignore_errors=True)
    commit()


def solver_args(job_args: list[str]) -> list[str]:
    """The job's settings minus what only decides where a run starts or stops."""
    out, skip = [], {"--force", "--from", "--only", "--until", "--sfm-part", "--split-min"}
    i = 0
    while i < len(job_args):
        if job_args[i] in skip:
            i += 2
            continue
        out.append(job_args[i])
        i += 1
    return out


def solve(jid: str, base: Path, api, work: Path = Path("/tmp/solve"), runner: list[str] | None = None) -> bool:
    """CPU side: map the matches of base/<jid>, heartbeat the job meanwhile, park the solved folder back.
    Returns True when the job is ready for a GPU again (the caller re-queues it and starts a worker)."""
    reload()
    mark = json.loads((base / jid / MARK).read_text())
    job = mark["job"]
    post = lambda **kw: api.req("POST", f"/v1/worker/jobs/{jid}", kw)
    out = work / jid
    _copy(base / jid, out)
    (out / MARK).unlink(missing_ok=True)
    (work / "in").mkdir(parents=True, exist_ok=True)
    alive = {"on": True, "msg": "sfm: solving the cameras on a CPU box"}

    def beat():
        while alive["on"]:
            try:
                post(stage="sfm", message=alive["msg"][:200])
            except Exception as e:
                print(f"[solver] heartbeat failed: {e}", file=sys.stderr, flush=True)
            for _ in range(20):
                if not alive["on"]:
                    return
                time.sleep(1)

    th = threading.Thread(target=beat, daemon=True)
    th.start()
    t0 = time.time()
    runner = runner or os.environ.get("PLY_PIPELINE_CMD", f"{sys.executable} -m plypipe").split()
    cmd = [*runner, "run", str(work / "in"), "--out", str(out), "--only", "sfm", "--reuse-matches", *solver_args(job.get("args") or [])]
    print("[solver] " + " ".join(cmd), flush=True)
    tail: list[str] = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        tail = (tail + [line.rstrip()[:300]])[-30:]
        if '"msg": "' in line and '"stage": "sfm"' in line:
            try:
                alive["msg"] = "sfm (CPU box): " + json.loads(line).get("msg", "")
            except ValueError:
                pass
    code = p.wait()
    alive["on"] = False
    th.join(timeout=5)
    done = out / ".done" / "sfm.json"
    if code != 0 or not done.exists():
        mark["state"] = "failed"
        (base / jid / MARK).write_text(json.dumps(mark))
        commit()
        post(status="failed", message=f"camera solve on the CPU box failed (exit {code}) :: " + "\n".join(tail[-12:]))
        return False
    rep = json.loads(done.read_text())
    rep.update(solvedOn="cpu", cpuMinutes=round((time.time() - t0) / 60, 1))
    done.write_text(json.dumps(rep))
    _copy(out, base / jid, drop_db=True)
    mark.update(state="solved", solvedAt=time.time())
    (base / jid / MARK).write_text(json.dumps(mark))
    commit()
    post(status="queued", stage="sfm", progress=0.27,
         message=f"cameras solved on a CPU box ({rep['cpuMinutes']} min, {rep.get('registered')}/{rep.get('images')} frames); waiting for a GPU")
    return True


def main(argv: list[str] | None = None) -> int:
    import argparse
    from plypipe.worker import Api
    ap = argparse.ArgumentParser(description="map a handed-over job's matches on this (CPU) machine")
    ap.add_argument("jid")
    a = ap.parse_args(argv)
    base = root()
    if base is None:
        raise SystemExit("PLY_HANDOFF_DIR is not set or missing")
    api = Api(os.environ["PLY_API"], os.environ["PLY_WORKER_TOKEN"])
    return 0 if solve(a.jid, base, api) else 1


if __name__ == "__main__":
    sys.exit(main())
