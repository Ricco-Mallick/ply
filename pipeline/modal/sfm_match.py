"""Features and matches of a saved job, made once on a GPU and saved, so every camera-solve variant maps the same database.

The chapel's saved job (chapel/high/job on the ply-bench Volume) kept the aligned solve but never its colmap/database.db, so
modal/sfm_cpu.py has nothing to re-solve from. This makes it again, exactly as the live worker's GPU half does for a long capture:
`plypipe run --only sfm --sfm-part match` with the options the job recorded (job.json), the features and matcher its first
matching used (colmap/matching.json), and no mapper. The frames, frames_index.json and .done/frames.json are restored from the
job; the database and matching.json go back to <src>/colmap/ with match-run.json (features, matcher, pairs, seconds, GPU).

    python pipeline/modal/sfm_match.py plan                                  # estimate, timeout and cost; no Modal
    MODAL_PROFILE=<account> modal run --detach pipeline/modal/sfm_match.py   # --src chapel/high/job; spawns and returns the call id
    ... --force                                                             # replace a database already saved there
    python pipeline/modal/sfm_match.py collect                               # match-run.json, from the Volume

An L4 with 8 cores and 32 GiB, as the worker has: 20 min estimate (10 min of extraction and matching on an L40S, 14 to 15 on an L4,
plus the frames in and the database out) is $0.48 at $1.43/h; the function timeout is 1.5 times that, 30 min ($0.72).
It refuses at once, before any container starts, when the frames are missing or a database is already there, and it returns as soon as
the run is spawned: a client waiting on `.remote()` cancels the run when it is killed (docs/MISTAKES.md, 2026-11-11).
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for root in (HERE.parent, Path("/root/pipeline")):  # a checkout; the container's mounted copy. Appended: pipeline/modal must not shadow `modal`
    if (root / "plypipe").is_dir():
        sys.path.append(str(root))
from ply_modal import image as worker_image  # noqa: E402
from plypipe import evalrun, solvebench  # noqa: E402

image = worker_image.add_local_file(HERE / "ply_modal.py", "/root/ply_modal.py")
GPUS = ["L4", "L40S"]  # an L40S when Modal has no L4 free (2026-11-13: an L4-only launch waited 12 minutes without starting)
app = modal.App("ply-sfm-match", image=image)
vol = modal.Volume.from_name("ply-bench")
VOL = Path("/vol")
JOB = Path("/tmp/work/match")
CODE = next(r for r in (Path("/root/pipeline"), HERE.parent) if (r / "plypipe").is_dir())
SAVE_MIN = 5  # the database goes back to the Volume after the matching deadline


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _gpu() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return evalrun.parse_gpu(out)


def _plypipe(args: list[str], deadline: float) -> str | None:
    """One plypipe run in its own process group; None on success, else why it stopped."""
    cmd = [sys.executable, "-m", "plypipe", "run", "/tmp/in", "--out", str(JOB), *args]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=str(CODE), PLY_CACHE_DIR="/tmp/ply-cache"), start_new_session=True)
    try:
        code = proc.wait(timeout=max(1.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return "stop-loss: the matching was still running at its deadline"
    return None if code == 0 else f"exit {code}"


def run_match(src: Path, force: bool, gpu: dict, extra: str = "", started: float | None = None, deadline: float | None = None) -> dict:
    """Restore the frames, make features and matches, save the database beside the job. Raises when it must not start; a failed
    run returns its record with `error` set and saves nothing but that record."""
    started = started or time.time()
    deadline = deadline or started + (solvebench.match_deadline_min() - SAVE_MIN) * 60
    why = solvebench.match_preflight(lambda rel: (src / rel).exists(), force)
    if why:
        raise FileNotFoundError(f"{src}: {why}")
    shutil.rmtree(JOB, ignore_errors=True)
    for rel in (*solvebench.MATCH_NEEDED, "job.json", "colmap/matching.json"):
        if (src / rel).is_dir():
            shutil.copytree(src / rel, JOB / rel)
        elif (src / rel).exists():
            (JOB / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, JOB / rel)
    args = solvebench.match_args(_json(src / "job.json"), _json(src / "colmap" / "matching.json"), extra)
    (JOB / "colmap" / "matching.json").unlink(missing_ok=True)  # the run writes its own; the saved one only chose the options
    Path("/tmp/in").mkdir(exist_ok=True)
    t = time.time()
    error = _plypipe(args, deadline)
    seconds = {"restore": round(t - started, 1), "match": round(time.time() - t, 1)}
    db, matching = JOB / "colmap" / "database.db", _json(JOB / "colmap" / "matching.json")
    if not error and (not db.exists() or not matching):
        error = "plypipe exited without a database and matching.json"
    if not error and (JOB / "colmap" / "sparse").exists():
        error = "a model was written: the mapper must not run in this step"
    record = {"src": str(src.relative_to(VOL)) if src.is_relative_to(VOL) else str(src), "args": args, "gpu": gpu, "error": error,
              "features": (matching or {}).get("features"), "matcher": (matching or {}).get("matcher"),
              "loopDetection": (matching or {}).get("loopDetection"), "seconds": seconds,
              "estimateMin": solvebench.MATCH_ESTIMATE_MIN}
    if not error:
        t = time.time()
        record.update(solvebench.database_counts(db), databaseMiB=round(db.stat().st_size / 2 ** 20))
        out = src / "colmap"
        out.mkdir(parents=True, exist_ok=True)
        part = out / "database.db.part"  # never a half-written database under its real name
        shutil.copy2(db, part)
        part.replace(out / "database.db")
        shutil.copy2(JOB / "colmap" / "matching.json", out / "matching.json")
        seconds["save"] = round(time.time() - t, 1)
    seconds["wall"] = round(time.time() - started, 1)
    record["costUsd"] = solvebench.match_cost_usd(seconds["wall"] / 60)
    (src / "colmap").mkdir(parents=True, exist_ok=True)
    (src / "colmap" / "match-run.json").write_text(json.dumps(record, indent=1))
    return record


@app.function(gpu=GPUS, cpu=solvebench.MATCH_CORES, memory=solvebench.MEMORY_GIB[solvebench.MATCH_CORES] * 1024,
              timeout=solvebench.match_deadline_min() * 60, volumes={"/vol": vol})
def match(job: dict) -> dict:
    started = time.time()
    gpu = _gpu()
    print(f"[sfm-match] {job['src']} on {gpu}", flush=True)
    record = run_match(VOL / job["src"], job["force"], gpu, job.get("args", ""), started)
    vol.commit()
    print("[sfm-match]", json.dumps(record), flush=True)
    return record


@app.local_entrypoint()
def main(src: str = solvebench.SOURCE, force: bool = False, args: str = ""):
    src = src.strip("/")
    print(solvebench.describe_match(), flush=True)
    try:
        why = solvebench.match_preflight(solvebench.volume_have(vol, src), force)
    except Exception as e:
        why = None
        print(f"[sfm-match] could not list {src} from here ({type(e).__name__}: {e}); the container checks again", flush=True)
    if why:
        raise SystemExit(f"[sfm-match] not started: {src}: {why}")
    call = match.spawn({"src": src, "force": force, "args": args})
    print(f"[sfm-match] started {call.object_id}: not waiting. `modal app logs ply-sfm-match` follows it; the database lands in "
          f"{src}/colmap/ with match-run.json (python pipeline/modal/sfm_match.py collect)", flush=True)


if __name__ == "__main__":  # `plan` and `collect` run here, not through `modal run`
    import argparse

    ap = argparse.ArgumentParser(prog="sfm_match.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    co = sub.add_parser("collect")
    co.add_argument("--src", default=solvebench.SOURCE)
    a = ap.parse_args()
    if a.cmd == "plan":
        print(solvebench.describe_match())
    else:
        print(b"".join(vol.read_file(f"{a.src}/colmap/match-run.json")).decode())
