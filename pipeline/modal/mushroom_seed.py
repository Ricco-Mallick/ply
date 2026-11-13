"""Seed the `ply-bench` Volume with the two MuSHRoom rooms (Zenodo, CC-BY-4.0): the iPhone photos, the Faro laser reference, and a camera
solve of our own (docs/IMPROVEMENT-PLAN.md section 5, scenario S3). Run once per account, in this order:

    python pipeline/modal/mushroom_seed.py plan                      # what each step fetches and costs; no Modal
    MODAL_PROFILE=<account> modal run --detach pipeline/modal/mushroom_seed.py --step fetch    # CPU: archives -> mushroom/<room>/in + reference
    MODAL_PROFILE=<account> modal run --detach pipeline/modal/mushroom_seed.py --step solve    # GPU: the pipeline's own solve -> mushroom/<room>/job
    PLY_EVAL_SCENARIO=mushroom-honka MODAL_PROFILE=<account> modal run --detach pipeline/modal/ply_eval.py --variant q0a    # then train + score

`--rooms honka` (or coffee_room) does one room. fetch: both archives of each room come over Zenodo's API in 16 MB ranges on 4 connections,
are checked against Zenodo's md5 and kept under mushroom/_dl; only the long capture's photos, the pose files, the alignment and the laser cloud
are unpacked. A rerun skips what is there and a killed run resumes at its finished ranges. solve: `plypipe run --until occupancy` on the
photos, the production path (ALIKED + LightGlue, then the camera-height prior for scale: the twin's scale is scored on this solve, never on
the dataset's poses), then the solve is scored against the reference cameras at once, before any training is paid for.

Each step starts one function per room and does not wait for it: a call the client waits on was cancelled about 90 s after the client was
killed (docs/MISTAKES.md, 2026-11-11), so `modal app logs ply-mushroom-seed` follows them and the Volume holds the result. The function
timeouts are the stop-loss, 1.5 times the estimate that `plan` prints (plus two minutes to save for solve).
"""
import json
import math
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
from plypipe import mushroom, mushroom_fetch  # noqa: E402

if modal.is_local():
    from ply_modal import image as worker_image  # noqa: E402
else:  # inside a container the image is already built
    worker_image = modal.Image.debian_slim()

PIPE = HERE.parent
SOLVE_MIN = 15            # estimate per room: ALIKED + LightGlue on 330 to 390 photos at 738 x 994; Dr Johnson's 263 photos took 6.2 min with exhaustive SIFT
SOLVE_GRACE_MIN = 2       # the function outlives the in-code deadline by this much, to save the job
SOLVE_GPUS = ["L4", "L40S"]
SOLVE_CPU, SOLVE_MEMORY_GIB = 8, 16
FETCH_CPU, FETCH_MEMORY_GIB = 2, 4
PER_CORE_HOUR, PER_GIB_HOUR = 0.047, 0.008           # Modal list prices, 2026-11
GPU_PER_HOUR = {"L4": 0.80, "L40S": 1.95}
SOLVE_ARGS = ["--no-labels"]                          # the label stage is after `--until occupancy` anyway; stated so a changed default cannot add it
ENV = {"PYTHONPATH": "/root/pipeline", "MPLBACKEND": "Agg", "PLY_CACHE_DIR": "/tmp/ply-cache"}
VOL = Path("/vol")
ROOT = VOL / "mushroom"
WORK = Path("/tmp/mushroom-work")
FETCH_TIMEOUT_S = max(mushroom_fetch.fetch_estimate(r)["timeoutSeconds"] for r in mushroom.ROOMS)
SOLVE_TIMEOUT_S = (math.ceil(mushroom_fetch.STOP_LOSS * SOLVE_MIN) + SOLVE_GRACE_MIN) * 60

cpu_image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy>=1.26", "scipy>=1.13")  # scipy: the nearest-point search that picks the alignment direction (the first fetch died without it).add_local_dir(PIPE / "plypipe", "/root/pipeline/plypipe"))
app = modal.App("ply-mushroom-seed")
vol = modal.Volume.from_name("ply-bench")  # not created here, so a wrong account fails at once


def describe(step: str, rooms: list[str]) -> str:
    """What a step fetches and what it will cost, for `plan` and printed before anything is started."""
    lines = []
    if step == "fetch":
        cost = (FETCH_CPU * PER_CORE_HOUR + FETCH_MEMORY_GIB * PER_GIB_HOUR)
        for r in rooms:
            e = mushroom_fetch.fetch_estimate(r)
            lines.append(f"{r}: fetches {e['bytes'] / 1e6:.0f} MB ({mushroom.ROOMS[r]['iphone']['name']} + {mushroom.ROOMS[r]['scan']['name']}); "
                         f"estimate {e['seconds'] / 60:.1f} min at one stream's {mushroom_fetch.BYTES_PER_SECOND / 1e6:.2f} MB/s, stop-loss {e['timeoutSeconds'] // 60} min, "
                         f"about ${e['seconds'] / 3600 * cost:.2f} (at most ${e['timeoutSeconds'] / 3600 * cost:.2f})")
        total = sum(mushroom_fetch.fetch_estimate(r)["bytes"] for r in rooms)
        lines.append(f"total {total / 1e6:.0f} MB onto the Volume's mushroom/_dl (kept), then the photos, poses and a thinned laser cloud per room")
    else:
        for g, rate in GPU_PER_HOUR.items():
            per_hour = rate + SOLVE_CPU * PER_CORE_HOUR + SOLVE_MEMORY_GIB * PER_GIB_HOUR
            lines.append(f"{g}: {SOLVE_MIN} min per room = ${SOLVE_MIN / 60 * per_hour:.2f}, stop-loss {math.ceil(mushroom_fetch.STOP_LOSS * SOLVE_MIN)} min = at most "
                         f"${math.ceil(mushroom_fetch.STOP_LOSS * SOLVE_MIN) / 60 * per_hour:.2f} (${per_hour:.2f}/h with {SOLVE_CPU} cores and {SOLVE_MEMORY_GIB} GiB)")
        lines.append(f"rooms: {', '.join(rooms)}; nothing is fetched; needs fetch done first (mushroom/<room>/in and reference)")
    return "\n".join(lines)


@app.function(image=cpu_image, cpu=FETCH_CPU, memory=FETCH_MEMORY_GIB * 1024, timeout=FETCH_TIMEOUT_S, volumes={"/vol": vol})
def fetch(room: str) -> dict:
    result = mushroom_fetch.seed_room(room, ROOT, WORK / "fetch" / room, commit=vol.commit)
    (ROOT / room).mkdir(parents=True, exist_ok=True)
    (ROOT / room / "seed.json").write_text(json.dumps({**result, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    vol.commit()
    print("[mushroom]", json.dumps(result), flush=True)
    return result


def _run(cmd: list[str], deadline: float) -> str | None:
    """One command in its own process group; None on success, else why it stopped."""
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env={**os.environ, **ENV}, start_new_session=True)
    try:
        code = proc.wait(timeout=max(1.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return "stop-loss: the solve was still running at its deadline"
    return None if code == 0 else f"plypipe exited with {code}"


@app.function(image=worker_image, gpu=SOLVE_GPUS, cpu=SOLVE_CPU, memory=SOLVE_MEMORY_GIB * 1024, timeout=SOLVE_TIMEOUT_S, volumes={"/vol": vol})
def solve(room: str) -> dict:
    started, src, job, dst = time.time(), ROOT / room, WORK / "job", ROOT / room / "job"
    absent = [p for p in ("in", "reference/meta.json", "reference/ref_points.npy") if not (src / p).exists()]
    if absent:
        raise FileNotFoundError(f"{src} lacks {', '.join(absent)}: run --step fetch first")
    if (dst / ".done" / "occupancy.json").exists():
        print(f"[mushroom] {room}: already solved ({dst}); delete it to solve again", flush=True)
        return json.loads((src / "solve.json").read_text()) if (src / "solve.json").exists() else {"room": room, "skipped": "already solved"}
    shutil.rmtree(job, ignore_errors=True)
    shutil.rmtree(WORK / "in", ignore_errors=True)
    shutil.copytree(src / "in", WORK / "in")
    error = _run([sys.executable, "-m", "plypipe", "run", str(WORK / "in"), "--out", str(job), "--until", "occupancy", *SOLVE_ARGS],
                 started + mushroom_fetch.STOP_LOSS * SOLVE_MIN * 60)
    report = json.loads((job / "report.json").read_text()) if (job / "report.json").exists() else {}
    result = {"room": room, "minutes": round((time.time() - started) / 60, 1), "error": error, "frames": report.get("frames"), "sfm": report.get("sfm"),
              "align": report.get("align")}
    if not error:
        try:  # the first scale reading, before any training is paid for
            result["score"] = mushroom.score_job(job, mushroom.load_reference(src / "reference"), splats=False)
        except Exception as e:  # noqa: BLE001: the solve is still worth saving
            result["score"] = {"error": f"{type(e).__name__}: {e}"}
    tmp = dst.with_name("job.tmp")
    job.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(job, tmp, symlinks=True, ignore=shutil.ignore_patterns("database.db", "sparse_all", "sparse_inc", "frames", "frames_raw"))
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    (src / "solve.json").write_text(json.dumps(result, indent=1))
    vol.commit()
    print("[mushroom]", json.dumps(result), flush=True)
    if error:
        raise RuntimeError(f"{room}: {error}; the partial job is in {dst}")
    return result


@app.local_entrypoint()
def main(step: str = "fetch", rooms: str = ",".join(mushroom.ROOMS)):
    names = [r.strip() for r in rooms.split(",") if r.strip()]
    unknown = [r for r in names if r not in mushroom.ROOMS]
    if step not in ("fetch", "solve") or unknown or not names:
        raise SystemExit(f"--step fetch|solve and --rooms from {', '.join(mushroom.ROOMS)} (got step={step!r}, rooms={rooms!r})")
    print(describe(step, names), flush=True)
    for room in names:
        call = (fetch if step == "fetch" else solve).spawn(room)
        print(f"[mushroom] {step} {room}: started {call.object_id}, not waiting; `modal app logs ply-mushroom-seed` follows it", flush=True)


if __name__ == "__main__":  # `plan` runs here, on this machine, not through `modal run` (which would build the image first)
    import argparse

    ap = argparse.ArgumentParser(prog="mushroom_seed.py plan")
    ap.add_argument("command", choices=["plan"])
    ap.add_argument("--rooms", default=",".join(mushroom.ROOMS))
    a = ap.parse_args()
    for step in ("fetch", "solve"):
        print(f"== {step}\n{describe(step, a.rooms.split(','))}")
