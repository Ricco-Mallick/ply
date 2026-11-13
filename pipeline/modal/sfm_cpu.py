"""Re-solve a saved job's cameras on a CPU-only container, from the matches already in its database.

The mapper (global positioning, bundle adjustment) runs on the CPU even on a GPU box: COLMAP's Ceres has no CUDA in
the pycolmap wheels. On the chapel it took 103 of the run's 119 min with the L40S idle, so this measures the same
solve where the hour costs $0.63 (8 cores + 32 GB) instead of $2.58.

    MODAL_PROFILE=<account> modal run --detach pipeline/modal/sfm_cpu.py --src chapel/quick-e2e/job --tag auto
    ... --args "--mapper global" --tag glob     # ordered video of 400+ frames starts incremental unless told otherwise

Reads <src> on the ply-bench Volume (frames/, frames_index.json, .done/frames.json, colmap/database.db) and writes
the solved job (without the database) to <src parent>/sfm-<tag>/job, ready for `plypipe run --from align`.

Phase 3 of docs/IMPROVEMENT-PLAN.md: one named variant of the solve, timed and scored, from the chapel's saved matches.

    python pipeline/modal/sfm_cpu.py plan --variant keyframes --cores 8       # estimate, stop-loss, cost; no Modal
    PLY_SFM_VARIANT=keyframes PLY_SFM_CPU=8 MODAL_PROFILE=<account> modal run --detach pipeline/modal/sfm_cpu.py \\
        --variant keyframes --cores 8                                          # spawns the run and returns its call id
    python pipeline/modal/sfm_cpu.py collect --out /tmp/sfm-solve              # the table of every variant on the Volume

The variant is incremental (1a, the reference), hierarchical (1g) or keyframes (1b); the cores 8 or 16. The container size
and timeout are fixed when this file loads, so PLY_SFM_VARIANT and PLY_SFM_CPU are read on this machine at launch and must
agree with --variant and --cores. It reads --src (default chapel/high/job: frames/, frames_index.json, .done/frames.json
and colmap/database.db must be there) and writes chapel/solve/<variant>-<cores>/: job/ (the solved job through align and
occupancy, laid out like chapel/high/job: data/sparse/0, dense/images, scene.json ... but no frames or database) and
result.json (seconds by part, frames registered, error, tracks, ATE in cm against --reference, default the aligned
chapel/high/job/data/sparse/0). To train on it: ply_eval.py ... --src chapel/solve/<variant>-<cores>/job --scenario chapel.
The entry spawns and returns: a client waiting on `.remote()` cancels the run when it is killed (docs/MISTAKES.md, 2026-11-11).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

PIPE = Path(__file__).resolve().parents[1]
for root in (PIPE, Path("/root/pipeline")):  # a checkout; the container's mounted copy. Appended: pipeline/modal must not shadow `modal`
    if (root / "plypipe").is_dir():
        sys.path.append(str(root))
from plypipe import solvebench  # noqa: E402

CORES = int(os.environ.get("PLY_SFM_CPU", "8"))
VARIANT = os.environ.get("PLY_SFM_VARIANT", "incremental")
solvebench.estimate_min(VARIANT, CORES)  # a wrong pair fails here, before anything is launched
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("pycolmap==4.2.1", "numpy>=1.26", "opencv-python-headless>=4.9")
         .add_local_dir(PIPE / "plypipe", "/root/pipeline/plypipe"))
app = modal.App("ply-sfm-cpu", image=image)
vol = modal.Volume.from_name("ply-bench")


@app.function(cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": vol})
def solve(src: str, tag: str, args: list[str]) -> dict:
    s, job = Path("/vol") / src, Path("/tmp/job")
    for rel in ("frames", "frames_index.json", ".done/frames.json", "colmap/database.db", "colmap/matching.json", "plan.json",
                "scene.json"):
        if (s / rel).is_dir():
            shutil.copytree(s / rel, job / rel)
        elif (s / rel).exists():
            (job / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s / rel, job / rel)
    prev = s / ".done" / "sfm.json"
    if prev.exists() and not (job / "colmap" / "matching.json").exists():  # jobs matched before matching.json existed
        r = json.loads(prev.read_text())
        (job / "colmap" / "matching.json").write_text(json.dumps({k: r.get(k) for k in ("features", "matcher", "loopDetection")}))
    Path("/tmp/in").mkdir(exist_ok=True)
    t0 = time.time()
    code = subprocess.run([sys.executable, "-m", "plypipe", "run", "/tmp/in", "--out", str(job), "--only", "sfm",
                           "--reuse-matches", *args], env=dict(os.environ, PYTHONPATH="/root/pipeline")).returncode
    minutes = round((time.time() - t0) / 60, 1)
    dst = s.parent / f"sfm-{tag}" / "job"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(job, dst, symlinks=True, ignore=shutil.ignore_patterns("database.db", "sparse_all", "sparse_inc"))
    vol.commit()
    done = job / ".done" / "sfm.json"
    return {"exit": code, "minutes": minutes, "report": json.loads(done.read_text()) if done.exists() else None, "saved": str(dst)}


JOB = Path("/tmp/work/job")
VOL = Path("/vol")
CODE = next(r for r in (Path("/root/pipeline"), PIPE) if (r / "plypipe").is_dir())  # the container's mounted copy, or this checkout


def _cpu_model() -> str:
    try:
        return next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name"))
    except (OSError, StopIteration):
        return "unknown"


def _plypipe(args: list[str], deadline: float | None) -> str | None:
    """One plypipe run in its own process group; None on success, else why it stopped."""
    import signal
    cmd = [sys.executable, "-m", "plypipe", "run", "/tmp/in", "--out", str(JOB), *args]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=str(CODE)), start_new_session=True)
    try:
        code = proc.wait(timeout=None if deadline is None else max(1.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return "stop-loss: the solve was still running at its deadline"
    return None if code == 0 else f"exit {code}"


@app.function(cpu=CORES, memory=solvebench.MEMORY_GIB[CORES] * 1024, timeout=solvebench.timeout_min(VARIANT, CORES) * 60, volumes={"/vol": vol})
def measure(job: dict) -> dict:
    """One variant of the camera solve from the saved matches: timed, scored against the reference, and saved."""
    import pycolmap
    from plypipe import ate
    started = time.time()
    v, c = job["variant"], job["cores"]
    src, dst = VOL / job["src"], VOL / solvebench.run_dir(v, c)
    absent = solvebench.missing(lambda rel: (src / rel).exists())
    if absent:
        raise FileNotFoundError(f"{src} lacks {', '.join(absent)}: the matches are not on this Volume (modal volume ls ply-bench {job['src']}/colmap); "
                                "copy the saved job's database, frames and frames_index.json here, or pass --src of a job that has them")
    shutil.rmtree(JOB, ignore_errors=True)
    for rel in solvebench.RESTORE:
        if (src / rel).is_dir():
            shutil.copytree(src / rel, JOB / rel)
        elif (src / rel).exists():
            (JOB / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, JOB / rel)
    Path("/tmp/in").mkdir(exist_ok=True)
    seconds = {"restore": round(time.time() - started, 1)}
    print(f"[sfm-cpu] {solvebench.describe(v, c)}", flush=True)
    t = time.time()
    error = _plypipe(["--only", "sfm", "--reuse-matches", *job["args"]], t + solvebench.solve_deadline_min(v, c) * 60)
    seconds["solve"] = round(time.time() - t, 1)
    report = model = ate_cm = None
    done = JOB / ".done" / "sfm.json"
    if not error and done.exists():
        report = json.loads(done.read_text())
        rec = pycolmap.Reconstruction(JOB / "colmap" / "sparse" / "0")
        model = solvebench.model_metrics(rec, report["images"])
        t = time.time()
        try:
            ref = pycolmap.Reconstruction(VOL / job["reference"])
            ate_cm = {k: (round(x * 100, 3) if k in ("rmse", "median", "max") else x) for k, x in ate.ate(ate.centres(rec), ate.centres(ref)).items()}
        except Exception as e:  # a missing or mismatched reference must not lose the solve
            error = f"ATE not scored: {type(e).__name__}: {e}"
        seconds["ate"] = round(time.time() - t, 1)
        t = time.time()
        # makes the solve usable by ply_eval.py: align writes data/sparse/0 and scene.json, occupancy the grid; labels need a GPU
        why = _plypipe(["--from", "align", "--until", "occupancy"], None)
        seconds["align"] = round(time.time() - t, 1)
        error = error or (f"align/occupancy: {why}" if why else None)
    seconds["wall"] = round(time.time() - started, 1)
    result = solvebench.assemble(job, report, model, ate_cm, seconds, _cpu_model(), error)
    t = time.time()
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    shutil.copytree(JOB, dst / "job", symlinks=True, ignore=shutil.ignore_patterns("database.db*", "sparse_all", "sparse_inc", "frames"))
    result["seconds"]["save"] = round(time.time() - t, 1)
    (dst / "result.json").write_text(json.dumps(result, indent=1))
    vol.commit()
    print("[sfm-cpu]", json.dumps(result), flush=True)
    return result


def _revision() -> dict:
    from plypipe import evalrun
    return evalrun.code_revision(PIPE.parent)


@app.local_entrypoint()
def main(src: str = "", tag: str = "auto", args: str = "", variant: str = "", cores: int = 8, reference: str = ""):
    if not variant:  # the original entry: re-solve a saved job and park it beside the source, waiting for it
        res = solve.remote(src or "chapel/quick-e2e/job", tag, args.split())
        print("[sfm-cpu]", json.dumps(res), flush=True)
        return
    if (variant, cores) != (VARIANT, CORES):  # the function was sized for these when this file was loaded
        raise SystemExit(f"--variant {variant} --cores {cores}: launch it with PLY_SFM_VARIANT={variant} PLY_SFM_CPU={cores}, "
                         "which sets the container and the timeout")
    job = solvebench.make_job(variant, cores, src, reference, args, _revision())
    print(solvebench.describe(variant, cores), flush=True)
    call = measure.spawn(job)
    print(f"[sfm-cpu] started {call.object_id}: not waiting. `modal app logs ply-sfm-cpu` follows it; the result lands in "
          f"{solvebench.run_dir(variant, cores)}/result.json on the Volume (python pipeline/modal/sfm_cpu.py collect --out DIR)", flush=True)


if __name__ == "__main__":  # `plan` and `collect` run here, on this machine, not through `modal run` (which would build the image first)
    import argparse

    ap = argparse.ArgumentParser(prog="sfm_cpu.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="the estimate, stop-loss and cost of one variant; no Modal")
    pl.add_argument("--variant", choices=solvebench.VARIANTS, required=True)
    pl.add_argument("--cores", type=int, choices=sorted(solvebench.MEMORY_GIB), default=8)
    co = sub.add_parser("collect", help="the table of every variant's result.json on the Volume")
    co.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        print(solvebench.describe(a.variant, a.cores))
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        got = solvebench.collect(vol, a.out)
        print(solvebench.table(got) if got else f"no result.json under {solvebench.SOLVE_ROOT} on the Volume yet")
