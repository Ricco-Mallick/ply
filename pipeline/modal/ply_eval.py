"""Image-quality experiments from one saved camera solve: ONE variant per run, scored on two hold-outs.

    python pipeline/modal/ply_eval.py plan --variant q0a        # prints the restore list, commands, stop-loss and cost; no Modal

    MODAL_PROFILE=<account> modal run --detach pipeline/modal/ply_eval.py \\
        --src chapel/high/job --scenario chapel --variant q0a   # one run; --args "..." makes an ad-hoc variant
    python pipeline/modal/ply_eval.py collect --scenario chapel --out /tmp/ply-eval      # metrics.json files, from the Volume
    python scripts/eval_table.py /tmp/ply-eval                                           # the results table

Runs on the worker's image (same cached layers) in its own app, never touching the job queue. The saved job is read
from the `ply-bench` Volume: its camera solve and photos are restored, its old training and export outputs are not,
then `plypipe --only train` and `--only export` run with the variant's arguments. Each finished stage is saved to
<scenario>/eval/<variant>/ and committed before the next starts; a relaunch layers that folder over the restore and
runs only what is not finished (training itself cannot resume part-way: the newest snapshot is kept only so a stopped
run leaves a model behind). The entrypoint starts exactly one long function, so `--detach` keeps it alive.

The stop-loss is in code: the function's timeout (1.5x the estimate, plus two minutes to save) and an earlier in-code
deadline for the variant. GPU: one type for comparable timings (L40S); PLY_EVAL_GPU=L40S,L4 allows a fallback, with a
longer timeout. Modal 1.6.3 can also set gpu and timeout per call (Function.with_options), but only one GPU name and
unproven under --detach here, so this file fixes them when it is loaded: PLY_EVAL_GPU and PLY_EVAL_TIMEOUT_MIN are read
at launch, on this machine, never inside the container.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for root in (HERE.parent, Path("/root/pipeline")):  # a checkout; the container's mounted copy. Appended: pipeline/modal must not shadow `modal`
    if (root / "plypipe").is_dir():
        sys.path.append(str(root))
from ply_modal import image as worker_image  # noqa: E402
from plypipe import evalrun  # noqa: E402

GPUS = [g.strip() for g in os.environ.get("PLY_EVAL_GPU", "L40S").split(",") if g.strip()]
CEILING_MIN = int(os.environ.get("PLY_EVAL_TIMEOUT_MIN") or evalrun.default_ceiling_min(tuple(GPUS)))
image = worker_image.add_local_file(HERE / "ply_modal.py", "/root/ply_modal.py")
app = modal.App("ply-eval", image=image)
vol = modal.Volume.from_name("ply-bench")  # seeded by `modal volume put`; not created here, so a wrong account fails at once
ENV = {"PYTHONPATH": "/root/pipeline", "MPLBACKEND": "Agg", "PLY_CACHE_DIR": "/tmp/ply-cache", "PLY_LIVE_SNAPSHOTS": "0.5,0.75"}
JOB = Path("/tmp/work/job")


def _run(stage: str, args: list[str], deadline: float) -> str | None:
    """One plypipe stage in its own process group; None on success, else why it stopped."""
    cmd = evalrun.plypipe_command(stage, str(JOB), args, sys.executable)
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env={**os.environ, **ENV}, start_new_session=True)
    try:
        code = proc.wait(timeout=max(1.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return f"stop-loss: {stage} still running at the deadline"
    return None if code == 0 else f"{stage} exited with {code}"


def _snapshots(dst: Path, stop: threading.Event) -> None:
    """While training: keep the newest snapshot PLY on the Volume, for the case that the run is stopped."""
    while not stop.wait(180):
        plys = sorted((JOB / "train" / "ply").glob("point_cloud_*.ply"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if not plys or (dst / "train-snapshot" / plys[-1].name).exists():
            continue
        out = dst / "train-snapshot"
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / (plys[-1].name + ".part")
        shutil.copy2(plys[-1], tmp)
        tmp.rename(out / plys[-1].name)
        for old in out.glob("point_cloud_*.ply"):
            if old.name != plys[-1].name:
                old.unlink()
        vol.commit()
        print(f"[eval] saved snapshot {plys[-1].name}", flush=True)


@app.function(gpu=GPUS if len(GPUS) > 1 else GPUS[0], cpu=evalrun.CPU, memory=evalrun.MEMORY_GIB * 1024,
              timeout=(CEILING_MIN + evalrun.GRACE_MIN) * 60, volumes={"/vol": vol})
def run_variant(job: dict) -> dict:
    started = time.time()
    gpu = evalrun.parse_gpu(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                           capture_output=True, text=True).stdout)
    print(f"[eval] {job['variant']} on {gpu}", flush=True)
    src = Path("/vol") / job["src"]
    saved = Path("/vol") / evalrun.variant_dir(job["scenario"], job["variant"])
    absent = [p for p in ("data/sparse/0/images.bin", "dense/images", "scene.json") if not (src / p).exists()]
    if absent:
        raise FileNotFoundError(f"/vol/{job['src']} lacks {', '.join(absent)}: seed the Volume with the saved job (modal volume ls ply-bench {job['src']})")
    if job["force"]:
        shutil.rmtree(saved / ".done", ignore_errors=True)
        (saved / "metrics.json").unlink(missing_ok=True)
    previous = evalrun.read_metrics(saved / "metrics.json")
    evalrun.check_resume(previous, job)
    if previous and previous["complete"]:
        print(f"[eval] {job['variant']} is complete on the Volume (--force reruns it)", flush=True)
        return previous

    t = time.time()
    got = evalrun.restore(src, JOB)
    layered = evalrun.layer(saved, JOB) if saved.exists() else 0
    problems = evalrun.check_job(JOB)
    if problems:
        raise RuntimeError(f"the restored job is unusable, missing: {', '.join(problems)}")
    todo = evalrun.stages_to_run(JOB)
    evalrun.clear_stale(JOB, todo)
    seconds = {"restore": round(time.time() - t, 1), "save": 0.0}
    print(f"[eval] restored {got['files']} files ({got['bytes'] / 2**20:.0f} MB) in {seconds['restore']} s, {layered} from an earlier run; "
          f"running {todo or 'nothing'}", flush=True)

    deadline = started + job["stopLossMin"] * 60
    stop = threading.Event()
    threading.Thread(target=_snapshots, args=(saved, stop), daemon=True).start()
    ran, error = [], None

    def metrics(report: dict) -> dict:
        sizes = {n: (JOB / n).stat().st_size for n in ("scene.ply", "scene.spz", "scene_lite.spz") if (JOB / n).exists()}
        record = evalrun.invocation_record(started, time.time() - started, gpu, ran, [s for s in ("train", "export") if s not in todo], error)
        return evalrun.assemble(job, report, gpu=gpu, seconds={**seconds, "wall": round(time.time() - started, 1)}, sizes=sizes,
                                invocation=record, previous=previous)

    def save() -> dict:
        """Metrics so far (cost included, for a relaunch to add to), then the job onto the Volume, then commit."""
        report_file = JOB / "report.json"
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
        if not error and (JOB / ".done" / "export.json").exists():  # the gate the CLI applies after a full run
            from plypipe import plan, verify
            report["quality"] = verify.check(report, plan.read(JOB))
            report_file.write_text(json.dumps(report, indent=2))
        result = metrics(report)
        (JOB / "metrics.json").write_text(json.dumps(result, indent=1))
        t0 = time.time()
        evalrun.save(JOB, saved)
        seconds["save"] = round(seconds["save"] + time.time() - t0, 1)
        vol.commit()
        return result

    for stage in todo:
        error = _run(stage, job["args"], deadline)
        if error:
            break
        ran.append(stage)
        save()
        print(f"[eval] {stage}: done and saved", flush=True)
    stop.set()
    result = save()
    print(evalrun.summary(result), flush=True)
    if error:
        raise RuntimeError(f"{job['variant']}: {error}; progress and partial metrics are on the Volume, relaunch to resume")
    return result


@app.local_entrypoint()
def main(src: str = "chapel/high/job", scenario: str = "chapel", variant: str = "", args: str = "", force: bool = False):
    job = evalrun.make_job(scenario, src, variant, args, force, tuple(GPUS), CEILING_MIN, evalrun.code_revision(HERE.parent.parent))
    print(evalrun.describe(job), flush=True)
    result = run_variant.remote(job)
    print(evalrun.summary(result), flush=True)


if __name__ == "__main__":  # `plan` and `collect` run here, on this machine, not through `modal run` (which would build the image first)
    import argparse

    if sys.argv[1:2] == ["collect"]:
        ap = argparse.ArgumentParser(prog="ply_eval.py collect")
        ap.add_argument("--scenario", default="chapel")
        ap.add_argument("--out", type=Path, required=True)
        a = ap.parse_args(sys.argv[2:])
        for name, files in evalrun.collect(vol, a.scenario, a.out).items():
            print(f"{name}: {', '.join(files) if files else 'no metrics.json (not finished)'}")
    else:
        sys.exit(evalrun.main(sys.argv[1:]))
