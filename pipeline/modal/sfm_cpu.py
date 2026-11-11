"""Re-solve a saved job's cameras on a CPU-only container, from the matches already in its database.

The mapper (global positioning, bundle adjustment) runs on the CPU even on a GPU box: COLMAP's Ceres has no CUDA in
the pycolmap wheels. On the chapel it took 103 of the run's 119 min with the L40S idle, so this measures the same
solve where the hour costs $0.63 (8 cores + 32 GB) instead of $2.58.

    MODAL_PROFILE=<account> modal run --detach pipeline/modal/sfm_cpu.py --src chapel/quick-e2e/job --tag auto
    ... --args "--mapper global" --tag glob     # ordered video of 400+ frames starts incremental unless told otherwise

Reads <src> on the ply-bench Volume (frames/, frames_index.json, .done/frames.json, colmap/database.db) and writes
the solved job (without the database) to <src parent>/sfm-<tag>/job, ready for `plypipe run --from align`.
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


@app.local_entrypoint()
def main(src: str = "chapel/quick-e2e/job", tag: str = "auto", args: str = ""):
    res = solve.remote(src, tag, args.split())
    print("[sfm-cpu]", json.dumps(res), flush=True)
