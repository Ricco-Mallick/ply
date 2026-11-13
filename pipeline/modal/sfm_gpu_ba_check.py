"""1f of docs/IMPROVEMENT-PLAN.md, a check and nothing more: does bundle adjustment run on the GPU in pycolmap-cuda12 4.2.1?

Takes the first 60 frames of a saved solve, nudges its 3D points by the same seeded noise, and gives the same start to three
solvers in turn: Ceres on the CPU, Ceres with `ceres.use_gpu` (what `ba_use_gpu` sets in the mapper), and the CASPAR backend.
Prints and saves one JSON: which ran, seconds each, the cost each ended at, and why one did not (the COLMAP CPU build raises
ValueError for CASPAR; the CUDA build may abort the process instead, so every solver runs in its own process with a timeout).

    MODAL_PROFILE=<account> modal run --detach pipeline/modal/sfm_gpu_ba_check.py            # spawns and returns the call id
    python pipeline/modal/sfm_gpu_ba_check.py collect                                        # the JSON, from the Volume
    python pipeline/modal/sfm_gpu_ba_check.py local --model <sparse model dir>               # the same check here (CPU build: GPU rows fail)

Runs on the eval runner's image (the worker's, plus its entry file) on an L4: about 5 minutes, $0.07 at $0.80/h. The stop-loss
is the function timeout (15 min, $0.20); each solver is also cut at 4 min. It reads the aligned model of chapel/high/job on the
ply-bench Volume, nothing else, and writes chapel/solve/ba-check.json. Time the chapel's 1,223 frames only if a GPU row ran and
ended at the CPU's cost or lower, faster than the CPU row.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ply_modal import image as worker_image  # noqa: E402

MODEL = "chapel/high/job/data/sparse/0"
RESULT = "chapel/solve/ba-check.json"
FRAMES, SEED = 60, 0
TIMEOUT_MIN = 15
SOLVER_TIMEOUT_S = 240
SOLVERS = ("ceres-cpu", "ceres-gpu", "caspar")

image = worker_image.add_local_file(HERE / "ply_modal.py", "/root/ply_modal.py")
app = modal.App("ply-sfm-ba-check", image=image)
vol = modal.Volume.from_name("ply-bench")
VOL = Path("/vol")

# one solver on the model in argv[1], printing one JSON line; a crash of the solver itself kills only this process
CHILD = r'''
import json, sys, time
import pycolmap
name, model = sys.argv[1], sys.argv[2]
rec = pycolmap.Reconstruction(model)
opts = pycolmap.BundleAdjustmentOptions()
opts.print_summary = False
if name == "ceres-gpu":
    opts.ceres.use_gpu = True
elif name == "caspar":
    opts.backend = pycolmap.BundleAdjustmentBackend.CASPAR
cfg = pycolmap.BundleAdjustmentConfig()
for image_id in rec.reg_image_ids():
    cfg.add_image(image_id)
rec.update_point_3d_errors()
before = float(rec.compute_mean_reprojection_error())
t = time.time()
summary = pycolmap.create_default_bundle_adjuster(opts, cfg, rec).solve()
seconds = time.time() - t
c = summary.ceres_summary
print("RESULT " + json.dumps({"solver": name, "seconds": round(seconds, 3), "usable": bool(summary.is_solution_usable()),
      "termination": str(summary.termination_type), "initialCost": float(c.initial_cost), "finalCost": float(c.final_cost),
      "iterations": int(c.num_successful_steps + c.num_unsuccessful_steps), "residuals": int(c.num_residuals),
      "linearSolver": str(c.linear_solver_type_used), "denseLibrary": str(c.dense_linear_algebra_library_type),
      "sparseLibrary": str(c.sparse_linear_algebra_library_type), "meanErrorPxBefore": before,
      "meanErrorPxAfter": (rec.update_point_3d_errors(), float(rec.compute_mean_reprojection_error()))[1]}), flush=True)
'''


def probe(pycolmap) -> dict:
    """Which GPU options this pycolmap exposes; looking must not raise on the CPU build."""
    ba = pycolmap.BundleAdjustmentOptions()
    inc = pycolmap.IncrementalPipelineOptions()
    return {"pycolmap": pycolmap.__version__, "hasCuda": bool(getattr(pycolmap, "has_cuda", False)),
            "backends": sorted(pycolmap.BundleAdjustmentBackend.__members__), "ceresUseGpu": hasattr(ba.ceres, "use_gpu"),
            "baUseGpu": hasattr(inc, "ba_use_gpu"), "baGlobalBackend": hasattr(inc, "ba_global_backend"),
            "caspar": hasattr(pycolmap, "CasparBundleAdjustmentOptions") and hasattr(ba, "caspar")}


def prepare(pycolmap, model: Path, out: Path, frames: int = FRAMES, seed: int = SEED) -> dict:
    """The first `frames` posed views of the model, its points moved by 1% of their spread (seeded), written to `out`."""
    import numpy as np
    rec = pycolmap.Reconstruction(str(model))
    posed = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    for im in posed[frames:]:
        rec.deregister_frame(im.frame_id)
    rng = np.random.default_rng(seed)
    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts])
    spread = float(np.median(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1))) if len(pts) else 0.0
    for p in pts:
        p.xyz = p.xyz + rng.normal(0, 0.01 * spread, 3)
    out.mkdir(parents=True, exist_ok=True)
    rec.update_point_3d_errors()
    rec.write(str(out))
    return {"frames": rec.num_reg_images(), "points": rec.num_points3D(), "pointNoise": 0.01 * spread,
            "meanErrorPx": float(rec.compute_mean_reprojection_error())}


def run_solver(name: str, model: Path, timeout_s: int = SOLVER_TIMEOUT_S) -> dict:
    t = time.time()
    try:
        r = subprocess.run([sys.executable, "-c", CHILD, name, str(model)], capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"solver": name, "ran": False, "why": f"no answer in {timeout_s} s"}
    lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    if r.returncode == 0 and lines:
        return {**json.loads(lines[-1][7:]), "ran": True}
    tail = [l for l in (r.stderr or r.stdout).splitlines() if l.strip()]
    return {"solver": name, "ran": False, "exit": r.returncode, "seconds": round(time.time() - t, 2), "why": (tail[-1] if tail else "no output")[:300]}


def check(model: Path, work: Path, frames: int = FRAMES) -> dict:
    import pycolmap
    info = probe(pycolmap)
    start = prepare(pycolmap, model, work / "start", frames)
    rows = {name: run_solver(name, work / "start") for name in SOLVERS}
    cpu = rows["ceres-cpu"]
    for row in rows.values():
        if row.get("ran") and cpu.get("ran"):
            row["speedupVsCpu"] = round(cpu["seconds"] / max(row["seconds"], 1e-9), 2)
            row["finalCostVsCpu"] = round(row["finalCost"] / max(cpu["finalCost"], 1e-12), 4)
    return {"probe": info, "start": start, "ran": [n for n, r in rows.items() if r.get("ran")], "solvers": rows}


@app.function(gpu="L4", timeout=TIMEOUT_MIN * 60, volumes={"/vol": vol})
def run() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    result = {"gpu": out, **check(VOL / MODEL, Path("/tmp/ba-check"))}
    (VOL / RESULT).parent.mkdir(parents=True, exist_ok=True)
    (VOL / RESULT).write_text(json.dumps(result, indent=1))
    vol.commit()
    print("[ba-check]", json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main():
    call = run.spawn()
    print(f"[ba-check] started {call.object_id}: not waiting. `modal app logs ply-sfm-ba-check` follows it; the JSON lands in {RESULT} on the "
          "Volume (python pipeline/modal/sfm_gpu_ba_check.py collect)", flush=True)


if __name__ == "__main__":  # `collect` and `local` run here, not through `modal run`
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(prog="sfm_gpu_ba_check.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    lo = sub.add_parser("local")
    lo.add_argument("--model", type=Path, required=True)
    lo.add_argument("--frames", type=int, default=FRAMES)
    a = ap.parse_args()
    if a.cmd == "collect":
        print(b"".join(vol.read_file(RESULT)).decode())
    else:
        with tempfile.TemporaryDirectory() as tmp:
            print(json.dumps(check(a.model, Path(tmp), a.frames), indent=1))
