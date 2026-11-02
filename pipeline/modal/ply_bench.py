"""Benchmark a dataset through plypipe on Modal: solve once, then train + export one twin per preset, in parallel.

    PLY_BENCH_DATA=/path/to/images MODAL_PROFILE=<account> \
        modal run pipeline/modal/ply_bench.py --name drjohnson --presets quick,standard,high

Runs on the worker's image (same cached layers, so nothing is rebuilt) in its own app, never touching the job queue.
The solved job (frames..label) is saved as bench-out/<name>/prep.tar; pass --from-tar to train more presets from it
without paying for SfM again. Each preset's report, plan, held-out renders and SPZ files land in bench-out/<name>/<preset>/.
"""
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from ply_modal import image as worker_image  # noqa: E402

DATA = os.environ.get("PLY_BENCH_DATA", "/nonexistent")
image = (worker_image
         .add_local_file(Path(__file__).with_name("ply_modal.py"), "/root/ply_modal.py")
         .add_local_dir(DATA, "/data/in"))
app = modal.App("ply-bench", image=image)
hf = modal.Secret.from_name("ply-hf")  # HF_TOKEN for the gated SAM 3 weights (label stage)
ENV = {"PYTHONPATH": "/root/pipeline", "MPLBACKEND": "Agg", "PLY_CACHE_DIR": "/tmp/ply-cache"}
JOB = Path("/tmp/work/job")


def _plypipe(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "plypipe", "run", "/data/in", "--out", str(JOB), *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env={**os.environ, **ENV}, check=False).returncode


@app.function(gpu="L4", cpu=8, memory=32768, timeout=3600, secrets=[hf])
def prep(extra: list[str]) -> bytes:
    """frames, sfm, align, occupancy, label: everything the presets share."""
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    code = _plypipe(["--until", "label", *extra])
    if code:
        raise RuntimeError(f"prep failed (exit {code})")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:  # uncompressed: the bulk is already-compressed JPEG
        tf.add(JOB, arcname="job", filter=lambda ti: None if ti.name.endswith("database.db") else ti)
    return buf.getvalue()


@app.function(gpu="L4", cpu=4, memory=16384, timeout=3 * 3600, secrets=[hf])
def train(prep_tar: bytes, preset: str, extra: list[str]) -> dict:
    with tarfile.open(fileobj=io.BytesIO(prep_tar)) as tf:
        tf.extractall(JOB.parent)
    code = _plypipe(["--from", "train", "--preset", preset, *extra])
    keep = ["report.json", "plan.json", "scene.json", "log.jsonl", "scene.spz", "scene_lite.spz", "shell.json",
            "occupancy.png", "occupancy.json"]
    files = {k: (JOB / k).read_bytes() for k in keep if (JOB / k).exists()}
    files.update({f"eval/{p.name}": p.read_bytes() for p in sorted((JOB / "eval").glob("*")) if p.is_file()})
    files.update({f"train/{p.name}": p.read_bytes() for p in sorted((JOB / "train" / "stats").glob("*.json"))})
    return {"exit": code, "files": files}


@app.local_entrypoint()
def main(name: str = "bench", presets: str = "quick,standard,high", args: str = "", out: str = "bench-out",
         from_tar: str = "", prep_only: bool = False):
    root = Path(out) / name
    root.mkdir(parents=True, exist_ok=True)
    extra = args.split()
    if from_tar:
        tar = Path(from_tar).read_bytes()
    else:
        tar = prep.remote(extra)
        (root / "prep.tar").write_bytes(tar)
        print(f"solved job saved: {root / 'prep.tar'} ({len(tar) >> 20} MB)")
    if prep_only:
        return
    names = [p for p in presets.split(",") if p]
    for preset, res in zip(names, train.starmap([(tar, p, extra) for p in names])):
        d = root / preset
        for k, v in res["files"].items():
            (d / k).parent.mkdir(parents=True, exist_ok=True)
            (d / k).write_bytes(v)
        print(f"{preset}: exit {res['exit']}, {len(res['files'])} files -> {d}")
