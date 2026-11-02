"""Benchmark a dataset through plypipe on Modal: solve once, then train + export one twin per preset.

    PLY_BENCH_DATA=/path/to/images MODAL_PROFILE=<account> \
        modal run pipeline/modal/ply_bench.py --name drjohnson --presets quick,standard,high

Runs on the worker's image (same cached layers, nothing rebuilt) in its own app, never touching the job queue.
Nothing is lost to a timeout: the job folder is saved to the `ply-bench` Volume after every stage, and while training
the newest snapshot PLY is saved every few minutes. Running the same command again resumes from what was saved.
Each preset's report, plan, held-out renders and SPZ files land in <out>/<name>/<preset>/.
"""
import os
import shutil
import subprocess
import sys
import threading
import time
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
vol = modal.Volume.from_name("ply-bench", create_if_missing=True)
ENV = {"PYTHONPATH": "/root/pipeline", "MPLBACKEND": "Agg", "PLY_CACHE_DIR": "/tmp/ply-cache",
       "PLY_LIVE_SNAPSHOTS": "0.25,0.5,0.75"}  # training leaves a usable model behind every quarter
JOB = Path("/tmp/work/job")
# the CI wheels are sm_89 only: L4 first, L40S when no L4 is free (about twice the price, about twice as fast)
GPUS = ["L4", "L40S"]
PREP = ("frames", "sfm", "align", "occupancy", "label")
SKIP = {"frames_raw", "database.db", "train"}  # re-creatable bulk; training state is saved as snapshots instead


def _plypipe(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "plypipe", "run", "/data/in", "--out", str(JOB), *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env={**os.environ, **ENV}, check=False).returncode


def _save(dst: Path, src: Path = JOB) -> None:
    """Mirror the job folder onto the Volume (changed files only) and commit."""
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if SKIP & set(rel.parts):
            continue
        q = dst / rel
        if p.is_symlink():
            if not q.is_symlink():
                q.parent.mkdir(parents=True, exist_ok=True)
                q.unlink(missing_ok=True)
                q.symlink_to(os.readlink(p))
        elif p.is_file() and (not q.exists() or q.stat().st_size != p.stat().st_size or q.stat().st_mtime < p.stat().st_mtime):
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
    vol.commit()


def _restore(src: Path, dst: Path = JOB) -> None:
    """Layer a saved folder onto the job (a preset over the shared prep): files replace, existing links stay."""
    if not src.exists():
        return
    for p in sorted(src.rglob("*")):
        q = dst / p.relative_to(src)
        if p.is_symlink():
            if not q.is_symlink() and not q.exists():
                q.parent.mkdir(parents=True, exist_ok=True)
                q.symlink_to(os.readlink(p))
        elif p.is_file() and not any(par.is_symlink() for par in q.parents):
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)


def _snapshots(dst: Path, stop: threading.Event) -> None:
    """While training: keep the newest snapshot PLY (and gsplat's stats) on the Volume."""
    while not stop.wait(180):
        plys = sorted((JOB / "train" / "ply").glob("point_cloud_*.ply"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if not plys:
            continue
        out = dst / "train-snapshot"
        out.mkdir(parents=True, exist_ok=True)
        if not (out / plys[-1].name).exists():
            tmp = out / (plys[-1].name + ".part")
            shutil.copy2(plys[-1], tmp)
            tmp.rename(out / plys[-1].name)
            for old in out.glob("point_cloud_*.ply"):
                if old.name != plys[-1].name:
                    old.unlink()
            vol.commit()
            print(f"[bench] saved snapshot {plys[-1].name}", flush=True)


@app.function(gpu=GPUS, cpu=8, memory=32768, timeout=3600, volumes={"/vol": vol}, secrets=[hf])
def prep(name: str, extra: list[str]) -> str:
    """frames, sfm, align, occupancy, label: everything the presets share, saved after each stage."""
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    saved = Path("/vol") / name / "prep"
    _restore(saved)
    for stage in PREP:
        if (JOB / ".done" / f"{stage}.json").exists():
            print(f"[bench] {stage}: saved earlier", flush=True)
            continue
        code = _plypipe(["--only", stage, *extra])
        if code:
            raise RuntimeError(f"{stage} failed (exit {code}); stages before it are saved")
        _save(saved)
        print(f"[bench] {stage}: saved", flush=True)
    return (JOB / "report.json").read_text()


@app.function(gpu=GPUS, cpu=4, memory=16384, timeout=3 * 3600, volumes={"/vol": vol}, secrets=[hf])
def train(name: str, preset: str, extra: list[str], tag: str = "") -> dict:
    root = Path("/vol") / name
    saved = root / (preset + tag)
    _restore(root / "prep")
    _restore(saved)  # a preset that already trained resumes at export
    stop = threading.Event()
    watcher = threading.Thread(target=_snapshots, args=(saved, stop), daemon=True)
    watcher.start()
    code = 0
    for stage in ("train", "export"):
        if (JOB / ".done" / f"{stage}.json").exists():
            continue
        code = _plypipe(["--only", stage, "--preset", preset, *extra])
        if code:
            break
        _save(saved)
    stop.set()
    if (JOB / ".done" / "export.json").exists():  # the gate, as the CLI applies it after a full run
        sys.path.insert(0, "/root/pipeline")
        import json
        from plypipe import plan, verify
        report = json.loads((JOB / "report.json").read_text())
        report["quality"] = verify.check(report, plan.read(JOB))
        (JOB / "report.json").write_text(json.dumps(report, indent=2))
        _save(saved)
    keep = ["report.json", "plan.json", "scene.json", "log.jsonl", "scene.spz", "scene_lite.spz", "shell.json",
            "occupancy.png", "occupancy.json", "places.json"]
    files = {k: (JOB / k).read_bytes() for k in keep if (JOB / k).exists()}
    files.update({f"eval/{p.name}": p.read_bytes() for p in sorted((JOB / "eval").glob("*")) if p.is_file()})
    return {"exit": code, "files": files}


@app.function(gpu=GPUS, cpu=4, memory=16384, timeout=1800, volumes={"/vol": vol}, secrets=[hf])
def rescore(name: str, folder: str) -> dict:
    """Export a trained preset again (current export code, including the viewer evaluation): no retraining."""
    root = Path("/vol") / name
    _restore(root / "prep")
    _restore(root / folder)
    (JOB / ".done" / "export.json").unlink(missing_ok=True)
    code = _plypipe(["--only", "export"])
    _save(root / folder)
    import json
    rep = json.loads((JOB / "report.json").read_text())
    files = {k: (JOB / k).read_bytes() for k in ("report.json", "scene.json", "scene.spz", "scene_lite.spz") if (JOB / k).exists()}
    return {"exit": code, "export": rep.get("export"), "train": rep.get("train"), "files": files}


@app.function(gpu=GPUS, cpu=4, memory=16384, timeout=1200, volumes={"/vol": vol})
def vieweval(name: str, folder: str, files: list[str]) -> dict:
    """The viewer-side score of saved PLYs (trained vs exported), without re-running anything."""
    sys.path.insert(0, "/root/pipeline")
    from plypipe import viewer_eval
    root = Path("/vol") / name
    _restore(root / "prep")
    return {f: viewer_eval.evaluate(root / folder / f, JOB / "data") for f in files}


@app.local_entrypoint()
def main(name: str = "bench", presets: str = "quick,standard,high", args: str = "", out: str = "bench-out",
         prep_only: bool = False, tag: str = "", rescore_only: str = ""):
    extra = args.split()
    if args.startswith("vieweval:"):  # vieweval:<folder>:<file>,<file>
        _, folder, files = args.split(":")
        print("[bench] vieweval", folder, vieweval.remote(name, folder, files.split(",")), flush=True)
        return
    if rescore_only:  # comma-separated folders already on the Volume
        import json
        for folder, res in zip(rescore_only.split(","), rescore.map(*zip(*[(name, f) for f in rescore_only.split(",")]))):
            ex = res["export"] or {}
            print(f"[bench] rescore {folder}: exit {res['exit']} viewer {json.dumps(ex.get('viewerEval'))} "
                  f"held-out {json.dumps((res['train'] or {}).get('eval'))}", flush=True)
            d = Path(out) / name / folder
            for k, v in (res.get("files") or {}).items():
                d.mkdir(parents=True, exist_ok=True)
                (d / k).write_bytes(v)
        return
    t0 = time.time()
    prep.remote(name, extra)
    print(f"[bench] prep done in {(time.time() - t0) / 60:.1f} min", flush=True)
    if prep_only:
        return
    names = [p for p in presets.split(",") if p]
    for preset, res in zip(names, train.starmap([(name, p, extra, tag) for p in names], return_exceptions=True)):
        if isinstance(res, Exception):
            print(f"[bench] {preset}: FAILED {res!r} (progress is on the ply-bench Volume; rerun to resume)", flush=True)
            continue
        d = Path(out) / name / (preset + tag)
        for k, v in res["files"].items():
            (d / k).parent.mkdir(parents=True, exist_ok=True)
            (d / k).write_bytes(v)
        print(f"[bench] {preset}: exit {res['exit']}, {len(res['files'])} files -> {d}", flush=True)
    print(f"[bench] total {(time.time() - t0) / 60:.1f} min", flush=True)
