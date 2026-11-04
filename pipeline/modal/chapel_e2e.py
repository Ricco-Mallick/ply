"""End-to-end test harness: the chapel clips from Drive -> a full plypipe run on one GPU, resumable.

    modal run pipeline/modal/chapel_e2e.py --step fetch                      # clips -> ply-bench Volume, chapel/in
    modal run --detach pipeline/modal/chapel_e2e.py --step run --args "--preset quick --iterations 7000" --tag quick-e2e

Every finished stage is copied to chapel/<tag>/job, and a relaunch with the same tag resumes from there (a seeded
job folder, e.g. one solved by sfm_cpu.py, works the same way). See docs/MISTAKES.md before a paid run."""
import os, shutil, subprocess, sys, threading, time
from pathlib import Path
import modal
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).parent))
    from ply_modal import image as worker_image  # noqa: E402
else:  # inside a container the image is already built; only the name is needed
    worker_image = modal.Image.debian_slim()

IDS = ["1890egZVkDLCBlTDs4OA6T8f1T50LIbX6", "1lPLICc2KGXToggrBcPEwUnWiftGELs03", "1ohEI7Uh7U1wuiz3tVbg8qVuXQ9xGIAiS",
       "16KFaN3uwaWU_UiqF3MaJdO4tJHFA1cTD", "1RJiWNIT6vdY0aVGaKm2nGuS5FAI6wMRD", "13c6OnjncNQCMpDo-z0r_xy4SZ8ZUMhrH",
       "12SQnFqePox7tqN__2iy_nELBp-v6B6s4", "1R-jgs16cMdof0cHm-_RffGQhnJDKyc-Y", "1rO0SMfuTovwHlU6thAJAwdureqmD254f"]
app = modal.App("ply-chapel-e2e", image=worker_image)
vol = modal.Volume.from_name("ply-bench", create_if_missing=True)
ROOT = Path("/vol/chapel")

@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/vol": vol}, timeout=1800, cpu=1)
def fetch(i: str) -> str:
    import re, urllib.request
    (ROOT / "in").mkdir(parents=True, exist_ok=True)
    r = urllib.request.urlopen(urllib.request.Request(f"https://drive.usercontent.google.com/download?id={i}&export=download&confirm=t",
                                                      headers={"User-Agent": "Mozilla/5.0"}), timeout=600)
    if "text/html" in (r.headers.get("content-type") or ""):
        return f"{i}: web page, not a file"
    name = re.search(r'filename="([^"]+)"', r.headers.get("content-disposition", "")).group(1)
    dest = ROOT / "in" / name
    if dest.exists() and dest.stat().st_size == int(r.headers.get("content-length", -1)):
        return f"{name}: already there"
    with open(str(dest) + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 8 << 20)
    os.replace(str(dest) + ".part", dest)
    vol.commit()
    return f"{name}: {dest.stat().st_size / 2**20:.0f} MB"

@app.function(gpu=["L4", "L40S"], cpu=8, memory=32768, timeout=4 * 3600, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("ply-hf")])
def run(args: list[str], tag: str) -> int:
    inp, out, keep = Path("/tmp/in"), Path("/tmp/out"), ROOT / tag
    keep.mkdir(parents=True, exist_ok=True)
    inp.mkdir(parents=True, exist_ok=True)
    if not (keep / "job" / ".done" / "frames.json").exists():  # frames already extracted: the clips are not read
        shutil.copytree(ROOT / "in", inp, dirs_exist_ok=True)
    if (keep / "job").exists():  # finished stages from an earlier attempt: the CLI skips them via .done/
        shutil.copytree(keep / "job", out, symlinks=True, dirs_exist_ok=True)
        print("[e2e] resumed stages:", sorted(p.stem for p in (out / ".done").glob("*.json")), flush=True)
    saved = set()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    env = dict(os.environ, PYTHONPATH="/root/pipeline", MPLBACKEND="Agg", PLY_CACHE_DIR="/tmp/ply-cache")
    stop = threading.Event()
    def mirror(final=False):
        for name in ("log.jsonl", "report.json", "plan.json", "scene.json", "frames_index.json"):
            if (out / name).exists(): shutil.copy2(out / name, keep / name)
        if (out / "logs").exists(): shutil.copytree(out / "logs", keep / "logs", dirs_exist_ok=True)
        if final:
            for p in out.iterdir():
                if p.is_file() and p.suffix in (".spz", ".json", ".png", ".ply") and p.stat().st_size < 400 << 20: shutil.copy2(p, keep / p.name)
            for d in ("eval", "sparse_txt"):
                if (out / d).exists(): shutil.copytree(out / d, keep / d, dirs_exist_ok=True)
            fr = out / "frames"
            if fr.exists():  # every 10th frame plus the per-frame sharpness, for the frame-selection review
                (keep / "frames_sample").mkdir(exist_ok=True)
                for j, p in enumerate(sorted(fr.glob("*.jpg"))):
                    if j % 10 == 0: shutil.copy2(p, keep / "frames_sample" / p.name)
            dsp = out / "data" / "sparse"
            if dsp.exists(): shutil.copytree(dsp, keep / "sparse", dirs_exist_ok=True)
        vol.commit()
    def checkpoint():
        done = {p.stem for p in (out / ".done").glob("*.json")} if (out / ".done").exists() else set()
        if done - saved:
            shutil.copytree(out, keep / "job", symlinks=True, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("frames_raw", "*.part", "train", "ckpts"))
            saved.update(done)
            vol.commit()
            print("[e2e] checkpointed", sorted(done), flush=True)
    def loop():
        while not stop.wait(60):
            try: mirror(); checkpoint()
            except Exception as e: print("mirror", e, flush=True)
    threading.Thread(target=loop, daemon=True).start()
    t0 = time.time()
    code = subprocess.run([sys.executable, "-m", "plypipe", "run", str(inp), "--out", str(out), *args], env=env).returncode
    stop.set()
    mirror(final=True)
    print(f"[e2e] exit {code} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return code

@app.local_entrypoint()
def main(step: str = "fetch", args: str = "--preset quick --iterations 7000", tag: str = "quick-e2e"):
    if step == "fetch":
        for r in fetch.map(IDS): print("[e2e]", r, flush=True)
    else:
        print("[e2e] run exit", run.remote(args.split(), tag), flush=True)
