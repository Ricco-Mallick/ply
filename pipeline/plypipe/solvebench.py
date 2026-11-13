# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Measuring one camera-solve variant on the chapel's saved matches (pipeline/modal/sfm_cpu.py), without Modal: what each
variant costs and may run, how a finished solve is scored, and the table of every variant on the Volume.

The estimates are the budget of a paid run, not predictions to trust: the incremental mapper took 58.8 min on 8 cores
and 32 GiB (docs/IMPROVEMENT.md, 2026-11-04, the same box as `sfm_cpu.py`; 93 min inside the GPU worker, which the run
log records as 103 min of the 119). The other rows are guesses from the plan's budget (`IMPROVEMENT-PLAN.md`, phase 3:
1g about $1 for both core counts, 1b about $0.60) and the work each variant skips. A run that outlives 1.5 times its
estimate is stopped, and that is itself the result: it was not at least 30% faster than the incremental mapper."""

from __future__ import annotations

import json
import math
from pathlib import Path

VARIANTS = ("incremental", "hierarchical", "keyframes")
MEMORY_GIB = {8: 32, 16: 64}          # the incremental solve fitted in 32 GiB on 8 cores; 16 cores run clusters side by side
PRICE_CORE_H, PRICE_GIB_H = 0.047, 0.008  # Modal CPU, as docs/IMPROVEMENT-PLAN.md records it
ESTIMATE_MIN = {("incremental", 8): 60, ("incremental", 16): 45, ("hierarchical", 8): 40, ("hierarchical", 16): 30,
                ("keyframes", 8): 45, ("keyframes", 16): 35}
STOP_LOSS = 1.5                       # the plan's rule for every paid run: the timeout is 1.5 times the estimate
GRACE_MIN = 10                        # align, occupancy, the trajectory error and the save, after the solve's own deadline
RESTORE = ("frames", "frames_index.json", ".done/frames.json", "colmap/database.db", "colmap/matching.json", "plan.json", "scene.json")
NEEDED = ("frames", "frames_index.json", ".done/frames.json", "colmap/database.db")
SOLVE_ROOT = "chapel/solve"
REFERENCE = "chapel/high/job/data/sparse/0"  # the aligned solve the twin chapel-high-2 was trained on, in metres
SOURCE = "chapel/high/job"
# The saved chapel job has the aligned solve and the frames but never kept colmap/database.db, so the features and matches are made
# once more (sfm_match.py) and every variant maps that one database. Matching the chapel's 1,236 frames took 10 min on an L40S
# (docs/IMPROVEMENT.md) and 14 to 15 min on an L4 in the live runs, plus the frames in and the database out: 20 min is the estimate.
MATCH_ESTIMATE_MIN = 20
MATCH_GPU_H = 0.80                    # an L4, as docs/IMPROVEMENT-PLAN.md records it; the box also has the worker's 8 cores and 32 GiB
MATCH_CORES = 8
MATCH_NEEDED = ("frames", "frames_index.json", ".done/frames.json")  # what the sfm stage reads besides the database it makes
SHIP_RULE = ("ships if: at least 30% faster than incremental on the same database and cores, registered frames within 0.5% of incremental's, "
             "ATE under 2 cm, then held-out PSNR within the noise floor on the fixed 7k-step train")


def hourly(cores: int) -> float:
    return cores * PRICE_CORE_H + MEMORY_GIB[cores] * PRICE_GIB_H


def estimate_min(variant: str, cores: int) -> int:
    if variant not in VARIANTS or cores not in MEMORY_GIB:
        raise ValueError(f"unknown variant/cores {variant!r}/{cores}: variants are {', '.join(VARIANTS)}, cores {', '.join(map(str, MEMORY_GIB))}")
    return ESTIMATE_MIN[(variant, cores)]


def solve_deadline_min(variant: str, cores: int) -> int:
    return math.ceil(STOP_LOSS * estimate_min(variant, cores))


def timeout_min(variant: str, cores: int) -> int:
    return solve_deadline_min(variant, cores) + GRACE_MIN


def cost_usd(minutes: float, cores: int) -> float:
    return round(minutes / 60 * hourly(cores), 2)


def match_hourly() -> float:
    return MATCH_GPU_H + hourly(MATCH_CORES)


def match_deadline_min() -> int:
    return math.ceil(STOP_LOSS * MATCH_ESTIMATE_MIN)


def match_cost_usd(minutes: float) -> float:
    return round(minutes / 60 * match_hourly(), 2)


def run_dir(variant: str, cores: int) -> str:
    return f"{SOLVE_ROOT}/{variant}-{cores}"


def describe(variant: str, cores: int) -> str:
    est = estimate_min(variant, cores)
    return (f"{variant} on {cores} cores / {MEMORY_GIB[cores]} GiB (${hourly(cores):.2f}/h): estimate {est} min = ${cost_usd(est, cores):.2f}; "
            f"the solve is stopped at {solve_deadline_min(variant, cores)} min (${cost_usd(solve_deadline_min(variant, cores), cores):.2f}), "
            f"function timeout {timeout_min(variant, cores)} min (${cost_usd(timeout_min(variant, cores), cores):.2f}); "
            f"model and result.json go to {run_dir(variant, cores)}/")


def describe_match() -> str:
    return (f"match on an L4 + {MATCH_CORES} cores / {MEMORY_GIB[MATCH_CORES]} GiB (${match_hourly():.2f}/h): estimate {MATCH_ESTIMATE_MIN} min = "
            f"${match_cost_usd(MATCH_ESTIMATE_MIN):.2f}; function timeout {match_deadline_min()} min (${match_cost_usd(match_deadline_min()):.2f}); "
            f"colmap/database.db goes to <src>/colmap/")


def plan() -> str:
    """Every paid step in launch order, one line each, then the rule the results are read by."""
    lines = [describe_match(), *(describe(v, c) for c in sorted(MEMORY_GIB) for v in VARIANTS)]
    return "\n".join(lines + [SHIP_RULE])


def model_metrics(rec, images: int) -> dict:
    """What a solve is scored on besides its trajectory: frames posed out of all, error, tracks (3D points)."""
    reg = int(rec.num_reg_images())
    return {"registered": reg, "images": images, "registeredRatio": reg / images if images else 0.0,
            "meanReprojErrorPx": float(rec.compute_mean_reprojection_error()), "tracks": int(rec.num_points3D()),
            "meanTrackLength": float(rec.compute_mean_track_length())}


def assemble(job: dict, report: dict | None, model: dict | None, ate_cm: dict | None, seconds: dict, cpu: str, error: str | None = None) -> dict:
    """The one JSON beside a solve. `seconds` holds wall time of the parts the function timed; the mapper's own parts are
    the stage report's `mapperSeconds`."""
    v, c = job["variant"], job["cores"]
    wall = seconds.get("wall", 0.0)
    return {"variant": v, "cores": c, "memoryGib": MEMORY_GIB[c], "cpu": cpu, "src": job["src"], "reference": job["reference"], "args": job["args"],
            "revision": job.get("revision"), "error": error,
            "seconds": {**seconds, "mapperParts": (report or {}).get("mapperSeconds")},
            "costUsd": cost_usd(wall / 60, c), "estimateMin": estimate_min(v, c), "stopLossMin": solve_deadline_min(v, c),
            "mapper": (report or {}).get("mapper"), "mapperFallback": (report or {}).get("mapperFallback"),
            "keyframes": (report or {}).get("keyframes"), "trajectoryJumps": (report or {}).get("trajectoryJumps"),
            "droppedViews": (report or {}).get("droppedViews"), **(model or {}), "ateCm": ate_cm}


def make_job(variant: str, cores: int, src: str, reference: str = "", extra_args: str = "", revision: dict | None = None) -> dict:
    estimate_min(variant, cores)
    return {"variant": variant, "cores": cores, "src": (src or "chapel/high/job").strip("/"), "reference": (reference or REFERENCE).strip("/"),
            "args": ["--mapper", variant, *extra_args.split()], "revision": revision}


def missing(have) -> list[str]:
    """What the saved job lacks (`have(rel)` says whether a path under it exists)."""
    return [p for p in NEEDED if not have(p)]


# the sfm options that decide what features and matches are made (job.json holds every option of the run, defaults included)
MATCH_OPTS = ("features", "matcher", "camera_model", "focal_35mm", "exhaustive_max")


def match_args(job_json: dict | None, matching: dict | None, extra: str = "") -> list[str]:
    """The plypipe arguments that make the saved job's database again: the options its run recorded (job.json, camera model and
    focal prior included, which change the cameras in the database), with the features and matcher the first matching really
    used (matching.json: `aliked` requested falls back to SIFT on a wheel without ONNX), then `--sfm-part match`, which stops
    the sfm stage after matching. `--split-min 1` because that stop applies only to captures with at least --split-min frames."""
    opts = dict((job_json or {}).get("opts") or {})
    done = matching or {}
    if done.get("matcher") == "lightglue":
        opts["features"], opts["matcher"] = "aliked-lg", None
    else:
        opts["features"] = done.get("features") if done.get("features") in ("aliked", "sift") else opts.get("features")
        opts["matcher"] = done.get("matcher") if done.get("matcher") in ("sequential", "exhaustive", "vocabtree") else opts.get("matcher")
    args = []
    for k in MATCH_OPTS:
        if opts.get(k) is not None:
            args += [f"--{k.replace('_', '-')}", str(opts[k])]
    if opts.get("no_loop_detection"):
        args.append("--no-loop-detection")
    return [*args, "--device", "cuda", "--only", "sfm", "--sfm-part", "match", "--split-min", "1", *extra.split()]


def match_preflight(have, force: bool) -> str | None:
    """Why the match step must not start, else None. `have(rel)` says whether a path under the source job exists."""
    absent = [p for p in MATCH_NEEDED if not have(p)]
    if absent:
        return (f"the source job lacks {', '.join(absent)}: copy the saved job's frames, frames_index.json and .done/frames.json to the "
                "Volume first (modal volume ls ply-bench <src>)")
    if have("colmap/database.db") and not force:
        return "colmap/database.db is already there: every variant must map one database, so it is not replaced without --force"
    return None


def volume_have(volume, src: str):
    """`have(rel)` for a path under `src` on a Modal Volume, asked from outside the container (before anything is launched)."""
    def have(rel: str) -> bool:
        parent, _, name = f"{src}/{rel}".rpartition("/")
        try:
            return any(Path(e.path).name == name for e in volume.listdir(parent))
        except Exception as e:  # a missing folder is an absent path; any other failure is for the container to decide
            if type(e).__name__ in ("FileNotFoundError", "NotFoundError"):
                return False
            raise
    return have


def database_counts(db: Path) -> dict:
    """What a matched database holds: images, keypoints, pairs matched and pairs geometrically verified."""
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        one = lambda q: int(con.execute(q).fetchone()[0] or 0)
        return {"images": one("select count(*) from images"), "keypoints": one("select sum(rows) from keypoints"),
                "pairsMatched": one("select count(*) from matches where rows > 0"),
                "pairsVerified": one("select count(*) from two_view_geometries where rows > 0")}
    finally:
        con.close()


def table(results: dict[str, dict]) -> str:
    """One row per variant found, fastest mapper first; the reference is whichever incremental row has the same cores."""
    def mins(r):
        s = r.get("seconds") or {}
        return (s.get("solve") or 0) / 60

    rows = [("variant", "cores", "solve min", "x incr.", "$", "registered", "error px", "tracks", "ATE rmse cm", "ATE med cm", "fallback")]
    base = {r["cores"]: mins(r) for r in results.values() if r.get("variant") == "incremental" and r.get("mapper")}
    for name, r in sorted(results.items(), key=lambda kv: (kv[1].get("cores", 0), mins(kv[1]))):
        a = r.get("ateCm") or {}
        fb = r.get("mapperFallback")
        ratio = f"{mins(r) / base[r['cores']]:.2f}" if base.get(r.get("cores")) else "-"
        rows.append((r.get("variant", name), str(r.get("cores", "?")), f"{mins(r):.1f}", ratio, f"{r.get('costUsd', 0):.2f}",
                     f"{r.get('registered', '-')}/{r.get('images', '-')}", f"{r['meanReprojErrorPx']:.3f}" if "meanReprojErrorPx" in r else "-",
                     str(r.get("tracks", "-")), f"{a['rmse']:.2f}" if a else "-", f"{a['median']:.2f}" if a else "-",
                     fb["used"] if fb else ("error" if r.get("error") else "none")))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows)


def collect(volume, out: Path) -> dict[str, dict]:
    """result.json of every variant under chapel/solve on the Volume, copied into out/<variant>-<cores>/result.json."""
    got: dict[str, dict] = {}
    try:
        entries = volume.listdir(SOLVE_ROOT)
    except Exception as e:  # nothing saved yet: modal raises FileNotFoundError or NotFoundError for the missing folder
        if type(e).__name__ not in ("FileNotFoundError", "NotFoundError"):
            raise
        return got
    for entry in entries:
        name = Path(entry.path).name
        try:
            data = b"".join(volume.read_file(f"{SOLVE_ROOT}/{name}/result.json"))
        except Exception:
            continue
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "result.json").write_bytes(data)
        got[name] = json.loads(data)
    return got
