# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""The pure half of pipeline/modal/ply_eval.py: train-only image-quality experiments that all start from one saved
camera solve (docs/IMPROVEMENT-PLAN.md, phase 1 and 4).

What lives here, so it can be tested without Modal: the table of variants, what a run restores from the saved job
folder and what it leaves out, resume decisions, saving after each stage, price and timeout estimates, and the
assembly of metrics.json from the job's report. `python -m plypipe.evalrun plan --variant q0a` prints what a run
would do without importing Modal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from plypipe.plan import LITE_FINETUNE_STEPS

SCHEMA = 1

# The plan's baseline: the saved High chapel run's settings (job.json: preset high, cap 2M from the plan, SH 3, no
# appearance model, antialiased, hold-out 30), 15k steps instead of 60k, plus the 20-frame hold-out block.
BASE = ["--preset", "high", "--iterations", "15000", "--cap-max", "2000000", "--holdout", "30", "--holdout-block", "20"]
# The small baseline (owner, 2026-11-11: try a variant on a small twin first and keep the credit for the final runs).
# Dr Johnson is 260 solved photos at 1332 px. The 600k cap binds as the chapel's 2M does (the plan's own budget for this
# set is 1.47M), every 8th view is held out as the 3DGS paper scores it (33 views), and the block is 10 photos in a row.
SMALL = ["--preset", "high", "--iterations", "15000", "--cap-max", "600000", "--holdout", "8", "--holdout-block", "10"]
# What each variant adds to its scenario's baseline ("steps" replaces the baseline's length); "args" is filled in below
VARIANTS: dict[str, dict] = {
    "smoke": {"add": [], "steps": 300,
              "note": "300 steps: proves restore, train, export, scoring and saving end to end", "timeout_min": 12},
    "q0a": {"add": [], "role": "baseline", "note": "baseline; with q0b it gives the noise floor"},
    "q0b": {"add": [], "role": "baseline", "note": "identical to q0a"},
    "q1": {"add": ["--bilateral-grid", "--exposure-fold", "2000"],
           "note": "exposure correction folded back: trained with the bilateral grid, the colour map to the photos folded into the "
                   "splats, then 2k steps without the grid"},
    "q1b": {"add": ["--exposure-affine", "--exposure-fold", "2000"],
            "note": "exposure correction the fold undoes exactly: trained with one affine colour transform per photo (the grid "
                    "at 1x1x1), folded into the splats, then 2k steps without it"},
    "q2": {"add": ["--pose-opt"], "note": "pose refinement"},
    "q4a": {"add": ["--depth-loss"], "note": "sparse depth loss at the SfM track points"},
    "q5a": {"add": ["--opacity-reg", "0.005"], "note": "opacity regulariser 0.005 (baseline 0.01)"},
    "q5b": {"add": ["--opacity-reg", "0.002"], "note": "opacity regulariser 0.002"},
    "q5c": {"add": ["--opacity-reg", "0.001"], "note": "opacity regulariser 0.001"},
    "q5d": {"add": ["--opacity-reg", "0"], "note": "no opacity regulariser"},
    # the phone file's fine-tune, since 2026-11-12 the export's default: named here so the runs on the Volume keep their
    # arguments. The small twin is under the phone size as trained: its phone file is cut to 240k, the chapel's ratio
    "q6": {"add": ["--lite-finetune", "3000"], "scenario_add": {"drjohnson": ["--lite-max", "240000"]},
           "note": "phone file trained on for 3k steps at its own size after it is cut down; scored against the cut-down one"},
}

# Modal list prices per hour, 2026-11. Only L4 and L40S match the CI wheels (sm_89).
GPU_PER_HOUR = {"H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95, "L4": 0.80}
CPU_PER_CORE_HOUR = 0.047
MEMORY_PER_GIB_HOUR = 0.008
CPU, MEMORY_GIB = 8, 32  # what the chapel's train and export ran on (the worker's own sizing)

# 1,717.9 s for 60,000 steps on an L40S (the saved High run's report). An L4 is about twice as slow (ply_bench's notes).
SECONDS_PER_STEP = {"L40S": 1717.9 / 60_000, "L4": 2 * 1717.9 / 60_000}
OVERHEAD_MIN = 5   # restore, export and scoring for a 2M-splat model
# Estimates, not measurements (no bilateral-grid run has been timed), set on the careful side so the stop-loss does not
# end a first q1 run: a step with the grid also slices one grid per photo and regularises all of them, and the exposure
# fold renders its views twice, rewrites the PLY and starts the trainer a second time.
GRID_SLOWDOWN = 1.5
# --exposure-affine's 1x1x1 grid runs on the pure-torch lib_bilagrid: q1b on Dr Johnson (L40S, 2026-11-12) took 464 s for
# 15,000 steps against 159 s plain, x2.9, where the fused 16x16x8 grid of q1 took 255 s (x1.6)
AFFINE_SLOWDOWN = 2.9
FOLD_MIN = 2
STOP_LOSS = 1.5    # IMPROVEMENT-PLAN rule 3: the timeout is 1.5 times the estimate
GRACE_MIN = 2      # the Modal function outlives the in-code deadline by this much, to save what a stopped run has

# What a scenario fixes: the saved solve on the Volume, the baseline, the container, and what its estimate is scaled from
SCENARIOS: dict[str, dict] = {
    "chapel": {"src": "chapel/high/job", "base": BASE, "cpu": CPU, "memory_gib": MEMORY_GIB,
               "seconds_per_step": SECONDS_PER_STEP, "overhead_min": OVERHEAD_MIN},
    # 6.9 min for 30,000 steps at a 733k cap on an L40S, trained and exported on 4 cores and 16 GiB (docs/BENCHMARK.md,
    # ply_bench). A photo set saved before frames_index.json existed: its block is taken in name order
    "drjohnson": {"src": "drjohnson/prep", "base": SMALL, "cpu": 4, "memory_gib": 16, "index": False,
                  "seconds_per_step": {"L40S": 6.9 * 60 / 30_000, "L4": 2 * 6.9 * 60 / 30_000}, "overhead_min": 4},
}


def variant_args(name: str, scenario: str = "chapel") -> list[str]:
    """A table variant's arguments on a scenario's baseline."""
    v, base = VARIANTS[name], list(SCENARIOS[scenario]["base"])
    if v.get("steps"):
        base[base.index("--iterations") + 1] = str(v["steps"])
    return base + v["add"] + v.get("scenario_add", {}).get(scenario, [])


for _name, _variant in VARIANTS.items():
    _variant["args"] = variant_args(_name)  # on the chapel

# What a run takes from the saved job folder: the camera solve and the files train and export read. Everything else
# (frames, colmap, the old splat, SPZ files, logs, training state) is left out; data/images is a link to dense/images.
RESTORE_FILES = ("report.json", "plan.json", "scene.json", "shell.json", "places.json", "occupancy.json", "occupancy.png",
                 "frames_index.json")
RESTORE_STAGES = ("frames", "sfm", "align", "occupancy", "label")  # .done markers; train, export and mesh are redone
RESTORE_TREES = ("data/sparse", "dense/images")
IMAGES_LINK = ("data/images", "dense/images")                       # `modal volume put` drops symlinks: made again here
STALE_REPORT_SECTIONS = ("train", "export", "mesh", "quality")      # what the old run's stages wrote into report.json

# What is kept on the Volume after each stage: everything the job made except the shared inputs and re-creatable bulk.
SAVE_SKIP_DIRS = ("data", "dense", "frames", "frames_raw", "colmap", "train", "train-lite", "train-snapshot")
SAVE_SKIP_FILES = ("scene_clean.ply", "scene_lite.ply")

# An export-only run (--export-from) takes the model another variant trained: what its train stage left in its folder.
# eval/lite is not the train stage's: the export scores the phone file there
TRAINED_FILES = ("scene.ply", ".done/train.json")
TRAINED_TREE, EXPORTED_TREE = "eval", "eval/lite"
# Its own options must be ones the export stage reads and training does not (tests/test_evalrun.py checks both stages'
# source): an option that changes training would be recorded against a model trained without it
EXPORT_ONLY_OPTIONS = ("--dense-occupancy", "--no-dense-occupancy", "--lite-finetune", "--lite-max", "--min-opacity", "--no-crop")

INPUT = "saved-solve"  # the CLI wants an input path; train and export never read it
NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,39}$")


# ------------------------------------------------------------------ variants

def resolve(variant: str, extra: str = "", scenario: str = "chapel") -> dict:
    """The named variant's arguments on the scenario's baseline, or for a new name that baseline plus `extra` (a later option wins)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}' (available: {', '.join(SCENARIOS)})")
    if not NAME.match(variant or ""):
        raise ValueError(f"variant name '{variant}' must be lower case letters, digits, '_', '.' or '-' (available: {', '.join(VARIANTS)})")
    extra_args = shlex.split(extra or "")
    if variant in VARIANTS:
        if extra_args:
            raise ValueError(f"'{variant}' is a table variant: its arguments are fixed. Pick a new name for an ad-hoc run")
        v = VARIANTS[variant]
        return {"name": variant, "args": variant_args(variant, scenario), "source": "table", "role": v.get("role"), "note": v.get("note"),
                "timeoutMin": v.get("timeout_min")}
    if not extra_args:
        raise ValueError(f"unknown variant '{variant}' (available: {', '.join(VARIANTS)}); for an ad-hoc run pass --args")
    return {"name": variant, "args": SCENARIOS[scenario]["base"] + extra_args, "source": "--args", "role": None, "note": "baseline arguments plus --args",
            "timeoutMin": None}


def resolve_export(variant: str, origin: str, extra: str = "", scenario: str = "chapel") -> dict:
    """An export-only run: the model the table variant `origin` trained, exported again under a new name with `extra`,
    which may hold export options only. A table variant, so its training arguments are known here and can be checked
    against the ones its folder on the Volume was trained with."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}' (available: {', '.join(SCENARIOS)})")
    if origin not in VARIANTS:
        raise ValueError(f"--export-from {origin}: not a table variant (available: {', '.join(VARIANTS)})")
    if not NAME.match(variant or "") or variant in VARIANTS:
        raise ValueError(f"an export-only run needs a new variant name, not '{variant}' (e.g. {origin}-reexport): "
                         f"its folder holds the new export beside {origin}'s")
    extra_args = shlex.split(extra or "")
    foreign = [a for a in extra_args if a.startswith("-") and a.split("=")[0] not in EXPORT_ONLY_OPTIONS]
    if foreign:
        raise ValueError(f"--export-from runs export alone: {', '.join(foreign)} would not reach the model "
                         f"(export options: {', '.join(EXPORT_ONLY_OPTIONS)})")
    train = variant_args(origin, scenario)
    return {"name": variant, "args": train + extra_args, "source": f"--export-from {origin}", "role": None,
            "note": f"export only, of the model {origin} trained" + (f", with {' '.join(extra_args)}" if extra_args else ""),
            "timeoutMin": None, "exportFrom": origin, "trainArgs": train}


def _int_option(args: list[str], flag: str) -> int | None:
    found = [args[i + 1] for i, a in enumerate(args[:-1]) if a == flag]
    return int(found[-1]) if found else None


def steps_of(args: list[str]) -> int | None:
    return _int_option(args, "--iterations")


# ------------------------------------------------------------------ price, estimate, stop-loss

def gpu_key(name: str | None) -> str | None:
    """Price-table key for an nvidia-smi name ('NVIDIA L40S', 'NVIDIA A100-SXM4-80GB'); L4 is not a prefix of L40S."""
    tokens = set(re.split(r"[^A-Za-z0-9]+", (name or "").upper()))
    for key in ("H100", "L40S", "L4"):
        if key in tokens:
            return key
    return "A100-80GB" if "A100" in tokens and "80GB" in tokens else None


def per_hour(gpu: str | None, cpu: float = CPU, memory_gib: float = MEMORY_GIB) -> float | None:
    key = gpu_key(gpu) or gpu
    return round(GPU_PER_HOUR[key] + cpu * CPU_PER_CORE_HOUR + memory_gib * MEMORY_PER_GIB_HOUR, 4) if key in GPU_PER_HOUR else None


def estimate(args: list[str], gpu: str = "L40S", timeout_min: int | None = None, scenario: str = "chapel",
             export_only: bool = False) -> dict:
    """Minutes and dollars for one run on `gpu`, and the stop-loss timeout (1.5x the estimate unless the variant sets one).
    `export_only`: the model is another run's, so only the export's own training (the phone file) counts."""
    steps, s = steps_of(args), SCENARIOS[scenario]
    if steps is None or gpu_key(gpu) not in s["seconds_per_step"]:
        raise ValueError(f"cannot estimate '{' '.join(args)}' on {gpu}: needs --iterations and an L4 or L40S")
    # the export trains the phone file on (the pipeline's default length unless --lite-finetune says otherwise). It
    # counts at the full model's pace: fewer splats train faster, and the difference pays for starting the trainer a
    # second time and scoring both phone models
    tune = _int_option(args, "--lite-finetune")
    tune = LITE_FINETUNE_STEPS if tune is None else tune
    # --exposure-fold trains the full model on, with no appearance model: its steps count at the plain pace
    affine = "--exposure-affine" in args and "--bilateral-grid" not in args
    grid, fold = "--bilateral-grid" in args or affine, _int_option(args, "--exposure-fold")
    if export_only:
        steps, grid, fold = 0, False, None
    slowdown = AFFINE_SLOWDOWN if affine else GRID_SLOWDOWN
    paced = steps * (slowdown if grid else 1) + max(fold or 0, 0) + tune
    train_min = math.ceil(paced * s["seconds_per_step"][gpu_key(gpu)] / 60)
    # the export's depth pass for the walking grid (on by default) took 4 s on Dr Johnson and 25 s on the chapel (L40S,
    # 2026-11-12): inside the overhead
    overhead = s["overhead_min"] + (FOLD_MIN if fold is not None else 0)
    total_min = train_min + overhead
    timeout = timeout_min or math.ceil(STOP_LOSS * total_min)
    rate = per_hour(gpu, s["cpu"], s["memory_gib"])
    return {"gpu": gpu, "steps": steps, **({"gridSlowdown": slowdown} if grid else {}), **({"exposureFoldSteps": max(fold, 0)} if fold is not None else {}),
            **({"fineTuneSteps": tune} if tune else {}), **({"exportOnly": True} if export_only else {}),
            "trainMin": train_min, "overheadMin": overhead,
            "totalMin": total_min,
            "timeoutMin": timeout, "usd": round(total_min / 60 * rate, 2), "maxUsd": round(timeout / 60 * rate, 2), "perHour": rate}


def make_job(scenario: str, src: str, variant: str, args: str = "", force: bool = False, gpus: tuple[str, ...] = ("L40S",),
             ceiling_min: int | None = None, revision: dict | None = None, export_from: str = "") -> dict:
    """Everything the Modal function needs, decided locally. Refuses a run whose stop-loss would not fit the function's timeout."""
    v = resolve_export(variant, export_from, args, scenario) if export_from else resolve(variant, args, scenario)
    s = SCENARIOS[scenario]
    estimates = {gpu_key(g): estimate(v["args"], g, v["timeoutMin"], scenario, export_only=bool(export_from)) for g in gpus}
    ceiling = ceiling_min or default_ceiling_min(gpus, scenario)
    slowest = max(estimates.values(), key=lambda e: e["timeoutMin"])
    if slowest["timeoutMin"] > ceiling:
        raise ValueError(f"{variant}: stop-loss {slowest['timeoutMin']} min on {slowest['gpu']} is above the function timeout of "
                         f"{ceiling} min (set PLY_EVAL_TIMEOUT_MIN when launching to raise it)")
    return {"scenario": scenario, "src": (src or s["src"]).strip("/"), "variant": variant, "args": v["args"], "argsSource": v["source"],
            "role": v["role"], "note": v["note"], "force": force, "gpus": list(gpus), "ceilingMin": ceiling,
            "cpu": s["cpu"], "memoryGib": s["memory_gib"], "needsIndex": s.get("index", True),
            "estimates": estimates, "revision": revision or {},
            **({"exportFrom": v["exportFrom"], "trainArgs": v["trainArgs"]} if export_from else {})}


def stop_loss_min(job: dict, gpu: dict) -> int:
    """The in-code deadline for the GPU the run actually got (the slowest allowed one when it is not in the price table)."""
    return (job["estimates"].get(gpu.get("priced")) or max(job["estimates"].values(), key=lambda e: e["timeoutMin"]))["timeoutMin"]


def default_ceiling_min(gpus: tuple[str, ...], scenario: str = "chapel") -> int:
    """The function's timeout: the stop-loss of the scenario's longest table variant on the slowest GPU allowed."""
    return max(estimate(variant_args(n, scenario), g, v.get("timeout_min"), scenario)["timeoutMin"] for n, v in VARIANTS.items() for g in gpus)


def plypipe_command(stage: str, job_dir: str, args: list[str], python: str = "python") -> list[str]:
    return [python, "-m", "plypipe", "run", INPUT, "--out", job_dir, "--only", stage, *args]


def variant_dir(scenario: str, variant: str) -> str:
    return f"{scenario}/eval/{variant}"


def describe(job: dict, job_dir: str = "/tmp/work/job") -> str:
    out = [f"variant {job['variant']} ({job['argsSource']}{', ' + job['note'] if job.get('note') else ''})",
           f"scenario {job['scenario']}, saved solve /vol/{job['src']}, results /vol/{variant_dir(job['scenario'], job['variant'])}/",
           f"{job.get('cpu', CPU)} cores, {job.get('memoryGib', MEMORY_GIB)} GiB; GPU {' then '.join(job['gpus'])}" + (" (a fallback: timings differ)" if len(job["gpus"]) > 1 else "")]
    origin = job.get("exportFrom")
    for e in job["estimates"].values():
        trained = (f"no training (the model {origin} trained)" if e.get("exportOnly") else
                   f"{e['steps']} steps{' at x' + str(e['gridSlowdown']) + ' for the bilateral grid (an estimate)' if e.get('gridSlowdown') else ''}"
                   f"{' + ' + str(e['exposureFoldSteps']) + ' after the exposure fold' if 'exposureFoldSteps' in e else ''}")
        out.append(f"  {e['gpu']}: {trained}"
                   f"{' + ' + str(e['fineTuneSteps']) + ' for the phone file' if e.get('fineTuneSteps') else ''}"
                   f" = {e['trainMin']} min training + {e['overheadMin']} min restore/export/scoring{' and the fold' if 'exposureFoldSteps' in e else ''} = "
                   f"{e['totalMin']} min, about ${e['usd']:.2f} at ${e['perHour']:.2f}/h; stop-loss {e['timeoutMin']} min, at most ${e['maxUsd']:.2f}")
    out += [f"the stop-loss is 1.5x the estimate unless the table sets one; the Modal function times out {GRACE_MIN} min later "
            f"({job['ceilingMin'] + GRACE_MIN} min), to save what a stopped run has",
            "restore from the saved folder: " + ", ".join([*RESTORE_FILES, *(f".done/{s}.json" for s in RESTORE_STAGES), *(f"{t}/" for t in RESTORE_TREES)]),
            f"  then {IMAGES_LINK[0]} -> {os.path.relpath(IMAGES_LINK[1], os.path.dirname(IMAGES_LINK[0]))} is made again (volume put drops links), "
            f"and report.json loses its {', '.join(STALE_REPORT_SECTIONS)} sections",
            "left out: frames/, frames_raw/, colmap/, the rest of dense/, the old scene*.ply and *.spz, eval/, logs/, log.jsonl, job.json, "
            ".done/{train,export,mesh}.json",
            f"a relaunch layers /vol/{variant_dir(job['scenario'], job['variant'])}/ over that and runs only the stages not finished"
            f"{' (--force ignores it)' if job['force'] else ''}"]
    if origin:
        out.append(f"export only: before anything is restored, /vol/{variant_dir(job['scenario'], origin)}/ must hold a finished training "
                   f"with {origin}'s arguments, or the run stops there; then its {', '.join(TRAINED_FILES)}, {TRAINED_TREE}/ "
                   f"(not {EXPORTED_TREE}/) and report.json's train section go into the job, nothing its export made")
    out.append("commands:")
    out += ["  " + " ".join(plypipe_command(s, job_dir, job["args"])) for s in (("export",) if origin else ("train", "export"))]
    out += [f"each stage is saved (report and markers last) without {', '.join(SAVE_SKIP_DIRS)}/, {', '.join(SAVE_SKIP_FILES)}; "
            "the newest training snapshot is kept every 3 min",
            "then verify.check gives the quality gate, and metrics.json is written and saved the same way"]
    if job.get("revision"):
        out.append(f"code {job['revision'].get('revision')} (hash {job['revision'].get('codeHash')})")
    return "\n".join(out)


# ------------------------------------------------------------------ restore, resume, save

def wanted(rel: str) -> bool:
    """Is this path (relative to the saved job folder, '/' separated) restored?"""
    parts = rel.split("/")
    if rel in RESTORE_FILES:
        return True
    if parts[0] == ".done":
        return len(parts) == 2 and parts[1].endswith(".json") and parts[1][:-5] in RESTORE_STAGES
    return any(rel == t or rel.startswith(t + "/") for t in RESTORE_TREES)


def _on_the_way(rel_dir: str) -> bool:
    """Can a wanted file sit in this folder or below it?"""
    return any(r == rel_dir or r.startswith(rel_dir + "/") or rel_dir.startswith(r + "/") for r in (".done", *RESTORE_TREES))


def prune_report(report: dict) -> dict:
    return {k: v for k, v in report.items() if k not in STALE_REPORT_SECTIONS}


def restore(src: Path, dst: Path) -> dict:
    """Copy the files a train-only run needs from the saved job folder, and make the images link the align stage makes."""
    files = size = 0
    for root, dirs, names in os.walk(src):
        rel_root = Path(root).relative_to(src)
        # folders with nothing wanted in them (frames/, dense/stereo/ ...) are not listed: each listing is a Volume round trip
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink() and _on_the_way((rel_root / d).as_posix())]
        for name in names:
            p = Path(root) / name
            rel = (rel_root / name).as_posix()
            if p.is_symlink() or not wanted(rel):
                continue
            q = dst / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            if rel == "report.json":
                q.write_text(json.dumps(prune_report(json.loads(p.read_text())), indent=2))
            else:
                shutil.copy2(p, q)
            files += 1
            size += p.stat().st_size
    link = dst / IMAGES_LINK[0]
    target = dst / IMAGES_LINK[1]
    if target.is_dir():
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink() if link.is_symlink() else shutil.rmtree(link)
        os.symlink(os.path.relpath(target, link.parent), link)
    elif (src / IMAGES_LINK[0]).is_dir() and not (src / IMAGES_LINK[0]).is_symlink():  # the upload followed the link
        shutil.copytree(src / IMAGES_LINK[0], link)
    return {"files": files, "bytes": size}


def check_job(root: Path, index: bool = True) -> list[str]:
    """What is missing for train and export to run on this folder (empty when it is usable). `index` false: a photo
    set, which has no frames_index.json."""
    problems = [f"data/sparse/0/{n}" for n in ("cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt")
                if not (root / "data" / "sparse" / "0" / n).exists()]
    images = root / IMAGES_LINK[0]
    if not images.is_dir() or not next(images.glob("*.jpg"), None):
        problems.append(f"{IMAGES_LINK[0]} (no photos: dense/images missing or the link broken)")
    for n in ("scene.json", "frames_index.json") if index else ("scene.json",):
        if not (root / n).exists():
            problems.append(n)
    return problems


def finished(job: Path) -> dict[str, bool]:
    done = job / ".done"
    return {"train": (done / "train.json").exists() and (job / "scene.ply").exists(),
            "export": (done / "export.json").exists() and (job / "scene.spz").exists() and (job / "scene_lite.spz").exists()}


def stages_to_run(job: Path) -> list[str]:
    """Which of train and export still have to run in a job folder restored with the saved variant folder layered on it."""
    state = finished(job)
    if state["train"] and state["export"]:
        return []
    return ["export"] if state["train"] else ["train", "export"]


def layer(saved: Path, job: Path) -> int:
    """Put a variant folder's saved files over the job (a relaunch). Snapshots are not state to resume from."""
    count = 0
    for p in sorted(saved.rglob("*")):
        rel = p.relative_to(saved)
        if p.is_file() and not p.is_symlink() and rel.parts[0] != "train-snapshot" and p.suffix != ".part":
            (job / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, job / rel)
            count += 1
    return count


def _trained(rel: str) -> bool:
    return rel in TRAINED_FILES or (rel.startswith(TRAINED_TREE + "/") and not rel.startswith(EXPORTED_TREE + "/"))


def check_training(origin: Path, train_args: list[str]) -> list[str]:
    """Why another variant's folder cannot give an export-only run its model (empty when it can): the train stage
    finished there, and with the arguments the run will record. A launch once ran a full training when the copy of a
    model it should have exported had failed (docs/MISTAKES.md, 2026-11-12)."""
    problems = [rel for rel in TRAINED_FILES if not (origin / rel).is_file()]
    try:
        train = json.loads((origin / "report.json").read_text()).get("train")
    except (OSError, ValueError):
        train = None
    if not train:
        problems.append("report.json with a train section")
    m = read_metrics(origin / "metrics.json")
    if m is None:
        problems.append("metrics.json (which arguments trained the model)")
    elif m.get("args") != train_args:
        problems.append(f"a model trained with other arguments ({' '.join(m.get('args') or [])})")
    return problems


def take_training(origin: Path, job: Path) -> int:
    """Put the model another variant trained into a restored job: what its train stage left (the PLY, its marker, the
    hold-out scores and the report's train section), nothing its export made. Returns the files copied."""
    count = 0
    for p in sorted(origin.rglob("*")):
        rel = p.relative_to(origin).as_posix()
        if p.is_file() and not p.is_symlink() and _trained(rel) and p.suffix != ".part":
            (job / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, job / rel)
            count += 1
    report = job / "report.json"
    merged = json.loads(report.read_text()) if report.exists() else {}
    merged["train"] = json.loads((origin / "report.json").read_text())["train"]
    report.write_text(json.dumps(merged, indent=2))
    return count


def clear_stale(job: Path, stages: list[str]) -> None:
    """Forget a stage that has to run again, and everything after it, so a half-saved folder cannot pass for a finished one."""
    for s in ("train", "export"):
        if s in stages or (s == "export" and "train" in stages):
            (job / ".done" / f"{s}.json").unlink(missing_ok=True)
    if "train" in stages:
        (job / "scene.ply").unlink(missing_ok=True)


def savable(rel: Path) -> bool:
    return rel.parts[0] not in SAVE_SKIP_DIRS and rel.name not in SAVE_SKIP_FILES and not rel.name.startswith("_")


def save(job: Path, dst: Path) -> dict:
    """Mirror the job onto the variant folder: changed files first, report.json then the .done markers last, so a
    marker never outlives the files it vouches for."""
    files = [p for p in sorted(job.rglob("*")) if p.is_file() and not p.is_symlink() and savable(p.relative_to(job))]
    late = lambda p: (p.relative_to(job).parts[0] == ".done", p.name == "report.json")
    count = size = 0
    for p in sorted(files, key=late):
        q = dst / p.relative_to(job)
        st = p.stat()
        if q.exists() and q.stat().st_size == st.st_size and q.stat().st_mtime >= st.st_mtime:
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        tmp = q.with_name(q.name + ".part")
        shutil.copy2(p, tmp)
        tmp.replace(q)
        count += 1
        size += st.st_size
    return {"files": count, "bytes": size}


# ------------------------------------------------------------------ metrics

SCORES = ("psnr", "ssim", "lpips", "ccPsnr", "ccSsim", "ccLpips", "views")


def parse_gpu(text: str) -> dict:
    """`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`, first line."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    name, _, mem = (x.strip() for x in line.partition(","))
    digits = re.sub(r"[^0-9]", "", mem)
    return {"name": name or None, "memoryMiB": int(digits) if digits else None, "priced": gpu_key(name)}


def _pick(d: dict | None, keys=SCORES) -> dict | None:
    return {k: d[k] for k in keys if isinstance(d, dict) and d.get(k) is not None} or None


def assemble(job: dict, report: dict, *, gpu: dict, seconds: dict, sizes: dict, invocation: dict, previous: dict | None = None) -> dict:
    """metrics.json from the job's report.json. A run that stopped early still gets one, with `complete` false and what is missing."""
    train, export = report.get("train") or {}, report.get("export") or {}
    shipped = export.get("shippedEval") or {}
    every = _pick(train.get("eval"))
    block = _pick(train.get("evalBlock"))
    if block is not None:
        block.update({"images": train.get("blockImages"), "clip": train.get("blockClip"), "seconds": train.get("blockSeconds")})
    missing = [name for name, value in (("train", train), ("scores.every", every), ("scores.block", block), ("export", export),
                                       ("viewer.desktop", shipped.get("full")), ("viewer.phone", shipped.get("lite"))) if not value]
    invocations = [*((previous or {}).get("invocations") or []), invocation]
    cost = round(sum(i.get("costUsd") or 0 for i in invocations), 2)
    quality = report.get("quality") or {}
    return {
        "schema": SCHEMA, "variant": job["variant"], "role": job.get("role"), "scenario": job["scenario"], "src": job["src"],
        "args": job["args"], "argsSource": job["argsSource"], "note": job.get("note"), "revision": job.get("revision") or {},
        **({"exportFrom": job["exportFrom"]} if job.get("exportFrom") else {}),
        "gpu": gpu, "steps": train.get("maxSteps") or steps_of(job["args"]), "capMax": train.get("capMax"),
        "complete": not missing, "missing": missing,
        "scores": {"every": every, "block": block},
        "viewer": {"desktop": shipped.get("full"), "phone": shipped.get("lite"), "cleaned": export.get("viewerEval"),
                   **({"phoneFinetune": export["liteFinetune"]} if export.get("liteFinetune") else {})},
        **({"exposureFold": train["exposureFold"]} if train.get("exposureFold") else {}),
        **({"denseOccupancy": export["denseOccupancy"]} if export.get("denseOccupancy") else {}),
        "splats": {"atEval": (train.get("eval") or {}).get("gaussians"), "trained": export.get("gaussiansIn"),
                   "shipped": export.get("gaussiansOut"), "keptFraction": export.get("keptFraction"), "phone": export.get("liteGaussians")},
        "files": sizes, "seconds": {**seconds, "train": train.get("seconds"), "export": export.get("seconds")},
        "gate": {"ok": quality.get("ok"), "issues": [{"level": i.get("level"), "code": i.get("code")} for i in quality.get("issues") or []]} if quality else None,
        "invocations": invocations, "costUsd": cost,
        "wallMinutes": round(sum(i.get("wallSeconds") or 0 for i in invocations) / 60, 1),
    }


def invocation_record(started: float, wall: float, gpu: dict, ran: list[str], skipped: list[str], error: str | None = None,
                      cpu: float = CPU, memory_gib: float = MEMORY_GIB) -> dict:
    rate = per_hour(gpu.get("priced") or gpu.get("name"), cpu, memory_gib)
    return {"startedAt": datetime.fromtimestamp(started, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "wallSeconds": round(wall, 1),
            "gpu": gpu.get("name"), "ran": ran, "resumedAfter": skipped, "costUsd": round(wall / 3600 * rate, 3) if rate else None,
            "pricePerHour": rate, **({"error": error} if error else {})}


def read_metrics(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == SCHEMA else None


def check_resume(previous: dict | None, job: dict) -> None:
    """A variant folder belongs to one set of arguments: a relaunch with others would mix two experiments."""
    if previous and previous.get("args") != job["args"]:
        raise ValueError(f"{job['variant']} already holds a run with other arguments ({' '.join(previous.get('args') or [])}); "
                         "pick a new variant name or relaunch with --force")


def dense_cell(d: dict) -> str:
    """--dense-occupancy's outcome in a line: which grid the twin got, and both grids' walk check."""
    def walk(w: dict | None) -> str:
        w = w or {}
        cams = w.get("cameras")
        return f"{'–' if cams is None else f'{cams:.0%}'} of capture positions walkable, {w.get('standableM2')} m2 to stand on"
    if d.get("used") == "depth":
        head = f"from depth ({d.get('views')} views)"
    else:
        head = f"sparse kept: {d.get('why') or 'no reason given'}"
    tail = f"; depth {walk(d['walk'])}, sparse {walk(d.get('sparseWalk'))}" if d.get("walk") and d.get("used") == "depth" else ""
    return head + tail + (f"; {d['seconds']} s" if d.get("seconds") is not None else "")


def summary(m: dict) -> str:
    def row(s: dict | None) -> str:
        return "n/a" if not s else f"cc-PSNR {s.get('ccPsnr')} cc-SSIM {s.get('ccSsim')} cc-LPIPS {s.get('ccLpips')} ({s.get('views')} views)"
    viewer = m.get("viewer") or {}
    tune = viewer.get("phoneFinetune") or {}
    fold = m.get("exposureFold") or {}
    folded = (f"gains {'/'.join(str(g) for g in fold.get('gains') or [])} fitted on {fold.get('views')} training views, brightness there "
              f"x{fold.get('brightnessBefore')} -> x{fold.get('brightnessAfter')} of the photos, "
              + (f"then {fold.get('steps')} steps without the appearance model" if fold.get("settled") else "no steps after it")) if fold.get("folded") else "NOT folded"
    return "\n".join([f"{m['variant']}: {'complete' if m['complete'] else 'INCOMPLETE, missing ' + ', '.join(m['missing'])}"
                      + (f" (export only: the hold-out scores are those of the model {m['exportFrom']} trained)" if m.get("exportFrom") else ""),
                      f"  every-Nth hold-out: {row(m['scores']['every'])}", f"  block hold-out:     {row(m['scores']['block'])}",
                      f"  viewer PSNR desktop {(viewer.get('desktop') or {}).get('psnr')}, phone {(viewer.get('phone') or {}).get('psnr')}; "
                      f"splats shipped {m['splats']['shipped']} of {m['splats']['trained']} (kept {m['splats']['keptFraction']})",
                      *([f"  phone file, {tune.get('steps')} fine-tune steps: PSNR {(tune.get('before') or {}).get('psnr')} before, "
                         f"{(tune.get('after') or {}).get('psnr')} fine-tuned; shipped the {tune.get('shipped')} one"
                         + (f" ({tune['why']})" if tune.get("why") else "")] if tune else []),
                      *([f"  exposure fold: {folded}" + (f" ({fold['why']})" if fold.get("why") else "")
                         + f"; viewer brightness desktop x{(viewer.get('desktop') or {}).get('brightness')}"] if fold else []),
                      *([f"  walking grid: {dense_cell(m['denseOccupancy'])}"] if m.get("denseOccupancy") else []),
                      f"  {m['wallMinutes']} min wall, about ${m['costUsd']}"])


def collect(volume, scenario: str, out: Path) -> dict[str, list[str]]:
    """Copy each variant's small result files (metrics.json, per-view scores, report) from the Volume into out/<variant>/.
    Returns what was fetched per variant; a variant without metrics.json has not finished."""
    got: dict[str, list[str]] = {}
    for entry in volume.listdir(f"{scenario}/eval"):
        name = Path(entry.path).name
        got[name] = []
        for rel in ("metrics.json", "eval/per_view.json", "report.json"):
            try:
                data = b"".join(volume.read_file(f"{scenario}/eval/{name}/{rel}"))
            except Exception:
                continue
            target = out / name / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            got[name].append(rel)
    return got


def code_revision(repo: Path) -> dict:
    """The commit (+dirty when the working tree differs) and a hash of the code that is shipped to the container."""
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=60).stdout.strip()
    sha, dirty = git("rev-parse", "--short=10", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    digest = hashlib.sha256()
    for p in sorted([*(repo / "pipeline" / "plypipe").rglob("*.py"), repo / "pipeline" / "modal" / "ply_eval.py"]):
        if p.exists():
            digest.update(p.relative_to(repo).as_posix().encode() + p.read_bytes())
    return {"revision": (sha + ("+dirty" if dirty else "")) if sha else "unknown", "codeHash": digest.hexdigest()[:12]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print what a ply_eval run would do. Nothing here contacts Modal.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--variant", required=True)
    p.add_argument("--args", default="")
    p.add_argument("--export-from", default="", metavar="VARIANT", help="export again the model this table variant trained; --args: export options only")
    # PLY_EVAL_SCENARIO is what sizes the container at launch (ply_eval.py): the plan follows it, as it follows PLY_EVAL_GPU
    p.add_argument("--scenario", default=os.environ.get("PLY_EVAL_SCENARIO", "chapel"), choices=sorted(SCENARIOS))
    p.add_argument("--src", default="", help="the saved job on the Volume (default: the scenario's)")
    p.add_argument("--gpu", default=os.environ.get("PLY_EVAL_GPU", "L40S"), help="comma-separated, tried in order")
    p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    gpus = tuple(g.strip() for g in a.gpu.split(",") if g.strip())
    repo = Path(__file__).resolve().parents[2]
    ceiling = int(os.environ["PLY_EVAL_TIMEOUT_MIN"]) if os.environ.get("PLY_EVAL_TIMEOUT_MIN") else None
    try:
        job = make_job(a.scenario, a.src, a.variant, a.args, a.force, gpus, ceiling, code_revision(repo), export_from=a.export_from)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(describe(job))
    return 0


if __name__ == "__main__":
    sys.exit(main())
