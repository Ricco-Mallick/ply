# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""device: what this machine can hold, as one more input to the plan.

plan.py sizes a run from the capture (surface area, views, pixels). Those numbers were tuned on L4 / L40S / T4
cards with 16-48 GB. A borrowed laptop or a college PC with a 4 GB GTX 1650 would run the same plan out of memory
in the first thousand steps. This module measures the GPU and RAM once and returns ceilings the plan takes the
minimum against, with the reason written into plan.json["device"]["why"]:

  * gaussian cap     from free VRAM, at an estimated cost per splat (calibrated by the first real run: the trainer
                     logs torch.cuda.max_memory_allocated, report.train.peakVramGB)
  * working size     only on cards under 3.5 GB, where the image buffers start to matter
  * packed mode      gsplat's packed rasterisation (less memory, slightly slower) under 12 GB
  * loader workers   from RAM: each DataLoader worker is its own process
  * SAM 3            people masks and labels need ~3.4 GB of fp32 weights on pre-Ampere cards; off where they
                     would not fit next to nothing else

On a big card none of these bind, so a Modal / Kaggle run plans exactly as before. Frames are never capped here:
fewer views is the wrong way to save memory (see docs/PROCESSING.md).

Overrides (testing, or a machine whose probe is wrong): PLY_DEVICE_VRAM_GB, PLY_DEVICE_RAM_GB, PLY_DEVICE_SM,
PLY_DEVICE_GPU. `--device-profile off` on `plypipe run` ignores the device entirely.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# bytes of VRAM one trained splat costs at SH degree 3: 59 floats x (param + grad + two Adam moments), MCMC state,
# projection buffers and tile intersections. An estimate on the safe side until a 4 GB run measures it
BYTES_PER_GAUSSIAN = 2400
# the CUDA context, torch's allocator slack, the current view's buffers and the scorer (LPIPS) at the last step
RESERVE_GB = 1.2
SMALL_CARD_GB = 3.5       # below this the 1280 px floor is lowered too
SMALL_CARD_SIDE = 1024
PACKED_UNDER_GB = 12.0
# SAM 3 (people masks, labels): ~848M parameters, fp32 on cards before Ampere (plypipe.people.sam3_dtype)
SAM3_MIN_GB_FP32 = 10.0
SAM3_MIN_GB_BF16 = 6.0
OOM_RETRY_FACTOR = 0.7    # the train stage retries once at this share of the cap after a CUDA out-of-memory
FREE_DISK_WARN_GB = 30.0


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def _nvidia_smi() -> dict | None:
    # WSL puts the Windows driver's nvidia-smi in /usr/lib/wsl/lib, which a login shell may not have on PATH
    exe = shutil.which("nvidia-smi") or next((p for p in ("/usr/lib/wsl/lib/nvidia-smi",) if Path(p).is_file()), None)
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.free,compute_cap,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({"name": parts[0], "totalGB": round(float(parts[1]) / 1024, 2), "freeGB": round(float(parts[2]) / 1024, 2),
                         "sm": float(parts[3]), "driver": parts[4]})
        except ValueError:
            continue
    if not gpus:
        return None
    # training is pinned to one card (CUDA_VISIBLE_DEVICES, PLY_TRAIN_GPU): describe that one
    idx = int(os.environ.get("PLY_TRAIN_GPU", "0") or 0)
    return gpus[idx if idx < len(gpus) else 0]


def _torch_gpu() -> dict | None:
    """A container without nvidia-smi on PATH still has a GPU torch can see: never plan a cloud card as "no GPU"."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        idx = 0
        free, total = torch.cuda.mem_get_info(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        return {"name": torch.cuda.get_device_name(idx), "totalGB": round(total / 2**30, 2), "freeGB": round(free / 2**30, 2),
                "sm": float(f"{major}.{minor}"), "driver": None, "via": "torch"}
    except Exception:
        return None


def _ram_gb() -> float | None:
    try:  # Linux and WSL: under WSL this is the VM's share (.wslconfig), which is the right number
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def probe(path: Path | None = None) -> dict:
    """The machine as measured now. Env overrides win (tests, or a probe that reads the wrong card)."""
    gpu = _nvidia_smi() or _torch_gpu()
    vram = _env_float("PLY_DEVICE_VRAM_GB")
    sm = _env_float("PLY_DEVICE_SM")
    if vram is not None or sm is not None or os.environ.get("PLY_DEVICE_GPU"):
        gpu = dict(gpu or {})
        gpu.setdefault("name", os.environ.get("PLY_DEVICE_GPU") or "override")
        if os.environ.get("PLY_DEVICE_GPU"):
            gpu["name"] = os.environ["PLY_DEVICE_GPU"]
        if vram is not None:
            gpu["totalGB"] = gpu["freeGB"] = vram
        if sm is not None:
            gpu["sm"] = sm
        gpu.setdefault("sm", 7.5)
        gpu.setdefault("totalGB", gpu.get("freeGB") or 0.0)
        gpu.setdefault("freeGB", gpu["totalGB"])
        gpu["override"] = True
    ram = _env_float("PLY_DEVICE_RAM_GB") or _ram_gb()
    disk = None
    try:
        disk = round(shutil.disk_usage(path or Path.cwd()).free / 2**30, 1)
    except OSError:
        pass
    return {"gpu": gpu, "ramGB": ram, "cpus": os.cpu_count(), "freeDiskGB": disk, "wsl": _is_wsl()}


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def device_plan(machine: dict) -> dict:
    """Ceilings for this machine. None means "no limit from the device" (the capture's own plan stands)."""
    gpu = machine.get("gpu") or None
    ram = machine.get("ramGB")
    notes: list[str] = []
    out: dict = {"machine": machine, "maxGaussians": None, "maxLongSide": None, "packed": False, "loaderWorkers": None,
                 "sam3": True, "trainable": gpu is not None}
    if gpu is None:
        out["sam3"] = False
        out["why"] = "no NVIDIA GPU found: frames, camera solve (slow, SIFT on the CPU) and floor plan only; no training"
        return out
    total = float(gpu.get("totalGB") or 0)
    # free, not total: a card that also drives the desktop has less than its label says. Never trust free above total
    free = min(float(gpu.get("freeGB") or total), total) if total else float(gpu.get("freeGB") or 0)
    sm = float(gpu.get("sm") or 0)
    usable = max(free - RESERVE_GB, 0.3)
    cap = int(usable * 2**30 / BYTES_PER_GAUSSIAN)
    out["maxGaussians"] = cap
    notes.append(f"{gpu.get('name', 'GPU')}: {free:g} of {total:g} GB free, holds about {cap / 1e6:.2f}M splats "
                 f"({BYTES_PER_GAUSSIAN} B each after {RESERVE_GB:g} GB reserve; the only splat bound)")
    if free < SMALL_CARD_GB:
        out["maxLongSide"] = SMALL_CARD_SIDE
        notes.append(f"under {SMALL_CARD_GB:g} GB: working size ≤{SMALL_CARD_SIDE} px")
    if total < PACKED_UNDER_GB:
        out["packed"] = True
        notes.append("packed rasterisation")
    sam_need = SAM3_MIN_GB_BF16 if sm >= 8 else SAM3_MIN_GB_FP32
    if free < sam_need:
        out["sam3"] = False
        notes.append(f"SAM 3 off (needs ~{sam_need:g} GB {'bf16' if sm >= 8 else 'fp32'} on sm {sm:g}): "
                     "people masks with YOLO, no object labels")
    if ram is not None:
        workers = 4 if ram >= 24 else 2 if ram >= 12 else 1
        if workers < 4:
            out["loaderWorkers"] = workers
            notes.append(f"{ram:g} GB RAM → {workers} loader worker{'s' if workers > 1 else ''}")
    disk = machine.get("freeDiskGB")
    if disk is not None and disk < FREE_DISK_WARN_GB:
        notes.append(f"only {disk:g} GB free disk: a multi-clip job can need 20-40 GB")
    if sm and sm < 7.0:
        out["trainable"] = False
        notes.append(f"compute capability {sm:g}: torch cu12x no longer ships kernels for it, no training")
    out["why"] = "; ".join(notes)
    return out


def oom(text: str) -> bool:
    t = text.lower()
    return "out of memory" in t or "cuda error: out of memory" in t or "cudaerrormemoryallocation" in t


def main(argv: list[str] | None = None) -> int:
    """`python -m plypipe.device [--json] [--dir PATH]`: what a run on this machine would be held to."""
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="python -m plypipe.device")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dir", type=Path, default=None, help="where the jobs will be written (free disk is measured there)")
    a = ap.parse_args(argv)
    dp = device_plan(probe(a.dir))
    if a.json:
        print(json.dumps(dp, indent=1))
        return 0
    m = dp["machine"]
    g = m.get("gpu") or {}
    print(f"GPU     {g.get('name', 'none')}  {g.get('freeGB', '?')} / {g.get('totalGB', '?')} GB free  sm {g.get('sm', '?')}  driver {g.get('driver') or '?'}")
    print(f"RAM     {m.get('ramGB')} GB   CPUs {m.get('cpus')}   free disk {m.get('freeDiskGB')} GB   WSL {m.get('wsl')}")
    print(f"train   {'yes' if dp['trainable'] else 'NO'}   splat ceiling {dp['maxGaussians'] or '-'}   packed {dp['packed']}   "
          f"loader workers {dp['loaderWorkers'] or 'default'}   SAM 3 {'on' if dp['sam3'] else 'off'}")
    print(f"why     {dp['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
