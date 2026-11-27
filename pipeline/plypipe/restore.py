# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Restore points for training: autosave on a timer, save-and-exit on a signal, resume at the exact step.

Runs inside the trainer process (plypipe.train_launcher), around gsplat's own simple_trainer, which has no resume of
its own. Both gsplat versions the pipeline runs (1.5.3 locally and on Kaggle, the pinned main on Modal) drive training
with `for step in tqdm.tqdm(range(init_step, max_steps))`. This module swaps that one progress bar for an iterator that,
between two steps (never inside one):

  * on its first step, when resuming, puts back the whole training state: every splat tensor at its saved size (MCMC
    replaces them as it relocates, so they are rebuilt and re-attached to their optimizers before the optimizer state is
    loaded), every optimizer, every learned module (pose, appearance, bilateral grid, PPISP), every learning-rate
    schedule and the densification strategy's state, then continues at the step after the saved one. Everything
    that follows the step number (SH degree, refinement, MCMC noise, LR decay) lines up again;
  * every PLY_AUTOSAVE_MIN minutes (default 10) writes a restore point atomically (temp file, then rename), one copy;
    that is what survives a power cut, a closed laptop lid or `wsl --shutdown`;
  * on SIGTERM / SIGINT / SIGHUP (the PC's window closed, Ctrl+C, a cloud container told to stop) saves and exits with
    PAUSED (75). A second signal exits at once. When its parent dies the trainer is sent SIGTERM (prctl), so an orphan
    never keeps the GPU;
  * prints one progress line every few seconds ({"plyTrain": {...}}: step, loss, splats, VRAM, speed, ETA) for the
    console and the worker.

Only the training loop's own bar is replaced; every other tqdm call goes to the real tqdm.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

PAUSED = 75
FILE, META = "restore.pt", "restore.json"
SKIP_MODULES = {"splats", "ssim", "psnr", "lpips"}
PROGRESS_EVERY_S = float(os.environ.get("PLY_PROGRESS_EVERY_S", 5))


class _S:
    runner = None
    stop: str | None = None
    schedulers: list = []
    dir: Path | None = None
    resume = False
    key = ""
    every_s = 600.0
    max_steps = None
    log = staticmethod(lambda m: print(m, flush=True))


def restore_key(cmd: list[str]) -> str:
    """What must match for a restore point to be resumed: the trainer's settings, minus the flags that only say where
    state lives. A restore point from other settings (another splat cap, other steps) is never used."""
    skip = {"--resume", "--restore"}
    keep, i = [], 0
    while i < len(cmd):
        if cmd[i] in skip:
            i += 1 if cmd[i] == "--resume" else 2
            continue
        a = str(cmd[i])
        keep.append(Path(a).name if "/" in a or "\\" in a else a)  # paths differ per machine; the settings do not
        i += 1
    return hashlib.sha256("\0".join(keep).encode()).hexdigest()[:16]


def read_meta(d: Path) -> dict | None:
    try:
        m = json.loads((Path(d) / META).read_text())
        return m if (Path(d) / FILE).is_file() else None
    except (OSError, ValueError):
        return None


def usable(d: Path, key: str) -> dict | None:
    m = read_meta(d)
    return m if m and m.get("key") == key else None


# ---- signals ------------------------------------------------------------------------------------------------------

def _on_signal(signum, _frame):
    if _S.stop:  # the second one: no more waiting
        os._exit(130)
    _S.stop = signal.Signals(signum).name
    _S.log(f"[restore] {_S.stop}: saving a restore point at the end of this step, then exiting")


def install_signals() -> None:
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _on_signal)
    try:  # Linux: the kernel sends SIGTERM when the parent (plypipe) dies
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


# ---- the training state ------------------------------------------------------------------------------------------

def _optimizer_sets(runner):
    for name, val in vars(runner).items():
        if name.endswith("optimizers") and isinstance(val, (dict, list, tuple)):
            yield name, val


def _modules(runner):
    import torch
    for name, val in vars(runner).items():
        if name in SKIP_MODULES or not isinstance(val, torch.nn.Module):
            continue
        if type(val).__module__.startswith("torchmetrics"):
            continue
        yield name, val


def snapshot(runner, step: int) -> dict:
    data = {"step": int(step), "splats": {k: v.detach() for k, v in runner.splats.items()}, "optim": {}, "modules": {},
            "sched": [s.state_dict() for s in _S.schedulers], "strategy": getattr(runner, "strategy_state", None)}
    for name, val in _optimizer_sets(runner):
        data["optim"][name] = {k: o.state_dict() for k, o in val.items()} if isinstance(val, dict) else [o.state_dict() for o in val]
    for name, mod in _modules(runner):
        data["modules"][name] = mod.state_dict()
    return data


def save(runner, step: int, why: str) -> dict:
    import torch
    d = _S.dir
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tmp = d / (FILE + ".tmp")
    torch.save(snapshot(runner, step), tmp)
    os.replace(tmp, d / FILE)  # atomic: a kill mid-save leaves the previous restore point whole
    meta = {"step": int(step), "of": _S.max_steps, "splats": int(len(runner.splats["means"])), "key": _S.key, "why": why,
            "savedAt": int(time.time()), "seconds": round(time.time() - t0, 1), "mb": round((d / FILE).stat().st_size / 2**20, 1)}
    (d / (META + ".tmp")).write_text(json.dumps(meta))
    os.replace(d / (META + ".tmp"), d / META)
    print(json.dumps({"plyRestore": meta}), flush=True)
    return meta


def load(runner, path: Path) -> int:
    import torch
    data = torch.load(path, map_location=runner.device, weights_only=False)
    for name, mod in _modules(runner):  # modules first: their optimizers keep pointing at the same parameters
        if name in data["modules"]:
            mod.load_state_dict(data["modules"][name])
    for k, t in data["splats"].items():  # MCMC changed the count: new parameters of the saved size
        runner.splats[k] = torch.nn.Parameter(t.to(runner.device).contiguous(), requires_grad=True)
    for name, saved in data["optim"].items():
        val = getattr(runner, name, None)
        if isinstance(val, dict):
            for k, opt in val.items():
                if name == "optimizers" and k in runner.splats:  # re-attach before loading: the state is per parameter
                    opt.state.clear()
                    opt.param_groups[0]["params"] = [runner.splats[k]]
                if k in saved:
                    opt.load_state_dict(saved[k])
        elif isinstance(val, (list, tuple)):
            for opt, sd in zip(val, saved):
                opt.load_state_dict(sd)
    if data.get("strategy") is not None:
        runner.strategy_state = data["strategy"]
    if len(data["sched"]) == len(_S.schedulers):
        for s, sd in zip(_S.schedulers, data["sched"]):
            s.load_state_dict(sd)
    else:
        _S.log(f"[restore] {len(data['sched'])} saved learning-rate schedules, {len(_S.schedulers)} now: schedules not restored")
    return int(data["step"])


def _track_schedulers() -> None:
    import torch
    base = torch.optim.lr_scheduler.LRScheduler
    if getattr(base, "_ply_tracked", False):
        return
    orig = base.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        _S.schedulers.append(self)

    base.__init__ = init
    base._ply_tracked = True


# ---- the loop ----------------------------------------------------------------------------------------------------

LOSS = re.compile(r"loss=([0-9.]+)")


class Loop:
    """Stands in for `tqdm.tqdm(range(...))` of the training loop. `saver`/`loader`/`clock` are replaceable (tests)."""

    def __init__(self, iterable, saver=None, loader=None, clock=time.time, out=None, **_kw):
        self.rng, self.desc = iterable, ""
        self.saver, self.loader, self.clock = saver or save, loader or load, clock
        self.out = out or (lambda line: print(line, flush=True))

    def set_description(self, desc, refresh=True):
        self.desc = str(desc)

    def set_postfix(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def close(self):
        pass

    def __iter__(self):
        runner, start, stop = _S.runner, self.rng.start, self.rng.stop
        if _S.resume and _S.dir is not None:
            m = usable(_S.dir, _S.key)
            if m:
                done = self.loader(runner, _S.dir / FILE)
                start = done + 1
                _S.log(f"[restore] resumed at step {start} of {stop} with {m.get('splats')} splats (saved {m.get('why')})")
                print(json.dumps({"plyRestore": {"resumedAt": start, "of": stop}}), flush=True)
        t0 = last_save = last_print = self.clock()
        first = start
        for step in range(start, stop):
            now = self.clock()
            if step > first and _S.dir is not None:
                if _S.stop:
                    self.saver(runner, step - 1, _S.stop)
                    _S.log(f"[restore] saved at step {step - 1}; exiting ({_S.stop}). Run the same job again to continue.")
                    sys.exit(PAUSED)
                if _S.every_s and now - last_save >= _S.every_s:
                    self.saver(runner, step - 1, "autosave")
                    last_save = self.clock()
            if now - last_print >= PROGRESS_EVERY_S or step == first:
                last_print = now
                self.out(json.dumps({"plyTrain": progress(runner, step, stop, first, now - t0, self.desc)}))
            yield step


def progress(runner, step: int, stop: int, first: int, elapsed: float, desc: str) -> dict:
    rate = (step - first) / elapsed if elapsed > 0 and step > first else None
    out = {"step": step, "of": stop, "round": os.environ.get("PLY_TRAIN_ROUND") or None}
    m = LOSS.search(desc or "")
    if m:
        out["loss"] = float(m.group(1))
    try:
        out["splats"] = int(len(runner.splats["means"]))
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["vramGB"] = round(torch.cuda.memory_reserved() / 2**30, 2)
    except Exception:
        pass
    if rate:
        out["stepsPerS"] = round(rate, 2)
        out["etaS"] = int((stop - step) / rate)
    return out


class _TqdmShim:
    """The `tqdm` module as simple_trainer sees it: the training loop's bar is a Loop, everything else real tqdm."""

    def __init__(self, real):
        self._real = real
        self._used = False

    def tqdm(self, iterable=None, *a, **k):
        if not self._used and isinstance(iterable, range) and _S.max_steps and iterable.stop == _S.max_steps:
            self._used = True
            return Loop(iterable)
        return self._real.tqdm(iterable, *a, **k)

    def __getattr__(self, name):
        return getattr(self._real, name)


def install(st, cfg, directory: str | None, resume: bool, key: str) -> dict:
    """Wire restore points into simple_trainer `st` for this run. Returns what the run's summary records."""
    _S.dir = Path(directory) if directory else None
    _S.resume, _S.key, _S.max_steps = bool(resume), key, int(cfg.max_steps)
    _S.every_s = float(os.environ.get("PLY_AUTOSAVE_MIN", 10)) * 60
    _track_schedulers()
    import tqdm as real
    st.tqdm = _TqdmShim(real)
    orig = st.Runner.train

    def train(self, *a, **k):
        _S.runner = self
        return orig(self, *a, **k)

    st.Runner.train = train
    install_signals()
    m = usable(_S.dir, key) if (_S.dir and resume) else None
    return {"autosaveMin": _S.every_s / 60, "dir": str(_S.dir) if _S.dir else None,
            **({"resumeFrom": m["step"]} if m else {})}
