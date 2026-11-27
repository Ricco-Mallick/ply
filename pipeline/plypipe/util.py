# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Shared plumbing: run context, stage markers (resumability), scene.json merging, subprocesses."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from plypipe.log import Log


@dataclass
class Ctx:
    out: Path
    log: Log
    opts: dict[str, Any] = field(default_factory=dict)

    # canonical layout -------------------------------------------------
    @property
    def frames_dir(self) -> Path:  # extracted, filtered, resized frames
        return self.out / "frames"

    @property
    def colmap_dir(self) -> Path:  # database + raw sparse models
        return self.out / "colmap"

    @property
    def dense_dir(self) -> Path:  # undistorted PINHOLE images + model (binary and text)
        return self.out / "dense"

    @property
    def data_dir(self) -> Path:  # gsplat data_dir: images/ + sparse/0 (aligned, scene frame)
        return self.out / "data"

    @property
    def train_dir(self) -> Path:
        return self.out / "train"

    def opt(self, key: str, default: Any = None) -> Any:
        v = self.opts.get(key)
        return default if v is None else v


# ------------------------------------------------------------------ markers

def marker(ctx: Ctx, stage: str) -> Path:
    return ctx.out / ".done" / f"{stage}.json"


def is_done(ctx: Ctx, stage: str) -> bool:
    return marker(ctx, stage).exists()


def mark_done(ctx: Ctx, stage: str, report: dict) -> None:
    m = marker(ctx, stage)
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps(report, indent=2, default=str))
    rep_path = ctx.out / "report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    rep[stage] = report
    rep_path.write_text(json.dumps(rep, indent=2, default=str))


def clear_from(ctx: Ctx, stages: Iterable[str]) -> None:
    for s in stages:
        m = marker(ctx, s)
        if m.exists():
            m.unlink()


# ------------------------------------------------------------------ scene.json

def update_scene(ctx: Ctx, patch: dict) -> dict:
    p = ctx.out / "scene.json"
    scene = json.loads(p.read_text()) if p.exists() else {}
    scene.update(patch)
    p.write_text(json.dumps(scene, indent=2))
    return scene


def read_scene(ctx: Ctx) -> dict:
    p = ctx.out / "scene.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ------------------------------------------------------------------ processes

def need(binary: str, hint: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise RuntimeError(f"`{binary}` not found on PATH. {hint}")
    return path


class Paused(RuntimeError):
    """The child saved a restore point and stopped on request (plypipe.restore.PAUSED): not a failure."""


PAUSED = 75


def run(cmd: list[str], ctx: Ctx, stage: str, cwd: str | Path | None = None, env: dict | None = None,
        progress_every: int = 20, own_session: bool = False) -> None:
    """Run a command, forwarding its output into the JSON log (every Nth line as progress, and every structured
    {"plyTrain"/"plyRestore": ...} line at once), all lines into <out>/logs/<stage>.log. Raises on a non-zero exit,
    Paused when the child saved and stopped.

    own_session: the child gets its own process group, so Ctrl+C at a terminal does not hit it (and its DataLoader
    workers) mid-step; this process forwards one SIGTERM instead and waits for it to save."""
    logs = ctx.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    ctx.log.emit(stage, "info", "exec", cmd=" ".join(map(str, cmd)))
    with (logs / f"{stage}.log").open("a") as raw:
        proc = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=full_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=own_session)
        assert proc.stdout is not None
        old_handlers = _forward_signals(proc, ctx, stage) if own_session else {}
        n = 0
        tail: list[str] = []
        try:
            for line in proc.stdout:
                raw.write(line)
                line = line.rstrip()
                if not line:
                    continue
                tail = (tail + [line])[-30:]
                n += 1
                if line.startswith('{"plyTrain"') or line.startswith('{"plyRestore"'):
                    try:
                        rec = json.loads(line)
                        kind = next(iter(rec))
                        ctx.log.emit(stage, "progress" if kind == "plyTrain" else "restore", "", **{"train" if kind == "plyTrain" else "restore": rec[kind]})
                        continue
                    except ValueError:
                        pass
                if n % progress_every == 0:
                    ctx.log.emit(stage, "progress", line[-300:])
            code = proc.wait()
        finally:
            for sig, h in old_handlers.items():
                signal.signal(sig, h)
    if code == PAUSED:
        raise Paused(f"{cmd[0]} saved a restore point and stopped")
    if code != 0:
        raise RuntimeError(f"{cmd[0]} exited with {code}; last output:\n" + "\n".join(tail))


def _forward_signals(proc, ctx: Ctx, stage: str) -> dict:
    """While the child runs: SIGINT/SIGTERM/SIGHUP here become one SIGTERM there (it saves and exits)."""
    import threading
    if threading.current_thread() is not threading.main_thread():
        return {}
    sent = {"n": 0}

    def fwd(signum, _frame):
        sent["n"] += 1
        if sent["n"] == 1:
            ctx.log.emit(stage, "warn", f"{signal.Signals(signum).name}: asking {stage} to save a restore point and stop")
            proc.send_signal(signal.SIGTERM)
        else:
            proc.kill()

    old = {}
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            old[sig] = signal.signal(sig, fwd)
    return old


def splat_transform_cmd(version: str = "3.10.1") -> list[str]:
    """The pinned splat-transform: the globally installed binary when it is that version (the GPU image), else npx.
    Several npx runs of one package at once race in npm's cache, so the binary is preferred."""
    import shutil
    import subprocess
    exe = shutil.which("splat-transform")
    if exe:
        try:
            if version in subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60).stdout:
                return [exe]
        except Exception:
            pass
    return ["npx", "-y", f"@playcanvas/splat-transform@{version}"]
