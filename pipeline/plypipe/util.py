"""Shared plumbing: run context, stage markers (resumability), scene.json merging, subprocesses."""

from __future__ import annotations

import json
import os
import shutil
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


def run(cmd: list[str], ctx: Ctx, stage: str, cwd: str | Path | None = None, env: dict | None = None,
        progress_every: int = 20) -> None:
    """Run a command, forwarding its output into the JSON log (every Nth line as progress,
    all lines into <out>/logs/<stage>.log). Raises on a non-zero exit."""
    logs = ctx.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    ctx.log.emit(stage, "info", "exec", cmd=" ".join(map(str, cmd)))
    with (logs / f"{stage}.log").open("a") as raw:
        proc = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=full_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        n = 0
        tail: list[str] = []
        for line in proc.stdout:
            raw.write(line)
            line = line.rstrip()
            if not line:
                continue
            tail = (tail + [line])[-30:]
            n += 1
            if n % progress_every == 0:
                ctx.log.emit(stage, "progress", line[-300:])
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{cmd[0]} exited with {code}; last output:\n" + "\n".join(tail))
