# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""Readable progress lines for a console window, one per event, appended (no cursor tricks: a wsl.exe window, a
log file and a pipe all show the same thing).

    09:41:07  train   42.1%  step 12,630/30,000 | loss 0.041 | 1.12M splats | VRAM 3.1 GB | 9.8 steps/s | ETA 29m
    09:41:12  train   restore point saved at step 12,630 (612 MB in 4.1 s)

plypipe prints these instead of JSON when its output is a terminal (or PLY_PRETTY=1); log.jsonl keeps the JSON
either way, and a worker reading the pipeline's output gets JSON (PLY_PRETTY=0, and its pipe is not a terminal).
"""

from __future__ import annotations

import os
import sys
import time

ASCII_BAR = 20


def enabled(stream=None) -> bool:
    v = os.environ.get("PLY_PRETTY")
    if v is not None:
        return v == "1"
    s = stream or sys.stdout
    return hasattr(s, "isatty") and s.isatty()


def _dur(s: float | int | None) -> str:
    if s is None:
        return "?"
    s = int(s)
    if s < 90:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _n(v) -> str:
    if v is None:
        return "?"
    v = float(v)
    return f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.0f}k" if v >= 1e4 else f"{int(v):,}"


def bar(frac: float) -> str:
    k = max(0, min(ASCII_BAR, int(round(frac * ASCII_BAR))))
    return "[" + "#" * k + "." * (ASCII_BAR - k) + "]"


def train_line(t: dict) -> str:
    step, of = t.get("step", 0), t.get("of") or 1
    parts = [f"{bar(step / of)} {100 * step / of:5.1f}%", f"step {step:,}/{of:,}"]
    if t.get("round"):
        parts.insert(0, f"round {t['round']}")
    if t.get("loss") is not None:
        parts.append(f"loss {t['loss']:.3f}")
    if t.get("splats") is not None:
        parts.append(f"{_n(t['splats'])} splats")
    if t.get("vramGB") is not None:
        parts.append(f"VRAM {t['vramGB']:.1f} GB")
    if t.get("stepsPerS"):
        parts.append(f"{t['stepsPerS']:.1f} steps/s")
    if t.get("etaS") is not None:
        parts.append(f"ETA {_dur(t['etaS'])}")
    return " | ".join(parts)


def format(ev: dict) -> str | None:
    """One event (a log.jsonl record) as a console line, or None to show nothing."""
    stage, event, msg = ev.get("stage", ""), ev.get("event", ""), str(ev.get("msg") or "")
    ts = str(ev.get("t", ""))[11:19] or time.strftime("%H:%M:%S")
    head = f"{ts}  {stage:<9}"
    if event == "progress" and isinstance(ev.get("train"), dict):
        return f"{head} {train_line(ev['train'])}"
    if event == "restore":
        r = ev.get("restore") or {}
        if "resumedAt" in r:
            return f"{head} resumed from the restore point at step {r['resumedAt']:,} of {r.get('of', 0):,}"
        return f"{head} restore point saved at step {r.get('step', 0):,} ({r.get('mb', '?')} MB in {r.get('seconds', '?')} s, {r.get('why', '')})"
    if event == "start":
        return f"{head} == {stage} =="
    if event == "skip":
        return f"{head} already done"
    if event == "done":
        bits = [f"done in {_dur(ev.get('seconds'))}"]
        for k, label in (("kept", "frames kept"), ("registeredRatio", "registered"), ("gaussiansOut", "splats")):
            if ev.get(k) is not None:
                v = ev[k]
                bits.append(f"{label} {v:.0%}" if k == "registeredRatio" else f"{label} {_n(v)}")
        e = ev.get("eval") or {}
        if e.get("psnr") is not None:
            bits.append(f"held-out {e.get('ccPsnr', e['psnr']):.2f} dB, LPIPS {e.get('lpips', 0):.3f}")
        g = ev.get("growth") or {}
        if g.get("rounds"):
            bits.append(f"{len(g['rounds']) - 1} growth rounds, best {g.get('best')}" + (" (wanted a bigger GPU)" if g.get("needsBiggerGpu") else ""))
        return f"{head} " + " | ".join(bits)
    if event == "paused":
        return f"{head} PAUSED: {msg}"
    if event == "error":
        return f"{head} ERROR: {msg[:400]}"
    if event == "warn":
        return f"{head} note: {msg[:400]}"
    if event == "progress":
        if not msg:
            return None
        if "done" in ev and "total" in ev and ev["total"]:
            return f"{head} {bar(ev['done'] / ev['total'])} {msg[:200]}"
        return f"{head} {msg[:240]}"
    if event == "info":
        return None  # exec lines and the like: in log.jsonl, not on screen
    return f"{head} {event} {msg[:200]}".rstrip()
