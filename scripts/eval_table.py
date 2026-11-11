# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""The results table of ply_eval runs, as Markdown on stdout.

    python scripts/eval_table.py /tmp/ply-eval [--baselines q0a,q0b]

Reads <folder>/<variant>/metrics.json (what `ply_eval.py collect` fetches). One row per variant: steps, the colour-corrected
scores on both hold-outs (every-Nth views, and the block of consecutive frames), viewer PSNR of the desktop and phone files,
the share of trained splats that survived export, wall minutes and estimated cost. With two or more complete baselines
(runs with role "baseline", or the names given) it also prints the noise floor, the spread of each metric across them, and
each other variant's change against their mean, flagged inside or outside twice the noise floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# label, path in metrics.json, decimals, True when a higher value is better
METRICS = [
    ("every cc-PSNR", ("scores", "every", "ccPsnr"), 2, True),
    ("every cc-SSIM", ("scores", "every", "ccSsim"), 4, True),
    ("every cc-LPIPS", ("scores", "every", "ccLpips"), 4, False),
    ("block cc-PSNR", ("scores", "block", "ccPsnr"), 2, True),
    ("block cc-SSIM", ("scores", "block", "ccSsim"), 4, True),
    ("block cc-LPIPS", ("scores", "block", "ccLpips"), 4, False),
    ("viewer PSNR desktop", ("viewer", "desktop", "psnr"), 2, True),
    ("viewer PSNR phone", ("viewer", "phone", "psnr"), 2, True),
]


def load_runs(folder: Path) -> dict[str, dict]:
    runs = {}
    for path in sorted(folder.glob("*/metrics.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            runs[data.get("variant") or path.parent.name] = data
    return runs


def value(run: dict, path: tuple[str, ...]) -> float | None:
    for key in path:
        run = run.get(key) if isinstance(run, dict) else None
    return float(run) if isinstance(run, (int, float)) else None


def fmt(v: float | None, places: int, sign: bool = False) -> str:
    return "–" if v is None else f"{v:+.{places}f}" if sign else f"{v:.{places}f}"


def baseline_names(runs: dict[str, dict], given: list[str] | None) -> list[str]:
    names = given if given is not None else sorted(n for n, r in runs.items() if r.get("role") == "baseline")
    return [n for n in names if n in runs and runs[n].get("complete")]


def order(runs: dict[str, dict], bases: list[str]) -> list[str]:
    return [*bases, *sorted(n for n in runs if n not in bases)]


def kept(run: dict) -> str:
    s = run.get("splats") or {}
    if s.get("keptFraction") is None:
        return "–"
    return f"{s['keptFraction']:.0%}" + (f" ({s['shipped'] / 1e6:.2f}M)" if s.get("shipped") else "")


def main_table(runs: dict[str, dict], names: list[str]) -> list[str]:
    head = ["variant", "steps", *(m[0] for m in METRICS), "splats kept", "minutes", "cost"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    for name in names:
        r = runs[name]
        cells = [name + ("" if r.get("complete") else " (incomplete)"), str(r.get("steps") or "–"),
                 *(fmt(value(r, path), places) for _, path, places, _ in METRICS), kept(r),
                 fmt(r.get("wallMinutes"), 1), f"${r['costUsd']:.2f}" if r.get("costUsd") is not None else "–"]
        out.append("| " + " | ".join(cells) + " |")
    return out


def noise_floor(runs: dict[str, dict], bases: list[str]) -> dict[str, tuple[float | None, float | None]]:
    """metric label -> (baseline mean, noise floor = spread across the baselines)."""
    out = {}
    for label, path, _, _ in METRICS:
        vals = [v for v in (value(runs[b], path) for b in bases) if v is not None]
        out[label] = (sum(vals) / len(vals), max(vals) - min(vals)) if len(vals) == len(bases) else (None, None)
    return out


def verdict(delta: float, floor: float, higher_better: bool) -> str:
    if abs(delta) <= 2 * floor:
        return "inside"
    return "outside, " + ("better" if (delta > 0) == higher_better else "worse")


def render(runs: dict[str, dict], given: list[str] | None = None) -> str:
    bases = baseline_names(runs, given)
    names = order(runs, bases)
    out = ["## Results", "", *main_table(runs, names), ""]
    if len(bases) < 2:
        have = ", ".join(bases) or "none"
        out += [f"Noise floor: needs two complete baselines (found: {have}). Run q0a and q0b first.", ""]
        return "\n".join(out)
    floors = noise_floor(runs, bases)
    out += [f"## Noise floor ({' vs '.join(bases)})", "", "| metric | baseline mean | noise floor | twice the floor |", "|---|---:|---:|---:|"]
    for label, path, places, _ in METRICS:
        mean, floor = floors[label]
        out.append(f"| {label} | {fmt(mean, places)} | {fmt(floor, places)} | {fmt(None if floor is None else 2 * floor, places)} |")
    others = [n for n in names if n not in bases]
    if others:
        out += ["", f"## Change against the baseline mean (inside = within twice the noise floor)", "",
                "| variant | " + " | ".join(m[0] for m in METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
        for name in others:
            cells = []
            for label, path, places, higher in METRICS:
                mean, floor = floors[label]
                v = value(runs[name], path)
                cells.append("–" if v is None or mean is None else f"{fmt(v - mean, places, True)} {verdict(v - mean, floor, higher)}")
            out.append(f"| {name}{'' if runs[name].get('complete') else ' (incomplete)'} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path, help="folder of <variant>/metrics.json")
    ap.add_argument("--baselines", help="comma-separated baseline variants (default: runs whose role is baseline)")
    a = ap.parse_args(argv)
    runs = load_runs(a.folder)
    if not runs:
        print(f"no <variant>/metrics.json under {a.folder}", file=sys.stderr)
        return 1
    print(render(runs, a.baselines.split(",") if a.baselines else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
