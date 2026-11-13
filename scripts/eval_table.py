# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""The results table of ply_eval runs, as Markdown on stdout.

    python scripts/eval_table.py /tmp/ply-eval [--baselines q0a,q0b]

Reads <folder>/<variant>/metrics.json (what `ply_eval.py collect` fetches). One row per variant: steps, the colour-corrected
scores on both hold-outs (every-Nth views, and the block of consecutive frames), viewer PSNR and brightness of the desktop
and phone files, the share of trained splats that survived export, wall minutes and estimated cost. Runs with an exposure
fold, a walking grid from depth or an export-only origin get a second table of those. With two or more complete baselines
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


def brightness(run: dict) -> str:
    """Desktop / phone file against the photos; the ship rule wants each within x0.95 to x1.05 (IMPROVEMENT-PLAN, phase 4)."""
    def one(file: str) -> str:
        v = value(run, ("viewer", file, "brightness"))
        return "–" if v is None else f"x{v:.3f}" + ("" if 0.95 <= v <= 1.05 else " (!)")
    return f"{one('desktop')} / {one('phone')}"


def fold_cell(run: dict) -> str:
    f = run.get("exposureFold")
    if not f:
        return "–"
    if not f.get("folded"):
        return "not folded" + (f": {f['why']}" if f.get("why") else "")
    settled = f"{f.get('steps')} steps after" if f.get("settled") else "no steps after" + (f" ({f['why']})" if f.get("why") else "")
    return f"x{f.get('brightnessBefore')} -> x{f.get('brightnessAfter')} on {f.get('views')} views, {settled}"


def details(runs: dict[str, dict], names: list[str]) -> list[str]:
    """Exposure fold, walking grid from depth and export-only origin, for the runs that have any of them."""
    shown = [n for n in names if any(runs[n].get(k) for k in ("exposureFold", "denseOccupancy", "exportFrom"))]
    if not shown:
        return []
    out = ["## Exposure fold, walking grid, origin", "", "| variant | model from | exposure fold | walking grid |", "|---|---|---|---|"]
    for n in shown:
        r = runs[n]
        dense = r.get("denseOccupancy")
        grid = "–" if not dense else dense_cell(dense)
        out.append(f"| {n} | {r.get('exportFrom') or 'its own training'} | {fold_cell(r)} | {grid} |")
    return [*out, ""]


def laser(runs: dict[str, dict], names: list[str]) -> list[str]:
    """Scale and geometry against a laser scan (the MuSHRoom scenarios): what plypipe.mushroom.score_job put in metrics.json."""
    shown = [n for n in names if runs[n].get("geometry")]
    if not shown:
        return []
    out = ["## Against the laser scan", "",
           "| variant | scale error % | cameras matched / inliers | camera fit rms cm | splat F 5 cm (P / R) | splat F 10 cm | chamfer cm |", "|---|---:|---:|---:|---:|---:|---:|"]
    for n in shown:
        g = runs[n]["geometry"]
        if g.get("error"):
            out.append(f"| {n} | scoring failed: {g['error']} | | | | | |")
            continue
        c, s = g.get("cameras") or {}, g.get("splats") or {}
        f5, f10 = s.get("f5cm") or {}, s.get("f10cm") or {}
        cm = lambda m: "–" if m is None else f"{m * 100:.1f}"  # noqa: E731
        out.append(f"| {n} | {fmt(g.get('scaleErrorPct'), 2, True)} | {c.get('matched', '–')} / {c.get('inliers', '–')} | {cm(c.get('rmsM'))} | "
                   f"{fmt(f5.get('f'), 3)} ({fmt(f5.get('precision'), 2)} / {fmt(f5.get('recall'), 2)}) | {fmt(f10.get('f'), 3)} | {cm(s.get('chamferM'))} |")
    return [*out, "", "Scale error is signed: positive means the twin's metres are longer than the room's. Splat scores are in the twin's own frame.", ""]


def dense_cell(d: dict) -> str:
    """Kept in step with plypipe.evalrun.dense_cell; this script reads metrics files and does not import the pipeline."""
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


def main_table(runs: dict[str, dict], names: list[str]) -> list[str]:
    head = ["variant", "steps", *(m[0] for m in METRICS), "brightness desktop / phone", "splats kept", "minutes", "cost"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    for name in names:
        r = runs[name]
        cells = [name + ("" if r.get("complete") else " (incomplete)"), str(r.get("steps") or "–"),
                 *(fmt(value(r, path), places) for _, path, places, _ in METRICS), brightness(r), kept(r),
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
    out = ["## Results", "", *main_table(runs, names), "", *details(runs, names), *laser(runs, names)]
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
        out += ["", "## Change against the baseline mean (inside = within twice the noise floor)", "",
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
