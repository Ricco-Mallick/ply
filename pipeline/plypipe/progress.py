"""Progress inside the sfm stage, read from COLMAP's own log lines (pycolmap logs to stderr, which the worker reads).

Without this the status page sits at the start of "Solving camera positions" for the whole of matching, which can take
half an hour. COLMAP already counts its work; this maps those counters onto one 0..1 fraction of the stage:

    extraction   0.00-0.10  "Processed file [12/180]"
    matching     0.10-0.60  "Processing block [2/4, 3/4]" (exhaustive) or "[n/N]" lines from the sequential matcher
    mapping      0.60-0.93  global mapper phases + bundle adjustment iterations; incremental: registered frames
    undistort    0.93-1.00  "Undistorting image [40/180]"
"""

from __future__ import annotations

import re

SPANS = {"extract": (0.0, 0.10), "match": (0.10, 0.60), "map": (0.60, 0.93), "undistort": (0.93, 1.0)}
LABELS = {"extract": "finding features", "match": "matching frames", "map": "placing cameras", "undistort": "undistorting frames"}
# global mapper phases, in order, as a fraction of the mapping span
GLOBAL_PHASES = [("rotation averaging", 0.05, "averaging rotations"), ("track establishment", 0.15, "linking feature tracks"),
                 ("global positioning", 0.25, "positioning cameras"), ("iterative bundle adjustment", 0.45, "bundle adjustment"),
                 ("iterative retriangulation", 0.85, "retriangulating points")]

SRC = re.compile(r"^[IWE]\d{8} [\d:.]+ +\d+ (\w+)\.cc:\d+\] (.*)$")
COUNT = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")
BLOCK = re.compile(r"Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]")
BA_ITER = re.compile(r"bundle adjustment iteration (\d+) / (\d+)")
REG = re.compile(r"num_reg_frames=(\d+)")


class SfmProgress:
    """Feed it every output line; it returns (fraction of the sfm stage, message, step key, fraction of that step) when
    a line moves progress, else None."""

    def __init__(self, n_images: int | None = None):
        self.n_images = n_images
        self.step: str | None = None

    def _out(self, step: str, f: float, detail: str = ""):
        lo, hi = SPANS[step]
        f = max(0.0, min(1.0, f))
        return lo + (hi - lo) * f, f"{LABELS[step]}{' · ' + detail if detail else ''}", step, f

    def feed(self, line: str):
        m = SRC.match(line.strip())
        if not m:
            return None
        src, msg = m.groups()
        if src == "feature_extraction":
            self.step = "extract"
            c = COUNT.search(msg) if msg.startswith("Processed file") else None
            return self._out("extract", int(c[1]) / int(c[2]), f"{c[1]}/{c[2]} frames") if c else None
        if src in ("feature_matching", "pairing"):
            self.step = "match"
            b = BLOCK.search(msg)
            if b:
                i, n, j, k = map(int, b.groups())
                f = ((i - 1) * k + j - 1) / (n * k)  # the block being processed has not finished yet
                return self._out("match", f, f"{round(f * 100)}%")
            c = COUNT.search(msg)
            if c and int(c[2]):
                return self._out("match", (int(c[1]) - 1) / int(c[2]), f"{c[1]}/{c[2]}")
            return None
        if src == "undistorters":
            self.step = "undistort"
            c = COUNT.search(msg)
            return self._out("undistort", int(c[1]) / int(c[2]), f"{c[1]}/{c[2]}") if c else None
        if src in ("global_mapper", "global_pipeline", "rotation_averaging", "global_positioning", "incremental_mapper",
                   "incremental_pipeline", "controllers"):
            self.step = "map"
            low = msg.lower()
            for key, at, label in GLOBAL_PHASES:
                if f"running {key}" in low:
                    return self._out("map", at, label)
            it = BA_ITER.search(msg)
            if it:  # bundle adjustment runs between 0.45 and 0.85 of mapping
                return self._out("map", 0.45 + 0.40 * int(it[1]) / max(1, int(it[2])), f"bundle adjustment {it[1]}/{it[2]}")
            r = REG.search(msg)
            if r and self.n_images:
                return self._out("map", int(r[1]) / self.n_images, f"{r[1]}/{self.n_images} frames placed")
        return None
