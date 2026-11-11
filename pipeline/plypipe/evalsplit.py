# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""A second hold-out for training runs: a block of consecutive frames from the middle of one clip, scored apart from
the every-Nth views, plus per-view scores for both sets.

Every-Nth views sit between two training frames half a second away, so they flatter a model that only interpolates.
A block of 20 consecutive frames (about 9 s of walking) has no training frame within a few seconds of its middle views.

`install` patches the trainer in-process (the launcher imports examples/simple_trainer.py), touching only what that
file exposes as module names: `Dataset` (its split), `Runner` (to build the block set and call its own `eval` a
second time with that set) and the three metric objects `Runner.eval` calls (wrapped to record each call).
gsplat's own means stay authoritative: the per-view series are checked against them and flagged when they disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

# gsplat's stats key -> the camelCase key the reports use
SCORE_KEYS = {"psnr": "psnr", "ssim": "ssim", "lpips": "lpips", "cc_psnr": "ccPsnr", "cc_ssim": "ccSsim", "cc_lpips": "ccLpips"}
METRICS = ("psnr", "ssim", "lpips")


def read_index(path: str | Path) -> list[dict] | None:
    """frames_index.json as a list of {frame, clip, t}; None when it is missing or not that (photo sets, older jobs)."""
    try:
        rows = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    ok = isinstance(rows, list) and rows and all(isinstance(d, dict) and "frame" in d and "clip" in d for d in rows)
    return rows if ok else None


def pick_block(index: list[dict] | None, names: list[str], size: int) -> dict:
    """`size` consecutive frames from the middle of the longest clip, all of them solved.

    `index` is frames_index.json (frame, clip, t); `names` are the registered image names. Frames the solve dropped stay
    in their clip's sequence, so a window never bridges a gap. A clip too short or too gappy for a window hands over
    to the next longest; photo sets (no index) count as one clip in name order."""
    if size < 2:
        raise ValueError("a block needs at least 2 frames")
    solved = set(names)
    clips: dict[str | None, list[str]] = {}
    when: dict[str, float] = {}
    if index:
        for d in index:
            clips.setdefault(d["clip"], []).append(d["frame"])
            if isinstance(d.get("t"), (int, float)):
                when[d["frame"]] = float(d["t"])
    else:
        clips[None] = list(names)
    ranked = sorted(clips, key=lambda c: -sum(f in solved for f in clips[c]))  # stable: ties keep recording order
    for clip in ranked:
        frames = sorted(clips[clip])
        ok = [f in solved for f in frames]
        ideal = (len(frames) - size) // 2
        starts = [s for s in range(len(frames) - size + 1) if all(ok[s:s + size])]
        if not starts:
            continue
        s = min(starts, key=lambda x: (abs(x - ideal), x))
        block = frames[s:s + size]
        span = round(when[block[-1]] - when[block[0]], 2) if block[0] in when and block[-1] in when else None
        return {"clip": clip, "frames": block, "seconds": span, "clipFrames": len(frames),
                "dropped": [f for f in frames if f not in solved]}
    raise ValueError(f"no {size} consecutive solved frames in any clip: the longest has {max((sum(f in solved for f in v) for v in clips.values()), default=0)} solved")


def per_view(names: list[str], calls: dict[str, list[float]], stats: dict, colour_corrected: bool) -> dict:
    """Per-view series from the metric call logs. With the colour-corrected metrics on, Runner.eval calls each metric
    twice per view (raw, then corrected), so a log alternates; the means must match gsplat's stats to be trusted."""
    stride = 2 if colour_corrected else 1
    out: dict = {"images": names}
    bad = []
    for metric in METRICS:
        series = {metric: calls.get(metric, [])[0::stride]}
        if colour_corrected:
            series["cc_" + metric] = calls.get(metric, [])[1::stride]
        for key, values in series.items():
            out[SCORE_KEYS[key]] = [round(v, 5) for v in values]
            want = stats.get(key)
            if len(values) != len(names) or want is None or abs(sum(values) / max(len(values), 1) - want) > 1e-3 * max(1.0, abs(want)):
                bad.append(key)
    out["verified"] = not bad
    if bad:
        out["mismatch"] = bad
    return out


class _Tap:
    def __init__(self, fn, log: list[float]):
        self.fn, self.log = fn, log

    def __call__(self, *args, **kwargs):
        value = self.fn(*args, **kwargs)
        self.log.append(float(value))
        return value

    def __getattr__(self, name):
        return getattr(self.fn, name)


def install(st, size: int, index: list[dict] | None) -> dict:
    """Hold out a block of `size` frames in simple_trainer `st`. Returns a dict that gains "block" (see pick_block) once
    the trainer has parsed the scene. Must run before the Runner is built, in the process that trains."""
    import numpy as np

    state: dict = {}
    base_dataset, base_runner = st.Dataset, st.Runner

    def rows_of(parser) -> list[int]:
        if "block" not in state:
            state["block"] = pick_block(index, list(parser.image_names), size)
        wanted = set(state["block"]["frames"])
        return [i for i, n in enumerate(parser.image_names) if n in wanted]

    class BlockDataset(base_dataset):
        """split "block" is the block; "train" and "val" are gsplat's own splits minus the block."""

        def __init__(self, parser, split="train", *args, **kwargs):
            super().__init__(parser, "val" if split == "block" else split, *args, **kwargs)
            block = set(rows_of(parser))
            self.indices = np.array(sorted(block) if split == "block" else [i for i in self.indices if i not in block], dtype=np.int64)

    class BlockRunner(base_runner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.blockset = st.Dataset(self.parser, split="block")
            total = len(self.parser.image_names)
            if len(self.trainset) + len(self.valset) + len(self.blockset) != total or not len(self.blockset):
                raise RuntimeError(f"hold-out split does not add up: {len(self.trainset)} train + {len(self.valset)} val + "
                                   f"{len(self.blockset)} block of {total} views")
            if self.world_rank == 0:
                Path(self.cfg.result_dir, "ply_block.json").write_text(json.dumps(state["block"], indent=1))
                print(json.dumps({"plyHoldoutBlock": state["block"]}), flush=True)

        def _scored(self, step: int, stage: str, dataset) -> dict:
            taps = {m: _Tap(getattr(self, m), []) for m in METRICS}
            originals, saved = {m: getattr(self, m) for m in METRICS}, self.valset
            for m, tap in taps.items():
                setattr(self, m, tap)
            self.valset = dataset
            try:
                super().eval(step, stage)
            finally:
                self.valset = saved
                for m, fn in originals.items():
                    setattr(self, m, fn)
            names = [self.parser.image_names[i] for i in dataset.indices]
            try:
                stats = json.loads(Path(self.stats_dir, f"{stage}_step{step:04d}.json").read_text())
                return per_view(names, {m: t.log for m, t in taps.items()}, stats, bool(self.cfg.use_color_correction_metric))
            except Exception as e:  # upstream's scores stand; only the per-view series is lost
                return {"images": names, "verified": False, "error": repr(e)}

        def eval(self, step: int, stage: str = "val"):
            if stage != "val":
                return super().eval(step, stage)
            sets = {"val": self._scored(step, "val", self.valset)}  # as upstream: a failure here fails the run
            try:
                sets["block"] = self._scored(step, "block", self.blockset)
                Path(self.stats_dir, f"per_view_step{step}.json").write_text(json.dumps({"step": step, "sets": sets}, indent=1))
            except Exception as e:  # the model is saved already: lose the extra scores, not the run
                print(json.dumps({"plyHoldoutBlockError": repr(e)}), flush=True)

    st.Dataset, st.Runner = BlockDataset, BlockRunner
    return state
