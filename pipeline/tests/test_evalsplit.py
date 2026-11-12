"""The held-out block: choosing it, keeping it out of training, scoring it apart, and per-view scores for both sets.

The first half runs anywhere (stand-ins for the trainer's classes). The last test drives the pinned gsplat
examples' real Parser, Dataset and Runner.eval on CPU, and is skipped where torch or the checkout is missing:
    PLY_GSPLAT_DIR=<gsplat checkout at the pinned SHA> python -m pytest tests/test_evalsplit.py
"""

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from plypipe import evalsplit


def index(*clips, start=0):
    rows, n = [], start
    for name, count in clips:
        rows += [{"frame": f"{n + i:06d}.jpg", "clip": name, "t": round(0.5 * i, 2)} for i in range(count)]
        n += count
    return rows


def names_of(rows, missing=()):
    return sorted(r["frame"] for r in rows if r["frame"] not in set(missing))


# ------------------------------------------------------------------ choosing the block

def test_block_is_the_middle_of_the_longest_clip():
    rows = index(("a.mp4", 30), ("b.mp4", 90), ("c.mp4", 40))
    block = evalsplit.pick_block(rows, names_of(rows), 20)
    assert block["clip"] == "b.mp4" and len(block["frames"]) == 20
    assert block["frames"][0] == f"{30 + 35:06d}.jpg" and block["frames"][-1] == f"{30 + 54:06d}.jpg"
    assert block["seconds"] == 9.5 and block["clipFrames"] == 90 and block["dropped"] == []


def test_block_is_deterministic_and_ties_keep_the_first_clip():
    rows = index(("a.mp4", 50), ("b.mp4", 50))
    first = evalsplit.pick_block(rows, names_of(rows), 10)
    assert first == evalsplit.pick_block(rows, list(reversed(names_of(rows))), 10) and first["clip"] == "a.mp4"


def test_frames_dropped_by_the_solve_never_sit_inside_the_block():
    rows = index(("a.mp4", 60))
    gone = {"000029.jpg", "000030.jpg"}                      # exactly the middle
    block = evalsplit.pick_block(rows, names_of(rows, gone), 20)
    assert not gone & set(block["frames"])
    nums = [int(f[:6]) for f in block["frames"]]
    assert nums == list(range(nums[0], nums[0] + 20))        # consecutive in the clip, no gap bridged
    assert block["dropped"] == sorted(gone)
    assert nums[0] == 9                                      # starts 9 and 31 are equally near the middle: the earlier wins


def test_a_gappy_clip_hands_over_to_the_next_longest():
    rows = index(("a.mp4", 60), ("b.mp4", 40))
    solved = [f for f in names_of(rows) if not (int(f[:6]) % 10 == 5 and int(f[:6]) < 60)]   # a gap every 10 frames in a
    block = evalsplit.pick_block(rows, solved, 20)
    assert block["clip"] == "b.mp4"
    with pytest.raises(ValueError, match="no 20 consecutive solved frames"):
        evalsplit.pick_block(index(("a.mp4", 15)), names_of(index(("a.mp4", 15))), 20)


def test_photo_sets_without_an_index_count_as_one_clip():
    names = [f"IMG_{i:04d}.jpg" for i in range(30)]
    block = evalsplit.pick_block(None, names, 10)
    assert block["clip"] is None and block["frames"] == names[10:20] and block["seconds"] is None


def test_read_index_accepts_only_the_frames_stage_format(tmp_path):
    good = tmp_path / "frames_index.json"
    good.write_text(json.dumps(index(("a.mp4", 3))))
    assert len(evalsplit.read_index(good)) == 3
    for bad in ("[]", "{}", '[{"frame": "x"}]', "not json"):
        good.write_text(bad)
        assert evalsplit.read_index(good) is None
    assert evalsplit.read_index(tmp_path / "missing.json") is None


# ------------------------------------------------------------------ per-view scores

def test_per_view_splits_raw_and_colour_corrected_calls_and_checks_the_means():
    calls = {"psnr": [20.0, 24.0, 22.0, 26.0], "ssim": [0.5, 0.6, 0.7, 0.8], "lpips": [0.3, 0.2, 0.4, 0.1]}
    stats = {"psnr": 21.0, "cc_psnr": 25.0, "ssim": 0.6, "cc_ssim": 0.7, "lpips": 0.35, "cc_lpips": 0.15}
    out = evalsplit.per_view(["a.jpg", "b.jpg"], calls, stats, True)
    assert out["psnr"] == [20.0, 22.0] and out["ccPsnr"] == [24.0, 26.0] and out["ccLpips"] == [0.2, 0.1]
    assert out["verified"] is True and "mismatch" not in out
    plain = evalsplit.per_view(["a.jpg", "b.jpg"], {k: v[:2] for k, v in calls.items()}, {"psnr": 22.0, "ssim": 0.55, "lpips": 0.25}, False)
    assert plain["verified"] is True and "ccPsnr" not in plain


def test_per_view_flags_series_that_disagree_with_gsplats_own_means():
    calls = {"psnr": [20.0, 24.0], "ssim": [0.5, 0.6], "lpips": [0.3, 0.2]}
    out = evalsplit.per_view(["a.jpg", "b.jpg"], calls, {"psnr": 30.0, "ssim": 0.55, "lpips": 0.25}, False)
    assert out["verified"] is False and out["mismatch"] == ["psnr"]
    out = evalsplit.per_view(["a.jpg", "b.jpg", "c.jpg"], calls, {"psnr": 22.0, "ssim": 0.55, "lpips": 0.25}, False)
    assert out["verified"] is False and set(out["mismatch"]) == {"psnr", "ssim", "lpips"}     # a view went missing


# ------------------------------------------------------------------ patching the trainer (stand-ins)

class FakeParser:
    def __init__(self, names, test_every):
        self.image_names, self.test_every = names, test_every


class FakeDataset:
    def __init__(self, parser, split="train", patch_size=None, load_depths=False):
        self.parser, self.split = parser, split
        ix = np.arange(len(parser.image_names))
        self.indices = ix[ix % parser.test_every != 0] if split == "train" else ix[ix % parser.test_every == 0]

    def __len__(self):
        return len(self.indices)


def fake_trainer(names, test_every, root, cc=True):
    """Just enough of simple_trainer: Dataset's split rule and Runner.eval's metric call order (raw, then corrected)."""
    seen = {"eval": []}

    class FakeRunner:
        def __init__(self, local_rank, world_rank, world_size, cfg):
            self.cfg, self.world_rank = cfg, world_rank
            self.parser = FakeParser(names, test_every)
            self.trainset = st.Dataset(self.parser, split="train")
            self.valset = st.Dataset(self.parser, split="val")
            self.stats_dir = str(root / "stats")
            os.makedirs(self.stats_dir, exist_ok=True)
            self.psnr = lambda a, b: np.float32(a)
            self.ssim = lambda a, b: np.float32(a / 100)
            self.lpips = lambda a, b: np.float32(a / 1000)

        def eval(self, step, stage="val"):
            seen["eval"].append((stage, len(self.valset)))
            metrics = {"psnr": [], "ssim": [], "lpips": [], "cc_psnr": [], "cc_ssim": [], "cc_lpips": []}
            for i in self.valset.indices:
                metrics["psnr"].append(self.psnr(20 + int(i), 0)); metrics["ssim"].append(self.ssim(20 + int(i), 0)); metrics["lpips"].append(self.lpips(20 + int(i), 0))
                if self.cfg.use_color_correction_metric:
                    metrics["cc_psnr"].append(self.psnr(30 + int(i), 0)); metrics["cc_ssim"].append(self.ssim(30 + int(i), 0)); metrics["cc_lpips"].append(self.lpips(30 + int(i), 0))
            stats = {k: float(np.mean(v)) for k, v in metrics.items() if v}
            Path(self.stats_dir, f"{stage}_step{step:04d}.json").write_text(json.dumps(stats))

    st = types.SimpleNamespace(Dataset=FakeDataset, Runner=FakeRunner)
    cfg = types.SimpleNamespace(result_dir=str(root), use_color_correction_metric=cc)
    return st, cfg, seen


def test_the_block_is_excluded_from_training_and_from_the_every_nth_set(tmp_path):
    rows = index(("a.mp4", 40), ("b.mp4", 20))
    names = names_of(rows)
    st, cfg, seen = fake_trainer(names, 10, tmp_path)
    state = evalsplit.install(st, 8, rows)
    runner = st.Runner(0, 0, 1, cfg)
    block = {int(f[:6]) for f in state["block"]["frames"]}
    assert len(block) == 8 and state["block"]["clip"] == "a.mp4"
    train, val, held = (set(map(int, d.indices)) for d in (runner.trainset, runner.valset, runner.blockset))
    assert held == block and not block & train and not block & val
    assert train | val | held == set(range(60)) and not train & val
    assert val == {i for i in range(60) if i % 10 == 0} - block    # gsplat's own every-Nth rule, minus any overlap


def test_the_runner_scores_both_sets_and_writes_per_view_files(tmp_path):
    rows = index(("a.mp4", 40))
    st, cfg, seen = fake_trainer(names_of(rows), 10, tmp_path)
    evalsplit.install(st, 6, rows)
    runner = st.Runner(0, 0, 1, cfg)
    runner.eval(99)
    assert [s for s, _ in seen["eval"]] == ["val", "block"] and seen["eval"][1][1] == 6
    assert (tmp_path / "stats" / "val_step0099.json").exists() and (tmp_path / "stats" / "block_step0099.json").exists()
    per = json.loads((tmp_path / "stats" / "per_view_step99.json").read_text())
    assert per["step"] == 99 and set(per["sets"]) == {"val", "block"}
    assert per["sets"]["val"]["verified"] and per["sets"]["block"]["verified"]
    assert per["sets"]["block"]["images"] == json.loads((tmp_path / "ply_block.json").read_text())["frames"]
    assert per["sets"]["block"]["ccPsnr"][0] == 30 + int(per["sets"]["block"]["images"][0][:6])
    assert runner.valset is not runner.blockset and len(runner.valset) == 3       # swapped back after the block pass (block 17-22 took 20)
    assert isinstance(runner.psnr, type(lambda: 0))                               # the metric objects are unwrapped again
    runner.eval(100, stage="other")                                                # other stages pass straight through
    assert seen["eval"][-1] == ("other", 3)


def _failing_on(stage, tmp_path):
    rows = index(("a.mp4", 40))
    st, cfg, _ = fake_trainer(names_of(rows), 10, tmp_path)
    original = st.Runner.eval

    def eval_that_fails(self, step, stage_name="val"):
        if stage_name == stage:
            raise RuntimeError(f"{stage} pass failed")
        original(self, step, stage_name)

    st.Runner.eval = eval_that_fails
    evalsplit.install(st, 6, rows)
    return st.Runner(0, 0, 1, cfg)


def test_a_failing_block_pass_keeps_the_run(tmp_path, capsys):
    _failing_on("block", tmp_path).eval(5)
    assert "plyHoldoutBlockError" in capsys.readouterr().out and not (tmp_path / "stats" / "per_view_step5.json").exists()
    assert (tmp_path / "stats" / "val_step0005.json").exists()


def test_a_failing_val_pass_fails_the_run_as_it_did_before(tmp_path):
    with pytest.raises(RuntimeError, match="val pass failed"):
        _failing_on("val", tmp_path).eval(5)


def test_a_split_that_does_not_add_up_stops_the_run_before_training(tmp_path):
    rows = index(("a.mp4", 40))
    st, cfg, _ = fake_trainer(names_of(rows), 10, tmp_path)
    evalsplit.install(st, 6, rows)
    original = st.Dataset.__init__
    def lose_a_view(self, parser, split="train", *a, **k):
        original(self, parser, split, *a, **k)
        if split == "train":
            self.indices = self.indices[1:]
    st.Dataset.__init__ = lose_a_view
    with pytest.raises(RuntimeError, match="does not add up"):
        st.Runner(0, 0, 1, cfg)


# ------------------------------------------------------------------ the pinned trainer itself, on CPU (conftest.py: trainer)

def test_pinned_gsplat_parser_dataset_and_eval_loop_on_cpu(trainer, tmp_path, monkeypatch):
    """Real Parser, Dataset and Runner.eval from the pinned simple_trainer; only the renderer and the three metrics
    are stand-ins. Proves the patch fits upstream's split rule, its eval loop's call order and its stats files."""
    import torch
    from gs_scene import write_trainable_scene
    st = trainer
    dataset, runner_cls = st.Dataset, st.Runner
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    names = write_trainable_scene(tmp_path / "scene", n_cams=40, clips=(28, 16), dropped=(30,))
    rows = json.loads((tmp_path / "scene" / "frames_index.json").read_text())
    try:
        state = evalsplit.install(st, 8, rows)
        parser = st.Parser(data_dir=str(tmp_path / "scene" / "data"), factor=1, normalize=False, test_every=10)
        assert list(parser.image_names) == names
        train, val, held = (st.Dataset(parser, split=s) for s in ("train", "val", "block"))
        assert len(train) + len(val) + len(held) == 40 and len(held) == 8
        block = state["block"]
        assert [parser.image_names[i] for i in held.indices] == block["frames"] and block["clip"] == "clip_0.mp4"
        assert block["dropped"] == []

        class Writer:
            def add_scalar(self, *a): pass
            def flush(self): pass

        class Stage:
            def render(self, scene_id, camtoworlds, Ks, width, height, **kw):
                tint = 0.4 + 0.02 * camtoworlds[:, 0, 3].reshape(-1, 1, 1, 1)
                return torch.full((1, height, width, 3), 0.5) * 0 + tint, None, None

        mse = lambda a, b: ((a - b) ** 2).mean()
        runner = object.__new__(st.Runner)
        runner.__dict__.update(
            cfg=types.SimpleNamespace(use_color_correction_metric=True, color_correct_method="affine", sh_degree=1, near_plane=0.01,
                                      far_plane=1e6, result_dir=str(tmp_path / "result")),
            device="cpu", world_rank=0, world_size=1, parser=parser, trainset=train, valset=val, blockset=held,
            stats_dir=str(tmp_path / "result" / "stats"), render_dir=str(tmp_path / "result" / "renders"), writer=Writer(),
            scene=types.SimpleNamespace(id="s"), splats={"means": torch.zeros(5, 3)}, stage=Stage(),
            psnr=lambda a, b: -10 * torch.log10(mse(a, b)), ssim=lambda a, b: 1 - mse(a, b), lpips=lambda a, b: mse(a, b))
        os.makedirs(runner.stats_dir), os.makedirs(runner.render_dir)
        runner.eval(step=7)
        stats = tmp_path / "result" / "stats"
        assert {p.name for p in stats.glob("*.json")} == {"val_step0007.json", "block_step0007.json", "per_view_step7.json"}
        per = json.loads((stats / "per_view_step7.json").read_text())["sets"]
        assert per["val"]["verified"] and per["block"]["verified"]
        assert per["block"]["images"] == block["frames"] and len(per["val"]["images"]) == len(val)
        assert len(per["block"]["ccPsnr"]) == 8 and len(list((tmp_path / "result" / "renders").glob("block_step7_*.png"))) == 8
        assert len(list((tmp_path / "result" / "renders").glob("val_step7_*.png"))) == len(val)
    finally:
        st.Dataset, st.Runner = dataset, runner_cls
