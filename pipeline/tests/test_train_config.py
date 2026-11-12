"""The GPU trainer cannot run here; check the config the launcher hands to gsplat's simple_trainer."""

import sys
import types

import pytest

from plypipe import train_launcher


class FakeConfig:
    means_lr = 1.6e-4

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.steps_scaler = 1.0


class FakeMCMC:
    def __init__(self, verbose=False, cap_max=1_000_000, refine_stop_iter=25_000, noise_lr=5e5):
        self.verbose, self.cap_max, self.refine_stop_iter, self.noise_lr = verbose, cap_max, refine_stop_iter, noise_lr


st = types.SimpleNamespace(Config=FakeConfig, MCMCStrategy=FakeMCMC)


@pytest.mark.parametrize("preset", ["quick", "standard", "high"])
def test_presets(preset):
    cfg, info = train_launcher.build_config(st, "/d", "/r", preset, None, False)
    p = train_launcher.PRESETS[preset]
    assert cfg.max_steps == p["max_steps"] and cfg.ply_steps == [p["max_steps"]] and cfg.save_ply
    assert cfg.normalize_world_space is False          # train in the aligned scene frame
    assert cfg.strategy.cap_max == p["cap_max"] and cfg.sh_degree == p["sh_degree"]
    assert cfg.disable_viewer and cfg.disable_video and cfg.eval_steps == []
    assert (cfg.post_processing == "ppisp") == p["ppisp"]
    if p["ppisp"]:
        assert cfg.ppisp_controller_activation_num_steps < cfg.max_steps


def test_ppisp_override():
    cfg, info = train_launcher.build_config(st, "/d", "/r", "standard", False, True)
    assert cfg.post_processing is None and info["ppisp"] is False and cfg.antialiased


def test_advanced_overrides_beat_the_preset():
    """The capture app's Advanced settings arrive as --iterations / --sh-degree / --cap-max."""
    cfg, info = train_launcher.build_config(st, "/d", "/r", "quick", None, False, iterations=12_000, sh_degree=3, cap_max=900_000)
    assert cfg.max_steps == 12_000 and cfg.ply_steps == [12_000] and cfg.sh_degree == 3 and cfg.strategy.cap_max == 900_000
    assert info["maxSteps"] == 12_000 and info["capMax"] == 900_000


def test_gsplat_without_ppisp_falls_back_to_appearance_model():
    """install_fast.sh trains with prebuilt gsplat 1.5.3, whose Config has no post_processing field."""
    import dataclasses

    @dataclasses.dataclass
    class OldConfig:
        strategy: object = None
        init_opa: float = 0.1
        init_scale: float = 1.0
        opacity_reg: float = 0.0
        scale_reg: float = 0.0
        app_opt: bool = False
        steps_scaler: float = 1.0

    old = types.SimpleNamespace(Config=OldConfig, MCMCStrategy=FakeMCMC)
    # PPISP asked for explicitly (it is off by default) on a gsplat without it: the appearance embedding stands in
    cfg, info = train_launcher.build_config(old, "/d", "/r", "standard", True, False)
    assert cfg.app_opt is True and info["ppisp"] is False and not hasattr(cfg, "post_processing")
    cfg, info = train_launcher.build_config(old, "/d", "/r", "standard", None, False)
    assert cfg.app_opt is False and info["ppisp"] is False


@pytest.mark.parametrize("steps", [7_000, 18_000, 30_000, 55_000])
def test_mcmc_stops_relocating_before_the_end(steps):
    """gsplat stops relocation at 25k of 30k; a fixed 25k never stops an 18k run, so it scales with the run."""
    cfg, _ = train_launcher.build_config(st, "/d", "/r", "standard", None, True, iterations=steps)
    assert cfg.strategy.refine_stop_iter == int(steps * 0.83) < steps


def test_holdout_scores_the_last_step_only_when_asked():
    cfg, info = train_launcher.build_config(st, "/d", "/r", "standard", None, True, iterations=30_000, holdout=30)
    assert cfg.test_every == 30 and cfg.eval_steps == [30_000] and cfg.use_color_correction_metric
    assert info["holdout"] == 30 and not cfg.pose_opt and not cfg.depth_loss
    cfg, info = train_launcher.build_config(st, "/d", "/r", "standard", None, True, iterations=30_000)
    assert cfg.test_every == 10**9 and cfg.eval_steps == [] and info["holdout"] == 0


def test_ab_options_reach_gsplat():
    cfg, info = train_launcher.build_config(st, "/d", "/r", "quick", None, True, pose_opt=True, depth_loss=True)
    assert cfg.pose_opt and cfg.depth_loss and info["poseOpt"] and info["depthLoss"]


def _depth_term(torch, loss, rendered, points, depths_gt):
    """What simple_trainer does around the loss: expected depth (2 m here), sampled bilinearly at the track points."""
    alpha = rendered.clone().requires_grad_(True)
    acc = (rendered * 2.0).requires_grad_(True)
    depth = acc / alpha.clamp(min=1e-10)
    height, width = depth.shape[1:3]
    grid = torch.stack([points[..., 0] / (width - 1) * 2 - 1, points[..., 1] / (height - 1) * 2 - 1], dim=-1).unsqueeze(2)
    sampled = torch.nn.functional.grid_sample(depth.permute(0, 3, 1, 2), grid, align_corners=True).squeeze(3).squeeze(1)
    value = loss(sampled, depths_gt, scene_scale=1.1)
    value.backward()
    return value.item(), acc.grad, alpha.grad


def test_depth_loss_keeps_gsplats_value_with_a_finite_gradient_beside_an_empty_pixel():
    torch = pytest.importorskip("torch")

    def gsplats(pred, gt, scene_scale=1.0):  # gsplat/losses.py at the pinned commit
        disp = torch.where(pred > 0.0, 1.0 / pred, torch.zeros_like(pred))
        disp_gt = torch.where(gt > 0.0, 1.0 / gt, torch.zeros_like(gt))
        return torch.nn.functional.l1_loss(disp, disp_gt) * scene_scale

    rendered = torch.zeros(1, 9, 9, 1)
    rendered[0, 3, 4, 0] = 0.5  # the only pixel a splat reaches
    # the second track point sits on column 3 exactly, between two empty pixels: its weight for the rendered pixel is 0
    points, depths = torch.tensor([[[4.0, 3.0], [3.0, 3.5]]]), torch.tensor([[2.5, 2.0]])
    theirs, their_acc, their_alpha = _depth_term(torch, gsplats, rendered, points, depths)
    assert torch.isnan(their_acc[0, 3, 4, 0]) and torch.isnan(their_alpha[0, 3, 4, 0])  # the defect being guarded against
    ours, acc, alpha = _depth_term(torch, train_launcher.depth_l1_loss, rendered, points, depths)
    assert ours == theirs and ours > 0
    assert torch.isfinite(acc).all() and torch.isfinite(alpha).all()
    assert acc[0, 3, 4, 0] != 0  # the track point on the rendered pixel still pulls its depth


def test_depth_loss_of_a_view_with_no_track_point_in_frame_is_zero():
    torch = pytest.importorskip("torch")
    rendered = torch.full((1, 9, 9, 1), 0.5)
    value, acc, alpha = _depth_term(torch, train_launcher.depth_l1_loss, rendered, torch.zeros(1, 0, 2), torch.zeros(1, 0))
    assert value == 0.0 and torch.isfinite(acc).all() and torch.isfinite(alpha).all()


def _launch(tmp_path, monkeypatch, *flags, gpus=1, trainer_has_the_loss=True, trainer_builds_splats=True):
    """main() up to the hand-over to gsplat, with stand-ins for simple_trainer and torch. Returns the stand-in trainer."""
    class Config(FakeConfig):
        def adjust_steps(self, factor):
            pass

    trainer = types.ModuleType("simple_trainer")
    trainer.Config, trainer.MCMCStrategy, trainer.main, trainer.cli = Config, FakeMCMC, None, lambda fn, cfg, verbose: None
    if trainer_has_the_loss:
        trainer.depth_l1_loss = "gsplat's"
    if trainer_builds_splats:
        trainer.create_splats_with_optimizers = "gsplat's"
    monkeypatch.setitem(sys.modules, "simple_trainer", trainer)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: gpus)))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    train_launcher.main(["--examples", str(tmp_path), "--data", str(tmp_path / "d"), "--result", str(tmp_path / "r"),
                         "--preset", "quick", *flags])
    return trainer


def test_depth_loss_replaces_gsplats_loss_only_when_asked(tmp_path, monkeypatch):
    assert _launch(tmp_path, monkeypatch).depth_l1_loss == "gsplat's"
    assert _launch(tmp_path, monkeypatch, "--depth-loss").depth_l1_loss is train_launcher.depth_l1_loss


def test_depth_loss_refuses_to_train_where_the_replacement_would_not_hold(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="one visible GPU"):  # gsplat's spawned workers import their own trainer
        _launch(tmp_path, monkeypatch, "--depth-loss", gpus=2)
    with pytest.raises(SystemExit, match="gsplat pin"):
        _launch(tmp_path, monkeypatch, "--depth-loss", trainer_has_the_loss=False)


def test_held_out_scores_are_read_from_gsplat_stats(tmp_path):
    import json
    from plypipe.stages import train as train_stage
    res = tmp_path / "train"
    (res / "stats").mkdir(parents=True)
    (res / "renders").mkdir()
    (res / "stats" / "val_step29999.json").write_text(json.dumps(
        {"psnr": 28.51234, "ssim": 0.8912, "lpips": 0.231, "cc_psnr": 29.1, "cc_ssim": 0.9, "cc_lpips": 0.22, "num_GS": 812345}))
    for i in range(9):
        (res / "renders" / f"val_step29999_{i:04d}.png").write_bytes(b"png")
    (res / "renders" / "val_step6999_0000.png").write_bytes(b"older run")
    out = train_stage.held_out_scores(res, tmp_path / "eval")
    assert out["psnr"] == 28.5123 and out["ccPsnr"] == 29.1 and out["gaussians"] == 812345 and out["views"] == 9
    kept = sorted(p.name for p in (tmp_path / "eval").glob("*.png"))
    assert len(kept) == 4 and all(n.startswith("val_step29999_") for n in kept)
    assert train_stage.held_out_scores(tmp_path / "missing", tmp_path / "eval2") is None


def test_mcmc_noise_follows_the_room_size():
    """gsplat's MCMC noise is lr x covariance: metres cubed. A 3 m room must get the unit-scene recipe, not 9x it."""
    unit, _ = train_launcher.build_config(st, "/d", "/r", "standard", None, True)
    room, info = train_launcher.build_config(st, "/d", "/r", "standard", None, True, world_scale=3.0)
    assert unit.strategy.noise_lr == 5e5 and unit.scale_reg == 0.01
    assert abs(room.strategy.noise_lr - 5e5 / 9) < 1e-6 and abs(room.scale_reg - 0.01 / 3) < 1e-12
    assert info["worldScale"] == 3.0


def test_camera_spread_flags_a_runaway_view(tmp_path):
    import numpy as np
    from plypipe.stages import train as train_stage
    from synth import write_synthetic_model
    write_synthetic_model(tmp_path / "m")
    spread = train_stage.camera_spread(tmp_path / "m")
    assert spread["medianM"] > 0 and spread["maxOverMedian"] < 10


def test_bilateral_grid_replaces_ppisp_and_random_backgrounds_reach_gsplat():
    import dataclasses

    @dataclasses.dataclass
    class NewConfig:
        strategy: object = None
        init_opa: float = 0.1
        init_scale: float = 1.0
        opacity_reg: float = 0.0
        scale_reg: float = 0.0
        post_processing: object = None
        app_opt: bool = False

    st2 = types.SimpleNamespace(Config=NewConfig, MCMCStrategy=FakeMCMC)
    cfg, info = train_launcher.build_config(st2, "/d", "/r", "high", None, True, bilateral_grid=True, random_bkgd=True)
    assert cfg.post_processing == "bilateral_grid" and cfg.random_bkgd and info["appearance"] == "bilateral_grid"
    assert info["ppisp"] is False and info["randomBkgd"] is True
    cfg, info = train_launcher.build_config(st2, "/d", "/r", "high", None, True)
    assert cfg.post_processing is None and not cfg.random_bkgd           # no appearance model by default
    cfg, info = train_launcher.build_config(st2, "/d", "/r", "high", True, True)
    assert cfg.post_processing == "ppisp" and info["ppisp"] is True


# ------------------------------------------------------------------ the held-out block and opacity options

def test_defaults_are_exactly_what_the_launcher_built_before_the_block_and_opacity_options():
    """Golden values from the launcher before --holdout-block and --opacity-reg existed: nothing may move while they are off."""
    cfg, info = train_launcher.build_config(st, "/d", "/r", "high", None, True, iterations=15_000, cap_max=2_000_000, holdout=30)
    assert {**vars(cfg), "strategy": vars(cfg.strategy)} == {
        "antialiased": True, "app_opt": False, "data_dir": "/d", "data_factor": 1, "depth_loss": False, "disable_video": True,
        "disable_viewer": True, "eval_steps": [15000], "init_opa": 0.5, "init_scale": 0.1, "max_steps": 15000,
        "normalize_world_space": False, "opacity_reg": 0.01, "ply_steps": [15000], "pose_opt": False, "post_processing": None,
        "random_bkgd": False, "result_dir": "/r", "save_ply": True, "save_steps": [15000], "scale_reg": 0.01, "sh_degree": 3,
        "steps_scaler": 1.0, "test_every": 30, "use_color_correction_metric": True,
        "strategy": {"cap_max": 2000000, "noise_lr": 500000.0, "refine_stop_iter": 12450, "verbose": True}}
    assert info == {"antialiased": True, "appearance": None, "capMax": 2000000, "depthLoss": False, "holdout": 30, "maxSteps": 15000,
                    "poseOpt": False, "ppisp": False, "preset": "high", "randomBkgd": False, "shDegree": 3, "worldScale": 1.0}


def test_opacity_regulariser_reaches_gsplat_and_is_reported_only_when_changed():
    cfg, info = train_launcher.build_config(st, "/d", "/r", "high", None, True, holdout=30, opacity_reg=0.005)
    assert cfg.opacity_reg == 0.005 and info["opacityReg"] == 0.005
    cfg, info = train_launcher.build_config(st, "/d", "/r", "high", None, True, holdout=30, opacity_reg=0.01)
    assert cfg.opacity_reg == 0.01 and "opacityReg" not in info


def test_the_block_needs_the_every_nth_holdout_and_is_reported():
    cfg, info = train_launcher.build_config(st, "/d", "/r", "high", None, True, holdout=30, holdout_block=20)
    assert info["holdoutBlock"] == 20 and cfg.eval_steps == [cfg.max_steps] and cfg.use_color_correction_metric
    assert "holdoutBlock" not in train_launcher.build_config(st, "/d", "/r", "high", None, True, holdout=30)[1]
    with pytest.raises(ValueError, match="--holdout"):
        train_launcher.build_config(st, "/d", "/r", "high", None, True, holdout=0, holdout_block=20)


class _Captured(Exception):
    pass


def _stage_command(tmp_path, monkeypatch, **opts):
    from plypipe.log import Log
    from plypipe.stages import train as train_stage
    from plypipe.util import Ctx
    (tmp_path / "gs" / "examples").mkdir(parents=True, exist_ok=True)
    (tmp_path / "gs" / "examples" / "simple_trainer.py").write_text("")
    seen = {}

    def capture(cmd, ctx, stage, **kw):
        seen["cmd"] = [str(c) for c in cmd]
        raise _Captured

    monkeypatch.setattr(train_stage, "run", capture)
    ctx = Ctx(out=tmp_path / "job", log=Log(tmp_path / "job", quiet=True), opts={"gsplat_dir": str(tmp_path / "gs"), "preset": "high", **opts})
    with pytest.raises(_Captured):
        train_stage.run_stage(ctx)
    return seen["cmd"]


def test_train_stage_command_line_is_unchanged_without_the_new_options(tmp_path, monkeypatch):
    cmd = _stage_command(tmp_path, monkeypatch)
    assert [c for c in cmd if c.startswith("--")] == ["--examples", "--data", "--result", "--preset", "--iterations", "--cap-max",
                                                      "--antialiased", "--holdout"]
    assert _stage_command(tmp_path, monkeypatch, exposure_fold=None) == cmd
    cmd = _stage_command(tmp_path, monkeypatch, holdout_block=0, opacity_reg=None)
    assert "--holdout-block" not in cmd and "--opacity-reg" not in cmd and "--frames-index" not in cmd


def test_train_stage_passes_the_block_and_opacity_options_on(tmp_path, monkeypatch):
    cmd = _stage_command(tmp_path, monkeypatch, holdout_block=20, opacity_reg=0.005, iterations=15_000)
    assert cmd[cmd.index("--holdout") + 1] == "30" and cmd[cmd.index("--holdout-block") + 1] == "20"
    assert cmd[cmd.index("--frames-index") + 1] == str(tmp_path / "job" / "frames_index.json")
    assert cmd[cmd.index("--opacity-reg") + 1] == "0.005" and cmd[cmd.index("--iterations") + 1] == "15000"
    with pytest.raises(RuntimeError, match="needs --holdout"):
        _stage_command(tmp_path, monkeypatch, holdout_block=20, holdout=0)


def test_train_report_carries_both_score_sets_the_block_images_and_the_per_view_file(tmp_path):
    import json
    from plypipe.stages import train as train_stage
    res = tmp_path / "train"
    (res / "stats").mkdir(parents=True)
    (res / "renders").mkdir()
    for stage, n, psnr in (("val", 4, 25.0), ("block", 20, 22.5)):
        (res / "stats" / f"{stage}_step14999.json").write_text(json.dumps(
            {"psnr": psnr, "ssim": 0.8, "lpips": 0.3, "cc_psnr": psnr + 0.5, "cc_ssim": 0.81, "cc_lpips": 0.29, "num_GS": 900000, "ellipse_time": 0.01}))
        for i in range(n):
            (res / "renders" / f"{stage}_step14999_{i:04d}.png").write_bytes(b"png")
    (res / "stats" / "per_view_step14999.json").write_text(json.dumps({"step": 14999, "sets": {}}))
    (res / "ply_block.json").write_text(json.dumps({"clip": "a.mp4", "frames": ["000001.jpg", "000002.jpg"], "seconds": 0.5}))
    out = train_stage.block_report(res, tmp_path / "eval")
    assert out["evalBlock"]["ccPsnr"] == 23.0 and out["evalBlock"]["views"] == 20 and out["evalBlock"]["gaussians"] == 900000
    assert out["blockImages"] == ["000001.jpg", "000002.jpg"] and out["blockClip"] == "a.mp4" and out["perView"] == "eval/per_view.json"
    assert train_stage.held_out_scores(res, tmp_path / "eval")["views"] == 4           # the every-Nth set is read as before
    assert (tmp_path / "eval" / "per_view.json").exists() and (tmp_path / "eval" / "block_stats.json").exists()
    assert sorted(p.name for p in (tmp_path / "eval").glob("block_step*.png")) == [f"block_step14999_{i:04d}.png" for i in (0, 5, 10, 15)]
    assert train_stage.block_report(tmp_path / "nothing", tmp_path / "eval2") == {}


# ------------------------------------------------------------------ the phone file's fine-tune (plypipe.finetune)

def test_a_fine_tune_caps_the_splats_at_the_models_count_and_trains_every_sh_band_from_the_start():
    cfg, summary = train_launcher.build_config(st, "/d", "/r", "high", False, True, iterations=3000, sh_degree=2, cap_max=2_000_000,
                                               from_splats=240_000)
    assert cfg.strategy.cap_max == 240_000 == summary["capMax"] == summary["fromSplats"]     # the model's count beats --cap-max
    assert cfg.strategy.refine_stop_iter == int(3000 * 0.83) and cfg.max_steps == 3000 and cfg.sh_degree == 2
    assert cfg.sh_degree_interval == 1 and cfg.means_lr == pytest.approx(1.6e-5)
    cfg, summary = train_launcher.build_config(st, "/d", "/r", "high", False, True)          # a fresh run is what it was
    assert "fromSplats" not in summary and not hasattr(cfg, "sh_degree_interval") and cfg.means_lr == 1.6e-4


def test_the_launcher_starts_from_a_model_only_when_asked_and_only_where_the_patch_holds(tmp_path, monkeypatch):
    import json
    from test_finetune import write_model
    write_model(tmp_path / "lite.ply", n=7, k=8)
    assert _launch(tmp_path, monkeypatch).create_splats_with_optimizers == "gsplat's"
    trainer = _launch(tmp_path, monkeypatch, "--from-ply", str(tmp_path / "lite.ply"), "--sh-degree", "2", "--iterations", "300")
    assert callable(trainer.create_splats_with_optimizers)
    preset = json.loads((tmp_path / "r" / "ply_preset.json").read_text())
    assert preset["fromSplats"] == 7 == preset["capMax"] and preset["maxSteps"] == 300
    with pytest.raises(SystemExit, match="--from-ply: needs one visible GPU"):
        _launch(tmp_path, monkeypatch, "--from-ply", str(tmp_path / "lite.ply"), "--sh-degree", "2", gpus=2)
    with pytest.raises(SystemExit, match="gsplat pin"):
        _launch(tmp_path, monkeypatch, "--from-ply", str(tmp_path / "lite.ply"), "--sh-degree", "2", trainer_builds_splats=False)
    with pytest.raises(ValueError, match="SH degree 3 needs 15"):                             # quick is SH 2; the model has no third band
        _launch(tmp_path, monkeypatch, "--from-ply", str(tmp_path / "lite.ply"), "--sh-degree", "3")


def test_the_fine_tune_runs_on_the_jobs_own_split_and_returns_the_trainers_model_and_scores(tmp_path, monkeypatch):
    import json
    from plypipe.log import Log
    from plypipe.stages import train as train_stage
    from plypipe.util import Ctx
    (tmp_path / "gs" / "examples").mkdir(parents=True)
    (tmp_path / "gs" / "examples" / "simple_trainer.py").write_text("")
    job, seen = tmp_path / "job", {}

    def trainer(cmd, ctx, stage, **kw):
        seen.update(cmd=[str(c) for c in cmd], stage=stage, env=kw["env"])
        result = job / "train-lite"
        for d in ("ply", "stats", "renders"):
            (result / d).mkdir(parents=True)
        (result / "ply" / "point_cloud_149.ply").write_bytes(b"part-way")
        (result / "ply" / "point_cloud_2999.ply").write_bytes(b"fitted")
        for name, psnr in (("val", 26.0), ("block", 23.0)):
            (result / "stats" / f"{name}_step2999.json").write_text(json.dumps({"psnr": psnr, "ssim": 0.8, "lpips": 0.3, "num_GS": 240000}))
            (result / "renders" / f"{name}_step2999_0000.png").write_bytes(b"png")

    monkeypatch.setattr(train_stage, "run", trainer)
    opts = {"gsplat_dir": str(tmp_path / "gs"), "preset": "high", "holdout": 8, "holdout_block": 10, "opacity_reg": 0.005,
            "pose_opt": True, "depth_loss": True, "bilateral_grid": True, "ppisp": True, "iterations": 15_000, "cap_max": 600_000}
    ctx = Ctx(out=job, log=Log(job, quiet=True), opts=opts)
    (job / "train-lite").mkdir()
    (job / "train-lite" / "stale.pt").write_bytes(b"an earlier attempt")
    out = train_stage.finetune(ctx, job / "scene_lite.ply", 3000, 2)
    cmd = seen["cmd"]
    value = lambda flag: cmd[cmd.index(flag) + 1]
    assert cmd[1:3] == ["-m", "plypipe.train_launcher"] and seen["stage"] == "export" and seen["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert seen["env"]["PLY_LIVE_SNAPSHOTS"] == ""                                  # the worker's part-way PLYs are for the train stage's live view
    assert value("--result") == str(job / "train-lite") and value("--from-ply") == str(job / "scene_lite.ply") and value("--preset") == "high"
    assert (value("--iterations"), value("--sh-degree"), value("--holdout"), value("--holdout-block")) == ("3000", "2", "8", "10")
    assert value("--frames-index") == str(job / "frames_index.json") and value("--opacity-reg") == "0.005" and "--antialiased" in cmd
    # the viewer applies no appearance model, and the model was fitted to the solved poses: none of that is trained again
    assert "--no-ppisp" in cmd and not {"--ppisp", "--pose-opt", "--depth-loss", "--bilateral-grid", "--cap-max"} & set(cmd)
    assert out["ply"] == job / "train-lite" / "ply" / "point_cloud_2999.ply" and not (job / "train-lite" / "stale.pt").exists()
    assert out["eval"]["psnr"] == 26.0 and out["eval"]["gaussians"] == 240000 and out["evalBlock"]["psnr"] == 23.0
    assert (job / "eval" / "lite" / "val_stats.json").exists() and (job / "eval" / "lite" / "block_stats.json").exists()
    with pytest.raises(RuntimeError, match="left no PLY"):
        monkeypatch.setattr(train_stage, "run", lambda *a, **k: None)
        train_stage.finetune(ctx, job / "scene_lite.ply", 3000, 2)


# ------------------------------------------------------------------ the exposure fold (plypipe.exposure, variant Q1)

FIT = {"views": 44, "pixels": 5_000_000, "usable": 0.83, "condition": 512.3, "matrix": [[0.84, 0.03, -0.02], [0.02, 0.81, 0.04], [-0.03, 0.05, 0.86]],
       "offset": [0.02, -0.01, 0.015], "gains": [0.85, 0.87, 0.88], "maeBefore": 0.07, "maeAfter": 0.03, "brightnessBefore": 1.21, "brightnessAfter": 1.0}


def _fold_job(tmp_path, monkeypatch, fit, settle_fails=False, names_the_block=True, **opts):
    """run_stage with a stand-in trainer (it leaves a model, its scores and the block's list) and a stand-in fit.
    Returns the stage's report, the launcher's command lines, the fit's calls and the job folder."""
    import json
    from pathlib import Path
    from test_finetune import write_model
    from plypipe import exposure
    from plypipe.log import Log
    from plypipe.stages import train as train_stage
    from plypipe.util import Ctx
    (tmp_path / "gs" / "examples").mkdir(parents=True, exist_ok=True)
    (tmp_path / "gs" / "examples" / "simple_trainer.py").write_text("")
    job, cmds, fits = tmp_path / "job", [], []

    def trainer(cmd, ctx, stage, **kw):
        cmd = [str(c) for c in cmd]
        cmds.append(cmd)
        result, steps, settle = Path(cmd[cmd.index("--result") + 1]), int(cmd[cmd.index("--iterations") + 1]), "--from-ply" in cmd
        if settle and settle_fails:
            raise RuntimeError("python exited with 1; last output:\nCUDA out of memory")
        for d in ("ply", "stats", "renders"):
            (result / d).mkdir(parents=True)
        write_model(result / "ply" / f"point_cloud_{steps - 1}.ply", n=6, k=15, offset=500.0 if settle else 0.0, normals=False)
        (result / "ply_preset.json").write_text(json.dumps({"maxSteps": steps, "appearance": None if settle else "bilateral_grid"}))
        if names_the_block:
            (result / "ply_block.json").write_text(json.dumps({"clip": "a.mp4", "frames": ["000004.jpg", "000005.jpg"], "seconds": 0.5}))
        for name, psnr in (("val", 26.0), ("block", 23.0)):
            (result / "stats" / f"{name}_step{steps - 1}.json").write_text(json.dumps({"psnr": psnr + settle, "ssim": 0.8, "lpips": 0.3, "num_GS": 6}))
            (result / "renders" / f"{name}_step{steps - 1}_0000.png").write_bytes(b"png")
        (result / "stats" / f"per_view_step{steps - 1}.json").write_text(json.dumps({"step": steps - 1, "sets": {}}))

    def fitted(ply, data_dir, every, block, antialiased=True):
        fits.append((Path(ply), Path(data_dir), every, list(block), antialiased))
        if isinstance(fit, Exception):
            raise fit
        return fit

    monkeypatch.setattr(train_stage, "run", trainer)
    monkeypatch.setattr(exposure, "fit_views", fitted)
    ctx = Ctx(out=job, log=Log(job, quiet=True), opts={"gsplat_dir": str(tmp_path / "gs"), "preset": "high", "iterations": 300, **opts})
    return train_stage.run_stage(ctx), cmds, fits, job


def test_the_cli_takes_the_exposure_folds_steps_and_leaves_it_off_by_default():
    from plypipe import cli
    parse = lambda *args: cli.build_parser().parse_args(["run", "in", "--out", "out", *args])
    assert parse().exposure_fold is None and parse("--bilateral-grid").exposure_fold is None
    assert parse("--bilateral-grid", "--exposure-fold", "2000").exposure_fold == 2000 and parse("--exposure-fold", "0").exposure_fold == 0


def test_exposure_fold_trains_as_before_then_folds_the_fitted_map_and_settles_without_an_appearance_model(tmp_path, monkeypatch):
    import json
    import numpy as np
    from plypipe import exposure, finetune
    opts = {"bilateral_grid": True, "holdout": 8, "holdout_block": 10, "opacity_reg": 0.005, "pose_opt": True, "depth_loss": True}
    plain = _stage_command(tmp_path, monkeypatch, iterations=300, **opts)
    report, cmds, fits, job = _fold_job(tmp_path, monkeypatch, FIT, exposure_fold=200, **opts)
    train = job / "train"
    assert len(cmds) == 2 and cmds[0] == plain and "--bilateral-grid" in plain              # the training itself is what it was without the fold
    # the fit sees the model the trainer wrote, the job's hold-out and the block the trainer named
    assert fits == [(train / "ply" / "point_cloud_299.ply", job / "data", 8, ["000004.jpg", "000005.jpg"], True)]
    was, now = finetune.read_splats(train / "ply" / "point_cloud_299.ply", 3), finetune.read_splats(train / "folded.ply", 3)
    sh0, shN = exposure.fold(was["sh0"], was["shN"], FIT["matrix"], FIT["offset"])
    assert np.array_equal(now["sh0"], sh0) and np.array_equal(now["shN"], shN) and np.array_equal(now["means"], was["means"])
    cmd = cmds[1]
    value = lambda flag: cmd[cmd.index(flag) + 1]
    assert cmd[1:3] == ["-m", "plypipe.train_launcher"] and value("--result") == str(train / "settle") and value("--from-ply") == str(train / "folded.ply")
    assert (value("--iterations"), value("--sh-degree"), value("--holdout"), value("--holdout-block"), value("--opacity-reg")) == ("200", "3", "8", "10", "0.005")
    assert "--no-ppisp" in cmd and "--antialiased" in cmd and not {"--bilateral-grid", "--ppisp", "--pose-opt", "--depth-loss", "--cap-max"} & set(cmd)
    assert (job / "scene.ply").read_bytes() == (train / "settle" / "ply" / "point_cloud_199.ply").read_bytes()   # the settled model ships
    fold = report["exposureFold"]
    assert {k: fold[k] for k in FIT} == FIT and (fold["steps"], fold["folded"], fold["settled"]) == (200, True, True) and "why" not in fold
    assert fold["fitSeconds"] >= 0 and fold["settleSeconds"] >= 0
    # the stage's scores are the shipped model's; the trainer's scores of the model before the fold sit beside the fit
    assert report["eval"]["psnr"] == 27.0 and report["evalBlock"]["psnr"] == 24.0 and report["perView"] == "eval/per_view.json"
    assert fold["trained"]["eval"]["psnr"] == 26.0 and fold["trained"]["evalBlock"]["psnr"] == 23.0
    assert json.loads((job / "eval" / "val_stats.json").read_text())["psnr"] == 27.0 and json.loads((job / "eval" / "trained" / "val_stats.json").read_text())["psnr"] == 26.0
    assert report["fromStep"] == 299 and report["maxSteps"] == 300 and report["appearance"] == "bilateral_grid" and report["blockImages"] == ["000004.jpg", "000005.jpg"]
    json.dumps(report)


def test_without_the_option_nothing_is_fitted_and_the_report_is_what_it_was(tmp_path, monkeypatch):
    report, cmds, fits, job = _fold_job(tmp_path, monkeypatch, FIT, bilateral_grid=True, holdout=8)
    assert len(cmds) == 1 and fits == [] and "exposureFold" not in report and not (job / "train" / "folded.ply").exists()
    assert (job / "scene.ply").read_bytes() == (job / "train" / "ply" / "point_cloud_299.ply").read_bytes() and report["eval"]["psnr"] == 26.0
    assert sorted(report) == ["appearance", "eval", "fromStep", "maxSteps", "plan", "ply", "sizeMB"]


def test_exposure_fold_0_ships_the_folded_model_without_training_on(tmp_path, monkeypatch):
    report, cmds, fits, job = _fold_job(tmp_path, monkeypatch, FIT, bilateral_grid=True, holdout=8, exposure_fold=0)
    fold = report["exposureFold"]
    assert len(cmds) == 1 and fits[0][2:] == (8, [], True) and (fold["steps"], fold["folded"], fold["settled"]) == (0, True, False)
    assert "why" not in fold and "trained" not in fold and report["eval"]["psnr"] == 26.0      # the trainer scored the model before the fold
    assert (job / "scene.ply").read_bytes() == (job / "train" / "folded.ply").read_bytes() != (job / "train" / "ply" / "point_cloud_299.ply").read_bytes()


@pytest.mark.parametrize("fit, why", [
    ({"views": 3, "pixels": 900, "usable": 0.01, "why": "too little to fit on: 900 usable pixels in 3 views (needs 100000 in 8)"}, "too little to fit on"),
    (None, "not fitted: needs a GPU with gsplat, and PINHOLE cameras"),
    (RuntimeError("CUDA out of memory"), "the fit failed: CUDA out of memory"),
])
def test_a_fit_that_is_refused_or_cannot_be_made_ships_the_model_as_trained_and_says_why(tmp_path, monkeypatch, fit, why):
    report, cmds, fits, job = _fold_job(tmp_path, monkeypatch, fit, bilateral_grid=True, holdout=8, exposure_fold=200)
    fold = report["exposureFold"]
    assert len(cmds) == 1 and len(fits) == 1 and not (job / "train" / "folded.ply").exists()
    assert (fold["folded"], fold["settled"], fold["steps"]) == (False, False, 200) and fold["why"].startswith(why) and report["eval"]["psnr"] == 26.0
    assert (job / "scene.ply").read_bytes() == (job / "train" / "ply" / "point_cloud_299.ply").read_bytes()
    if isinstance(fit, dict):
        assert fold["views"] == 3 and fold["usable"] == 0.01                                # what the refusal rests on is in the report


def test_a_block_the_trainer_did_not_name_is_never_guessed_at(tmp_path, monkeypatch):
    report, cmds, fits, job = _fold_job(tmp_path, monkeypatch, FIT, names_the_block=False, bilateral_grid=True, holdout=8, holdout_block=10, exposure_fold=200)
    assert fits == [] and len(cmds) == 1 and report["exposureFold"]["folded"] is False
    assert "left no list of the held-out block's views" in report["exposureFold"]["why"]


def test_steps_after_the_fold_that_fail_ship_the_folded_model_and_say_so(tmp_path, monkeypatch):
    report, cmds, fits, job = _fold_job(tmp_path, monkeypatch, FIT, settle_fails=True, bilateral_grid=True, holdout=8, exposure_fold=200)
    fold = report["exposureFold"]
    assert len(cmds) == 2 and (fold["folded"], fold["settled"]) == (True, False) and "CUDA out of memory" in fold["why"]
    assert fold["why"].startswith("the 200 steps after the fold failed") and report["eval"]["psnr"] == 26.0 and "trained" not in fold
    assert (job / "scene.ply").read_bytes() == (job / "train" / "folded.ply").read_bytes()
