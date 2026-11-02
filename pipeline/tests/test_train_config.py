"""The GPU trainer cannot run here; check the config the launcher hands to gsplat's simple_trainer."""

import types

import pytest

from plypipe import train_launcher


class FakeConfig:
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
    cfg, info = train_launcher.build_config(old, "/d", "/r", "standard", None, False)
    assert cfg.app_opt is True and info["ppisp"] is False and not hasattr(cfg, "post_processing")
    cfg, info = train_launcher.build_config(old, "/d", "/r", "quick", None, False)
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
