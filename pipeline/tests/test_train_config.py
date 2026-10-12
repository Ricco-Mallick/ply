"""The GPU trainer cannot run here; check the config the launcher hands to gsplat's simple_trainer."""

import types

import pytest

from plypipe import train_launcher


class FakeConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.steps_scaler = 1.0


class FakeMCMC:
    def __init__(self, verbose=False, cap_max=1_000_000):
        self.verbose, self.cap_max = verbose, cap_max


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
