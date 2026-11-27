"""plypipe.device: the machine as a ceiling on the plan. Pure functions, so every card is tested here on a CPU."""

import json

import pytest

from plypipe import device, plan


def gpu(total, free=None, sm=7.5, name="GPU"):
    return {"gpu": {"name": name, "totalGB": total, "freeGB": total if free is None else free, "sm": sm, "driver": "x"},
            "ramGB": 16.0, "cpus": 12, "freeDiskGB": 120.0, "wsl": True}


def test_gtx_1650_holds_about_a_million_splats_packed_and_no_sam3():
    dp = device.device_plan(gpu(4.0, 3.7, 7.5, "NVIDIA GeForce GTX 1650"))
    assert 0.9e6 < dp["maxGaussians"] < 1.4e6
    assert dp["packed"] is True
    assert dp["sam3"] is False
    assert dp["maxLongSide"] is None  # 3.7 GB free: the 1280 px floor stands
    assert dp["loaderWorkers"] == 2  # 16 GB RAM
    assert dp["trainable"] is True
    assert "SAM 3 off" in dp["why"]


def test_card_driving_the_desktop_plans_from_free_memory():
    a = device.device_plan(gpu(4.0, 4.0))
    b = device.device_plan(gpu(4.0, 3.0))
    assert b["maxGaussians"] < a["maxGaussians"]
    assert b["maxLongSide"] == device.SMALL_CARD_SIDE  # under 3.5 GB free


@pytest.mark.parametrize("total,sam3", [(6, False), (8, False), (12, True), (16, True)])
def test_sam3_needs_fp32_room_before_ampere(total, sam3):
    assert device.device_plan(gpu(total, sm=7.5))["sam3"] is sam3


def test_ampere_runs_sam3_in_bf16_from_6gb():
    assert device.device_plan(gpu(8, sm=8.6))["sam3"] is True
    assert device.device_plan(gpu(4, sm=8.6))["sam3"] is False


def test_big_cards_plan_exactly_as_before():
    for total in (24, 48):
        dp = device.device_plan({**gpu(total, sm=8.9), "ramGB": 64})
        assert dp["packed"] is False and dp["loaderWorkers"] is None and dp["maxLongSide"] is None and dp["sam3"]
        mp = plan.model_plan(900, {"min": [0, 0, 0], "max": [20, 3, 15]}, "high", max_gaussians=dp["maxGaussians"])
        assert mp["capMax"] == plan.model_plan(900, {"min": [0, 0, 0], "max": [20, 3, 15]}, "high")["capMax"]
        assert "deviceCapMax" not in mp


def test_no_gpu_cannot_train():
    dp = device.device_plan({"gpu": None, "ramGB": 8, "cpus": 4, "freeDiskGB": 50})
    assert dp["trainable"] is False and dp["sam3"] is False


def test_old_architecture_is_refused():
    assert device.device_plan(gpu(8, sm=6.1))["trainable"] is False


def test_model_plan_takes_the_device_ceiling_and_says_so():
    bounds = {"min": [0, 0, 0], "max": [12, 3, 9]}
    free = plan.model_plan(800, bounds, "standard")
    held = plan.model_plan(800, bounds, "standard", max_gaussians=600_000)
    assert free["capMax"] > 600_000
    assert held["capMax"] == held["deviceCapMax"] == 600_000
    assert "GPU's memory" in held["why"]
    assert held["iterations"] == free["iterations"]  # steps follow the views, not the card


def test_explicit_cap_still_wins():
    mp = plan.model_plan(800, {"min": [0, 0, 0], "max": [12, 3, 9]}, "standard", cap_max=1_500_000, max_gaussians=600_000)
    assert mp["capMax"] == 1_500_000


def test_input_plan_lowers_pixels_never_frames():
    free = plan.input_plan(300, 3, 3840, None)
    held = plan.input_plan(300, 3, 3840, None, max_side=1024)
    assert held["longSide"] == 1024 < free["longSide"]
    assert held["keptFps"] == free["keptFps"] and held["expectedFrames"] == free["expectedFrames"]
    assert "GPU's memory" in held["why"]


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PLY_DEVICE_VRAM_GB", "4")
    monkeypatch.setenv("PLY_DEVICE_SM", "7.5")
    monkeypatch.setenv("PLY_DEVICE_RAM_GB", "16")
    m = device.probe(tmp_path)
    assert m["gpu"]["totalGB"] == 4 and m["gpu"]["sm"] == 7.5 and m["ramGB"] == 16
    assert device.device_plan(m)["packed"] is True


def test_oom_text():
    assert device.oom("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB")
    assert not device.oom("RuntimeError: something else")


def test_cli_turns_sam3_off_on_a_small_card_and_keeps_masks_made_elsewhere(monkeypatch, tmp_path):
    from plypipe import cli
    from plypipe.log import Log
    from plypipe.util import Ctx
    monkeypatch.setenv("PLY_DEVICE_VRAM_GB", "4")
    monkeypatch.setenv("PLY_DEVICE_SM", "7.5")
    ctx = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={})
    cli._device_profile(ctx, ["frames", "sfm", "label", "train", "export"])
    assert ctx.opts["no_people_mask"] and ctx.opts["no_labels"]
    assert json.loads((tmp_path / "plan.json").read_text())["device"]["packed"] is True

    (tmp_path / "masks").mkdir()
    (tmp_path / "masks" / "people.json").write_text("{}")
    ctx2 = Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={})
    cli._device_profile(ctx2, ["sfm", "label", "train"])
    assert not ctx2.opts.get("no_people_mask")
