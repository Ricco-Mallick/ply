"""The phone file's fine-tune: reading a trained model as gsplat's parameters, starting the trainer from it, and the
export step that ships the fine-tuned model only when it is the better one.

The tests that take the `trainer` fixture run the pinned simple_trainer itself on a CPU (see conftest.py)."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from plypipe import finetune


def write_model(path, n=6, k=15, offset=0.0, normals=True):
    """A 3DGS PLY with k SH coefficients per colour above the base one; every value is distinct."""
    props = ["x", "y", "z", *(["nx", "ny", "nz"] if normals else []), "f_dc_0", "f_dc_1", "f_dc_2",
             *(f"f_rest_{i}" for i in range(3 * k)), "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = (np.arange(n * len(props), dtype=np.float32).reshape(n, len(props)) + offset) / 100
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n" + "".join(f"property float {p}\n" for p in props) + "end_header\n"
    Path(path).write_bytes(header.encode() + data.tobytes())
    return props, data


def test_a_model_is_read_as_gsplats_parameters_with_its_sh_cut_to_the_phone_degree(tmp_path):
    props, data = write_model(tmp_path / "m.ply", n=6, k=15)
    col = lambda name: data[:, props.index(name)]
    s = finetune.read_splats(tmp_path / "m.ply", 2)
    assert {k: v.shape for k, v in s.items()} == {"means": (6, 3), "scales": (6, 3), "quats": (6, 4), "opacities": (6,),
                                                  "sh0": (6, 1, 3), "shN": (6, 8, 3)}
    assert np.array_equal(s["means"][:, 1], col("y")) and np.array_equal(s["quats"][:, 3], col("rot_3"))
    assert np.array_equal(s["scales"][:, 2], col("scale_2")) and np.array_equal(s["opacities"], col("opacity"))
    assert np.array_equal(s["sh0"][:, 0, 1], col("f_dc_1"))
    # the file is channel-major (15 R coefficients, then G, then B); gsplat holds [splat, coefficient, channel]
    for j, c in ((0, 0), (7, 0), (0, 1), (7, 2)):
        assert np.array_equal(s["shN"][:, j, c], col(f"f_rest_{c * 15 + j}")), (j, c)
    assert finetune.read_splats(tmp_path / "m.ply", 3)["shN"].shape == (6, 15, 3)
    assert finetune.read_splats(tmp_path / "m.ply", 0)["shN"].shape == (6, 0, 3)
    write_model(tmp_path / "plain.ply", n=4, k=8, normals=False)                     # as gsplat writes it: no normals
    assert finetune.read_splats(tmp_path / "plain.ply", 2)["shN"].shape == (4, 8, 3)


def test_a_model_without_the_bands_asked_for_or_that_is_no_splat_model_is_refused(tmp_path):
    write_model(tmp_path / "sh1.ply", k=3)
    with pytest.raises(ValueError, match="SH degree 2 needs 8"):
        finetune.read_splats(tmp_path / "sh1.ply", 2)
    (tmp_path / "points.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
                                          b"property float z\nend_header\n" + np.zeros(3, np.float32).tobytes())
    with pytest.raises(ValueError, match="not a 3DGS model"):
        finetune.read_splats(tmp_path / "points.ply", 0)


def test_the_trainer_keeps_its_own_optimizers_and_gets_the_models_values(tmp_path):
    torch = pytest.importorskip("torch")
    import types
    write_model(tmp_path / "m.ply", n=6, k=8)
    splats = finetune.read_splats(tmp_path / "m.ply", 2)

    def upstream(parser, sh_degree=3, **kw):  # the shape of gsplat's: built from other points, one optimizer per parameter
        k = (sh_degree + 1) ** 2 - 1
        params = torch.nn.ParameterDict({name: torch.nn.Parameter(torch.zeros(3, *shape)) for name, shape in
                                         (("means", (3,)), ("scales", (3,)), ("quats", (4,)), ("opacities", ()), ("sh0", (1, 3)), ("shN", (k, 3)))})
        return params, {name: torch.optim.Adam([{"params": params[name], "lr": 0.5, "name": name}]) for name in params}

    st = types.SimpleNamespace(create_splats_with_optimizers=upstream)
    finetune.install(st, splats)
    params, optimizers = st.create_splats_with_optimizers(None, sh_degree=2)
    for name, value in splats.items():
        assert torch.equal(params[name].data, torch.from_numpy(value)) and params[name].requires_grad, name
        assert optimizers[name].param_groups[0]["params"][0] is params[name] and optimizers[name].param_groups[0]["lr"] == 0.5
    before = params["means"].detach().clone()
    (params["means"] ** 2).sum().backward()
    optimizers["means"].step()
    assert params["means"].shape == (6, 3) and not torch.equal(params["means"].detach(), before)
    with pytest.raises(RuntimeError, match="SH degree"):       # the trainer was configured for other bands than the model has
        st.create_splats_with_optimizers(None, sh_degree=3)
    features = types.SimpleNamespace(create_splats_with_optimizers=lambda *a, **k: ({"means": None, "features": None}, {}))
    finetune.install(features, splats)
    with pytest.raises(RuntimeError, match="plain SH colours"):
        features.create_splats_with_optimizers(None)


# ------------------------------------------------------------------ the export step

def _phone(tmp_path, monkeypatch, scores, fitted=None):
    """export._finetune_lite on a cut-down model, with stand-ins for the GPU scoring and the training run. Returns its
    report, whether scene_lite.ply was replaced, and the calls the stand-in trainer got."""
    from plypipe import viewer_eval
    from plypipe.log import Log
    from plypipe.stages import export, train as train_stage
    from plypipe.util import Ctx
    lite, calls = tmp_path / "scene_lite.ply", []
    write_model(lite, n=5, k=8)

    def trained(ctx, ply, steps, sh_degree):
        calls.append((ply.name, steps, sh_degree))
        out = ctx.out / "train-lite" / "ply" / "point_cloud_299.ply"
        out.parent.mkdir(parents=True)
        write_model(out, n=5, k=8, offset=1000.0, normals=False)
        return {"ply": out, "eval": {"psnr": 27.0, "views": 33}}

    monkeypatch.setattr(train_stage, "finetune", fitted or trained)
    monkeypatch.setattr(viewer_eval, "evaluate", lambda ply, *a, **k: scores.get(Path(ply).name))
    before = lite.read_bytes()
    report = export._finetune_lite(Ctx(out=tmp_path, log=Log(tmp_path, quiet=True), opts={}), lite, 300, 8, True)
    assert not (tmp_path / "train-lite").exists()                                    # checkpoints never stay in the job
    return report, lite.read_bytes() != before, calls


def test_the_fine_tuned_phone_model_ships_when_the_viewer_shows_it_better(tmp_path, monkeypatch):
    from plypipe.stages import export
    scores = {"scene_lite.ply": {"psnr": 24.0, "holes": 0.01}, "point_cloud_299.ply": {"psnr": 25.5, "holes": 0.012}}
    report, replaced, calls = _phone(tmp_path, monkeypatch, scores)
    assert calls == [("scene_lite.ply", 300, export.LITE_SH)] and replaced
    assert report["shipped"] == "finetuned" and report["steps"] == 300 and "why" not in report and report["seconds"] >= 0
    assert report["before"]["psnr"] == 24.0 and report["after"]["psnr"] == 25.5 and report["eval"] == {"psnr": 27.0, "views": 33}


@pytest.mark.parametrize("after", [{"psnr": 23.9, "holes": 0.01}, {"psnr": 25.5, "holes": 0.05}, None])
def test_a_fine_tune_that_is_not_better_leaves_the_cut_down_phone_model(tmp_path, monkeypatch, after):
    report, replaced, calls = _phone(tmp_path, monkeypatch, {"scene_lite.ply": {"psnr": 24.0, "holes": 0.01}, "point_cloud_299.ply": after})
    assert len(calls) == 1 and not replaced and report["shipped"] == "decimated" and report["why"].startswith("measured")
    assert report["before"]["psnr"] == 24.0 and report["after"] == after


def test_a_fine_tune_that_fails_or_cannot_be_measured_leaves_the_export_as_it_was(tmp_path, monkeypatch):
    def out_of_memory(*a, **k):
        raise RuntimeError("python exited with 1; last output:\nCUDA out of memory")

    report, replaced, _ = _phone(tmp_path, monkeypatch, {"scene_lite.ply": {"psnr": 24.0, "holes": 0.01}}, fitted=out_of_memory)
    assert not replaced and report["shipped"] == "decimated" and "CUDA out of memory" in report["why"]
    report, replaced, calls = _phone(tmp_path, monkeypatch, {})                      # no GPU: nothing to compare, so no training
    assert not replaced and calls == [] and report == {"steps": 300, "shipped": "decimated", "why": "not measured: needs a GPU and held-out views"}


# ------------------------------------------------------------------ the pinned trainer itself, on CPU

def test_the_pinned_trainer_starts_from_a_model_gsplat_wrote(trainer, tmp_path, monkeypatch):
    """Real Parser, create_splats_with_optimizers, MCMCStrategy and PLY writer from the pinned gsplat: the model read
    back is the model written, and the trainer's own optimizers train it."""
    import torch
    from gsplat import export_splats
    from gs_scene import write_trainable_scene
    from plypipe import train_launcher
    st, n = trainer, 50
    write_trainable_scene(tmp_path / "scene", n_cams=12, clips=(12,), dropped=())
    parser = st.Parser(data_dir=str(tmp_path / "scene" / "data"), factor=1, normalize=False, test_every=8)
    g = torch.Generator().manual_seed(0)
    model = {name: torch.randn(n, *shape, generator=g) for name, shape in
             (("means", (3,)), ("scales", (3,)), ("quats", (4,)), ("opacities", ()), ("sh0", (1, 3)), ("shN", (15, 3)))}
    export_splats(**model, format="ply", save_to=str(tmp_path / "model.ply"))
    splats = finetune.read_splats(tmp_path / "model.ply", 2)
    for name, value in model.items():
        assert np.allclose(splats[name], value.numpy()[:, :8] if name == "shN" else value.numpy()), name

    upstream = st.create_splats_with_optimizers
    monkeypatch.setattr(st, "create_splats_with_optimizers", upstream)               # put back after the test
    theirs, their_optimizers = upstream(parser, sh_degree=2, scene_scale=2.0, device="cpu")
    finetune.install(st, splats)
    params, optimizers = st.create_splats_with_optimizers(parser, sh_degree=2, scene_scale=2.0, device="cpu")
    assert len(theirs["means"]) == len(parser.points) != n
    for name, value in splats.items():
        group, ref = optimizers[name].param_groups[0], their_optimizers[name].param_groups[0]
        assert torch.equal(params[name].data, torch.from_numpy(value)) and params[name].requires_grad, name
        assert group["params"][0] is params[name] and (group["lr"], group["eps"], group["betas"]) == (ref["lr"], ref["eps"], ref["betas"])
    st.MCMCStrategy(cap_max=n).check_sanity(params, optimizers)
    before = params["means"].detach().clone()
    (params["means"] ** 2).sum().backward()
    optimizers["means"].step()
    assert params["means"].shape == (n, 3) and not torch.equal(params["means"].detach(), before)
    with pytest.raises(RuntimeError, match="SH degree"):
        st.create_splats_with_optimizers(parser, sh_degree=3, scene_scale=2.0, device="cpu")

    # the config a fine-tune runs with, on gsplat's own Config and strategy
    cfg, summary = train_launcher.build_config(st, "/d", "/r", "high", False, True, iterations=3000, sh_degree=2, cap_max=2_000_000, from_splats=n)
    cfg.adjust_steps(cfg.steps_scaler)
    assert cfg.strategy.cap_max == n == summary["capMax"] == summary["fromSplats"] and cfg.sh_degree == 2 and cfg.sh_degree_interval == 1
    assert cfg.means_lr == pytest.approx(st.Config().means_lr * finetune.MEANS_LR) and cfg.max_steps == 3000


def test_the_launcher_rehearses_a_fine_tune_on_the_pinned_trainer(trainer, tmp_path, monkeypatch, capsys):
    from gs_scene import write_trainable_scene
    from plypipe import train_launcher
    monkeypatch.setattr(trainer, "create_splats_with_optimizers", trainer.create_splats_with_optimizers)
    monkeypatch.setattr(trainer, "Dataset", trainer.Dataset)
    monkeypatch.setattr(trainer, "Runner", trainer.Runner)
    monkeypatch.delitem(sys.modules, "fused_ssim", raising=False)                    # the fixture's stand-in has no spec to find
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    write_trainable_scene(tmp_path / "scene", n_cams=40, clips=(28, 16), dropped=(30,))
    write_model(tmp_path / "lite.ply", n=40, k=8, normals=False)
    assert train_launcher.main(["--examples", str(Path(os.environ["PLY_GSPLAT_DIR"]) / "examples"), "--data", str(tmp_path / "scene" / "data"),
                                "--result", str(tmp_path / "train-lite"), "--preset", "high", "--no-ppisp", "--from-ply", str(tmp_path / "lite.ply"),
                                "--iterations", "300", "--sh-degree", "2", "--antialiased", "--holdout", "8", "--holdout-block", "8",
                                "--frames-index", str(tmp_path / "scene" / "frames_index.json"), "--dry-run"]) == 0
    printed = {k: v for line in capsys.readouterr().out.splitlines() if line.startswith("{") for k, v in json.loads(line).items()}
    config, found = printed["plyTrainConfig"], printed["plyTrainDryRun"]
    assert config["fromSplats"] == 40 == config["capMax"] and config["shDegree"] == 2 and config["maxSteps"] == 300
    assert found["startSplats"] == 40 and found["startShCoefficients"] == 9 and found["blockImages"] == 8
    assert json.loads((tmp_path / "train-lite" / "ply_preset.json").read_text()) == config
