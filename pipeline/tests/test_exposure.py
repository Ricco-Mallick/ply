"""Exposure correction folded back (plan variant Q1): the fit of the colour map on training views, its guards, and the
fold into the SH colours. All numpy, so CI runs them; the two that check against gsplat itself need torch and
PLY_GSPLAT_DIR on the path, and are skipped without."""

from pathlib import Path

import numpy as np
import pytest

from plypipe import exposure, finetune, viewer_eval
from test_finetune import write_model

# the viewer showed the twin about x1.2 brighter than the photos: the map back is a gain near 0.83, with some mixing
M_TRUE = np.array([[0.84, 0.03, -0.02], [0.02, 0.81, 0.04], [-0.03, 0.05, 0.86]])
B_TRUE = np.array([0.02, -0.01, 0.015])
TOLERANCE = 1.5e-3  # on every entry of M and b, from 153,600 pixels with noise 0.02


def views_of(rng, m=M_TRUE, b=B_TRUE, n=8, shape=(120, 160), noise=0.02, lo=0.15, hi=0.9):
    out = []
    for _ in range(n):
        render = rng.uniform(lo, hi, (*shape, 3)).astype(np.float32)
        photo = (render @ np.asarray(m).T + b + rng.normal(0, noise, render.shape)).astype(np.float32)
        out.append((render, np.ones(shape, np.float32), photo))
    return out


def fit(views, **kw):
    acc = exposure.sums()
    for view in views:
        exposure.accumulate(acc, *view)
    return exposure.solve(acc, **kw)


def off(found, m=M_TRUE, b=B_TRUE):
    return max(np.abs(np.array(found["matrix"]) - m).max(), np.abs(np.array(found["offset"]) - b).max())


# ------------------------------------------------------------------ the fit

def test_a_known_colour_map_is_recovered_from_noisy_renders_and_photos():
    views = views_of(np.random.default_rng(0))
    found = fit(views)
    assert "why" not in found and found["views"] == 8 and found["pixels"] == 8 * 120 * 160 and found["usable"] == 1.0
    assert off(found) < TOLERANCE and found["condition"] < 1e3
    assert np.allclose(found["gains"], M_TRUE.sum(1), atol=2 * TOLERANCE)
    change = exposure.measure(views, found["matrix"], found["offset"])
    # the photos are darker than the renders by the map; after it only the noise is left (its mean absolute value is 0.016)
    assert change["maeBefore"] == pytest.approx(0.064, abs=2e-3) and change["maeAfter"] == pytest.approx(0.02 * np.sqrt(2 / np.pi), abs=5e-4)
    assert change["brightnessBefore"] == pytest.approx(1.133, abs=0.005) and change["brightnessAfter"] == pytest.approx(1.0, abs=0.002)


def test_clipped_photo_pixels_and_pixels_the_model_does_not_cover_stay_out_of_the_fit(monkeypatch):
    rng = np.random.default_rng(1)
    views = views_of(rng)
    for render, alpha, photo in views:
        render[:40, :60] = rng.uniform(1.3, 1.6, (40, 60, 3))      # a window: the model renders it bright, the photo clipped
        photo[:40, :60] = 1.0
        photo[100:, 100:] = 0.0                                     # crushed blacks under a pew
        alpha[60:90, :50] = 0.6                                     # a hole: 60 % of a colour over the black background,
        render[60:90, :50] *= 0.6                                   # and the photo shows what is really behind it
        photo[60:90, :50] = rng.uniform(0.1, 0.9, (30, 50, 3))
    found = fit(views)
    assert found["pixels"] == 8 * (120 * 160 - 40 * 60 - 20 * 60 - 30 * 50) and found["usable"] == pytest.approx(0.7344, abs=1e-4)
    assert off(found) < TOLERANCE
    monkeypatch.setattr(exposure, "PHOTO_RANGE", (-1.0, 2.0))       # with the glare and the blacks in, the gain is wrong
    assert off(fit(views)) > 0.1
    monkeypatch.undo()
    monkeypatch.setattr(exposure, "ALPHA_MIN", 0.0)                 # and so it is with the hole in
    assert off(fit(views)) > 0.02
    monkeypatch.undo()
    views[0][0][50, 100] = np.nan                                   # one broken pixel is left out, not summed
    assert fit(views)["pixels"] == found["pixels"] - 1 and off(fit(views)) < TOLERANCE


def test_the_offset_counts_by_coverage_as_the_fold_adds_it_to_every_splat():
    rng = np.random.default_rng(2)
    views = []
    for _ in range(8):
        colour, alpha = rng.uniform(0.15, 0.9, (120, 160, 3)), rng.uniform(0.99, 1.0, (120, 160))
        render = colour * alpha[..., None]
        views.append((render.astype(np.float32), alpha.astype(np.float32), (render @ M_TRUE.T + alpha[..., None] * B_TRUE).astype(np.float32)))
    assert off(fit(views)) < 1e-4


@pytest.mark.parametrize("m, b, why", [
    (np.diag([1.0, 1.0, -1.0]), [0.0, 0.0, 1.0], "mirrors or flattens colour"),
    (2.5 * np.eye(3), [0.0, 0.0, 0.0], "channel gains of 2.50, 2.50, 2.50: outside 0.5 to 2.0"),
    (0.3 * np.eye(3), [0.1, 0.1, 0.1], "outside 0.5 to 2.0"),
    ([[3.0, -2.0, 0.0], [-2.0, 3.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0], "scales a colour direction by 5.00"),
    (0.6 * np.eye(3), [0.3, 0.3, 0.3], "an offset of 0.30: beyond 0.25"),
])
def test_an_implausible_map_is_refused_with_its_reason(m, b, why):
    views = views_of(np.random.default_rng(3), m=np.array(m), b=np.array(b), lo=0.25, hi=0.35, noise=0.001)
    found = fit(views)
    assert why in found["why"] and off(found, np.array(m), np.array(b)) < 0.02     # the map is recorded next to the reason
    assert why in exposure.implausible(np.array(m, float), np.array(b, float))


def test_a_fit_on_too_little_or_on_renders_without_colour_is_refused():
    rng = np.random.default_rng(4)
    assert "why" not in fit(views_of(rng))
    few = fit(views_of(rng, n=7, shape=(150, 160)))
    assert few["why"] == "too little to fit on: 168000 usable pixels in 7 views (needs 100000 in 8)" and "matrix" not in few
    small = fit(views_of(rng, n=8, shape=(100, 120)))
    assert small["why"].startswith("too little to fit on: 96000 usable pixels in 8 views") and "matrix" not in small
    assert "why" not in fit(views_of(rng, n=3, shape=(40, 40)), min_views=3, min_pixels=4000)
    grey = [(np.repeat(r[..., :1], 3, -1), a, p) for r, a, p in views_of(rng)]
    found = fit(grey)
    assert "do not determine a colour map" in found["why"] and "matrix" not in found
    assert exposure.implausible(np.full((3, 3), np.nan), np.zeros(3)) == "the fitted colour map is not finite"
    assert exposure.implausible(np.eye(3), np.array([0.0, np.inf, 0.0])) == "the fitted colour map is not finite"
    assert exposure.implausible(np.eye(3), np.zeros(3)) is None and exposure.implausible(M_TRUE, B_TRUE) is None


# ------------------------------------------------------------------ the view loop

def scene(root, monkeypatch):
    """A 40-view scene with a stand-in rasteriser whose render of a view is that view's photo taken back through the
    known map. Returns the data folder, the image names, what the rasteriser was asked for, and the photos read."""
    cv2 = pytest.importorskip("cv2")
    from gs_scene import write_trainable_scene
    from plypipe import colmap_txt
    names = write_trainable_scene(root, n_cams=40)
    data = root / "data"
    by_pose = {tuple(np.round(im.tvec, 6)): im.name for im in colmap_txt.read_model(data / "sparse" / "0").images}
    asked, read, imread = [], [], cv2.imread

    def render(view, K, width, height):
        name = by_pose[tuple(np.round(view[:3, 3], 6))]
        asked.append((name, width, height, K))
        photo = cv2.cvtColor(cv2.imdecode(np.fromfile(data / "images" / name, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        photo = cv2.resize(photo, (width, height), interpolation=cv2.INTER_AREA).astype(np.float64) / 255
        return ((photo - B_TRUE) @ np.linalg.inv(M_TRUE).T).astype(np.float32), np.ones((height, width), np.float32)

    def spy(path, *args):
        if args == (cv2.IMREAD_COLOR,):                 # the fit's reads (the train stage also sizes the first photo, in grey)
            read.append(Path(path).name)
        return imread(path, *args)

    monkeypatch.setattr(cv2, "imread", spy)
    monkeypatch.setattr(exposure, "_renderer", lambda ply, antialiased: render)
    return data, names, asked, read


def test_training_views_are_gsplats_split_without_the_block():
    names = [f"{i:06d}.jpg" for i in range(20)]
    assert viewer_eval.training_views(names, 8) == [i for i in range(20) if i not in (0, 8, 16)]
    assert viewer_eval.training_views(names, 8, names[3:6]) == [i for i in range(20) if i not in (0, 8, 16, 3, 4, 5)]
    assert viewer_eval.training_views(names, 0, names[:2]) == list(range(2, 20))   # no hold-out: only the block is kept out


def test_the_fit_reads_training_views_only_and_recovers_the_map_through_the_view_loop(tmp_path, monkeypatch):
    data, names, asked, read = scene(tmp_path / "scene", monkeypatch)
    block = names[10:16]
    held = {names[i] for i in (0, 8, 16, 24, 32)} | set(block)
    found = exposure.fit_views(tmp_path / "model.ply", data, every=8, block=block, scale=1.0)
    train = [n for n in names if n not in held]
    assert len(train) == 29 and found["views"] == 29 and "why" not in found
    assert read == train * 2 and [a[0] for a in asked] == train * 2             # one pass to fit, one to measure; never a held-out view
    assert not held & set(read) and all(a[1:3] == (320, 240) for a in asked)
    assert off(found) < 1e-4 and found["maeAfter"] < 1e-4 < 0.05 < found["maeBefore"]
    assert found["brightnessBefore"] > 1.1 and found["brightnessAfter"] == pytest.approx(1.0, abs=1e-3)


def test_the_fit_spreads_its_views_over_the_capture_and_renders_them_at_reduced_resolution(tmp_path, monkeypatch):
    data, names, asked, read = scene(tmp_path / "scene", monkeypatch)
    found = exposure.fit_views(tmp_path / "model.ply", data, every=8, max_views=12)   # scale 0.5, no block
    train = [n for i, n in enumerate(names) if i % 8]
    assert len(train) == 35 and read == train[::3] * 2 and found["views"] == 12 and found["pixels"] > 100_000
    name, width, height, K = asked[0]
    assert (width, height) == (160, 120) and np.allclose(K, [[120, 0, 80], [0, 120, 60], [0, 0, 1]])   # the scene's 240 px focal, halved
    assert off(found) < 5e-3                                                    # the photo is resampled before the map here, after it there
    held_everything = exposure.fit_views(tmp_path / "model.ply", data, every=8, block=names)
    assert held_everything["views"] == 0 and held_everything["why"].startswith("too little to fit on: 0 usable pixels in 0 views")
    assert len(read) == 24                                                      # and nothing more was read


def test_without_a_gpu_there_is_no_fit(tmp_path):
    pytest.importorskip("cv2")
    from gs_scene import write_trainable_scene
    write_trainable_scene(tmp_path / "scene", n_cams=12, clips=(12,), dropped=())
    write_model(tmp_path / "model.ply", n=6, k=8)
    assert exposure.fit_views(tmp_path / "model.ply", tmp_path / "scene" / "data", every=8) is None


# ------------------------------------------------------------------ the fold

def sh_basis(dirs):
    """The 16 real SH basis functions gsplat and the viewer use (Sloan's recurrences, as gsplat/cuda/_torch_impl.py)."""
    x, y, z = dirs.T
    z2, c1, s1 = z * z, x * x - y * y, 2 * x * y
    c2, s2 = x * c1 - y * s1, x * s1 + y * c1
    a, b = -2.285228997322329 * z2 + 0.4570457994644658, 1.445305721320277 * z
    return np.stack([np.full_like(x, 0.2820947917738781), -0.48860251190292 * y, 0.48860251190292 * z, -0.48860251190292 * x,
                     0.5462742152960395 * s1, -1.092548430592079 * z * y, 0.9461746957575601 * z2 - 0.3153915652525201,
                     -1.092548430592079 * z * x, 0.5462742152960395 * c1, -0.5900435899266435 * s2, b * s1, a * y,
                     z * (1.865881662950577 * z2 - 1.119528997770346), a * x, b * c1, -0.5900435899266435 * c2], 1)


def colour(sh0, shN, dirs):
    """Colour before the clamp at zero: 0.5 + the SH sum, one direction per splat."""
    coeffs = np.concatenate([sh0, shN], 1).astype(np.float64)
    return 0.5 + np.einsum("nk,nkc->nc", sh_basis(dirs)[:, :coeffs.shape[1]], coeffs)


def random_model(rng, degree, n=500, dtype=np.float64):
    dirs = rng.normal(size=(n, 3))
    return (rng.normal(0, 1.0, (n, 1, 3)).astype(dtype), rng.normal(0, 0.3, (n, (degree + 1) ** 2 - 1, 3)).astype(dtype),
            dirs / np.linalg.norm(dirs, axis=1, keepdims=True))


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_the_folded_coefficients_give_the_mapped_colour_in_every_direction(degree):
    rng = np.random.default_rng(degree)
    sh0, shN, dirs = random_model(rng, degree)
    f0, fN = exposure.fold(sh0, shN, M_TRUE, B_TRUE)
    before, after = colour(sh0, shN, dirs), colour(f0, fN, dirs)
    assert f0.shape == sh0.shape and fN.shape == shN.shape and f0.dtype == np.float64
    assert np.abs(after - (before @ M_TRUE.T + B_TRUE)).max() < 1e-13
    # the viewer clamps a colour at zero: where neither side is clamped, the clamped colours follow the map too
    lit = (before > 0).all(1) & (after > 0).all(1)
    assert lit.sum() > 50 and np.allclose(np.maximum(after, 0)[lit], np.maximum(before, 0)[lit] @ M_TRUE.T + B_TRUE, atol=1e-13)
    s0, sN = sh0.astype(np.float32), shN.astype(np.float32)                    # as the PLY holds them
    g0, gN = exposure.fold(s0, sN, M_TRUE, B_TRUE)
    assert g0.dtype == np.float32 and gN.dtype == np.float32
    assert np.abs(colour(g0, gN, dirs) - (colour(s0, sN, dirs) @ M_TRUE.T + B_TRUE)).max() < 2e-6


def test_folding_the_identity_changes_nothing_and_a_map_then_its_inverse_returns_the_coefficients():
    rng = np.random.default_rng(7)
    sh0, shN, dirs = random_model(rng, 3)
    same0, sameN = exposure.fold(sh0, shN, np.eye(3), np.zeros(3))
    assert np.array_equal(same0, sh0) and np.array_equal(sameN, shN)           # so the rendered brightness is unchanged
    f0, fN = exposure.fold(sh0, shN, M_TRUE, B_TRUE)
    inverse = np.linalg.inv(M_TRUE)
    back0, backN = exposure.fold(f0, fN, inverse, -inverse @ B_TRUE)
    assert np.abs(back0 - sh0).max() < 1e-13 and np.abs(backN - shN).max() < 1e-13
    grey0, greyN = exposure.fold(sh0, shN, 0.8 * np.eye(3), np.zeros(3))       # a plain gain: every colour 0.8 times as bright
    assert np.allclose(colour(grey0, greyN, dirs), 0.8 * colour(sh0, shN, dirs), atol=1e-13)


@pytest.mark.parametrize("k", [0, 8, 15])
def test_a_folded_ply_keeps_its_layout_and_splats_and_changes_only_the_colours(tmp_path, k):
    from plypipe import clean
    degree = {0: 0, 8: 2, 15: 3}[k]
    props, data = write_model(tmp_path / "m.ply", n=6, k=k)
    assert exposure.fold_ply(tmp_path / "m.ply", tmp_path / "folded.ply", M_TRUE, B_TRUE) == {"splats": 6, "shDegree": degree}
    header, cols = clean.read_ply(tmp_path / "folded.ply")
    assert header == clean.read_ply(tmp_path / "m.ply")[0] and list(cols) == props
    for i, p in enumerate(props):
        assert np.array_equal(cols[p], data[:, i]) != p.startswith("f_"), p     # every colour column changed, nothing else did
    was, now = finetune.read_splats(tmp_path / "m.ply", degree), finetune.read_splats(tmp_path / "folded.ply", degree)
    want0, wantN = exposure.fold(was["sh0"], was["shN"], M_TRUE, B_TRUE)
    assert np.array_equal(now["sh0"], want0) and np.array_equal(now["shN"], wantN)
    dirs = np.random.default_rng(k).normal(size=(6, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mapped = colour(was["sh0"], was["shN"], dirs) @ M_TRUE.T + B_TRUE
    assert np.allclose(colour(now["sh0"], now["shN"], dirs), mapped, rtol=1e-5, atol=1e-5)
    (tmp_path / "points.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n" + np.zeros(1, np.float32).tobytes())
    with pytest.raises(ValueError, match="not a 3DGS model"):
        exposure.fold_ply(tmp_path / "points.ply", tmp_path / "out.ply", M_TRUE, B_TRUE)


# ------------------------------------------------------------------ against gsplat itself (torch, on a CPU)

def test_gsplats_own_sh_evaluation_of_a_folded_model_is_the_mapped_colour():
    torch = pytest.importorskip("torch")
    impl = pytest.importorskip("gsplat.cuda._torch_impl")
    rng = np.random.default_rng(11)
    for degree in range(4):
        sh0, shN, dirs = random_model(rng, degree, n=200, dtype=np.float32)
        gsplat = lambda a, b: impl._spherical_harmonics(degree, torch.from_numpy(dirs), torch.from_numpy(np.concatenate([a, b], 1))).double().numpy() + 0.5
        assert np.abs(gsplat(sh0, shN) - colour(sh0, shN, dirs)).max() < 1e-5   # the basis these tests use is gsplat's
        f0, fN = exposure.fold(sh0, shN, M_TRUE, B_TRUE)
        assert np.abs(gsplat(f0, fN) - (gsplat(sh0, shN) @ M_TRUE.T + B_TRUE)).max() < 1e-5


def test_the_rasteriser_is_handed_the_model_as_the_viewer_draws_it(tmp_path, monkeypatch):
    """exposure._renderer up to gsplat's rasterization(), which needs CUDA: a stand-in takes its arguments on the CPU."""
    torch = pytest.importorskip("torch")
    gsplat = pytest.importorskip("gsplat")
    write_model(tmp_path / "m.ply", n=6, k=15, normals=False)
    model, seen, as_tensor = finetune.read_splats(tmp_path / "m.ply", 3), {}, torch.as_tensor

    def rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, sh_degree=None, rasterize_mode="classic"):
        seen.update(means=means, quats=quats, scales=scales, opacities=opacities, colors=colors, viewmats=viewmats, Ks=Ks,
                    sh_degree=sh_degree, mode=rasterize_mode, grad=torch.is_grad_enabled())
        return torch.full((1, height, width, 3), 1.25), torch.full((1, height, width, 1), 0.5), {}

    monkeypatch.setattr(gsplat, "rasterization", rasterization)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "as_tensor", lambda a, dtype=None, device=None: as_tensor(a, dtype=dtype))   # "cuda" is not here
    view, K = np.eye(4), np.array([[120.0, 0, 80], [0, 120, 60], [0, 0, 1]])
    view[:3, 3] = 1, 2, 3
    rgb, alpha = exposure._renderer(tmp_path / "m.ply", True)(view, K, 160, 120)
    assert rgb.shape == (120, 160, 3) and alpha.shape == (120, 160) and rgb.dtype == np.float32
    assert (rgb == 1.25).all() and (alpha == 0.5).all()                         # as rasterised: nothing clamped
    assert np.array_equal(seen["means"], model["means"]) and np.array_equal(seen["quats"], model["quats"])
    assert np.allclose(seen["scales"], np.exp(model["scales"])) and np.allclose(seen["opacities"], 1 / (1 + np.exp(-model["opacities"])))
    assert np.array_equal(seen["colors"], np.concatenate([model["sh0"], model["shN"]], 1)) and seen["sh_degree"] == 3
    assert seen["viewmats"].shape == (1, 4, 4) and np.array_equal(seen["viewmats"][0], view) and np.array_equal(seen["Ks"][0], K)
    assert seen["mode"] == "antialiased" and not seen["grad"]
    exposure._renderer(tmp_path / "m.ply", False)(view, K, 16, 12)
    assert seen["mode"] == "classic"


def test_the_train_stage_folds_and_the_pinned_trainer_continues_from_the_folded_colours_with_no_appearance_model(trainer, tmp_path, monkeypatch, capsys):
    """The train stage with --bilateral-grid --exposure-fold, its launcher command lines run by the real launcher on
    the pinned trainer (--dry-run: gsplat's Config, parser, datasets and parameters on a CPU). Only the rasteriser and
    the training steps are stand-ins."""
    import json
    import os
    import shutil
    import sys
    import torch
    from gsplat import export_splats
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from plypipe import train_launcher
    from plypipe.log import Log
    from plypipe.stages import train as train_stage
    from plypipe.util import Ctx
    st, job, n = trainer, tmp_path / "job", 50
    originals = {k: getattr(st, k) for k in ("Dataset", "Runner", "create_splats_with_optimizers")}
    for k, v in originals.items():
        monkeypatch.setattr(st, k, v)                                             # put back after the test
    monkeypatch.delitem(sys.modules, "fused_ssim", raising=False)                 # the fixture's stand-in has no spec to find
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    data, names, asked, read = scene(job, monkeypatch)
    g = torch.Generator().manual_seed(0)
    model = {name: torch.randn(n, *shape, generator=g) for name, shape in
             (("means", (3,)), ("scales", (3,)), ("quats", (4,)), ("opacities", ()), ("sh0", (1, 3)), ("shN", (15, 3)))}
    runs = []

    def launch(cmd, ctx, stage, **kw):
        """What a run of the launcher's process does up to the first training step, then a model where gsplat would save one."""
        cmd = [str(c) for c in cmd]
        for k, v in originals.items():
            setattr(st, k, v)
        sys.modules.pop("fused_ssim", None)
        assert cmd[1:3] == ["-m", "plypipe.train_launcher"] and train_launcher.main(cmd[3:] + ["--dry-run"]) == 0
        printed = {k: v for line in capsys.readouterr().out.splitlines() if line.startswith("{") for k, v in json.loads(line).items()}
        result = Path(cmd[cmd.index("--result") + 1])
        parser = st.Parser(data_dir=str(data), factor=1, normalize=False, test_every=8)
        run = {"cmd": cmd, **printed, "train": [parser.image_names[i] for i in st.Dataset(parser, split="train").indices]}
        (result / "ply").mkdir(parents=True)
        if "--from-ply" in cmd:
            run["params"], _ = st.create_splats_with_optimizers(parser, sh_degree=3, scene_scale=float(parser.scene_scale), device="cpu")
            shutil.copy2(cmd[cmd.index("--from-ply") + 1], result / "ply" / "point_cloud_199.ply")
        else:
            export_splats(**model, format="ply", save_to=str(result / "ply" / "point_cloud_299.ply"))
            (result / "ply_block.json").write_text(json.dumps(printed["plyTrainDryRun"]["block"]))   # as the Runner writes it
        runs.append(run)

    monkeypatch.setattr(train_stage, "run", launch)
    opts = {"gsplat_dir": os.environ["PLY_GSPLAT_DIR"], "preset": "high", "iterations": 300, "holdout": 8, "holdout_block": 8,
            "bilateral_grid": True, "exposure_fold": 200}
    report = train_stage.run_stage(Ctx(out=job, log=Log(job, quiet=True), opts=opts))
    trained, settle = runs
    fold = report["exposureFold"]
    # trained as configured, with the grid; the fit then read views of the trainer's own training split only
    assert trained["plyTrainConfig"]["appearance"] == "bilateral_grid" and "--bilateral-grid" in trained["cmd"]
    held = set(names) - set(trained["train"])
    block = trained["plyTrainDryRun"]["block"]["frames"]
    assert len(block) == 8 and held == {names[i] for i in range(0, 40, 8)} | set(block)
    assert set(read) == set(trained["train"]) and not held & set(read) and fold["views"] == len(trained["train"]) == 28
    assert off(fold) < 5e-3 and (fold["folded"], fold["settled"], fold["steps"]) == (True, True, 200)
    assert fold["brightnessBefore"] > 1.1 and fold["brightnessAfter"] == pytest.approx(1.0, abs=5e-3)
    # the continuation: no appearance model, the model's own splat count and SH bands, the same split
    config = settle["plyTrainConfig"]
    assert config["appearance"] is None and config["ppisp"] is False and config["fromSplats"] == n == config["capMax"]
    assert (config["maxSteps"], config["shDegree"], config["holdout"], config["holdoutBlock"]) == (200, 3, 8, 8)
    assert settle["plyTrainDryRun"]["startSplats"] == n and settle["plyTrainDryRun"]["startShCoefficients"] == 16
    assert settle["train"] == trained["train"] and "--bilateral-grid" not in settle["cmd"]
    # gsplat's parameters hold the folded colours: its own SH evaluation of them is the fitted map of the trained model's
    dirs = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
    shown = lambda sh0, shN: (_spherical_harmonics(3, dirs, torch.cat([sh0, shN], 1)) + 0.5).double().numpy()
    params = settle["params"]
    assert all(p.requires_grad for p in params.values()) and torch.allclose(params["means"].data, model["means"])
    want = shown(model["sh0"], model["shN"]) @ np.array(fold["matrix"]).T + np.array(fold["offset"])
    assert np.abs(shown(params["sh0"].data, params["shN"].data) - want).max() < 1e-5
    assert (job / "scene.ply").read_bytes() == (job / "train" / "folded.ply").read_bytes()   # the stand-in "settled" model is the folded one


def test_the_pinned_trainers_bilateral_grid_is_not_part_of_the_model_it_saves(trainer):
    """A checkpoint holds the grid under "post_processing", beside "splats"; the PLY holds the splats alone. So a
    model trained with the grid continues without one as it is, in the colours the grid left it in."""
    import inspect
    import lib_bilagrid
    grid = lib_bilagrid.BilateralGrid(4, grid_X=16, grid_Y=16, grid_W=8)
    assert {k: tuple(v.shape) for k, v in grid.state_dict().items()} == {"grids": (4, 12, 8, 16, 16), "rgb2gray_weight": (1, 3)}
    source = inspect.getsource(trainer.Runner.train)
    assert 'data["post_processing"] = self.post_processing_module.state_dict()' in source and '"splats": self.splats.state_dict()' in source
    assert "sh0 = self.splats[\"sh0\"]" in source and "export_splats(" in source


def test_a_one_cell_grid_is_exactly_one_affine_colour_transform_per_photo(trainer):
    """--exposure-affine trains the pinned trainer's own bilateral grid at 1x1x1, so the appearance model holds exactly
    what plypipe.exposure's fold can undo: the identity at initialisation, zero total variation (the TV regulariser is
    a no-op on it), and after a perturbation one affine map explains its output, where the default 16x16x8 grid leaves
    a residual two orders of magnitude outside one map."""
    import lib_bilagrid
    import torch
    g = torch.Generator().manual_seed(7)
    xy, rgb = torch.rand(4096, 2, generator=g), torch.rand(4096, 3, generator=g)
    idx = torch.zeros(4096, 1, dtype=torch.long)
    residual = {}
    for shape in ((1, 1, 1), (16, 16, 8)):
        grid = lib_bilagrid.BilateralGrid(2, grid_X=shape[0], grid_Y=shape[1], grid_W=shape[2])
        assert torch.allclose(lib_bilagrid.slice(grid, xy, rgb, idx)["rgb"], rgb, atol=1e-6)    # identity at initialisation
        with torch.no_grad():
            grid.grids += 0.1 * torch.randn(grid.grids.shape, generator=g)
        out = lib_bilagrid.slice(grid, xy, rgb, idx)["rgb"]
        ones = torch.cat([rgb, torch.ones(4096, 1)], 1)
        fit = torch.linalg.lstsq(ones, out).solution                       # the one affine map closest to what the grid did
        residual[shape] = (ones @ fit - out).abs().max().item()
        tv = grid.tv_loss()
        assert torch.isfinite(tv) and (tv.item() == 0.0) == (shape == (1, 1, 1))
    assert residual[(1, 1, 1)] < 1e-5 < 1e-2 < residual[(16, 16, 8)]
