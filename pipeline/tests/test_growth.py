"""plypipe.growth: the round decision, the patch metric and the density split, on synthetic data."""

import json

import numpy as np
import pytest

from plypipe import growth, verify


def ev(db, lp=0.2):
    return {"eval": {"ccPsnr": db, "lpips": lp}}


def test_a_round_is_kept_only_when_the_held_out_views_improve():
    assert growth.judge(ev(28.0), ev(28.3))[0]
    assert not growth.judge(ev(28.0), ev(28.1))[0]                    # inside the noise
    assert not growth.judge(ev(28.0, 0.20), ev(28.5, 0.21))[0]        # sharper by dB, worse to the eye
    ok, why = growth.judge({**ev(28.0), "patches": {"share": 0.08}}, {**ev(28.05), "patches": {"share": 0.05}})
    assert ok and "patches" in why                                     # fewer soft patches counts too
    assert not growth.judge({**ev(28.0), "patches": {"share": 0.08}}, {**ev(27.9), "patches": {"share": 0.01}})[0]


def test_no_ceiling_except_the_gpu():
    assert growth.next_cap(4_000_000, None) == (6_000_000, "4,000,000 -> 6,000,000")
    cap, why = growth.next_cap(1_000_000, 1_200_000)
    assert cap == 1_200_000 and "fit this GPU" in why
    assert growth.next_cap(1_190_000, 1_200_000)[0] is None             # full: stop and say so


def test_tile_psnr_finds_a_soft_patch_and_ignores_flat_and_masked_tiles():
    rng = np.random.default_rng(0)
    photo = rng.random((256, 256, 3)).astype(np.float32)
    render = photo + rng.normal(0, 0.01, photo.shape).astype(np.float32)
    render[0:64, 0:64] = 0.5                                            # one blurry tile
    tiles = growth.tile_psnr(photo, render)
    rep = growth.view_patches(tiles)
    assert rep["patches"] == 1 and rep["tiles"] == 16
    keep = np.ones((256, 256, 1), np.float32)
    keep[0:64, 0:64] = 0                                                # a person stood there: not judged
    assert growth.view_patches(growth.tile_psnr(photo, render, keep))["patches"] == 0
    flat = np.full((256, 256, 3), 0.5, np.float32)
    assert growth.view_patches(growth.tile_psnr(flat, flat))["tiles"] == 0


def test_patches_reads_gsplat_canvases(tmp_path):
    import cv2
    (tmp_path / "stats").mkdir()
    (tmp_path / "renders").mkdir()
    (tmp_path / "stats" / "val_step29999.json").write_text("{}")
    rng = np.random.default_rng(1)
    photo = (rng.random((128, 256, 3)) * 255).astype(np.uint8)
    render = photo.copy()
    render[0:64, 0:64] = 128
    cv2.imwrite(str(tmp_path / "renders" / "val_step29999_0000.png"), np.concatenate([photo, render], axis=1))
    rep = growth.patches(tmp_path)
    assert rep["views"] == 1 and rep["patches"] == 1 and rep["share"] == 0.125


def test_density_separates_thin_well_filmed_cells_from_unfilmed_ones():
    rng = np.random.default_rng(2)
    xs, zs = np.meshgrid(np.arange(0, 10, 0.5) + 0.25, np.arange(0, 4, 0.5) + 0.25)
    cells = np.stack([xs.ravel(), zs.ravel()], 1)
    pts = []
    for x, z in cells:
        n = 200
        if 2.0 < x < 2.6 and 1.0 < z < 1.6:
            n = 10                                                      # thin, though the walk passed right by
        if x > 9.0:
            n = 10                                                      # thin, and nobody walked there
        pts.append(np.column_stack([rng.uniform(x - .24, x + .24, n), rng.uniform(0, 2.5, n), rng.uniform(z - .24, z + .24, n)]))
    means = np.vstack(pts)
    cams = np.column_stack([np.linspace(0.5, 5.5, 60), np.full(60, 1.5), np.full(60, 2.0)])  # walked the left half
    d = growth.density(means, cams)
    assert d["underDense"] >= 1 and any(abs(x - 2.25) < .3 and abs(z - 1.25) < .3 for x, z in d["underDenseAt"])
    assert d["underFilmed"] >= 1 and all(x > 9.0 for x, _ in d["underFilmedAt"])


def test_the_gate_says_when_a_bigger_gpu_or_more_filming_is_needed():
    rep = {"train": {"growth": {"needsBiggerGpu": True, "stop": {"why": "this GPU holds 1,100,000"}, "best": 1,
                                "rounds": [{"patchShare": 0.1}, {"patchShare": 0.08}]},
                     "density": {"underFilmed": 3, "underFilmedAt": [[9.2, 1.0]]}}}
    codes = {i["code"] for i in verify.check(rep, {})["issues"]}
    assert {"gpu-bound", "patches", "under-filmed"} <= codes


@pytest.fixture
def rig(tmp_path, monkeypatch):
    from plypipe.log import Log
    from plypipe.stages import train
    from plypipe.util import Ctx
    out = tmp_path / "out"
    base = out / "train"
    base.mkdir(parents=True)
    ply = base / "base.ply"
    ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1000000\nend_header\n")
    scores = {}

    def fake_train_on(ctx, src, steps, sh, result, stage, grow_cap=None, round_no=None):
        result.mkdir(parents=True, exist_ok=True)
        p = result / "r.ply"
        p.write_bytes(f"ply\nelement vertex {grow_cap}\nend_header\n".encode())
        scores[str(result)] = rig.db.pop(0)
        if isinstance(scores[str(result)], Exception):
            raise scores[str(result)]
        return p

    monkeypatch.setattr(train, "train_on", fake_train_on)
    monkeypatch.setattr(train, "held_out_scores", lambda result, keep, **k: {"ccPsnr": scores.get(str(result), 28.0), "lpips": 0.2})
    monkeypatch.setattr(growth, "patches", lambda *a, **k: None)
    ctx = Ctx(out=out, log=Log(out, quiet=True), opts={})

    class R:
        pass
    rig = R()
    rig.db, rig.ctx, rig.base, rig.ply, rig.train = [], ctx, base, ply, train
    return rig


def test_rounds_continue_while_they_pay_and_the_best_one_ships(rig):
    rig.db = [28.4, 28.7, 28.75]  # +0.4 kept, +0.3 kept, +0.05 not
    ply, result, rep = rig.train.grow(rig.ctx, rig.base, rig.ply, 30_000, 3, {}, {})
    assert [r["kept"] for r in rep["rounds"]] == [True, True, True, False]
    assert rep["best"] == 2 and result.name == "grow-2"
    assert [r["splats"] for r in rep["rounds"]] == [1_000_000, 1_500_000, 2_250_000, 3_375_000]  # no ceiling
    assert rep["stop"]["reason"] == "converged" and not rep["needsBiggerGpu"]


def test_a_full_gpu_stops_growth_and_says_the_twin_wanted_more(rig):
    rig.db = [28.5]
    _, _, rep = rig.train.grow(rig.ctx, rig.base, rig.ply, 30_000, 3, {"maxGaussians": 1_400_000}, {})
    assert rep["rounds"][1]["splats"] == 1_400_000 and rep["rounds"][1]["kept"]
    assert rep["stop"]["reason"] == "gpu-full" and rep["needsBiggerGpu"]


def test_an_out_of_memory_round_is_reverted_not_fatal(rig):
    rig.db = [RuntimeError("torch.OutOfMemoryError: CUDA out of memory")]
    ply, result, rep = rig.train.grow(rig.ctx, rig.base, rig.ply, 30_000, 3, {}, {})
    assert result == rig.base and rep["stop"]["reason"] == "gpu-full" and rep["best"] == 0


def test_growth_resumes_from_its_record_and_a_paused_round_is_not_lost(rig):
    from plypipe.util import Paused
    rig.db = [28.4, Paused("saved a restore point")]
    with pytest.raises(Paused):
        rig.train.grow(rig.ctx, rig.base, rig.ply, 30_000, 3, {}, {})
    rec = json.loads((rig.base / "growth.json").read_text())
    assert [r["round"] for r in rec["rounds"]] == [0, 1]  # the paused round 2 is not recorded as tried
    rig.db = [28.9, 28.95]  # the same run again: round 1 is not redone, round 2 continues (its restore point)
    _, result, rep = rig.train.grow(rig.ctx, rig.base, rig.ply, 30_000, 3, {}, {})
    assert [r["round"] for r in rep["rounds"]] == [0, 1, 2, 3]
    assert result.name == "grow-2" and rep["stop"]["reason"] == "converged"
