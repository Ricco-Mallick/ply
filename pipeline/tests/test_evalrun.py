"""The pure half of pipeline/modal/ply_eval.py: variants, restore and skip lists, resume, saving, estimates, metrics."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from plypipe import cli, evalrun


# ------------------------------------------------------------------ variants

def opts(args):
    ns = cli.build_parser().parse_args(["run", "in", "--out", "out", "--only", "train", *args])
    return {k: v for k, v in vars(ns).items() if k not in ("cmd", "input", "out", "only")}


def test_the_baseline_reproduces_the_saved_runs_settings_apart_from_steps_and_the_block():
    saved = opts(["--preset", "high"])                    # the saved High chapel run's job.json: every other option at its default
    for name in ("q0a", "q0b", "smoke"):
        ours = opts(evalrun.VARIANTS[name]["args"])
        differ = {k for k in saved if saved[k] != ours[k]}
        assert differ == {"iterations", "cap_max", "holdout_block"}, name
        assert ours["cap_max"] == 2_000_000 and ours["holdout_block"] == 20 and ours["holdout"] == 30   # the plan's cap for that run
    assert opts(evalrun.VARIANTS["q0a"]["args"])["iterations"] == 15_000 and opts(evalrun.VARIANTS["smoke"]["args"])["iterations"] == 300
    assert evalrun.VARIANTS["q0a"]["args"] == evalrun.VARIANTS["q0b"]["args"]


def test_each_variant_changes_one_thing_against_the_baseline():
    base = opts(evalrun.BASE)
    def diff(name):
        ours = opts(evalrun.VARIANTS[name]["args"])
        return {k: ours[k] for k in base if base[k] != ours[k]}
    assert diff("q2") == {"pose_opt": True} and diff("q4a") == {"depth_loss": True}
    assert diff("q5a") == {"opacity_reg": 0.005} and diff("q5b") == {"opacity_reg": 0.002}
    assert diff("q5c") == {"opacity_reg": 0.001} and diff("q5d") == {"opacity_reg": 0.0}
    assert evalrun.VARIANTS["q5a"]["args"][-2:] == ["--opacity-reg", "0.005"]
    assert diff("q1") == {"bilateral_grid": True, "exposure_fold": 2000}       # the fold is what makes the grid's twin shippable: one change
    assert evalrun.VARIANTS["q1"]["args"][-3:] == ["--bilateral-grid", "--exposure-fold", "2000"]


def test_resolve_names_and_the_args_escape_hatch():
    assert evalrun.resolve("q2")["source"] == "table"
    adhoc = evalrun.resolve("q9", "--sh-degree 2 --iterations 8000")
    assert adhoc["source"] == "--args" and adhoc["args"] == evalrun.BASE + ["--sh-degree", "2", "--iterations", "8000"]
    assert evalrun.steps_of(adhoc["args"]) == 8000                       # a later option wins, as argparse reads it
    assert opts(adhoc["args"])["iterations"] == 8000
    for bad in ("", "Q2", "a b", "../x", "x" * 41):
        with pytest.raises(ValueError):
            evalrun.resolve(bad, "--pose-opt")
    with pytest.raises(ValueError, match="fixed"):
        evalrun.resolve("q0a", "--sh-degree 2")
    with pytest.raises(ValueError, match="unknown variant"):
        evalrun.resolve("q9")


def test_the_small_scenario_runs_the_same_variants_on_its_own_baseline():
    small = opts(evalrun.SMALL)
    assert {k for k, v in opts(evalrun.BASE).items() if v != small[k]} == {"cap_max", "holdout", "holdout_block"}
    assert small["cap_max"] == 600_000 and small["holdout"] == 8 and small["holdout_block"] == 10
    for name, v in evalrun.VARIANTS.items():
        ours, chapel = evalrun.resolve(name, scenario="drjohnson")["args"], evalrun.resolve(name)["args"]
        small_only = v.get("scenario_add", {}).get("drjohnson", [])
        assert chapel[len(evalrun.BASE):] == v["add"] and ours[len(evalrun.SMALL):] == v["add"] + small_only, name   # a variant adds the same thing to both
        assert chapel == v["args"]
    # the one exception: the small twin trains under the phone size, so only there the phone file's size is set, for it to be cut down at all
    assert [n for n, v in evalrun.VARIANTS.items() if v.get("scenario_add")] == ["q6"]
    assert evalrun.resolve("q6", scenario="drjohnson")["args"] == evalrun.SMALL + ["--lite-finetune", "3000", "--lite-max", "240000"]
    assert evalrun.resolve("q6")["args"] == evalrun.BASE + ["--lite-finetune", "3000"]
    assert evalrun.resolve("smoke", scenario="drjohnson")["args"] == ["--preset", "high", "--iterations", "300", "--cap-max", "600000",
                                                                      "--holdout", "8", "--holdout-block", "10"]
    # unchanged, so the smoke run saved on the chapel's Volume is still the same experiment
    assert evalrun.VARIANTS["smoke"]["args"] == ["--preset", "high", "--iterations", "300", "--cap-max", "2000000", "--holdout", "30", "--holdout-block", "20"]
    assert evalrun.resolve("q9", "--sh-degree 2", "drjohnson")["args"] == evalrun.SMALL + ["--sh-degree", "2"]
    with pytest.raises(ValueError, match="unknown scenario"):
        evalrun.resolve("q0a", scenario="garden")


def test_the_small_scenario_has_its_own_saved_solve_container_and_estimate():
    job = evalrun.make_job("drjohnson", "", "q5b", gpus=("L4", "L40S"))
    assert job["src"] == "drjohnson/prep" and (job["cpu"], job["memoryGib"], job["needsIndex"]) == (4, 16, False)
    # every export trains the phone file on for 3k steps, so every estimate counts them; --lite-finetune 0 is the run without
    assert job["args"] == evalrun.SMALL + ["--opacity-reg", "0.002"] and job["ceilingMin"] == 29   # the function's timeout is q1's
    l4, l40s = job["estimates"]["L4"], job["estimates"]["L40S"]
    assert (l4["fineTuneSteps"], l4["trainMin"], l4["totalMin"], l4["timeoutMin"]) == (3000, 9, 13, 20)
    assert (l40s["trainMin"], l40s["totalMin"], l40s["timeoutMin"]) == (5, 9, 14)
    assert l4["perHour"] == pytest.approx(0.80 + 4 * 0.047 + 16 * 0.008) and l4["usd"] == 0.24 and l40s["usd"] == 0.34   # the slower GPU is the cheaper run
    assert "15000 steps + 3000 for the phone file" in evalrun.describe(job)
    off = evalrun.estimate(job["args"] + ["--lite-finetune", "0"], "L4", scenario="drjohnson")
    assert (off["trainMin"], off["totalMin"], off["timeoutMin"], off["usd"]) == (7, 11, 17, 0.20) and "fineTuneSteps" not in off
    assert "4 cores, 16 GiB" in evalrun.describe(job) and "/vol/drjohnson/eval/q5b/" in evalrun.describe(job)
    chapel = evalrun.make_job("chapel", "", "q0a")
    assert chapel["src"] == "chapel/high/job" and (chapel["cpu"], chapel["memoryGib"], chapel["needsIndex"]) == (8, 32, True)
    assert chapel["estimates"]["L40S"]["usd"] > 2 * l4["usd"]
    hour = evalrun.invocation_record(1.0, 3600.0, {"name": "NVIDIA L4", "priced": "L4"}, [], [], None, job["cpu"], job["memoryGib"])
    assert hour["costUsd"] == pytest.approx(1.116) and hour["pricePerHour"] == l4["perHour"]


# ------------------------------------------------------------------ price, estimate, stop-loss

def test_gpu_names_map_to_the_price_table():
    assert [evalrun.gpu_key(n) for n in ("NVIDIA L40S", "NVIDIA L4", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-40GB", None)] == \
        ["L40S", "L4", "A100-80GB", "H100", None, None]
    assert evalrun.per_hour("NVIDIA L40S") == pytest.approx(1.95 + 8 * 0.047 + 32 * 0.008)
    assert evalrun.per_hour("L4", cpu=0, memory_gib=0) == 0.80 and evalrun.per_hour("T4") is None
    assert evalrun.parse_gpu("NVIDIA L40S, 46068 MiB\n") == {"name": "NVIDIA L40S", "memoryMiB": 46068, "priced": "L40S"}
    assert evalrun.parse_gpu("") == {"name": None, "memoryMiB": None, "priced": None}


def test_a_15k_run_with_the_phone_files_3k_is_9_minutes_of_training_and_14_in_all_with_a_1_5x_stop_loss():
    e = evalrun.estimate(evalrun.VARIANTS["q0a"]["args"], "L40S")
    assert (e["trainMin"], e["totalMin"], e["timeoutMin"]) == (9, 14, 21) and e["usd"] == 0.60 and e["maxUsd"] == 0.90
    e = evalrun.estimate(evalrun.VARIANTS["q0a"]["args"] + ["--lite-finetune", "0"], "L40S")       # the 15k steps alone
    assert (e["trainMin"], e["totalMin"], e["timeoutMin"]) == (8, 13, 20) and e["usd"] == 0.56 and e["maxUsd"] == 0.86
    assert evalrun.estimate(evalrun.VARIANTS["smoke"]["args"], "L40S")["timeoutMin"] == 11
    assert evalrun.make_job("chapel", "chapel/high/job", "smoke")["estimates"]["L40S"]["timeoutMin"] == 12     # the table sets its own
    assert evalrun.estimate(evalrun.BASE, "NVIDIA L4")["trainMin"] == 18
    with pytest.raises(ValueError, match="L4 or L40S"):
        evalrun.estimate(evalrun.BASE, "H100")
    with pytest.raises(ValueError, match="--iterations"):
        evalrun.estimate(["--preset", "high"], "L40S")


def test_q1_counts_the_grids_slower_steps_the_fold_and_the_steps_after_it():
    e = evalrun.estimate(evalrun.VARIANTS["q1"]["args"], "L40S")
    assert (e["steps"], e["gridSlowdown"], e["exposureFoldSteps"], e["fineTuneSteps"]) == (15000, 1.5, 2000, 3000)
    # (15000 x 1.5 + 2000 + 3000) steps at the chapel's 28.6 ms; the fold's fit, PLY and second start add 2 min to the 5
    assert (e["trainMin"], e["overheadMin"], e["totalMin"], e["timeoutMin"]) == (14, 7, 21, 32) and e["usd"] == 0.90 and e["maxUsd"] == 1.38
    job = evalrun.make_job("drjohnson", "", "q1", gpus=("L4", "L40S"))
    l4, l40s = job["estimates"]["L4"], job["estimates"]["L40S"]
    assert (l4["trainMin"], l4["overheadMin"], l4["totalMin"], l4["timeoutMin"], l4["usd"]) == (13, 6, 19, 29, 0.35)
    assert (l40s["trainMin"], l40s["totalMin"], l40s["timeoutMin"], l40s["usd"]) == (7, 13, 20, 0.49) and job["ceilingMin"] == 29
    text = evalrun.describe(job)
    assert "15000 steps at x1.5 for the bilateral grid (an estimate) + 2000 after the exposure fold + 3000 for the phone file" in text
    assert "6 min restore/export/scoring and the fold" in text and "--bilateral-grid --exposure-fold 2000" in text
    fold_only = evalrun.estimate(evalrun.BASE + ["--bilateral-grid", "--exposure-fold", "0"], "L40S")    # no steps after it: the fold still takes its time
    assert (fold_only["exposureFoldSteps"], fold_only["trainMin"], fold_only["overheadMin"]) == (0, 13, 7)
    for name in ("q0a", "q2", "q6"):                                                 # a run without the options is estimated as before
        e = evalrun.estimate(evalrun.VARIANTS[name]["args"], "L40S")
        assert "gridSlowdown" not in e and "exposureFoldSteps" not in e and (e["trainMin"], e["overheadMin"], e["timeoutMin"]) == (9, 5, 21), name


def test_the_function_timeout_follows_the_slowest_allowed_gpu():
    assert evalrun.default_ceiling_min(("L40S",)) == 32 and evalrun.default_ceiling_min(("L40S", "L4")) == 51     # q1 is the longest variant
    job = evalrun.make_job("chapel", "/chapel/high/job/", "q0a")
    assert job["src"] == "chapel/high/job" and job["ceilingMin"] == 32 and set(job["estimates"]) == {"L40S"}
    assert job["estimates"]["L40S"]["timeoutMin"] == 21                              # its own stop-loss is what it was
    with pytest.raises(ValueError, match="PLY_EVAL_TIMEOUT_MIN"):
        evalrun.make_job("chapel", "chapel/high/job", "q0a", gpus=("L40S", "L4"), ceiling_min=20)
    both = evalrun.make_job("chapel", "chapel/high/job", "q0a", gpus=("L40S", "L4"))
    assert evalrun.stop_loss_min(both, {"priced": "L40S"}) == 21 and evalrun.stop_loss_min(both, {"priced": "L4"}) == 35
    assert evalrun.stop_loss_min(both, {"priced": None}) == 35                       # unknown GPU: the slowest budget


def test_plan_prints_what_would_run_without_modal(capsys):
    assert evalrun.main(["plan", "--variant", "q5a"]) == 0
    text = capsys.readouterr().out
    for needle in ("q5a", "--opacity-reg 0.005", "--only train", "--only export", "stop-loss 21 min", "$0.60", "15000 steps + 3000 for the phone file", "data/images -> ../dense/images",
                   "dense/images/", "frames/", "/vol/chapel/eval/q5a/"):
        assert needle in text, needle


def test_the_plan_follows_the_scenario_the_launch_would_use(capsys, monkeypatch):
    monkeypatch.setenv("PLY_EVAL_SCENARIO", "drjohnson")
    assert evalrun.main(["plan", "--variant", "q1"]) == 0
    text = capsys.readouterr().out
    for needle in ("/vol/drjohnson/eval/q1/", "--cap-max 600000 --holdout 8 --holdout-block 10 --bilateral-grid --exposure-fold 2000", "stop-loss 20 min"):
        assert needle in text, needle
    assert evalrun.main(["plan", "--variant", "q1", "--scenario", "chapel"]) == 0 and "/vol/chapel/eval/q1/" in capsys.readouterr().out


def test_the_pure_half_never_imports_modal():
    root = str(Path(__file__).resolve().parents[1])
    code = "import sys, plypipe.evalrun, plypipe.evalsplit, plypipe.train_launcher; assert 'modal' not in sys.modules and 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": root})


# ------------------------------------------------------------------ what a run restores

def saved_job(root: Path, links: bool = True) -> Path:
    def put(rel, data=b"x"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())
    for rel in ("scene.json", "shell.json", "places.json", "plan.json", "occupancy.json", "occupancy.png", "frames_index.json", "job.json",
                "log.jsonl", "scene.ply", "scene_clean.ply", "scene.spz", "scene_lite.spz", "eval/val_stats.json", "logs/train.log",
                "frames/000000.jpg", "frames_raw/clip.mp4", "colmap/database.db", "dense/sparse/0/images.bin", "dense/stereo/depth_maps/a.bin",
                "dense/sparse_txt/images.txt", "train/ply/point_cloud_9.ply", "train-snapshot/point_cloud_5.ply", "_probe.ply"):
        put(rel)
    for rel in ("cameras", "images", "points3D"):
        put(f"data/sparse/0/{rel}.bin")
        put(f"data/sparse/0/{rel}.txt")
    for n in range(3):
        put(f"dense/images/{n:06d}.jpg", b"jpg")
    for stage in ("frames", "sfm", "align", "occupancy", "label", "train", "export", "mesh"):
        put(f".done/{stage}.json", "{}")
    put("report.json", json.dumps({"frames": {"kept": 3}, "sfm": {"registered": 3}, "train": {"eval": {"psnr": 1}}, "export": {"gaussiansOut": 5},
                                   "quality": {"ok": True}, "mesh": {}}))
    if links:
        os.symlink("../dense/images", root / "data" / "images")
    return root


def files_under(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}


def test_restore_takes_the_solve_and_the_photos_and_leaves_the_rest(tmp_path):
    src = saved_job(tmp_path / "saved")
    got = evalrun.restore(src, tmp_path / "job")
    assert files_under(tmp_path / "job") == {
        "report.json", "plan.json", "scene.json", "shell.json", "places.json", "occupancy.json", "occupancy.png", "frames_index.json",
        ".done/frames.json", ".done/sfm.json", ".done/align.json", ".done/occupancy.json", ".done/label.json",
        *(f"data/sparse/0/{n}.{ext}" for n in ("cameras", "images", "points3D") for ext in ("bin", "txt")),
        *(f"dense/images/{n:06d}.jpg" for n in range(3)), "data/images"}
    assert got["files"] == 22 and got["bytes"] > 0
    assert json.loads((tmp_path / "job" / "report.json").read_text()) == {"frames": {"kept": 3}, "sfm": {"registered": 3}}


def test_the_images_link_is_made_again_when_the_upload_dropped_it(tmp_path):
    src = saved_job(tmp_path / "saved", links=False)
    assert not (src / "data" / "images").exists()
    evalrun.restore(src, tmp_path / "job")
    link = tmp_path / "job" / "data" / "images"
    assert link.is_symlink() and os.readlink(link) == "../dense/images" and len(os.readlink(link)) == 15
    assert sorted(p.name for p in link.glob("*.jpg")) == ["000000.jpg", "000001.jpg", "000002.jpg"]
    evalrun.restore(src, tmp_path / "job")                                           # idempotent
    assert link.is_symlink() and evalrun.check_job(tmp_path / "job") == []


def test_a_saved_link_is_not_followed_and_a_dereferenced_upload_is_copied(tmp_path):
    src = saved_job(tmp_path / "saved")                                              # data/images is a link here
    evalrun.restore(src, tmp_path / "a")
    assert os.readlink(tmp_path / "a" / "data" / "images") == "../dense/images"
    plain = saved_job(tmp_path / "plain", links=False)                               # photos only under data/images, as a real folder
    (plain / "data" / "images").mkdir()
    (plain / "data" / "images" / "000000.jpg").write_bytes(b"jpg")
    for p in (plain / "dense" / "images").iterdir():
        p.unlink()
    (plain / "dense" / "images").rmdir()
    evalrun.restore(plain, tmp_path / "b")
    assert not (tmp_path / "b" / "data" / "images").is_symlink() and (tmp_path / "b" / "data" / "images" / "000000.jpg").exists()


def test_a_restored_job_is_checked_before_any_stage_runs(tmp_path):
    evalrun.restore(saved_job(tmp_path / "saved"), tmp_path / "job")
    assert evalrun.check_job(tmp_path / "job") == []
    (tmp_path / "job" / "data" / "sparse" / "0" / "images.bin").unlink()
    (tmp_path / "job" / "frames_index.json").unlink()
    assert evalrun.check_job(tmp_path / "job") == ["data/sparse/0/images.bin", "frames_index.json"]
    assert evalrun.check_job(tmp_path / "job", index=False) == ["data/sparse/0/images.bin"]     # a photo set has no index
    shutil.rmtree(tmp_path / "job" / "dense")
    assert any("no photos" in p for p in evalrun.check_job(tmp_path / "job"))


def test_restore_does_not_list_folders_it_will_not_copy(tmp_path, monkeypatch):
    src = saved_job(tmp_path / "saved")
    listed = []
    real = os.walk
    def spy(top, *a, **k):
        for root, dirs, files in real(top, *a, **k):
            listed.append(Path(root).relative_to(top).as_posix())
            yield root, dirs, files
    monkeypatch.setattr(evalrun.os, "walk", spy)
    evalrun.restore(src, tmp_path / "job")
    assert not {"frames", "frames_raw", "colmap", "eval", "logs", "train", "dense/stereo", "dense/sparse", "dense/sparse_txt"} & set(listed)
    assert {".", ".done", "data", "data/sparse/0", "dense", "dense/images"} <= set(listed)


# ------------------------------------------------------------------ resume and save

def job_with(tmp_path, train=False, export=False):
    job = tmp_path / "job"
    (job / ".done").mkdir(parents=True)
    if train:
        (job / ".done" / "train.json").write_text("{}")
        (job / "scene.ply").write_bytes(b"ply")
    if export:
        (job / ".done" / "export.json").write_text("{}")
        (job / "scene.spz").write_bytes(b"spz")
        (job / "scene_lite.spz").write_bytes(b"spz")
    return job


def test_stages_to_run_follows_what_is_really_finished(tmp_path):
    assert evalrun.stages_to_run(job_with(tmp_path / "a")) == ["train", "export"]
    assert evalrun.stages_to_run(job_with(tmp_path / "b", train=True)) == ["export"]
    assert evalrun.stages_to_run(job_with(tmp_path / "c", train=True, export=True)) == []
    half = job_with(tmp_path / "d", train=True)
    (half / "scene.ply").unlink()                                       # a marker without the file it vouches for
    assert evalrun.stages_to_run(half) == ["train", "export"]
    lost = job_with(tmp_path / "e", train=True, export=True)
    (lost / "scene_lite.spz").unlink()
    assert evalrun.stages_to_run(lost) == ["export"]


def test_clear_stale_forgets_a_stage_and_everything_after_it(tmp_path):
    job = job_with(tmp_path, train=True, export=True)
    evalrun.clear_stale(job, ["export"])
    assert not (job / ".done" / "export.json").exists() and (job / ".done" / "train.json").exists() and (job / "scene.ply").exists()
    evalrun.clear_stale(job, ["train", "export"])
    assert not (job / ".done" / "train.json").exists() and not (job / "scene.ply").exists()


def test_a_relaunch_layers_the_saved_variant_over_the_restore_but_not_snapshots(tmp_path):
    saved = tmp_path / "eval" / "q0a"
    for rel, data in ((".done/train.json", "{}"), ("scene.ply", "ply"), ("report.json", '{"train": {}}'), ("train-snapshot/point_cloud_5.ply", "s"),
                      ("eval/per_view.json", "{}"), ("scene.spz.part", "x")):
        (saved / rel).parent.mkdir(parents=True, exist_ok=True)
        (saved / rel).write_text(data)
    job = tmp_path / "job"
    job.mkdir()
    (job / "report.json").write_text('{"frames": {}}')
    assert evalrun.layer(saved, job) == 4
    assert json.loads((job / "report.json").read_text()) == {"train": {}} and (job / "scene.ply").exists()
    assert not (job / "train-snapshot").exists() and not (job / "scene.spz.part").exists()
    assert evalrun.stages_to_run(job) == ["export"]


def test_save_skips_the_shared_inputs_and_bulk_and_writes_markers_last(tmp_path, monkeypatch):
    job = saved_job(tmp_path / "job")
    (job / "metrics.json").write_text("{}")
    order = []
    real = evalrun.shutil.copy2
    monkeypatch.setattr(evalrun.shutil, "copy2", lambda a, b, *k: (order.append(Path(b).relative_to(tmp_path / "vol").as_posix().removesuffix(".part")), real(a, b))[1])
    got = evalrun.save(job, tmp_path / "vol")
    assert files_under(tmp_path / "vol") == {"scene.json", "shell.json", "places.json", "plan.json", "occupancy.json", "occupancy.png", "frames_index.json",
                                             "job.json", "log.jsonl", "scene.ply", "scene.spz", "scene_lite.spz", "eval/val_stats.json", "logs/train.log",
                                             "report.json", "metrics.json", *(f".done/{s}.json" for s in ("frames", "sfm", "align", "occupancy", "label", "train", "export", "mesh"))}
    assert got["files"] == len(order) == 24
    assert order[-1].startswith(".done/") and order[-9] == "report.json" and all(o.startswith(".done/") for o in order[-8:])
    assert evalrun.save(job, tmp_path / "vol") == {"files": 0, "bytes": 0}              # nothing changed, nothing copied
    (job / "scene.ply").write_bytes(b"a longer ply")
    assert evalrun.save(job, tmp_path / "vol")["files"] == 1 and (tmp_path / "vol" / "scene.ply").read_bytes() == b"a longer ply"
    assert not list((tmp_path / "vol").rglob("*.part"))


# ------------------------------------------------------------------ metrics

def finished_report():
    score = {"psnr": 24.0, "ssim": 0.85, "lpips": 0.25, "ccPsnr": 24.5, "ccSsim": 0.86, "ccLpips": 0.24, "gaussians": 1_900_000.0, "secondsPerView": 0.007, "views": 41}
    return {"frames": {"kept": 1236}, "sfm": {"registered": 1223},
            "train": {"maxSteps": 15000, "capMax": 2000000, "eval": score, "evalBlock": {**score, "psnr": 22.0, "ccPsnr": 22.5, "views": 20},
                      "blockImages": ["000117.jpg", "000118.jpg"], "blockClip": "a.mp4", "blockSeconds": 8.9, "seconds": 480.0},
            "export": {"gaussiansIn": 2_000_000, "gaussiansOut": 1_500_000, "keptFraction": 0.75, "liteGaussians": 800_000, "seconds": 210.5,
                       "viewerEval": {"psnr": 23.9, "views": 12},
                       "shippedEval": {"full": {"psnr": 23.5, "ssim": 0.8, "brightness": 1.0, "holes": 0.01, "views": 12},
                                       "lite": {"psnr": 21.5, "ssim": 0.76, "brightness": 1.0, "holes": 0.02, "views": 12}}},
            "quality": {"ok": True, "issues": [{"level": "warn", "code": "pruned-some", "message": "x"}]}}


def job_for(variant="q0a", args=None):
    return evalrun.make_job("chapel", "chapel/high/job", variant, revision={"revision": "abc+dirty", "codeHash": "123"})


def test_metrics_carry_both_hold_outs_the_viewer_scores_splats_files_time_and_cost():
    gpu = {"name": "NVIDIA L40S", "memoryMiB": 46068, "priced": "L40S"}
    inv = evalrun.invocation_record(1_800_000_000, 900.0, gpu, ["train", "export"], [])
    m = evalrun.assemble(job_for(), finished_report(), gpu=gpu, seconds={"restore": 60.0, "save": 30.0, "wall": 900.0},
                         sizes={"scene.ply": 100, "scene.spz": 10}, invocation=inv)
    assert m["complete"] and m["missing"] == [] and m["schema"] == 1 and m["variant"] == "q0a" and m["role"] == "baseline"
    assert m["args"] == evalrun.VARIANTS["q0a"]["args"] and m["argsSource"] == "table" and m["revision"]["revision"] == "abc+dirty"
    assert m["gpu"] == gpu and m["steps"] == 15000 and m["capMax"] == 2000000
    assert m["scores"]["every"]["ccPsnr"] == 24.5 and m["scores"]["every"]["views"] == 41 and "gaussians" not in m["scores"]["every"]
    assert m["scores"]["block"]["ccPsnr"] == 22.5 and m["scores"]["block"]["views"] == 20 and m["scores"]["block"]["images"] == ["000117.jpg", "000118.jpg"]
    assert m["viewer"]["desktop"]["psnr"] == 23.5 and m["viewer"]["phone"]["psnr"] == 21.5 and m["viewer"]["cleaned"]["psnr"] == 23.9
    assert m["splats"] == {"atEval": 1_900_000.0, "trained": 2_000_000, "shipped": 1_500_000, "keptFraction": 0.75, "phone": 800_000}
    assert m["files"] == {"scene.ply": 100, "scene.spz": 10} and m["seconds"] == {"restore": 60.0, "save": 30.0, "wall": 900.0, "train": 480.0, "export": 210.5}
    assert m["costUsd"] == round(900 / 3600 * (1.95 + 8 * 0.047 + 32 * 0.008), 2) == 0.65 and m["wallMinutes"] == 15.0
    assert m["gate"] == {"ok": True, "issues": [{"level": "warn", "code": "pruned-some"}]}
    assert "q0a: complete" in evalrun.summary(m) and "cc-PSNR 22.5" in evalrun.summary(m)
    assert "phoneFinetune" not in m["viewer"] and "fine-tune" not in evalrun.summary(m)


def test_metrics_carry_the_phone_files_fine_tune_next_to_the_cut_down_score():
    gpu = {"name": "NVIDIA L40S", "memoryMiB": 46068, "priced": "L40S"}
    report = finished_report()
    report["export"]["liteFinetune"] = {"steps": 3000, "shipped": "finetuned", "before": {"psnr": 21.4, "holes": 0.02},
                                        "after": {"psnr": 23.1, "holes": 0.01}, "eval": {"psnr": 23.6, "views": 41}, "seconds": 95.0}
    job = evalrun.make_job("chapel", "chapel/high/job", "q6")
    m = evalrun.assemble(job, report, gpu=gpu, seconds={}, sizes={}, invocation=evalrun.invocation_record(1_800_000_000, 900.0, gpu, ["train", "export"], []))
    assert m["complete"] and m["viewer"]["phoneFinetune"] == report["export"]["liteFinetune"] and m["viewer"]["phone"]["psnr"] == 21.5
    assert "phone file, 3000 fine-tune steps: PSNR 21.4 before, 23.1 fine-tuned; shipped the finetuned one" in evalrun.summary(m)
    report["export"]["liteFinetune"] = {"steps": 3000, "shipped": "untuned", "why": "CUDA out of memory"}
    m = evalrun.assemble(job, report, gpu=gpu, seconds={}, sizes={}, invocation=evalrun.invocation_record(1_800_000_000, 900.0, gpu, ["export"], []))
    assert "shipped the untuned one (CUDA out of memory)" in evalrun.summary(m)
    assert not evalrun.savable(Path("train-lite/ckpts/ckpt_2999_rank0.pt")) and evalrun.savable(Path("eval/lite/val_stats.json"))


def test_metrics_carry_the_exposure_folds_fit_and_the_summary_shows_it_with_the_viewers_brightness():
    gpu = {"name": "NVIDIA L40S", "memoryMiB": 46068, "priced": "L40S"}
    inv = evalrun.invocation_record(1_800_000_000, 900.0, gpu, ["train", "export"], [])
    job = evalrun.make_job("chapel", "chapel/high/job", "q1")
    report = finished_report()
    assert "exposureFold" not in evalrun.assemble(job, report, gpu=gpu, seconds={}, sizes={}, invocation=inv)
    assert "exposure fold" not in evalrun.summary(evalrun.assemble(job, report, gpu=gpu, seconds={}, sizes={}, invocation=inv))
    fold = {"steps": 2000, "folded": True, "settled": True, "views": 44, "pixels": 5_000_000, "usable": 0.83, "condition": 512.3,
            "matrix": [[0.84, 0.03, -0.02], [0.02, 0.81, 0.04], [-0.03, 0.05, 0.86]], "offset": [0.02, -0.01, 0.015], "gains": [0.85, 0.87, 0.88],
            "maeBefore": 0.07, "maeAfter": 0.03, "brightnessBefore": 1.21, "brightnessAfter": 1.0, "trained": {"eval": {"psnr": 23.0}}}
    report["train"]["exposureFold"] = fold
    m = evalrun.assemble(job, report, gpu=gpu, seconds={}, sizes={}, invocation=inv)
    assert m["complete"] and m["exposureFold"] == fold and m["args"][-3:] == ["--bilateral-grid", "--exposure-fold", "2000"]
    assert ("  exposure fold: gains 0.85/0.87/0.88 fitted on 44 training views, brightness there x1.21 -> x1.0 of the photos, "
            "then 2000 steps without the appearance model; viewer brightness desktop x1.0") in evalrun.summary(m)
    report["train"]["exposureFold"] = {"steps": 2000, "folded": False, "settled": False, "views": 3, "pixels": 900, "why": "too little to fit on"}
    m = evalrun.assemble(job, report, gpu=gpu, seconds={}, sizes={}, invocation=inv)
    assert "  exposure fold: NOT folded (too little to fit on); viewer brightness desktop x1.0" in evalrun.summary(m)
    report["train"]["exposureFold"] = {**fold, "settled": False, "why": "the 2000 steps after the fold failed: CUDA out of memory"}
    m = evalrun.assemble(job, report, gpu=gpu, seconds={}, sizes={}, invocation=inv)
    assert "no steps after it (the 2000 steps after the fold failed: CUDA out of memory)" in evalrun.summary(m)
    assert not evalrun.savable(Path("train/settle/ckpts/ckpt_1999_rank0.pt")) and not evalrun.savable(Path("train/folded.ply"))
    assert evalrun.savable(Path("eval/trained/val_stats.json"))                    # the scores of the model before the fold are kept


def test_a_stopped_run_still_gets_metrics_that_say_what_is_missing():
    gpu = {"name": "NVIDIA L40S", "memoryMiB": 46068, "priced": "L40S"}
    report = finished_report()
    del report["export"], report["quality"], report["train"]["evalBlock"]
    inv = evalrun.invocation_record(1_800_000_000, 600.0, gpu, ["train"], [], "stop-loss: export still running")
    m = evalrun.assemble(job_for(), report, gpu=gpu, seconds={}, sizes={}, invocation=inv)
    assert not m["complete"] and m["missing"] == ["scores.block", "export", "viewer.desktop", "viewer.phone"]
    assert m["scores"]["block"] is None and m["gate"] is None and m["invocations"][0]["error"].startswith("stop-loss")
    assert evalrun.assemble(job_for(), {}, gpu=gpu, seconds={}, sizes={}, invocation=inv)["missing"][0] == "train"
    assert "INCOMPLETE" in evalrun.summary(m)


def test_a_relaunch_adds_its_cost_to_the_first_runs():
    gpu = {"name": "NVIDIA L40S", "memoryMiB": 46068, "priced": "L40S"}
    first = evalrun.assemble(job_for(), {}, gpu=gpu, seconds={}, sizes={}, invocation=evalrun.invocation_record(1.0, 600.0, gpu, ["train"], [], "exit 1"))
    second = evalrun.assemble(job_for(), finished_report(), gpu=gpu, seconds={}, sizes={}, previous=first,
                              invocation=evalrun.invocation_record(2.0, 300.0, gpu, ["export"], ["train"]))
    assert [i["ran"] for i in second["invocations"]] == [["train"], ["export"]] and second["invocations"][1]["resumedAfter"] == ["train"]
    assert second["costUsd"] == round(900 / 3600 * 2.582, 2) and second["wallMinutes"] == 15.0
    unpriced = evalrun.invocation_record(1.0, 60.0, {"name": "NVIDIA T4", "priced": None}, [], [])
    assert unpriced["costUsd"] is None


def test_a_variant_folder_belongs_to_one_set_of_arguments(tmp_path):
    path = tmp_path / "metrics.json"
    assert evalrun.read_metrics(path) is None
    path.write_text(json.dumps({"schema": 1, "args": evalrun.BASE}))
    previous = evalrun.read_metrics(path)
    evalrun.check_resume(previous, {"variant": "q0a", "args": evalrun.BASE})
    with pytest.raises(ValueError, match="other arguments"):
        evalrun.check_resume(previous, {"variant": "q0a", "args": evalrun.BASE + ["--pose-opt"]})
    evalrun.check_resume(None, {"variant": "q0a", "args": []})
    path.write_text("[1]")
    assert evalrun.read_metrics(path) is None


def test_collect_copies_the_small_result_files(tmp_path):
    files = {"chapel/eval/q0a/metrics.json": b'{"a": 1}', "chapel/eval/q0a/eval/per_view.json": b"[]", "chapel/eval/q0b/report.json": b"{}"}

    class Volume:
        def listdir(self, path):
            assert path == "chapel/eval"
            return [SimpleNamespace(path="chapel/eval/q0a"), SimpleNamespace(path="chapel/eval/q0b")]

        def read_file(self, path):
            if path not in files:
                raise FileNotFoundError(path)
            yield files[path][:3]
            yield files[path][3:]

    got = evalrun.collect(Volume(), "chapel", tmp_path / "out")
    assert got == {"q0a": ["metrics.json", "eval/per_view.json"], "q0b": ["report.json"]}
    assert (tmp_path / "out" / "q0a" / "metrics.json").read_bytes() == b'{"a": 1}' and (tmp_path / "out" / "q0a" / "eval" / "per_view.json").exists()


def test_code_revision_names_the_commit_and_flags_edits(tmp_path):
    (tmp_path / "pipeline" / "plypipe").mkdir(parents=True)
    (tmp_path / "pipeline" / "plypipe" / "a.py").write_text("x = 1\n")
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t", *a], check=True, capture_output=True)
    assert evalrun.code_revision(tmp_path)["revision"] == "unknown"
    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "x")
    clean = evalrun.code_revision(tmp_path)
    assert len(clean["revision"]) == 10 and not clean["revision"].endswith("+dirty")
    (tmp_path / "pipeline" / "plypipe" / "a.py").write_text("x = 2\n")
    dirty = evalrun.code_revision(tmp_path)
    assert dirty["revision"] == clean["revision"] + "+dirty" and dirty["codeHash"] != clean["codeHash"]
