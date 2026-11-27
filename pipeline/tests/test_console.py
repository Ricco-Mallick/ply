"""plypipe.console: the lines a person watches, and JSON wherever a program reads."""

import json

from plypipe import console
from plypipe.log import Log


def test_training_progress_reads_as_one_line():
    line = console.format({"t": "2026-11-27T09:41:07+0530", "stage": "train", "event": "progress", "msg": "",
                           "train": {"step": 12630, "of": 30000, "loss": 0.0412, "splats": 1_120_000, "vramGB": 3.1,
                                     "stepsPerS": 9.8, "etaS": 1773, "round": "2"}})
    assert line.startswith("09:41:07  train")
    for bit in ("round 2", "42.1%", "step 12,630/30,000", "loss 0.041", "1.12M splats", "VRAM 3.1 GB", "9.8 steps/s", "ETA 29m"):
        assert bit in line


def test_restore_pause_and_done_lines():
    assert "restore point saved at step 12,630" in console.format(
        {"stage": "train", "event": "restore", "restore": {"step": 12630, "mb": 612, "seconds": 4.1, "why": "autosave"}})
    assert "resumed from the restore point at step 12,631" in console.format(
        {"stage": "train", "event": "restore", "restore": {"resumedAt": 12631, "of": 30000}})
    assert "PAUSED" in console.format({"stage": "train", "event": "paused", "msg": "saved"})
    done = console.format({"stage": "train", "event": "done", "seconds": 4400, "eval": {"psnr": 28.1, "ccPsnr": 29.3, "lpips": 0.21},
                           "growth": {"rounds": [{}, {}, {}], "best": 1, "needsBiggerGpu": True}})
    assert "1h13m" in done and "29.30 dB" in done and "2 growth rounds" in done and "bigger GPU" in done
    assert console.format({"stage": "train", "event": "info", "msg": "exec"}) is None


def test_log_is_json_for_programs_and_lines_for_people(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PLY_PRETTY", "0")
    Log(tmp_path).emit("frames", "start")
    assert json.loads(capsys.readouterr().out)["stage"] == "frames"
    monkeypatch.setenv("PLY_PRETTY", "1")
    Log(tmp_path).emit("frames", "start")
    assert capsys.readouterr().out.strip().endswith("== frames ==")
    assert all(json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines())  # the file stays JSON
