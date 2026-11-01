import json

from plypipe.cli import main

from synth import write_synthetic_model


def events(out):
    return [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]


def test_cli_runs_resumes_and_forces(tmp_path):
    out = tmp_path / "scene"
    write_synthetic_model(out / "dense" / "sparse_txt")
    (out / "dense" / "images").mkdir(parents=True)
    args = ["run", str(tmp_path / "unused.mp4"), "--out", str(out), "--from", "align", "--until", "occupancy", "--quiet"]
    assert main(args) == 0
    ev = events(out)
    assert [(e["stage"], e["event"]) for e in ev if e["event"] in ("done", "skip")] == [("align", "done"), ("occupancy", "done")]
    for f in ("scene.json", "occupancy.png", "occupancy.json", "report.json", "job.json"):
        assert (out / f).exists(), f
    rep = json.loads((out / "report.json").read_text())
    assert rep["align"]["scaleSource"] == "camera-height-prior" and rep["occupancy"]["free"] > 0

    n = len(ev)
    assert main(args) == 0  # second run: everything skipped
    assert [(e["stage"], e["event"]) for e in events(out)[n:] if e["event"] in ("done", "skip")] == [("align", "skip"), ("occupancy", "skip")]

    n = len(events(out))
    assert main(args + ["--force", "occupancy"]) == 0
    assert [(e["stage"], e["event"]) for e in events(out)[n:] if e["event"] in ("done", "skip")] == [("align", "skip"), ("occupancy", "done")]


def test_cli_reports_errors_and_exits_nonzero(tmp_path):
    out = tmp_path / "empty"
    assert main(["run", str(tmp_path / "nothing"), "--out", str(out), "--only", "align", "--quiet"]) == 1
    err = [e for e in events(out) if e["event"] == "error"]
    assert err and "sparse_txt" in err[0]["msg"]


def test_frame_rate_and_resolution_default_to_the_plan():
    """argparse defaults would beat plan.input_plan() (ctx.opt only falls back on None)."""
    from plypipe import cli as cli_mod
    a = cli_mod.build_parser().parse_args(["run", "walk.mp4", "--out", "o"])
    assert a.fps is None and a.long_side is None
    a = cli_mod.build_parser().parse_args(["run", "walk.mp4", "--out", "o", "--fps", "3", "--long-side", "1920"])
    assert a.fps == 3.0 and a.long_side == 1920
