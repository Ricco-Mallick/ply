"""scripts/eval_table.py: the results table, the noise floor and the flags against it, from fixture metrics.json files."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval_table.py"
spec = importlib.util.spec_from_file_location("eval_table", SCRIPT)
eval_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_table)


def metrics(variant, every=(24.50, 0.8600, 0.2400), block=(22.00, 0.8000, 0.3000), desktop=23.5, phone=21.4, role=None, complete=True,
            kept=0.75, minutes=14.0, cost=0.6, steps=15000):
    return {"schema": 1, "variant": variant, "role": role, "steps": steps, "complete": complete, "missing": [],
            "scores": {"every": dict(zip(("ccPsnr", "ccSsim", "ccLpips"), every)), "block": dict(zip(("ccPsnr", "ccSsim", "ccLpips"), block))},
            "viewer": {"desktop": {"psnr": desktop}, "phone": {"psnr": phone}},
            "splats": {"keptFraction": kept, "shipped": int(2_000_000 * kept), "trained": 2_000_000}, "wallMinutes": minutes, "costUsd": cost}


def write(folder: Path, *runs):
    for r in runs:
        (folder / r["variant"]).mkdir(parents=True)
        (folder / r["variant"] / "metrics.json").write_text(json.dumps(r))
    return folder


def rows(text):
    return [line for line in text.splitlines() if line.startswith("|")]


def test_one_row_per_variant_with_both_hold_outs_viewer_scores_splats_minutes_and_cost(tmp_path):
    write(tmp_path, metrics("q0a", role="baseline"), metrics("q2", every=(24.9, 0.862, 0.236)))
    out = eval_table.render(eval_table.load_runs(tmp_path))
    first = rows(out)[:4]
    assert first[0].startswith("| variant | steps | every cc-PSNR | every cc-SSIM | every cc-LPIPS | block cc-PSNR | block cc-SSIM | block cc-LPIPS | "
                               "viewer PSNR desktop | viewer PSNR phone | brightness desktop / phone | splats kept | minutes | cost |")
    assert first[2] == "| q0a | 15000 | 24.50 | 0.8600 | 0.2400 | 22.00 | 0.8000 | 0.3000 | 23.50 | 21.40 | – / – | 75% (1.50M) | 14.0 | $0.60 |"
    assert first[3].startswith("| q2 | 15000 | 24.90 | 0.8620 | 0.2360 |")
    assert "Noise floor: needs two complete baselines (found: q0a)" in out


def test_noise_floor_is_the_gap_between_the_baselines_and_deltas_are_flagged_against_twice_it(tmp_path):
    write(tmp_path,
          metrics("q0a", role="baseline", every=(24.50, 0.8600, 0.2400), block=(22.00, 0.8000, 0.3000), desktop=23.5, phone=21.4),
          metrics("q0b", role="baseline", every=(24.60, 0.8610, 0.2390), block=(22.20, 0.8010, 0.2990), desktop=23.7, phone=21.6),
          metrics("q5a", every=(24.90, 0.8610, 0.2300), block=(22.10, 0.8000, 0.3200), desktop=23.6, phone=21.5))
    out = eval_table.render(eval_table.load_runs(tmp_path))
    assert "## Noise floor (q0a vs q0b)" in out
    cells = lambda row: [c.strip() for c in row.split("|")[2:-1]]
    floor = {r.split("|")[1].strip(): cells(r) for r in rows(out) if r.split("|")[1].strip() in dict((m[0], 1) for m in eval_table.METRICS)}
    assert floor["every cc-PSNR"] == ["24.55", "0.10", "0.20"] and floor["every cc-LPIPS"] == ["0.2395", "0.0010", "0.0020"]
    assert floor["block cc-PSNR"][1] == "0.20" and floor["viewer PSNR desktop"][1] == "0.20" and floor["viewer PSNR phone"][1] == "0.20"
    change = cells([r for r in rows(out) if r.startswith("| q5a |")][-1])
    assert change == ["+0.35 outside, better",          # 24.90 against the mean 24.55: more than twice the 0.10 floor
                      "+0.0005 inside",                  # inside twice the 0.001 floor
                      "-0.0095 outside, better",         # LPIPS down by more than twice its floor
                      "+0.00 inside", "-0.0005 inside",
                      "+0.0205 outside, worse",          # block LPIPS up
                      "+0.00 inside", "+0.00 inside"]


def test_identical_baselines_make_every_change_outside(tmp_path):
    write(tmp_path, metrics("q0a", role="baseline"), metrics("q0b", role="baseline"), metrics("q2", every=(24.5, 0.86, 0.2401)), metrics("q4a"))
    out = eval_table.render(eval_table.load_runs(tmp_path))
    q2 = next(r for r in rows(out) if r.startswith("| q2 | ") and "outside" in r)
    assert "+0.0001 outside, worse" in q2
    q4a = next(r for r in rows(out) if r.startswith("| q4a | ") and "inside" in r)
    assert "outside" not in q4a                                                  # exactly the baseline: zero change is inside zero noise


def test_incomplete_runs_and_missing_values_are_marked_not_dropped(tmp_path):
    partial = metrics("q2", complete=False)
    partial["scores"]["block"] = None
    partial["viewer"] = {"desktop": None, "phone": None}
    partial["costUsd"] = None
    write(tmp_path, metrics("q0a", role="baseline"), metrics("q0b", role="baseline", complete=False), partial)
    out = eval_table.render(eval_table.load_runs(tmp_path))
    assert "(incomplete)" in out and "q0b (incomplete)" in out
    assert "Noise floor: needs two complete baselines (found: q0a)" in out           # an incomplete baseline does not count
    q2 = next(r for r in rows(out) if r.startswith("| q2 (incomplete)"))
    assert q2.count("–") >= 5


def test_baselines_can_be_named_and_the_command_prints_to_stdout(tmp_path, capsys):
    write(tmp_path, metrics("a1"), metrics("a2", every=(24.6, 0.861, 0.239)), metrics("x1", every=(25.5, 0.86, 0.24)))
    assert eval_table.main([str(tmp_path), "--baselines", "a1,a2"]) == 0
    out = capsys.readouterr().out
    assert "## Noise floor (a1 vs a2)" in out and "| x1 |" in out and out.index("| a1 |") < out.index("| a2 |") < out.index("| x1 |")
    assert eval_table.main([str(tmp_path / "empty")]) == 1


def test_unreadable_files_are_skipped(tmp_path):
    write(tmp_path, metrics("q0a"))
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "metrics.json").write_text("{not json")
    assert list(eval_table.load_runs(tmp_path)) == ["q0a"]


def test_brightness_is_shown_for_both_files_and_flagged_outside_the_ship_rule(tmp_path):
    run = metrics("q1b")
    run["viewer"]["desktop"]["brightness"], run["viewer"]["phone"]["brightness"] = 1.006, 1.21
    write(tmp_path, run)
    row = rows(eval_table.render(eval_table.load_runs(tmp_path)))[2]
    assert "| x1.006 / x1.210 (!) |" in row                                          # the phone file is outside x0.95 to x1.05


def test_the_fold_the_walking_grid_and_an_export_only_origin_get_their_own_table(tmp_path):
    folded = metrics("q1b")
    folded["exposureFold"] = {"folded": True, "settled": True, "steps": 2000, "views": 73, "brightnessBefore": 1.2, "brightnessAfter": 1.0}
    failed = metrics("q1c")
    failed["exposureFold"] = {"folded": False, "why": "not fitted: needs a GPU with gsplat"}
    dense = metrics("q0a-dense")
    dense["exportFrom"] = "q0a"
    dense["denseOccupancy"] = {"used": "depth", "views": 52, "seconds": 41.5, "walk": {"cameras": 1.0, "standableM2": 61.2},
                               "sparseWalk": {"cameras": 0.97, "standableM2": 48.0}}
    rejected = metrics("q0a-dense2")
    rejected["denseOccupancy"] = {"used": "sparse", "why": "it walks worse than the sparse grid"}
    write(tmp_path, metrics("q0a", role="baseline"), folded, failed, dense, rejected)
    out = eval_table.render(eval_table.load_runs(tmp_path))
    table = {line.split(" | ")[0][2:]: line for line in rows(out.split("## Exposure fold, walking grid, origin")[1])[2:]}
    assert set(table) == {"q1b", "q1c", "q0a-dense", "q0a-dense2"}                   # q0a has none of the three
    assert "| its own training | x1.2 -> x1.0 on 73 views, 2000 steps after | – |" in table["q1b"]
    assert "not folded: not fitted: needs a GPU with gsplat" in table["q1c"]
    assert table["q0a-dense"].endswith("| q0a | – | from depth (52 views); depth 100% of capture positions walkable, 61.2 m2 to stand on, "
                                       "sparse 97% of capture positions walkable, 48.0 m2 to stand on; 41.5 s |")
    assert "sparse kept: it walks worse than the sparse grid |" in table["q0a-dense2"]
    write(tmp_path / "plain", metrics("q0a"))
    assert "Exposure fold" not in eval_table.render(eval_table.load_runs(tmp_path / "plain"))


def test_the_walking_grid_reads_the_same_here_and_in_the_runs_summary():
    from plypipe import evalrun
    for d in ({"used": "depth", "views": 52, "seconds": 41.5, "walk": {"cameras": 1.0, "standableM2": 61.2}, "sparseWalk": {"cameras": None, "standableM2": 0.0}},
              {"used": "sparse", "why": "failed: x"}, {"used": "sparse"}):
        assert eval_table.dense_cell(d) == evalrun.dense_cell(d)


def test_runs_scored_against_a_laser_scan_get_their_own_table_with_a_signed_scale_error(tmp_path):
    geometry = {"scaleErrorPct": -3.456, "scale": 1.0358, "cameras": {"twin": 330, "matched": 320, "inliers": 300, "rmsM": 0.031, "medianM": 0.02},
                "splats": {"points": 400000, "chamferM": 0.0421, "f5cm": {"precision": 0.61, "recall": 0.55, "f": 0.5784}, "f10cm": {"precision": 0.8, "recall": 0.7, "f": 0.7467}}}
    write(tmp_path, {**metrics("q0a", role="baseline"), "geometry": geometry}, {**metrics("q2"), "geometry": {"error": "ValueError: no cameras match"}}, metrics("q4a"))
    out = eval_table.render(eval_table.load_runs(tmp_path))
    assert "## Against the laser scan" in out
    assert "| q0a | -3.46 | 320 / 300 | 3.1 | 0.578 (0.61 / 0.55) | 0.747 | 4.2 |" in out
    assert "| q2 | scoring failed: ValueError: no cameras match |" in out and "| q4a |" not in out.split("## Against the laser scan")[1]
    write(tmp_path / "plain", metrics("q0a"))
    assert "laser" not in eval_table.render(eval_table.load_runs(tmp_path / "plain"))
