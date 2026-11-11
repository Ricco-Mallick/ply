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
                               "viewer PSNR desktop | viewer PSNR phone | splats kept | minutes | cost |")
    assert first[2] == "| q0a | 15000 | 24.50 | 0.8600 | 0.2400 | 22.00 | 0.8000 | 0.3000 | 23.50 | 21.40 | 75% (1.50M) | 14.0 | $0.60 |"
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
