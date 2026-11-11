"""The worker's limits must fit the containers that run them: fails when a limit is raised or a timeout lowered out of fit.

ply_modal.py is read as source (it needs `modal` to import), so the numbers asserted here are the ones that get deployed."""

import ast
import time
from pathlib import Path

import pytest

from plypipe import limits as lm
from plypipe import worker

MODAL = Path(__file__).resolve().parents[1] / "modal" / "ply_modal.py"
TREE = ast.parse(MODAL.read_text())


def constant(name: str) -> float:
    node = next(n.value for n in TREE.body if isinstance(n, ast.Assign) and [t.id for t in n.targets if isinstance(t, ast.Name)] == [name])
    return eval(compile(ast.Expression(node), str(MODAL), "eval"), {"__builtins__": {}})


def env_default(name: str) -> str:
    """The default ply_modal.py gives `os.environ.get(name, default)` at deploy time."""
    for n in ast.walk(TREE):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and len(n.args) == 2
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == name):
            return n.args[1].value
    raise AssertionError(f"ply_modal.py reads no {name}")


def timeouts() -> dict[str, str]:
    """function name -> the name its `timeout=` refers to."""
    return {n.name: kw.value.id for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) for d in n.decorator_list
            if isinstance(d, ast.Call) for kw in d.keywords if kw.arg == "timeout" and isinstance(kw.value, ast.Name)}


def fit(limits=None, cap=None, container=None, cpu=None):
    return lm.fit_problems(limits or lm.stage_limits(env_default("PLY_STAGE_LIMITS")), cap or lm.job_cap_min(env_default("PLY_JOB_MAX_MIN")),
                           container or constant("TIMEOUT_S"), constant("DEADLINE_MARGIN_S"), cpu or constant("CPU_SOLVE_TIMEOUT_S"))


def test_the_deployed_limits_fit_their_containers():
    assert fit() == []


def test_both_containers_take_their_timeout_from_the_constants_that_are_checked():
    assert timeouts()["work"] == "TIMEOUT_S" and timeouts()["solve"] == "CPU_SOLVE_TIMEOUT_S"
    src = MODAL.read_text()
    assert "PLY_DEADLINE=str(time.time() + TIMEOUT_S - DEADLINE_MARGIN_S)" in src


def test_no_stage_limit_is_hard_coded_over_the_deploy_time_setting():
    """Regression: work() set PLY_STAGE_LIMITS="sfm=300,export=60" itself, so the deploy-time value never arrived."""
    hard_coded = [kw.arg for n in ast.walk(TREE) if isinstance(n, ast.Call) for kw in n.keywords
                  if kw.arg in ("PLY_STAGE_LIMITS", "PLY_JOB_MAX_MIN", "PLY_MAX_INPUT_MB") and isinstance(kw.value, ast.Constant)]
    assert hard_coded == []
    assert "**DEPLOY_LIMITS" in MODAL.read_text() and env_default("PLY_STAGE_LIMITS") == ""   # empty: the worker's defaults apply


def test_raising_a_limit_or_lowering_a_timeout_is_caught():
    base = lm.stage_limits("")
    assert fit(limits={**base, "sfm": 300})                              # the old deployed sfm=300: past the cap and the CPU function
    assert fit(limits={**base, "train": 300})                            # longer than the cap, so it could never fire
    assert fit(cap=400)                                                  # a cap bigger than the container's usable time
    assert fit(container=3 * 3600)                                       # the container shrank under the cap
    assert fit(cpu=150 * 60)                                             # the CPU function can no longer hold the sfm limit
    assert fit(limits={**base, "export": 235})                           # no time left for training before the cap
    assert fit(limits={**base, "sfm_match": 200})
    assert not fit(limits={**base, "sfm": 150}) and not fit(cap=300)


def test_every_default_keeps_one_and_a_half_times_the_slowest_healthy_time_seen_on_an_l4():
    """The chapel (1,236 frames of 4K), minutes: a limit under 1.5x these would kill a healthy job."""
    observed = {"frames": 9.1, "sfm_match": 15, "sfm": 117, "train": 58, "export": 18}
    for stage, minutes in observed.items():
        assert lm.STAGE_LIMITS_MIN[stage] >= 1.5 * minutes, stage
    assert lm.JOB_MAX_MIN >= 1.5 * 135                                   # GPU time of the whole job at High


def test_worst_case_is_the_sum_of_the_stage_limits_but_never_past_the_cap():
    small = {"frames": 1, "sfm_match": 5, "sfm": 2, "align": 1, "occupancy": 1, "label": 1, "train": 3, "export": 1, "mesh": 0}
    assert lm.worst_case_min({**small, "mesh": 1}, 240) == 11             # sfm_match is a tighter sfm, not another stage
    assert lm.worst_case_min(small, 240) == 240                           # a limit of 0 means none: the cap bounds it
    assert lm.worst_case_min(lm.stage_limits(""), 240) == 240


def test_overrides_and_bad_values(monkeypatch):
    assert lm.stage_limits("sfm=30, train=60,")["sfm"] == 30 and lm.stage_limits("sfm=30")["export"] == lm.STAGE_LIMITS_MIN["export"]
    assert lm.job_cap_min("90") == 90 and lm.job_cap_min("") == lm.JOB_MAX_MIN and lm.max_input_mb(" ") == lm.MAX_INPUT_MB
    for bad in ("0", "-5"):
        with pytest.raises(ValueError):
            lm.job_cap_min(bad)
    monkeypatch.setenv("PLY_JOB_MAX_MIN", "123")
    monkeypatch.setenv("PLY_STAGE_LIMITS", "mesh=7")
    assert lm.job_cap_min() == 123 and worker.stage_limits()["mesh"] == 7


class Claims:
    """A stand-in Api that has no work: counts the claims a worker makes."""
    count = 0

    def __init__(self, base, token):
        pass

    def req(self, method, path, body=None, ctype=None, tries=5):
        Claims.count += 1
        return 204, None


def run_main(monkeypatch, window_min, **env):
    Claims.count = 0
    monkeypatch.setattr(worker, "Api", Claims)
    monkeypatch.setenv("PLY_LIVE_SNAPSHOTS", "0.1,0.3,0.6,0.85")        # main() would set it for the rest of the session
    monkeypatch.delenv("PLY_STAGE_LIMITS", raising=False)
    monkeypatch.delenv("PLY_JOB_MAX_MIN", raising=False)
    monkeypatch.setenv("PLY_DEADLINE", str(time.time() + window_min * 60))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    code = worker.main(["--api", "http://api.invalid", "--token", "t", "--idle-exit", "1e-9"])
    return code, Claims.count


def test_the_deadline_check_uses_the_effective_limits_not_the_defaults(monkeypatch):
    assert run_main(monkeypatch, 355) == (0, 1)                           # defaults: the cap (240) fits a fresh container
    assert run_main(monkeypatch, 200) == (2, 0)                           # ...and not 200 minutes: said loudly, nothing claimed
    # an override that shrinks every stage lowers the worst case, so a short window is enough
    tiny = "frames=1,sfm=1,sfm_match=1,align=1,occupancy=1,label=1,train=1,export=1,mesh=1"
    assert run_main(monkeypatch, 20, PLY_STAGE_LIMITS=tiny) == (0, 1)
    # overrides that grow the limits and the cap past the container are refused, where the defaults would have run
    assert run_main(monkeypatch, 355, PLY_STAGE_LIMITS="sfm=300,export=60", PLY_JOB_MAX_MIN="600") == (2, 0)
    assert run_main(monkeypatch, 355, PLY_JOB_MAX_MIN="600", PLY_STAGE_LIMITS="frames=1,sfm=1,sfm_match=1,align=1,occupancy=1,label=1,train=1,export=1,mesh=1") == (0, 1)


def test_a_worker_without_a_deadline_never_refuses(monkeypatch):
    Claims.count = 0
    monkeypatch.setattr(worker, "Api", Claims)
    monkeypatch.setenv("PLY_LIVE_SNAPSHOTS", "0.1,0.3,0.6,0.85")
    monkeypatch.delenv("PLY_DEADLINE", raising=False)
    assert worker.main(["--api", "http://api.invalid", "--token", "t", "--idle-exit", "1e-9"]) == 0 and Claims.count == 1
