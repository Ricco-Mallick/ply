"""Run gsplat's examples/simple_trainer.py (pinned SHA) with a Ply preset, in-process.

Invoked by the `train` stage as a subprocess inside the GPU environment:

    python -m plypipe.train_launcher --examples /opt/gsplat/examples --data <out>/data \
        --result <out>/train --preset standard [--no-ppisp]

Building the Config object here (instead of passing CLI flags) avoids depending on tyro's flag
spelling for booleans, and lets presets scale PPISP's controller schedule with max_steps.

Training happens in the scene frame (normalize_world_space=False): the aligned model already
has +Y up, floor at y=0 and metric scale, so the exported PLY is directly viewer-ready.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PRESETS = {
    # steps, SH degree, max Gaussians (MCMC cap), PPISP on by default
    "quick": {"max_steps": 7_000, "sh_degree": 2, "cap_max": 600_000, "ppisp": False},
    "standard": {"max_steps": 30_000, "sh_degree": 3, "cap_max": 1_500_000, "ppisp": True},
    "high": {"max_steps": 40_000, "sh_degree": 3, "cap_max": 3_000_000, "ppisp": True},
}


def build_config(st, data: str, result: str, preset: str, ppisp: bool | None, antialiased: bool,
                 iterations: int | None = None, sh_degree: int | None = None, cap_max: int | None = None):
    p = dict(PRESETS[preset])
    # the capture app's advanced settings: explicit values win over the preset
    for k, v in (("max_steps", iterations), ("sh_degree", sh_degree), ("cap_max", cap_max)):
        if v is not None:
            p[k] = v
    use_ppisp = p["ppisp"] if ppisp is None else ppisp
    steps = p["max_steps"]
    cfg = st.Config(
        init_opa=0.5, init_scale=0.1, opacity_reg=0.01, scale_reg=0.01,  # the "mcmc" preset of simple_trainer
        strategy=st.MCMCStrategy(verbose=True, cap_max=p["cap_max"]),
    )
    cfg.data_dir = data
    cfg.data_factor = 1  # frames are already capped at 1600 px
    cfg.result_dir = result
    cfg.normalize_world_space = False
    cfg.test_every = 10**9  # train on every frame (index 0 is the only hold-out)
    cfg.max_steps = steps
    cfg.eval_steps = []
    cfg.save_steps = [steps]
    cfg.save_ply = True
    cfg.ply_steps = [steps]
    cfg.sh_degree = p["sh_degree"]
    cfg.disable_viewer = True
    cfg.disable_video = True
    cfg.antialiased = antialiased
    if use_ppisp:
        cfg.post_processing = "ppisp"
        cfg.ppisp_controller_activation_num_steps = int(steps * 0.83)
    else:
        cfg.post_processing = None
        cfg.app_opt = False
    return cfg, {"preset": preset, "maxSteps": steps, "shDegree": p["sh_degree"], "capMax": p["cap_max"],
                 "ppisp": use_ppisp, "antialiased": antialiased}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True, help="gsplat/examples directory of the pinned checkout")
    ap.add_argument("--data", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="standard")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ppisp", dest="ppisp", action="store_true", default=None)
    g.add_argument("--no-ppisp", dest="ppisp", action="store_false")
    ap.add_argument("--antialiased", action="store_true")
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--sh-degree", type=int)
    ap.add_argument("--cap-max", type=int)
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.abspath(a.examples))
    os.chdir(a.examples)  # simple_trainer imports sibling modules (datasets, utils, gsplat_viewer)
    import simple_trainer as st  # noqa: E402

    cfg, summary = build_config(st, os.path.abspath(a.data), os.path.abspath(a.result), a.preset, a.ppisp, a.antialiased,
                                a.iterations, a.sh_degree, a.cap_max)
    cfg.adjust_steps(cfg.steps_scaler)
    os.makedirs(a.result, exist_ok=True)
    with open(os.path.join(a.result, "ply_preset.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"plyTrainConfig": summary}), flush=True)
    st.cli(st.main, cfg, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
