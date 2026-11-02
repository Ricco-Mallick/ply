# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
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
    "high": {"max_steps": 40_000, "sh_degree": 3, "cap_max": 2_000_000, "ppisp": True},
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
        # MCMC relocates until refine_stop_iter (25k by default, i.e. 25k of gsplat's 30k). Below 25k steps it never
        # stops, so the last splats are teleported with no time left to fit; keep upstream's ratio at any length
        strategy=st.MCMCStrategy(verbose=True, cap_max=p["cap_max"], refine_stop_iter=int(steps * 0.83)),
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
    # PLY_LIVE_SNAPSHOTS="0.15,0.4,0.7": also save PLYs part-way, for the live view on the status page
    cfg.ply_steps = sorted({*(max(1, int(steps * float(f))) for f in os.environ.get("PLY_LIVE_SNAPSHOTS", "").split(",") if f.strip()), steps})
    cfg.sh_degree = p["sh_degree"]
    cfg.disable_viewer = True
    cfg.disable_video = True
    cfg.antialiased = antialiased
    has_ppisp = "post_processing" in getattr(st.Config, "__dataclass_fields__", {"post_processing": None})
    if use_ppisp and not has_ppisp:  # gsplat <= 1.5.x: exposure drift handled by the appearance embedding instead
        use_ppisp = False
        cfg.app_opt = True
    elif use_ppisp:
        cfg.post_processing = "ppisp"
        cfg.ppisp_controller_activation_num_steps = int(steps * 0.83)
    elif has_ppisp:
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
    ap.add_argument("--dry-run", action="store_true",
                    help="everything except the GPU work: import the trainer, build the config, load the scene with gsplat's parser")
    a = ap.parse_args(argv)
    if a.dry_run:  # CPU rehearsal: CUDA-only extensions may be missing; stand in for them so the import itself is tested
        import importlib.util, types
        # only fused_ssim is imported at module level (fused_bilagrid only when the bilateral grid is on)
        if importlib.util.find_spec("fused_ssim") is None:
            stub = types.ModuleType("fused_ssim")
            stub.fused_ssim = lambda *x, **k: None
            sys.modules["fused_ssim"] = stub

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
    if a.dry_run:
        parser = st.Parser(data_dir=cfg.data_dir, factor=cfg.data_factor, normalize=cfg.normalize_world_space, test_every=cfg.test_every)
        ds = st.Dataset(parser, split="train")
        item = ds[0]
        print(json.dumps({"plyTrainDryRun": {"images": len(parser.image_names), "trainImages": len(ds), "points": int(len(parser.points)),
                                             "imageShape": list(item["image"].shape), "sceneScale": float(parser.scene_scale)}}), flush=True)
        return 0
    st.cli(st.main, cfg, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
