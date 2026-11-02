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
                 iterations: int | None = None, sh_degree: int | None = None, cap_max: int | None = None,
                 holdout: int = 0, pose_opt: bool = False, depth_loss: bool = False, world_scale: float | None = None,
                 bilateral_grid: bool = False, random_bkgd: bool = False):
    p = dict(PRESETS[preset])
    # the capture app's advanced settings: explicit values win over the preset
    for k, v in (("max_steps", iterations), ("sh_degree", sh_degree), ("cap_max", cap_max)):
        if v is not None:
            p[k] = v
    use_ppisp = p["ppisp"] if ppisp is None else ppisp
    steps = p["max_steps"]
    # world_scale: the median camera distance in metres. The MCMC recipe (noise_lr 5e5, scale_reg 0.01) is tuned
    # for gsplat's normalised world, where that distance is 1. Training here runs in metres, and gsplat's position
    # noise is lr x covariance (metres cubed), so a room 3 m across got 9x the intended noise and a hall far more.
    # Dividing by the scale (squared for the noise) gives every room the noise the recipe was tuned with.
    ws = world_scale if world_scale and world_scale > 0 else 1.0
    cfg = st.Config(
        init_opa=0.5, init_scale=0.1, opacity_reg=0.01, scale_reg=0.01 / ws,  # the "mcmc" preset of simple_trainer
        # MCMC relocates until refine_stop_iter (25k by default, i.e. 25k of gsplat's 30k). Below 25k steps it never
        # stops, so the last splats are teleported with no time left to fit; keep upstream's ratio at any length
        strategy=st.MCMCStrategy(verbose=True, cap_max=p["cap_max"], refine_stop_iter=int(steps * 0.83), noise_lr=5e5 / ws**2),
    )
    cfg.data_dir = data
    cfg.data_factor = 1  # frames are already capped at 1600 px
    cfg.result_dir = result
    cfg.normalize_world_space = False
    # holdout N: every Nth view is kept out of training and scored (PSNR/SSIM/LPIPS) once, at the last step.
    # 0 trains on every view (index 0 is still gsplat's only hold-out) and scores nothing
    cfg.test_every = holdout if holdout and holdout > 1 else 10**9
    cfg.max_steps = steps
    cfg.eval_steps = [steps] if holdout and holdout > 1 else []
    # PPISP has no per-frame exposure for a held-out view: the colour-corrected scores separate tone from detail
    cfg.use_color_correction_metric = bool(holdout and holdout > 1)
    cfg.pose_opt = bool(pose_opt)
    cfg.depth_loss = bool(depth_loss)  # disparity at the views' own SfM track points (sparse), already in gsplat
    cfg.save_steps = [steps]
    cfg.save_ply = True
    # PLY_LIVE_SNAPSHOTS="0.15,0.4,0.7": also save PLYs part-way, for the live view on the status page
    cfg.ply_steps = sorted({*(max(1, int(steps * float(f))) for f in os.environ.get("PLY_LIVE_SNAPSHOTS", "").split(",") if f.strip()), steps})
    cfg.sh_degree = p["sh_degree"]
    cfg.disable_viewer = True
    cfg.disable_video = True
    cfg.antialiased = antialiased
    has_ppisp = "post_processing" in getattr(st.Config, "__dataclass_fields__", {"post_processing": None})
    # random backgrounds: a splat cannot pass as "dark" by being transparent over the black training background,
    # which is what shows as see-through gaps in the viewer
    cfg.random_bkgd = bool(random_bkgd)
    if bilateral_grid and has_ppisp:
        # per-photo affine colour grids, identity-initialised: they absorb exposure and white balance while the splats
        # stay at the photos' average exposure. PPISP's splats live in pre-ISP radiance and open brighter in the viewer
        use_ppisp = False
        cfg.post_processing = "bilateral_grid"
        try:
            import fused_bilagrid  # noqa: F401
            cfg.bilateral_grid_fused = True
        except ImportError:
            cfg.bilateral_grid_fused = False
    elif use_ppisp and not has_ppisp:  # gsplat <= 1.5.x: exposure drift handled by the appearance embedding instead
        use_ppisp = False
        cfg.app_opt = True
    elif use_ppisp:
        cfg.post_processing = "ppisp"
        cfg.ppisp_controller_activation_num_steps = int(steps * 0.83)
    elif has_ppisp:
        cfg.post_processing = None
        cfg.app_opt = False
    return cfg, {"preset": preset, "maxSteps": steps, "shDegree": p["sh_degree"], "capMax": p["cap_max"],
                 "ppisp": use_ppisp, "antialiased": antialiased, "holdout": holdout if holdout and holdout > 1 else 0,
                 "poseOpt": bool(pose_opt), "depthLoss": bool(depth_loss), "worldScale": round(ws, 4),
                 "appearance": cfg.post_processing or ("app_opt" if getattr(cfg, "app_opt", False) else None),
                 "randomBkgd": bool(random_bkgd)}


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
    ap.add_argument("--holdout", type=int, default=0, help="score every Nth view, held out of training (0: off)")
    ap.add_argument("--pose-opt", action="store_true")
    ap.add_argument("--depth-loss", action="store_true")
    ap.add_argument("--bilateral-grid", action="store_true")
    ap.add_argument("--random-bkgd", action="store_true")
    ap.add_argument("--world-scale", type=float, help="median camera distance (scene units): MCMC noise/scale_reg follow it")
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
                                a.iterations, a.sh_degree, a.cap_max, a.holdout, a.pose_opt, a.depth_loss, a.world_scale,
                                a.bilateral_grid, a.random_bkgd)
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
