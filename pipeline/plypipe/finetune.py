# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
"""finetune: start gsplat's simple_trainer from a trained model instead of from the SfM points.

The phone file was the desktop model cut down to the phone's splat count and shipped as cut: 2.15 dB below the desktop
file on the chapel (docs/PROJECT-REVIEW.md, T2). With this, the cut-down model is where a short training run starts,
with the splat cap at its own size, so the splats that are left refit the photos (IMPROVEMENT-PLAN.md, variant Q6).
A twin under the phone's count gets the same run: its phone file loses only the third SH band, and the bands that are
left refit (docs/EXPERIMENTS.md: +0.80 dB on the chapel, +0.73 and +0.52 dB on Dr Johnson).

simple_trainer loads a checkpoint only to score it. `install` wraps its create_splats_with_optimizers instead: upstream
still builds the parameters and their optimizers (from the SfM points, with its own learning rates), and the values
are then replaced by the model's, the way upstream's checkpoint path does it (`runner.splats[k].data = ...`). Adam
keeps no state before its first step, so the optimizers follow the new tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plypipe import clean

# The position learning rate a fine-tune starts at, as a share of a fresh run's. gsplat decays it to 1 % over the run,
# whatever its length: started at the full rate, a fitted model is shaken apart before it can settle again.
MEANS_LR = 0.1


def read_splats(ply: Path, sh_degree: int) -> dict[str, np.ndarray]:
    """A 3DGS PLY as gsplat's parameters (log scales, logit opacities), with the SH bands above `sh_degree` dropped.

    f_rest is channel-major in the file (all R coefficients, then G, then B) and [N, K - 1, 3] in gsplat. A file with
    fewer bands than asked for is refused: the missing ones cannot be made up."""
    _, c = clean.read_ply(Path(ply))
    wanted = ["x", "y", "z", "opacity", *(f"f_dc_{i}" for i in range(3)), *(f"scale_{i}" for i in range(3)),
              *(f"rot_{i}" for i in range(4))]
    missing = [p for p in wanted if p not in c]
    if missing:
        raise ValueError(f"{ply}: not a 3DGS model (no {', '.join(missing)})")
    rest = sorted((k for k in c if k.startswith("f_rest_")), key=lambda k: int(k.rsplit("_", 1)[1]))
    n, k, want = len(c["x"]), len(rest) // 3, (sh_degree + 1) ** 2 - 1
    if k < want:
        raise ValueError(f"{ply}: holds {k} SH coefficients per colour above the base one, SH degree {sh_degree} needs {want}")
    higher = np.stack([c[r] for r in rest], 1).reshape(n, 3, k).transpose(0, 2, 1)[:, :want] if want else np.zeros((n, 0, 3), np.float32)
    return {"means": np.stack([c["x"], c["y"], c["z"]], 1), "scales": np.stack([c[f"scale_{i}"] for i in range(3)], 1),
            "quats": np.stack([c[f"rot_{i}"] for i in range(4)], 1), "opacities": np.array(c["opacity"]),
            "sh0": np.stack([c[f"f_dc_{i}"] for i in range(3)], 1)[:, None, :], "shN": np.ascontiguousarray(higher)}


def install(st, splats: dict[str, np.ndarray]) -> None:
    """Make simple_trainer `st` start from `splats` (read_splats). Must run before the Runner is built, in the process
    that trains; the launcher sets the splat cap to their count, so MCMC moves splats and adds none."""
    upstream = st.create_splats_with_optimizers

    def from_model(*args, **kwargs):
        import torch

        params, optimizers = upstream(*args, **kwargs)
        if set(params.keys()) != set(splats):  # e.g. the appearance model's features and colours
            raise RuntimeError(f"the trainer's parameters ({', '.join(sorted(params.keys()))}) are not the model's "
                               f"({', '.join(sorted(splats))}): a fine-tune trains plain SH colours")
        for name, value in splats.items():
            if tuple(value.shape[1:]) != tuple(params[name].shape[1:]):
                raise RuntimeError(f"{name}: the model holds {tuple(value.shape[1:])} per splat, the trainer expects "
                                   f"{tuple(params[name].shape[1:])} (SH degree?)")
            params[name].data = torch.as_tensor(np.ascontiguousarray(value), dtype=params[name].dtype, device=params[name].device)
        print(f"Fine-tune: starting from {len(splats['means'])} trained splats", flush=True)
        return params, optimizers

    st.create_splats_with_optimizers = from_model
