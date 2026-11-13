# Phase 7: the combined run, ready to launch

Everything below is decided; nothing here needs a design choice. Costs are `evalrun` estimates on an L40S and were
recomputed on 2026-11-13 after the wheels switch.

## What goes in, and why

| Lever | State | Source |
|---|---|---|
| Q6, the phone file fine-tuned | **in**, export default | `EXPERIMENTS.md`, 2026-11-12 |
| Walking grid from depth | **in**, export default | hand-off item 3 |
| Q1b, one affine colour map per photo, folded | **in**, the owner's blind pick | `STATUS.md`, 2026-11-13 |
| Opacity regulariser 0.005 | **out** | fails its see-through condition on the chapel (1.8% of pixels, 14% in the worst view) |
| Mesh | **out** | `MESH-DIAGNOSIS.md`: 84,443 fragments, not a surface |
| Q3, higher resolution | **out** | the chapel has no 4K frames |

## The two runs

**Full length, from the saved chapel solve** (30,000 steps, Q1b's exposure model, both export defaults):

```
PLY_EVAL_SCENARIO=chapel PLY_EVAL_GPU=L40S MODAL_PROFILE=opservemakesalt \
  modal run --detach pipeline/modal/ply_eval.py --variant q1b --args=--iterations 30000
```

Estimated **44 min training, 51 min total, $2.19**, stop-loss 77 min.

**End to end at Quick length**, to prove the deployed worker still runs the whole chain:

```
PLY_EVAL_SCENARIO=drjohnson PLY_EVAL_GPU=L4,L40S MODAL_PROFILE=opservemakesalt \
  modal run --detach pipeline/modal/ply_eval.py --variant q6 --args=--iterations 7000
```

About **$0.35**, 20 min.

## Why `--args` and not a new table variant

A `phase7` entry was written and reverted: the eval table's tests pin the estimate and ceiling of every variant
(`test_the_function_timeout_follows_the_slowest_allowed_gpu` asserts the longest variant's ceiling), and a 30,000-step
entry becomes the slowest, so it changes expectations in four places. Those numbers are load-bearing for the stop-loss
that protects a paid run, so they should not be loosened to accommodate a run that has not happened. `--args` applies
the same arguments without touching the table.

## After the run

1. `collect` and `eval_table.py`, then compare against the two baselines (`q0a`, `q0b`): the ship rule is the noise
   floor of 0.14 dB and 0.0011 LPIPS.
2. If it clears, the win is recorded in `EXPERIMENTS.md` and the twin is published; `chapel-high-2` stays as the
   fallback.
3. Then the worker deploy and its smoke (`plyctl smoke`), which is plan 7.3, and `STATUS.md` is frozen for the day.
