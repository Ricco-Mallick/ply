# Status

Updated 2026-11-11 (evening). What is deployed, what is running and what is next. Update this file at the end of every
session, so the next one starts here and not from an exported transcript.

## Deployed

| What | Version | Note |
|---|---|---|
| Site (https://ricco.codes/ply/) | `main` at `e7bddf1` (PR #22), 2026-11-11 | still opens the chapel in Walk and draws the grid in the minimap until the second batch is merged |
| API (`ply-api`, Cloudflare) | `35c9c1d`, 2026-11-05 | read from `GET /` |
| GPU worker (Modal app `ply`, riccomarcelino) | not redeployed since before PR #21 | the mapper choice, one-commit evidence upload, job limits and top view reach jobs only after a redeploy |
| Twins on Hugging Face (`ply-data`) | `4c46886`, 2026-11-11 | 13 twins have `topview.webp` and a `topview` block in `scene.json`; `4c09354` is the commit before. `chapel-high-2` has the rebuilt walking grid (`4c09354`; `e063f08` before it) |
| CUDA wheels | `wheels/cp312-cu128-sm89` at `e063f08` | L4 and L40S; what `pipeline/wheels.lock.json` pins and every image built so far uses |
| CUDA wheels, rebuilt | `wheels/cp312-cu128-sm80-89-90` at `716b3e4` | checked on an L40S and an A100. **Not usable on an H100**: its `fused-ssim` has no 9.0 kernel (`fused-bilagrid` has none either). The lock for it is `pipeline/wheels.lock.sm80-89-90.json`; the default lock is unchanged |

## Accounts

| Modal account | Holds | Used for |
|---|---|---|
| riccomarcelino (about $6) | the deployed worker, `ply-bench` and `ply-handoff` Volumes, secrets `ply-hf` and `ply-worker` | live jobs |
| opservemakesalt ($30 on 2026-11-11) | Volume `ply-bench` with `chapel/high/job` (the solve and photos only), secret `ply-hf` (read token), the worker image | experiments (`pipeline/modal/ply_eval.py`) |
| ricco-mallick (about $1.70) | `ply-bench:drjohnson/prep` | nothing at present |

## Running

- **Rebuild of `fused-ssim` and `fused-bilagrid` with an H100 kernel**: the `wheels` workflow, dispatched on the
  work branch with `pkgs=["fused-ssim","fused-bilagrid"]` after `install_gpu.sh` learned to pass the architecture
  list to both. About 7 minutes each once GitHub runs them.

## Experiments so far (`EXPERIMENTS.md`)

- Noise floor from two identical baselines: 0.14 dB and 0.0011 LPIPS on the every-30th hold-out.
- Opacity regulariser 0.002 (`q5b`) is the first clear gain: +0.55 dB, −0.033 LPIPS, 80% of splats kept against 61%.
- Pose refinement (`q2`): no gain. Sparse depth loss (`q4a`): diverges, a defect to fix.
- Spent on opservemakesalt on 2026-11-11: $2.98 by Modal's billing.

## Next

1. **Finish the H100 wheels**: download the two rebuilt wheels, put them with `gsplat` and `ppisp` from
   `wheels/cp312-cu128-sm80-89-90` into `pipeline/wheels`, `modal run scripts/wheels.py::publish --tag <new tag>`,
   build an image from them and run a GPU check on L40S, A100 and H100 (gsplat render, `fused_ssim` forward and
   backward, a bilateral-grid step). Only then switch `pipeline/wheels.lock.json`; every account rebuilds its image
   once after that.
2. **Phase 4, next variants** from the same saved solve: opacity regulariser 0.001 and 0, a look at the see-through
   views of `q5a` and `q5b`, then one full-length run of the winner before the trainer's default changes.
   `MODAL_PROFILE=opservemakesalt modal run --detach pipeline/modal/ply_eval.py --variant <name>` returns at once;
   `python pipeline/modal/ply_eval.py collect --scenario chapel --out /tmp/ply-eval` and
   `python scripts/eval_table.py /tmp/ply-eval` give the table. Fix `--depth-loss` before Q4. Q1, Q3 and Q6 need code.
3. **Redeploy the worker** (riccomarcelino) and run the smoke capture (0.6) before any long job. The first long
   ordered capture after that shows whether skipping the global mapper saves the expected 32 minutes.
4. **Make the `ci` checks required** on `main` (repository settings, owner).
5. **Rest of phase 0**: the API side of 0.3 and 0.4 (it needs a D1 migration and deploys on merge), the smoke capture
   and `plyctl` (0.6, 0.7; they need the team code as a project variable).
6. **Walk** (2.1): walking speed and sliding along edges first. Free is the default until the owner says otherwise.

## Known and left alone

- The capture app cannot ask for the global mapper on a long video: it sends `--mapper` only for `incremental`, and
  its "Global (fast, default)" label no longer describes what a 400-frame walk gets (`app/src/capture/lib/backend.ts`).
- `chapel_e2e.py`, `sfm_cpu.py` and `ply_bench.py` wait on their run with `.remote()`: a killed client cancels them (`MISTAKES.md`, 2026-11-11).
- The model page accepts a module from anywhere on the site's origin, not only under `/ply/`. There is no content
  security policy, and `?scene=` accepts any origin by design.
- Two old twins (`c09a4c82…`, `0ffc07fb…`) live on the bucket, not on Hugging Face, and have no top view.
- Capture and upload findings C1 to C5 of `PROJECT-REVIEW.md` are open.
