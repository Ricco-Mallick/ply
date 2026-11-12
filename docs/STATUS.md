# Status

Updated 2026-11-11 (night). What is deployed, what is running and what is next. Update this file at the end of every
session, so the next one starts here and not from an exported transcript.

## Deployed

| What | Version | Note |
|---|---|---|
| Site (https://ricco.codes/ply/) | `main` at `0597c98` (PR #23), 2026-11-11 | every twin opens in Free and the minimap shows the twin's top view. The deploy succeeded and the pages answer; not looked at in a browser since |
| API (`ply-api`, Cloudflare) | `35c9c1d`, 2026-11-05 | read from `GET /` |
| GPU worker (Modal app `ply`, riccomarcelino) | not redeployed since before PR #21 | the mapper choice, one-commit evidence upload, job limits, top view and the export filter fix reach jobs only after a redeploy |
| Twins on Hugging Face (`ply-data`) | `4c46886`, 2026-11-11 | 13 twins have `topview.webp` and a `topview` block in `scene.json`; `4c09354` is the commit before. `chapel-high-2` has the rebuilt walking grid (`4c09354`; `e063f08` before it) |
| CUDA wheels | `wheels/cp312-cu128-sm89` at `e063f08` | L4 and L40S; what `pipeline/wheels.lock.json` pins and every image built so far uses |
| CUDA wheels, rebuilt | `wheels/cp312-cu128-sm80-89-90-r2` at `73b576a` | for A100, L4/L40S and H100. An image built from it passed a GPU check on an L40S, an A100 and an H100 request (served by an H200): gsplat render, `fused_ssim` and a bilateral-grid kernel forward and backward, the trainer's imports. No training run has used it. Its lock is `pipeline/wheels.lock.sm80-89-90.json`; the default lock is unchanged. (`…sm80-89-90` without `-r2` is the first rebuild, whose `fused-ssim` failed on the H100) |

## Accounts

| Modal account | Holds | Used for |
|---|---|---|
| riccomarcelino (about $6) | the deployed worker, `ply-bench` and `ply-handoff` Volumes, secrets `ply-hf` and `ply-worker` | live jobs |
| opservemakesalt ($30 on 2026-11-11; $5.34 billed by the end of that day) | Volume `ply-bench` with `chapel/high/job` and `drjohnson/prep` (each the solve and photos only), secret `ply-hf` (read token), the worker image | experiments (`pipeline/modal/ply_eval.py`) |
| ricco-mallick (about $1.70) | `ply-bench:drjohnson/prep` and the Dr Johnson benchmark twins | nothing at present |

## Running

Nothing. No Modal app is running on either account and no workflow is in progress.

## Experiments so far (`EXPERIMENTS.md`)

- **Small scenario (Dr Johnson):** variants go here first (plan decision 8). Noise floor 0.02 dB and 0.0004 LPIPS; a
  run is billed about $0.20.
- **Chapel:** noise floor from two identical baselines 0.14 dB and 0.0011 LPIPS on the every-30th hold-out.
- **Opacity regulariser.** At 15k steps 0.002 wins on both scenes (chapel +0.55 dB and −0.033 LPIPS, Dr Johnson
  +0.10 dB and −0.023 LPIPS), and the trend ends there: 0.001 is no better and none is worse. At 30k steps on
  Dr Johnson it is a trade between sharpness and fidelity, and 0.005 is the better balance (see `EXPERIMENTS.md`).
  **The trainer's default stays at 0.01.**
- **Export** shipped the 0.002 twin with 7% of its pixels see-through. Fixed: the value filter is loosened when the
  viewer pays for it (27.72 to 29.47 dB on the same model). The chapel's exports have not been run again.
- **Sparse depth loss (`q4a`):** the NaN is fixed and the fix holds on a GPU; no gain on Dr Johnson.
  **Pose refinement (`q2`):** no gain on the chapel.
- Billed on opservemakesalt on 2026-11-11: $5.34 ($3.17 for the chapel runs and checks, $2.17 for the small scenario).
  About $0.50 more falls on 2026-11-12 (the 30k runs' last minutes) and was not in the report yet.

## Next

1. **Switch to the rebuilt wheels** when an A100 or H100 is wanted, or at the next worker deploy:
   `cp pipeline/wheels.lock.sm80-89-90.json pipeline/wheels.lock.json`, `modal run scripts/wheels.py::fetch`, then
   the 300-step smoke variant (`ply_eval.py --variant smoke --force`) before anything longer. Every account rebuilds
   its image once after the switch.
2. **Phase 4, on the small scenario first** (one launch at a time):
   `PLY_EVAL_SCENARIO=drjohnson PLY_EVAL_GPU=L4,L40S MODAL_PROFILE=opservemakesalt modal run --detach
   pipeline/modal/ply_eval.py --variant <name>` returns at once; `python pipeline/modal/ply_eval.py collect --scenario
   drjohnson --out /tmp/ply-eval` and `python scripts/eval_table.py /tmp/ply-eval` give the table.
   Next run: a second 30k baseline (`--variant q0-30k-b --args "--iterations 30000"`, `PLY_EVAL_TIMEOUT_MIN=27`), for
   the noise floor that decides whether an opacity regulariser of 0.005 may become the default. Then the variants that
   need code and can be judged on this scene: Q6 (a phone file fine-tuned, not decimated) and Q1 (exposure folded
   back). Q3 (higher resolution) needs the chapel's 4K frames and waits.
   On the chapel, before any training run there: export `q5a` and `q5b` again under the new filter rule (export only)
   to see whether their see-through pixels go. No full-length run before the rest is in (decision 8).
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
