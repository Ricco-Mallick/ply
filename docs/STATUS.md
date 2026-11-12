# Status

Updated 2026-11-12 (evening). What is deployed, what is running and what is next. Update this file at the end of every
session, so the next one starts here and not from an exported transcript.

## Deployed

| What | Version | Note |
|---|---|---|
| Site (https://ricco.codes/ply/) | `main` at `804869e` (PR #30), 2026-11-12 | every twin opens in Free and the minimap shows the twin's top view. Walk moves at Free's speed and slides along walls. The deploy succeeded; Walk was checked in a browser on the dev build's demo room only, and the live site has not been looked at in a browser since PR #23 |
| API (`ply-api`, Cloudflare) | `35c9c1d`, 2026-11-05 | read from `GET /` |
| GPU worker (Modal app `ply`, riccomarcelino) | v19, `c5cccb3`, 2026-11-12 (the head of PR #29: `main` at `1bf549d` plus the phone file's fine-tune by default; redeploy from `main` if that pull request changes or closes) | an L4, or an L40S when Modal has no L4 free. Smoke test passed after the deploy (job `8582285e…`): every stage, the gate and the phone file's fine-tune ran on a real job, the fine-tune in 44 s of a 62 s export. On that clip (2,000 steps, not a twin) the fine-tuned file scored 1.5 dB higher with 1.7 points more see-through pixels, which is over the export's limit, so the untuned file shipped as designed. The job took 20 minutes from upload to review, most of it the container's start and publishing |
| Twins on Hugging Face (`ply-data`) | `4c46886`, 2026-11-11 | 13 twins have `topview.webp` and a `topview` block in `scene.json`; `4c09354` is the commit before. `chapel-high-2` has the rebuilt walking grid (`4c09354`; `e063f08` before it) |
| CUDA wheels | `wheels/cp312-cu128-sm89` at `e063f08` | L4 and L40S; what `pipeline/wheels.lock.json` pins and every image built so far uses |
| CUDA wheels, rebuilt | `wheels/cp312-cu128-sm80-89-90-r2` at `73b576a` | for A100, L4/L40S and H100. An image built from it passed a GPU check on an L40S, an A100 and an H100 request (served by an H200): gsplat render, `fused_ssim` and a bilateral-grid kernel forward and backward, the trainer's imports. No training run has used it. Its lock is `pipeline/wheels.lock.sm80-89-90.json`; the default lock is unchanged. (`…sm80-89-90` without `-r2` is the first rebuild, whose `fused-ssim` failed on the H100) |

## Accounts

| Modal account | Holds | Used for |
|---|---|---|
| riccomarcelino (about $6 before 2026-11-12; $1.37 billed that day for two deploys and two smoke jobs) | the deployed worker, `ply-bench` (with `smoke/smoke-chapel-30s.mp4`) and `ply-handoff` Volumes, secrets `ply-hf` and `ply-worker` | live jobs |
| opservemakesalt ($30 on 2026-11-11; $5.34 billed by the end of that day and $1.59 on 2026-11-12) | Volume `ply-bench` with `chapel/high/job` and `drjohnson/prep` (each the solve and photos only) and the smoke clip, secret `ply-hf` (read token), the worker image | experiments (`pipeline/modal/ply_eval.py`) |
| ricco-mallick (about $1.70) | `ply-bench:drjohnson/prep` and the Dr Johnson benchmark twins | nothing at present |

## Running

Nothing. No Modal app is running on either account and no workflow is in progress.

## Experiments so far (`EXPERIMENTS.md`)

- **Small scenario (Dr Johnson):** variants go here first (plan decision 8). Noise floor 0.02 dB and 0.0004 LPIPS; a
  run is billed about $0.20.
- **Chapel:** noise floor from two identical baselines 0.14 dB and 0.0011 LPIPS on the every-30th hold-out.
- **Opacity regulariser.** At 15k steps 0.002 wins on both scenes (chapel +0.55 dB and −0.033 LPIPS, Dr Johnson
  +0.10 dB and −0.023 LPIPS), and the trend ends there: 0.001 is no better and none is worse. At 30k steps on
  Dr Johnson, with the floor now measured from two baselines (0.09 dB, 0.002 LPIPS), 0.005 clears the ship rule
  (LPIPS −0.012, nothing else outside the floor) and 0.002 loses SSIM. On the chapel at 15k the 0.005 model has 1.8%
  see-through pixels (14% in the worst view) before any export step, against 0.02% and 0.13% for the two baselines.
  **The trainer's default stays at 0.01** (decision 10); 0.005 is a candidate for the combined run of phase 7.
- **Export** shipped the 0.002 twin with 7% of its pixels see-through. Fixed: the value filter is loosened when the
  viewer pays for it (27.72 to 29.47 dB on the same model). The chapel's 0.005 twin, exported again under that rule,
  is the same file: its see-through pixels are in the trained model, which export cannot repair.
- **Sparse depth loss (`q4a`):** the NaN is fixed and the fix holds on a GPU; no gain on Dr Johnson.
  **Pose refinement (`q2`):** no gain on the chapel.
- **Exposure folded back (`q1`):** the fit, the fold and the 2,000 steps run on a GPU and the viewer shows the twin
  at ×1.006 of the photos' brightness, but on Dr Johnson it loses 0.50 dB on held-out views and 0.90 dB in the
  viewer: the bilateral grid holds more than one colour map can undo. Not shipped. `q1b` (`--exposure-affine`, one
  affine transform per photo, the trainer's grid at 1×1×1) is built and tested on a CPU and has not run.
- **Walking grid from depth** (plan 2.2 and 2.3): built as the opt-in `--dense-occupancy` (`plypipe/dense.py`,
  `occupancy.from_depth`), tested on a CPU with a synthetic two-room scene. It has never run on a GPU or a real twin.
- **Phone file (`q6`):** trained on for 3,000 steps at its own size, it gains 0.80 dB on the chapel (800k splats,
  now 0.48 dB below the desktop file instead of 1.25) and 0.73 dB on Dr Johnson cut to 240k. A twin under the phone
  size, whose phone file only loses the third SH band, gains 0.52 dB and ends 0.05 dB below the desktop file.
  See-through pixels rise a little on the chapel (0.16% to 0.29%). **On by default since 2026-11-12** (decision 9),
  because no other lever here buys as much per dollar (`EXPERIMENTS.md`): the export fine-tunes the phone file wherever
  it has fewer splats or SH bands than the twin and a GPU can score it, and ships the result only when it scores
  better. A Quick twin (SH 2, under 800k) is its own phone file and pays nothing. `--lite-finetune 0` turns it off.
  It adds about five minutes to a chapel-size export on an L4 (scaled from 151 s on an L40S; not timed on an L4).
  Billed $1.11 on opservemakesalt on 2026-11-12 for the three runs.
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
   Next run: `--variant q1b`. The estimate is 13 minutes and $0.49 on an L40S, 19 minutes and $0.35 on an L4. Its
   grid runs in pure torch and has not been timed (it is budgeted at the full grid's pace), so read the pace of the
   first steps in the log. What to read and what it decides is item 1 of the hand-off queue below.
   Q6 (the phone file fine-tuned)
   is the default and deployed. To read from the first real Standard or High job: `report.export.liteFinetune`
   (its `seconds`, estimated at about five minutes for an 800k phone file on an L4, and that it did not fall back for
   lack of GPU memory).
   Q3 (higher resolution) needs the chapel's 4K frames and waits.
   On the chapel, before any training run there: export `q5a` and `q5b` again under the new filter rule (export only)
   to see whether their see-through pixels go. No full-length run before the rest is in (decision 8).
3. **After every worker deploy:** `python3 scripts/plyctl.py smoke --clip smoke-chapel-30s.mp4` (the clip is on both
   accounts' `ply-bench` Volume under `smoke/`; the team code in `PLY_TEAM_KEY` or `--key-file`). It takes about six
   minutes of GPU. If a job was queued before the deploy, start a worker by hand (`scripts/spawn_worker.py`):
   the call that was already waiting did not start under the new settings (`MISTAKES.md`, 2026-11-12).
   Still open from the deploy: the first long ordered capture shows whether skipping the global mapper saves the
   expected 32 minutes, and the hand-off to the CPU solver has not run since (a capture under 400 frames skips it).
4. **Make the `ci` checks required** on `main` (repository settings, owner).
5. **Rest of phase 0**: the API side of 0.3 and 0.4 (it needs a D1 migration and deploys on merge); `plyctl` has
   `status` and `smoke`, and `logs`, `stop` and `cost` are not built (0.7).
6. **Walk** (2.1): speed, sliding along walls, sub-step movement, the wheel limit and the collider interface are in
   (PR #30), checked in a browser on the demo room. Left: Walk on a real twin on the live site, and the plan's
   guided-route test. Free is the default until the owner says otherwise.

## Hand-off queue

Work that is decided and specified, in order. Each item says what done means; none needs a design decision. A paid
run needs the owner's go-ahead and follows plan section 13: one launch, its log read, then the next.

1. **Run `q1b` on Dr Johnson** (command in Next, 2). It worked if `metrics.json` has `exposureFold.folded` and
   `.settled` true with no `why`, and `viewer.desktop.brightness` and `viewer.phone.brightness` within 0.95 to 1.05.
   Compare `scores.every`, `scores.block` and the viewer's PSNR with `q0a` and `q0b`. Dr Johnson's photos do not
   differ in exposure, so `q1b` cannot win there: the run measures what the method costs where there is nothing to
   correct. Within 0.1 dB of the baseline on all three: run it once on the chapel at 15k, where the windows are, and
   apply the ship rule of plan section 8. More than 0.2 dB under on any: Q1 is closed and the twin keeps no
   appearance model. In between: stop and report, with the score of the model as it left training, before the fold
   (`q1` was 2.3 dB under the baseline there). Record the result in `EXPERIMENTS.md` and replace the one-cell grid's
   budget in `evalrun.estimate` with the measured pace. (Thresholds set by the agent; the owner can overturn them.)
2. ~~An export-only rerun in `ply_eval.py`~~ Done 2026-11-12: `--export-from VARIANT` with a new variant name and
   `--args=` export options only. The function stops before restoring anything unless the variant's folder holds a
   finished training with its table arguments, takes only what training left there, and never runs `train`.
3. **`--dense-occupancy` on a GPU**: `PLY_EVAL_SCENARIO=drjohnson PLY_EVAL_GPU=L4,L40S MODAL_PROFILE=opservemakesalt
   modal run --detach pipeline/modal/ply_eval.py --variant q0a-dense --export-from q0a --args=--dense-occupancy`
   (about 9 minutes and $0.17 on an L4), then the same on the chapel (`PLY_EVAL_SCENARIO=chapel`). Read
   `denseOccupancy` in its `metrics.json` (or the second table of `eval_table.py`): `used`, `why`, `views`, `walk`
   against `sparseWalk`, `seconds`, and put the measured seconds in place of `evalrun.DENSE_MIN`. Report the
   numbers; whether the depth grid becomes the default is a decision (below).
4. ~~`scripts/eval_table.py`~~ Done 2026-11-12: a brightness column for both files, flagged outside ×0.95 to ×1.05,
   and a second table with the exposure fold, the walking grid and an export-only run's origin.
5. ~~`pipeline/README.md`~~ Done 2026-11-12.
6. **Walk on the live site**: open a real twin (`chapel-high-2`) in a browser, walk into walls and through a door,
   and write the plan's guided-route test (2.1).
7. **Mesh (2.4) and layout (2.5)** from the dense points, as the plan specifies them, with `test_mesh_fusion.py` and
   `test_layout.py`. After item 3 has shown the depth is usable on a real twin.
8. **Rest of phase 0** (Next, 5) and the Hugging Face `.gitattributes` fix (Known and left alone).

Not for hand-off, because each needs a judgement or the owner:

- `q1b` landing between the two thresholds of item 1, and what goes into the combined run of phase 7 (Q6 is in; an
  opacity regulariser of 0.005 only with see-through no higher than the baseline's, decision 10).
- The depth grid as the default walking grid: the gate of plan section 6 on real twins, and whether a navmesh (5c)
  is needed.
- Phases 3, 5 and 6: nothing was done on them on 2026-11-12. The plan's cut order (section 3) applies once the date
  of the final is known.
- The worker runs `c5cccb3`. `main` has since gained options that are off by default (`--exposure-fold`,
  `--exposure-affine`, `--dense-occupancy`); it is redeployed when one of them is wanted on live jobs.
- Owner: the date of the final, Walk as the default mode (decision 6), the `ci` checks required on `main`, and the
  go-ahead for each paid run.

## Known and left alone

- The export reads a `no_clean` option and its docstring names `--no-clean`, but the CLI defines no such flag, so the
  cleanup cannot be turned off from the command line (found 2026-11-12; `--export-from` does not offer it).
- The capture app cannot ask for the global mapper on a long video: it sends `--mapper` only for `incremental`, and
  its "Global (fast, default)" label no longer describes what a 400-frame walk gets (`app/src/capture/lib/backend.ts`).
- `chapel_e2e.py`, `sfm_cpu.py` and `ply_bench.py` wait on their run with `.remote()`: a killed client cancels them (`MISTAKES.md`, 2026-11-11).
- The model page accepts a module from anywhere on the site's origin, not only under `/ply/`. There is no content
  security policy, and `?scene=` accepts any origin by design.
- Two old twins (`c09a4c82…`, `0ffc07fb…`) live on the bucket, not on Hugging Face, and have no top view.
- Capture and upload findings C1 to C5 of `PROJECT-REVIEW.md` are open.
- Hugging Face refuses a small model file: `scene.ply` and `scene_clean.ply` under about 10 MB are sent as plain
  files and rejected as binary ("use Xet"), and one `live-sparse.bin` was refused over a missing LFS pointer. The
  worker stores them on the bucket and the API serves them from there, so no job fails. Seen on the smoke job
  (3.6 MB model); a twin of normal size goes through LFS. The likely fix is to track `*.ply` and `*.bin` in the
  dataset's `.gitattributes`; not done.
- The smoke job (`5265343b…`, "Smoke test 20261112-100140") waits in review, with a failed gate as expected for
  2,000 steps. Rejecting it would queue it again.
