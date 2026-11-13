# Status

Updated 2026-11-14. What is deployed, what is running and what is next. Update this file at the end of every
session, so the next one starts here and not from an exported transcript.

## Deployed

| What | Version | Note |
|---|---|---|
| Site (https://ricco.codes/ply/) | `main` at `f92ccff` (PR #33), 2026-11-13 | `ab.html` shows two twins side by side on one camera (the chapel exposure look-test), and every twin opens in Free and the minimap shows the twin's top view. Walk moves at Free's speed and slides along walls. The deploy succeeded; Walk was checked in a browser on the dev build's demo room only, and the live site has not been looked at in a browser since PR #23 |
| API (`ply-api`, Cloudflare) | `35c9c1d`, 2026-11-05 | read from `GET /` |
| GPU worker (Modal app `ply`, riccomarcelino) | v19, `c5cccb3`, 2026-11-12 (the head of PR #29: `main` at `1bf549d` plus the phone file's fine-tune by default; redeploy from `main` if that pull request changes or closes) | an L4, or an L40S when Modal has no L4 free. Smoke test passed after the deploy (job `8582285e…`): every stage, the gate and the phone file's fine-tune ran on a real job, the fine-tune in 44 s of a 62 s export. On that clip (2,000 steps, not a twin) the fine-tuned file scored 1.5 dB higher with 1.7 points more see-through pixels, which is over the export's limit, so the untuned file shipped as designed. The job took 20 minutes from upload to review, most of it the container's start and publishing |
| Twins on Hugging Face (`ply-data`) | `4c46886`, 2026-11-11 | 13 twins have `topview.webp` and a `topview` block in `scene.json`; `4c09354` is the commit before. `chapel-high-2` has the rebuilt walking grid (`4c09354`; `e063f08` before it) |
| CUDA wheels | `wheels/cp312-cu128-sm89` at `e063f08` | L4 and L40S; what `pipeline/wheels.lock.json` pins and every image built so far uses |
| CUDA wheels, rebuilt | `wheels/cp312-cu128-sm80-89-90-r2` at `73b576a` | for A100, L4/L40S and H100. An image built from it passed a GPU check on an L40S, an A100 and an H100 request (served by an H200): gsplat render, `fused_ssim` and a bilateral-grid kernel forward and backward, the trainer's imports. No training run has used it. Its lock is `pipeline/wheels.lock.sm80-89-90.json`; the default lock is unchanged. (`…sm80-89-90` without `-r2` is the first rebuild, whose `fused-ssim` failed on the H100) |

## Accounts

| Modal account | Holds | Used for |
|---|---|---|
| riccomarcelino (about $6 before 2026-11-12; $1.37 billed that day for two deploys and two smoke jobs) | the deployed worker, `ply-bench` (with `smoke/smoke-chapel-30s.mp4`) and `ply-handoff` Volumes, secrets `ply-hf` and `ply-worker` | live jobs |
| opservemakesalt ($30 on 2026-11-11; $5.34 billed by the end of that day and $1.59 on 2026-11-12, plus about $2.30 by estimate for the evening's Q1b runs on both scenes and four export-only runs) | Volume `ply-bench` with `chapel/high/job` and `drjohnson/prep` (each the solve and photos only) and the smoke clip, secret `ply-hf` (read token), the worker image | experiments (`pipeline/modal/ply_eval.py`) |
| ricco-mallick (about $1.70) | `ply-bench:drjohnson/prep` and the Dr Johnson benchmark twins | nothing at present |

## Running

Nothing: every run of 2026-11-13 on the experiments account has finished (the three chapel solves, the bundle-adjustment
check, the scan's export and twin, the MuSHRoom rooms, the chapel's mesh exports). Nothing on the live account.

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
   No run is queued: `q1b` and the walking grid from depth ran on 2026-11-12 (hand-off queue, items 1 and 3), and
   what follows from them waits on the owner (the list after the queue). An export-only rerun of a trained variant is
   `--variant <new name> --export-from <variant> --args=<export options>`.
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

1. ~~Run `q1b` on Dr Johnson~~ Done 2026-11-12, then the owner's go-ahead for the diagnostic and the chapel
   (`EXPERIMENTS.md`, Q1b). Trained with an affine map per photo, the chapel model is clearly better once each view's
   colours are corrected (+0.75 dB, −0.010 LPIPS), but the file it ships is 1.42 dB under in the viewer, which scores
   it against the photos as they were exposed. **Q1 does not ship; the twin keeps no appearance model.**
2. ~~An export-only rerun in `ply_eval.py`~~ Done 2026-11-12: `--export-from VARIANT` with a new variant name and
   `--args=` export options only. The function stops before restoring anything unless the variant's folder holds a
   finished training with its table arguments, takes only what training left there, and never runs `train`.
3. ~~The walking grid from depth on a GPU~~ Done 2026-11-12 on Dr Johnson and the chapel (`EXPERIMENTS.md`). It now
   takes the sparse grid's floor where the depth saw nothing and keeps its walls; that grid is right on both twins
   (the chapel's pews blocked, Dr Johnson's corridor kept), Walk and guided routes hold on it in the viewer, and it
   is **on by default** in export (`--no-dense-occupancy` turns it off). Live jobs get it from the next worker deploy.
4. ~~`scripts/eval_table.py`~~ Done 2026-11-12: a brightness column for both files, flagged outside ×0.95 to ×1.05,
   and a second table with the exposure fold, the walking grid and an export-only run's origin.
5. ~~`pipeline/README.md`~~ Done 2026-11-12.
6. ~~Walk on the live site~~ Done 2026-11-12 on `chapel-high-2` in the dev build: 16 headings held into walls at
   9.7 m/s for 5 s each never put the walker inside a wall, and Walk crossed the chapel along four routes without
   stopping. The plan's guided-route test is in `app/test/walk.test.mts`; it found that "Take me there" routes ran
   up to 12 cm inside the walker's width on the chapel (5 cm from a wall on a 5 cm grid), fixed in
   `app/src/lab/places.ts`.
7. ~~Get the two MuSHRoom rooms~~ Done 2026-11-13 (`EXPERIMENTS.md`): honka and coffee_room are on the experiments
   account's `ply-bench` under `mushroom/`, solved by our pipeline and trained at Quick, and every eval table of the
   scenarios `mushroom-honka` and `mushroom-coffee-room` has "Against the laser scan". Today's scale guess is +5.16%
   and +3.12% off the laser.
8. ~~Phase 6, step 0~~ Done 2026-11-13: `lidar-office-1` is one unified cloud (184.6 M points, colour and intensity,
   no normals, one pose) with **no photos** (`images2D` is empty), so route 8b (training on the scanner's photos)
   is not possible with this file. Route 8a (`plypipe lidar`, one disk per 1 cm voxel) is built and tested on a
   synthetic scanned room and run on the real file: 1.92M splats in 128 s on 4 CPUs, published as `lidar-office-8a`
   and looked at in the viewer (`EXPERIMENTS.md`). Left: a walking route (the file has no scanner station a person
   stood at), the phone file on a real phone, and wall crossings.
9. **Mesh (2.4) and layout (2.5)**: built, tested on synthetic rooms, and **not good enough on a real twin**; both are
   opt-in (`--mesh tsdf`, `--dense-layout`) and neither is scored against a laser yet. On the chapel (`EXPERIMENTS.md`):
   the mesh's first run was simplified to 176 triangles; with that fixed it reaches its counts (298k and 20k, 100 s on
   an L4) but is a soup of crossing faces (5.7 triangles per vertex, 182k pieces), which export now refuses to write.
   `--mesh-keep-raw` saves the extraction (`chapel/eval/q0a-meshraw/mesh_raw.npz` on the Volume) so the next step
   works on real data: find out whether the fused surface or the vertex clustering is at fault. The dense layout
   validates on the chapel and is worse than the sparse shell there (fragments, no end walls).
10. **Rest of phase 0** (Next, 5) and the Hugging Face `.gitattributes` fix (Known and left alone).
11. ~~Phase 3, the camera solve~~ Done 2026-11-13, **nothing to ship** (`EXPERIMENTS.md`). No matches of the chapel
    had been saved, so they were made again once (`pipeline/modal/sfm_match.py`, $0.31) and are now in
    `chapel/high/job/colmap/database.db`. On that database keyframes (1b) is 13% faster than incremental, not 30%;
    hierarchical (1g) produced 15 path jumps and fell back; GPU bundle adjustment (1f) does not exist in the
    installed pycolmap. Both mappers stay as opt-in options. Two identical incremental solves took 60.5 and 34.3
    minutes on two 8-core Modal boxes, so any future timing must compare variants on one box.
12. **Phase 5, scale** (2026-11-13): `plypipe rescale` (2r, the reference measurement) is built and is the shipped
    path by the plan's gate; the viewer's side of it (tapping two points) is not built. The scale from MoGe-2 depth
    (2b, `--priors`, opt-in) is within 3% of the laser on both rooms when applied, but falls back on coffee_room
    under the 5% spread rule (`EXPERIMENTS.md`). The worker image does not have MoGe-2; `ply_priors.py` installs it
    at start.

Not for hand-off, because each needs a judgement or the owner:

- ~~The exposure look-test~~ **Decided 2026-11-14: the owner preferred B.** The page
  (https://ricco.codes/ply/ab.html) showed `chapel-ab-a` = `chapel/eval/q0a-merged` (the baseline) against
  `chapel-ab-b` = `chapel/eval/q1b-nosettle` (one affine colour map per photo, folded back). B is the appearance
  model, so **Q1b ships** under the ship rule that was written down before the pick: it is 0.3 dB under the
  baselines on held-out views (29.13 against 29.25) and 0.0057 better on LPIPS (0.2356 against 0.2412). This is a
  look decision, taken blind, not a metric win; the big-grid `q1` still does not ship. What remains is to make
  `--exposure-affine --exposure-fold 2000` the default for the combined run of phase 7 and to re-export the twins
  that should carry it.
- What goes into the combined run of phase 7: Q6 and the depth walking grid are in; an opacity regulariser of 0.005
  only with see-through no higher than the baseline's (decision 10); whatever of phase 3 passes its ship rule. Phase
  4's Q3 still waits for the chapel's 4K frames and Q4b for depth priors (phase 5's `priors` stage now saves them).
- Whether 2b's 5% spread rule should be a rule on the median's own uncertainty: two rooms are too few to decide.
- Whether the navmesh (5c) is needed: the gate of plan section 6 on the MuSHRoom rooms.
- The worker runs `c5cccb3`. `main` has gained the walking grid from depth, on by default, and two opt-in options
  (`--exposure-fold`, `--exposure-affine`); a redeploy and the smoke test bring the grid to live jobs. Published
  twins keep their grids unless they are exported again.
- Owner: the date of the final, Walk as the default mode (decision 6), the `ci` checks required on `main`, and the
  Modal tokens of all three accounts, which were pasted into chat on 2026-11-12 and need rotating.

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
