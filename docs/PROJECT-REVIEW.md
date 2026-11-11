# Project review, 2026-11-10

A review of the whole repository and of the earlier working sessions, made before rewriting `IMPROVEMENT-PLAN.md`.
It lists what is solid, what is wrong, where the evidence is, and what each fix costs. The plan takes its order of
work from this file.

## How this was done

| Part | Method | Result |
|---|---|---|
| `pipeline/` | Read `occupancy.py`, the mapper logic in `sfm.py`, `train_launcher.py` and `plan.py`; searched the other stages for specific settings. Not a line-by-line read of every stage | findings below |
| `pipeline/tests` | `pytest -q -m "not slow"` with pycolmap 4.2.1 (CPU wheel) | 125 passed, 1 failed, 1 deselected in 205 s. The failure is `test_export_decimates_the_phone_file` (no WebGPU adapter), already recorded in `IMPROVEMENT.md` |
| `cloud/` | Full read by a reviewer; the three test files run with Node 22 flags | all three pass |
| `app/` | Full read by a reviewer; `npm ci`, `tsc --noEmit`, `vite build` | all pass. No linter and no tests exist |
| pycolmap 4.2.1 | Listed the options of the installed wheel | see "Camera solve" |
| GitHub | Visibility, branch protection and workflow artifacts through the API | see "This week" |
| Earlier sessions | The exported transcript, `MISTAKES.md`, `IMPROVEMENT.md` | see "How the work was run" |

Not done: no browser session, no paid run, no call to the live API. Items marked *(reviewer)* come from the delegated
read and were not re-checked line by line; everything else was read or run directly. Security findings are listed as
work to do. The step-by-step detail is kept out of this public repository until the fixes are merged.

## What is solid

- **Measured gates.** `verify.py` refuses to publish a run that fails its checks, the export scores what the viewer
  shows (`viewer_eval.py`), and every 30th view is held out and scored.
- **Changes are recorded with evidence.** `IMPROVEMENT.md` names the defect, the measurement and the result for each
  change. `MISTAKES.md` turns each lost run into a rule.
- **Resumable work.** Stages checkpoint to a Volume, a stopped training run exports its newest snapshot, and the CPU
  camera-solve split passed a live job end to end.
- **The fast test suite is real**: 126 tests in about 3.5 minutes on a CPU, with a synthetic room.
- **API basics** *(reviewer)*: prepared statements everywhere, an atomic job claim, presigned uploads that bypass the
  Worker, no secret in the tracked files or the commit history.

## This week

These break the demo or cost money whatever the plan says.

| # | Finding | Evidence | Fix | Effort |
|---|---|---|---|---|
| U1 | **The CUDA wheels expire on 2026-11-12.** They exist only as workflow artifacts with 30-day retention. The workflow is manual and defaults to Python 3.13 and arch 7.5, while the Modal image is Python 3.12. After expiry no worker can be deployed until they are rebuilt | Artifacts API: `ply-wheels-gsplat-py3.12` expires 2026-11-12T23:23:51Z; `.github/workflows/wheels.yml:8-17`; `MISTAKES.md` 2026-11-03 | Keep the wheels somewhere durable with checksums, fetch them in one command, and set the workflow defaults to the Modal target | S |
| U2 | **Credentials posted in chat are not confirmed rotated**: two Modal token pairs and the team and judges' invite codes (and Cloudflare/R2 keys in an earlier session). Rotation was asked for three times | the exported transcript | Rotate all of them; pass secrets as project variables only | S (owner) |
| U3 | **Spend is not bounded by the API.** There is no limit on jobs per invite or per day, trainer flags sent with a job reach the worker as sent, and a job can hold a GPU up to the 6-hour container timeout. The stage limits add up to 590 min against that 360 min timeout, and the file's own header still says "3-hour hard cap" | `cloud/src/index.ts:189-200`; `pipeline/plypipe/worker.py:49`; `pipeline/modal/ply_modal.py:10,87,98` | Job limits per invite and per day, settings mapped to flags on the server, one wall-clock cap per job, stage limits that fit inside the timeout | M |
| U4 | **Every new twin is labelled "Bad capture · very old pipeline".** Any scene id that is not hard-coded gets that badge, so each new job's walkthrough carries it | `app/src/shared/scene-status.ts:47-58` | No badge for unknown ids; take the label from the pipeline version in `scene.json` | S |
| U5 | **The model page loads a room module from any address in the link, and the walkthrough writes place names as HTML**, on the site origin where the invite code is stored | `app/src/model/main.ts:80-85`; `app/src/lab/overlay.ts` *(reviewer)* | Same-origin modules only, text nodes instead of HTML, a content security policy | S |
| U6 | **Job results are served by job id whatever the job's status**, and any invite can decide on any job *(reviewer)* | `cloud/src/index.ts:134-139` | Serve the viewer's files only, for published or in-review jobs; record an owner per job | M |
| U7 | **Nothing is checked on a pull request.** The API tests run on push to `main` only, the site workflow only builds, the pipeline tests run nowhere, and branch protection requires no check | `.github/workflows/api.yml:6-9`; branch API: protected, no required contexts | One PR workflow: API tests, app typecheck and build, pipeline fast tests; make it required | S |
| U8 | **The repository is public** while `LICENSE` and `README.md` call it proprietary and confidential. It holds room photos, account names, budgets, the API address and a competitor teardown | `gh repo view`: PUBLIC | Owner's decision: make it private (Pages then needs a paid plan or a separate public site repo), or drop the confidentiality claim | owner |

Status, 2026-11-10: U1 is fixed. The wheels are on Hugging Face, restored by `modal run scripts/wheels.py::fetch`
and pinned by `pipeline/wheels.lock.json`. U2 and U8 are the owner's decisions: the credentials are not rotated,
and the repository stays public for now at the competition organiser's request.

Status, 2026-11-11: U4 and U5 are fixed in the app. U7 has its workflow (`ci.yml`); making the checks required is
still a repository setting. S1 and T6 are changed in code and reach jobs at the next worker deploy. U3 and U6 are open.

## Walking and layout

The first plan blamed wall-crossing on "37% of cells unknown". That is not what the code does: the viewer treats an
unknown cell as blocked (`app/src/walk/scene.ts:70` blocks every value under 128, and unknown is written as 127 by
`pipeline/plypipe/stages/occupancy.py`). So unknown cells make Walk mode too tight, not too loose. The owner confirmed
on 2026-11-10 that this is the real complaint: in Walk the walker gets stuck and cannot get from place to place, and
the viewer opens in Free as a workaround. W0 is that problem; W1 to W3 are the ways through a wall. None of them has
been measured on the chapel yet.

| # | Finding | Evidence | Fix |
|---|---|---|---|
| W0 | **Walk mode confines the walker to a narrow corridor and closes it easily.** A cell is free only where the solve has floor points or within 0.3 m of the camera path; the rest is unknown and therefore blocked. On a plain floor that is a corridor about 0.7 m wide for a walker 0.44 m wide. Two sparse points block a cell, and only the one-cell path line overrides them | `occupancy.py` (`path_radius=0.3`, `min_pts=2`, the final `grid[path] = FREE`); `app/src/walk/scene.ts:35-38,70`; `app/src/walk/engine.ts:15` | Free space from what the cameras saw through, not only from floor points (plan, 2.0 and 2.3) |
| W1 | **The viewer opens in Free mode, where there is no collision at all**, because Walk is not usable yet. A tour also ends in Free mode | `app/src/walk/engine.ts:22,183-187`; `app/src/walk/tour.ts:71` | Open in Walk once a scene's grid passes the walking gate |
| W2 | **Plain interior walls are bridged.** A cell is blocked only when it holds two or more sparse points between 0.2 and 1.8 m, and free space is then closed with a 0.3 m element, which fills gaps up to about 0.6 m wide. A plain wall has few sparse points, so where both sides were walked the gap is filled as free | `occupancy.py` (`min_pts=2`, `close_m=0.3`) | Build the grid from depth rendered from the trained splat, with free space carved along the view rays (plan, phase 2) |
| W3 | **No sub-stepping.** At wheel speed ×4 one frame moves up to 0.5 m, and a one-cell wall is skipped *(reviewer, simulated)* | `app/src/walk/engine.ts:163,183-186` | Sub-step so no step exceeds half a cell; cap wheel speed in Walk |
| W4 | A start inside a wall can leave the walker frozen: `unstick()` searches 6 m and then gives up silently | `app/src/walk/engine.ts:141-144` | Fall back to the spawn point |
| W5 | The "Measured" layer draws `shell.json` unchecked, and the bundled classroom shell has a 7.19 m ceiling (`ceilingY` in `room-hq-classroom-1/shell.json`). Openings are not drawn | `app/src/walk/Measure.tsx` | Validate the shell in the pipeline; hide the layer when validation fails |
| W6 | SAM 3 already detects doors and windows in the frames (they are in the label prompts), but the layout does not use them | `pipeline/plypipe/stages/label.py` | Lift door and window detections onto the wall planes (plan, phase 2) |

## Camera solve

| # | Finding | Evidence | Fix |
|---|---|---|---|
| S1 | **The global mapper is tried first on every capture and is rejected on every chapel solve** (57, 27, 83, 21 and 29 path jumps), which costs about 32 of the 102 minutes before the incremental mapper starts | `pipeline/plypipe/stages/sfm.py:391,397,420`; `IMPROVEMENT.md` | Ordered video of 400 frames or more goes straight to the incremental mapper. `--mapper incremental` already exists |
| S2 | The incremental mapper runs with defaults plus a lighter bundle-adjustment schedule. Nothing sets a GPU solver, a thread count or a backend | `sfm.py:254-263` | Test what pycolmap 4.2.1 already offers (next row) |
| S3 | pycolmap 4.2.1 exposes `hierarchical_mapping` (clusters of up to 500 images solved in parallel, then merged), `ba_use_gpu`, a second bundle-adjustment backend (`CASPAR`), and `image_names` with `fix_existing_frames` for a keyframe solve. Whether the CUDA wheel has the GPU solvers compiled in is not known: the CPU wheel reports `has_cuda False`, and `IMPROVEMENT.md` records that its Ceres has no CUDA | option listing of the installed wheel | Plan, phase 3: variants 1g, 1f and 1b |
| S4 | Frames are extracted with `fps` and `scale` filters and JPEG quality 2. There is no colour conversion, and the clips' transfer function is not recorded. The chapel clips are 10-bit HEVC, and about 10% of that twin is blown out at the windows. If the clips are HLG or D-Log, highlight detail is being thrown away at this step. Not checked | `pipeline/plypipe/stages/frames.py:104,247` | Record `color_transfer` from ffprobe in the report; convert when it is not BT.709 (free to check on a saved clip) |
| S5 | Scale is the median camera height set to an assumed 1.45 m, with no uncertainty recorded. AR poses are used when a capture has them | `stages/align.py` (`--height-cam`) | Plan, phase 5 |

## Training and export

| # | Finding | Evidence | Fix |
|---|---|---|---|
| T1 | **The High chapel run trained to a 2M cap and shipped 1.06M splats.** The gate warns below 85% kept; this run kept 53%. Why is not recorded in the docs. Two 3M-splat runs also went unstable earlier. Raising the cap to 4M or 6M, as the first plan proposed, would mostly buy splats that are then deleted | `IMPROVEMENT.md:152,283-284`; `train_launcher.py:50` (`opacity_reg=0.01`) | Read the export breakdown of the saved run first (free), then test the opacity regulariser before the cap |
| T2 | **The phone file is 2.15 dB worse than the desktop file** on the High chapel (23.92 against 26.07 dB), about the same as the whole Quick-to-High difference. It is made by cutting the trained model to 800k splats afterwards, which is the "fitting, not filtering" mistake `PROCESSING.md` warns about | `IMPROVEMENT.md:150-151`; `PROCESSING.md:40-42` | Fine-tune a model at the phone size instead of decimating |
| T3 | **No noise floor.** Held-out views are every 30th frame of the same walk, so they sit between two training frames 0.46 s apart. No run has been repeated to see how much the score moves by itself, and an earlier A/B was called a tie at 0.08 dB | `train_launcher.py:61`; `BENCHMARK.md:53-55` | Train the baseline twice; add a held-out block of consecutive frames |
| T4 | Exposure correction is off because twins opened 1.2–1.3× too bright, so varying exposure is fitted as geometry. The trainer flags exist (`--ppisp`, `--bilateral-grid`) but nothing folds the correction back into the colours | `train_launcher.py:25-29,80-98` | Plan, phase 4, variant Q1 |
| T5 | `--pose-opt` and `--depth-loss` exist and have never been scored on a video capture | `train_launcher.py:66-67,120-121` | Plan, phase 4, variants Q2 and Q4a |
| T6 | **Publishing bills the GPU for about 18 minutes**: 48 evidence frames go to Hugging Face one commit per file | `pipeline/plypipe/worker.py:618`; `pipeline/plypipe/hf.py:38-40`; `IMPROVEMENT.md:118-119` | One commit for the folder, or upload after the GPU is released |

## Capture and upload

These matter on the final day, when a new space is captured and uploaded from a phone. All *(reviewer)*.

| # | Finding | Evidence | Fix |
|---|---|---|---|
| C1 | A clip the browser cannot decode is dropped, although the SOP says HEVC is fine; a clip with an infinite duration or a stalled seek hangs with no cancel | `app/src/capture/App.tsx:47-53`; `app/src/capture/lib/session.ts:71-76`; `docs/SOP.md:40` | Make the in-browser analysis best-effort; never drop a clip for it |
| C2 | Creating a job is retried up to five times and the server makes a new job each time, so a lost response starts duplicate GPU runs | `app/src/capture/lib/backend.ts:75-87,136` | An idempotency key from the client |
| C3 | Storage and recorder failures while recording are not handled | `app/src/capture/lib/store.ts:53-54`; `app/src/capture/ui/Live.tsx:35-39` | Surface write errors, check free space |
| C4 | The status page stops polling after "Send it back to the queue" and retries errors every 4 s for ever | `app/src/capture/ui/Status.tsx:32-34,42` | Resume polling, back off, treat a missing job as final |
| C5 | The capture page loads about 1.2 MB of gzipped script (three.js and the splat renderer) before the invite screen | `app/src/capture/ui/LiveTheatre.tsx:8-9` | Load the viewers lazily |

## Viewer

- No handling of a lost WebGL context, and `dispose()` leaves listeners and the renderer behind *(reviewer)*:
  `app/src/walk/engine.ts:147-165,244`.
- `scene.json` carries `scale`, `scaleSource`, `transform` and a `shell` key; the viewer reads none of them and
  ignores `version` *(reviewer)*: `app/src/walk/scene.ts:86-107`. A missing optional `occupancy.json` fails the whole
  load.
- Nothing in the viewer can load a mesh, a collider or a navmesh yet: there is no glTF loader, the engine calls the
  grid directly in four places, and routes are flat 2D paths *(reviewer)*. The plan's viewer work starts with one
  collider interface.
- `tutorial.html` links a stylesheet that is not in the build, and `model.html` with no query throws *(reviewer)*.

## Repository and operations

- **Thirteen `watch*`, `wait*` and night-check scripts** are leftovers of single runs. Some fail open or stop the wrong thing: one runs
  `modal app stop ply`, which takes the deployed app and its wake-up endpoint down *(reviewer)*. Replace them with
  one tool (`scripts/plyctl.py`: status, logs, stop, cost, smoke test).
- **The room model is in the repository twice**, byte for byte: `room-hq-classroom-1/model/parts/` and
  `app/public/models/parts/` *(reviewer)*.
- **No cost record per job.** Spend is read from `modal billing report` by hand. A `job_runs` row per attempt
  (machine, seconds, estimated cost) would make every estimate in the plan checkable.
- **No single status file.** Three sessions began with "continue where it left off" and an exported transcript. A
  short `docs/STATUS.md` (what is deployed, what is running, what is next), updated at the end of each session,
  replaces that.

## How the work was run

From the transcript and `MISTAKES.md`.

| Pattern | What happened | What to change |
|---|---|---|
| Paid runs were the integration test | Three hand-off defects were found only by two live chapel jobs (about $2.45). The tests used stand-ins for Modal and the pipeline | A 60-frame smoke capture through the real deployed path, with the split forced on, for about $0.15, before any long run and after every deploy |
| Rules were written down and broken again | `pkill -f` killed its own shell five times; the rule was in the checklist from the first time | Tools, not rules: `plyctl stop` kills by PID |
| Estimates were off by 2 to 5 times | The chapel test was put at $1–2 and cost $9.46; $4.23 of that was a run lost to a disconnect | Estimate from the per-stage minutes of the closest saved report, and enforce the stop-loss in code (function timeout), not in a checklist |
| Runs nobody asked for | Five training runs when one was asked for (about $2) | One run per approval; the plan below names every paid run |
| Secrets in chat | Twice | Project variables only |
| The plan was written before the viewer was read | The first plan's diagnosis of wall-crossing was wrong, and it sized experiments (splat cap, hardware) without reading what the saved runs already show | Each plan item cites the code or the measurement it rests on |

About $9 of the logged spend went to runs that were lost, unwanted, or stopped by defects a rehearsal would have
caught.

## Docs that disagree with the code

| Doc | Says | Code |
|---|---|---|
| `PROCESSING.md:20-21` | `high` trains about 25% more steps with 15% more gaussians | `plan.py:47-48`: 1.25× steps, 2.0× gaussians (capped at 2M) |
| `train_launcher.py:56` (comment) | frames are already capped at 1600 px | the plan picks 1280–2048 px (`PROCESSING.md:15`) |
| `ply_modal.py:10` | 3-hour hard cap per wake-up | `ply_modal.py:87`: 6 hours |
| `ply_modal.py:22-24` | deploy-time `PLY_STAGE_LIMITS` is honoured | `ply_modal.py:98` hard-codes `sfm=300,export=60` |
| `wheels.yml` header | matches the Kaggle image, Python 3.13 | the Modal image and the artifacts in use are Python 3.12 |
| `IMPROVEMENT-PLAN.md` (first version) | unknown cells let the walker through walls | `app/src/walk/scene.ts:70`: unknown is blocked |
| `README.md` | proprietary and confidential | the repository is public |
