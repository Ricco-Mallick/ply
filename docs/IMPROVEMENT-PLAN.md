# Improvement plan: faster solve, better twins, real layout and walking

Rewritten 2026-11-10 after a review of the whole project (`PROJECT-REVIEW.md`). The first version (2026-11-06) set
eight workstreams and about thirty variants side by side. This version keeps the workstreams and their tests, puts
them in one order with a gate after each phase, corrects what the first version got wrong, and cuts what cannot ship
before the final. Nothing ships on a paper's claim alone: each change is measured on our own captures and on one
public dataset with a laser-scanned ground truth, and must not make another metric worse.

Budget: about $38 until the final day ($6.29 on riccomarcelino and $1.71 on ricco-mallick as of 2026-11-06, about $30
of free credit on a third account as of 2026-11-08), then about $60 on the final day (IIT Bombay, $30 on each of the
first two accounts). Prices are Modal list prices per hour (2026-11): H100 $3.95, A100 80 GB $2.50, L40S $1.95,
L4 $0.80, CPU $0.047 per core plus $0.008 per GiB. Follow the checklist in `MISTAKES.md` before every paid run.

## 1. Where a job's time goes today

The chapel at Quick (1,236 frames, job of 2026-11-05/06): 190 minutes from submission to review, $3.07.

| Part | Minutes | Billed on | Note |
|---|---|---|---|
| Matching | 15 | GPU (L4) | |
| Camera solve | 102 | CPU box, $1.12 | about 32 min is a global mapper run that is rejected; it has been rejected on all five chapel solves |
| Training (7k steps) | 14 | GPU | 29 min at High on an L40S |
| Publishing | 18 | GPU | 48 evidence frames, one upload per file |
| Everything else | about 40 | mostly GPU | cold starts, clip download, frames (4 min), align, occupancy, labels, export |

Training is 7% of the job, so a bigger GPU shortens very little of it. More cores are unlikely to shorten the solve
much either, because the incremental mapper registers frames one after another (measured in phase 3, not assumed).
The cheapest minutes are the 32 spent on a mapper that always fails and the 18 spent uploading with a GPU attached:
50 of 190 minutes, for two small code changes (0.8 and 0.9 below).

## 2. What the review changes

| First version | What the code or the record says | Consequence |
|---|---|---|
| 37% of occupancy cells are unknown, so the walker crosses walls | The viewer blocks unknown cells, so they are why Walk mode gets stuck, not why walls are crossed. Free space is known only on textured floor and within 0.3 m of the camera path, which leaves a corridor about 0.7 m wide for a walker 0.44 m wide, and two stray points close it. Walk was the intended mode; the viewer opens in Free, where there is no collision, because Walk is not usable yet (`PROJECT-REVIEW.md`, W0–W3) | Fix the grid first (phase 2), then make Walk the default again. A navmesh is needed only for spaces with more than one floor level |
| Raise the splat cap from 2M to 4M and 6M | The High chapel run kept 1.06M of its 2M splats, and two 3M runs went unstable | Find out why half the budget is deleted (free), then test the opacity regulariser; raise the cap only if the splats survive |
| Ship a training change at +0.2 dB or −0.005 LPIPS | No run has been repeated, so the noise is unknown; an earlier A/B differed by 0.08 dB. Held-out views sit between training frames | Train the baseline twice, hold out a block of consecutive frames too, and read every threshold as the stated value or twice the measured spread, whichever is larger |
| Compare L40S, A100 and H100 on full runs; compare 8, 16 and 32 cores | Training is 7% of the job, and the solve's waste is a mapper choice | Pick hardware from a two-minute timing probe per GPU. Test cores only with a mapper that can use them (1g) |
| Layout from SpatialLM on mesh points | Its weights are non-commercial and it was trained on box rooms. We already hold three strong signals: dense wall points, the camera path (a wall line the camera crossed is a door) and SAM 3's door and window detections | The deterministic layout (4b) is the main line; SpatialLM is one comparison run |
| not covered | The phone file is 2.15 dB worse than the desktop file | Variant Q6: fine-tune a model at the phone size instead of decimating |
| not covered | The wheels expire on 2026-11-12, spend is unbounded, new twins carry a "bad capture" badge, nothing is checked on a pull request | Phase 0 |

## 3. Order of work

| Phase | What | Spend | Gate before the next phase |
|---|---|---|---|
| 0 | Protect the demo: wheels, secrets, spend limits, judge-facing bugs, CI, smoke test, cheaper solve and publish | ~$1 | CI green on a pull request; the smoke capture passes on every account |
| 1 | Measure: harness, baselines, noise floor | ~$5 | Baseline table for the chapel, Dr Johnson and two MuSHRoom rooms; noise floor known |
| 2 | Walking, mesh and layout | ~$2 | Wall crossings under 0.5%, camera path at least 99% walkable, every door passable |
| 3 | Camera solve | ~$2.50 | At least 30% faster at equal quality, or the variant is dropped |
| 4 | Splat quality | ~$11.50 | Each kept variant beats the baseline by more than the noise |
| 5 | Scale | ~$0.60 | Scale error at most 3% on both MuSHRoom rooms, or the reference measurement is the shipped path |
| 6 | LiDAR scan to splat | ~$2 | The phone file loads on a phone; geometry unchanged |
| 7 | Combine, rehearse, freeze | ~$3.50 | One end-to-end run of the `best` preset passes every gate |
| Reserve | a failed run, or a re-run after a fix | ~$8 | |
| **Final day** | tape measurements, then the `best` preset on the chapel and one new capture | ~$50 of $60 | |

That is about $36 of the $38. Phases 2, 3 and 4 do not depend on each other once phase 1 is done. Phase 5 feeds
variant Q4b, and phases 2 and 5 feed phase 6. If time runs short, cut in this order: phase 6, then phase 3 (0.8 stays),
then Q3 to Q5 of phase 4. Phases 0 to 2 and variants Q1, Q2 and Q6 are the minimum worth taking to the final.

## 4. Phase 0: protect the demo

| # | Task | Where | Check | Cost |
|---|---|---|---|---|
| 0.1 | **Keep the wheels.** Done 2026-11-10: the four wheels are in the `ply-data` dataset under `wheels/cp312-cu128-sm89`, pinned with checksums by `pipeline/wheels.lock.json`. `modal run scripts/wheels.py::fetch` restores them and `::publish` stores a rebuild. Open: they are built for sm_89 only (L4, L40S); see phase 7 | `scripts/wheels.py`, `pipeline/wheels.lock.json` | fetched into an empty folder; checksums equal to the originals | $0 |
| 0.2 | **Credentials.** Owner's decision, 2026-11-10: the tokens and invite codes posted in chat stay as they are. New secrets go in project variables, never in chat. That leaves 0.3 as the protection against a leaked code | owner | — | $0 |
| 0.3 | **Spend limits.** Worker side done 2026-11-11 (`plypipe/limits.py`: a 240-minute GPU-time cap per job carried across the CPU solve, stage limits checked against their containers at deploy time, a 12 GB input cap); the API side (per-invite limits, flags mapped on the server, the idempotency key) is open. Per invite: one active job and a daily count. Settings are mapped to trainer flags on the server; raw flags stay for the team key. One wall-clock cap per job. Stage limits that add up to less than the container timeout. A size cap on downloads. An idempotency key on job creation | `cloud/src/index.ts`, `pipeline/plypipe/worker.py:49`, `pipeline/modal/ply_modal.py:87,98`, `app/src/capture/lib/backend.ts` | new cases in `cloud/test/` and `tests/test_worker.py` | $0 |
| 0.4 | **Judge-facing bugs.** App side done 2026-11-11; the two API items (results by status, an owner per job) are open. No "bad capture" badge on new twins. Room modules from our own origin only. Place names as text. Results served only for published and in-review jobs. An owner per job | `app/src/shared/scene-status.ts`, `app/src/model/main.ts`, `app/src/lab/overlay.ts`, `cloud/src/index.ts:134-139` | unit tests, and a Playwright check of a new job's walkthrough | $0 |
| 0.5 | **Checks on pull requests**: API tests, app typecheck and build, pipeline fast tests (205 s on a CPU). Required by branch protection. Workflow added 2026-11-11; making it required is a repository setting (owner) | `.github/workflows/ci.yml` | the workflow | $0 |
| 0.6 | **Smoke capture.** A 30-second clip (60 to 90 frames) kept on every account. `plyctl smoke` submits it through the real API with the CPU split forced on and 2k training steps, then checks every stage, the hand-off, the uploaded files and `verify` | `scripts/plyctl.py`, `PLY_CPU_SFM_MIN` | after every deploy and before every long run | ~$0.15 a run |
| 0.7 | **One operations tool.** `plyctl status / logs / stop / cost / smoke` replaces the thirteen watch scripts and stops processes by PID | `scripts/` | — | $0 |
| 0.8 | **Skip the mapper that always fails (1a).** Ordered video of 400 frames or more goes straight to the incremental mapper. In code 2026-11-11; live at the next worker deploy, saving not yet measured | `pipeline/plypipe/stages/sfm.py:391-433` | `test_sfm_plan.py`: mapper choice by source and frame count | saves ~32 min and ~$0.35 a run |
| 0.9 | **Publish without the GPU waiting.** Evidence frames go up in one commit. In code 2026-11-11; live at the next worker deploy, saving not yet measured | `pipeline/plypipe/worker.py:618`, `pipeline/plypipe/hf.py` | `test_worker.py`: one upload call for the folder | saves ~15 GPU-min a run |
| 0.10 | **`docs/STATUS.md`**: what is deployed, what is running, what is next. Updated at the end of every session. Started 2026-11-11 | `docs/` | — | $0 |
| 0.11 | **Record each clip's colour transfer** (`color_transfer` from ffprobe) in the frames report, and read it for the chapel clips. Done 2026-11-11: the tags are recorded with a warning for HLG and PQ; the chapel clips are `bt709` (`EXPERIMENTS.md`), so no conversion was added. Converting HLG and PQ clips before JPEG is open until a capture needs it | `pipeline/plypipe/stages/frames.py:104,247` | `test_frames.py` | $0 |

## 5. Phase 1: measure

### Scenarios

| ID | Scenario | Source | Ground truth | Tests |
|---|---|---|---|---|
| S1 | **Chapel**: 8 DJI 4K clips, vaulted, glare, some blur, 1,236 frames | `ply-bench:chapel/in` (riccomarcelino) | held-out photos; tape measurements on the final day | solve speed, splat quality, layout on a room that is not a box |
| S2 | **Dr Johnson** (Deep Blending photo set) | `ply-bench:drjohnson/prep` (ricco-mallick) | held-out photos; published scores | splat quality against public numbers |
| S3 | **MuSHRoom**, 2 rooms (iPhone video + Faro laser scan) | Zenodo, open ([records](https://zenodo.org/communities/mushroom/records)) | laser mesh, aligned poses | scale, mesh, floor plan, walking |
| S4 | **Classroom** (phone walk, box room) | our earlier capture, if the clips are still stored | tape, if revisitable | everyday case; regression check |
| S5 | **Synthetic rooms** (`tests/synth.py`): the box room, plus two rooms joined by a door through a plain partition wall, with depth renders | generated | exact | CPU unit tests in CI |
| S6 | **LiDAR office** (`lidar-office-1`) | E57 on HF | scanner cloud | phase 6; a phone-walk test only if a video of the same office appears |

S3 is the only geometric ground truth we have. If the download or its pose format takes more than two hours to get
working, stop and score geometry on S5 and on the chapel's hand-drawn plan only.

### Metrics

| Area | Metric | How |
|---|---|---|
| Image | cc-PSNR, SSIM, LPIPS on every 30th frame | existing (`report.json`) |
| Image | the same on a **held-out block**: 20 consecutive frames from the middle of one clip, poses from the solve | new: a split option in `train_launcher.py` |
| Image | **hard-region LPIPS**: the held-out views that show the altar and the windows, masked to that region | new: view ids picked once, stored in `eval/regions.json` |
| Image | viewer PSNR, brightness and see-through pixels, for the desktop file and the phone file | existing (`viewer_eval.py`, `shippedEval`) |
| Poses | registered frames, reprojection error, path jumps; ATE and rotation error against a reference solve after a Sim3 fit | new `eval/poses.py` |
| Scale | relative error of known distances (MuSHRoom: laser; chapel: tape) | new `eval/scale.py` |
| Mesh | Chamfer distance, F-score at 5 and 10 cm, normal consistency against the laser mesh; wall flatness (RMS distance to the fitted plane, per wall) | new `eval/geometry.py` (scipy cKDTree) |
| Layout | wall endpoint error, room polygon IoU, door and window precision and recall | new `eval/layout.py`; ground truth drawn by hand from the laser mesh (about 1 h per room) |
| Walking | wall-crossing rate over 10,000 random agent steps, with the viewer's own movement code at normal and maximum speed | new `eval/nav.py`, plus one Playwright walk |
| Walking | free-space IoU against the laser floor plan; camera-path walkability; every doorway of 0.7 m or more passable; route success (the share of pairs of labelled places joined by a walkable route) | new |
| Cost | minutes and $ per stage, from `report.json` and `modal billing report` | existing |

### Tooling

- `pipeline/plypipe/eval/`: the modules above. Each takes a job folder and a ground-truth folder and writes
  `metrics.json`. Pure numpy and scipy, so it runs in CI.
- `pipeline/modal/ply_eval.py`: runs a list of variants against one scenario, saves to the `ply-bench` Volume after
  every stage and resumes. **Every variant starts from the same saved stage**, so a training experiment never pays
  for a new camera solve.
- `scripts/eval_table.py`: collects every `metrics.json` into `docs/EXPERIMENTS.md`, one table per phase, with the
  cost of each run.

Built 2026-11-11, for image quality only: `ply_eval.py` runs one named variant per launch from the saved chapel solve
(`plypipe/evalrun.py` holds the logic and prints the plan and cost without Modal), `--holdout-block 20` is the second
hold-out with per-view scores, and `eval_table.py` prints the table and the noise floor. A 300-step smoke variant
proved the chain on the third account for $0.26. The geometry, layout and walking modules (`plypipe/eval/`) are not
built; they wait for the MuSHRoom rooms.

### Runs

| Run | What it answers | Cost |
|---|---|---|
| Free readings from saved runs | Which export filter removed 47% of the High chapel's splats; its opacity histogram; step time from its training log; the chapel clips' colour transfer; wall crossings on the current chapel grid | $0 |
| Baseline twice (chapel, saved solve, 15k steps, High settings) | The noise floor of every image metric | ~$2 |
| Two MuSHRoom rooms end to end at Quick | Baseline scale, mesh, layout and walking against the laser scan | ~$3 |

**Done when** `docs/EXPERIMENTS.md` holds the baseline row for S1, S2 and S3 and the noise floor, and the harness's
unit tests pass on S5.

## 6. Phase 2: walking, mesh and layout

One pass over the trained splat feeds all three, so none of them needs a new model or a second training run.

**What is wrong with Walk today.** Walk was the intended mode. The owner's report (2026-11-10) is that the walker gets
stuck and cannot get from place to place, which is why the viewer opens in Free. The code explains it. A cell is
free only where the solve has floor points or within 0.3 m of the camera path; everything else is unknown, and the
viewer blocks unknown. On a plain floor that leaves a corridor about 0.7 m wide along the walked path, for a walker
0.44 m wide. Two sparse points in a cell block it, and only the one-cell path line overrides them, so a stray point
closes the corridor (`stages/occupancy.py`; `app/src/walk/scene.ts:35-38,70`; `app/src/walk/engine.ts:15`).

**2.0 Stopgap for existing twins (free, CPU only). Done 2026-11-11** (`IMPROVEMENT.md`). The grid is rebuilt from the
saved solve: a walked corridor 0.2 m either side of the camera path that wins over blocked cells, free space from
sight lines below 0.6 m (blocked still wins), no morphological close. `plypipe/walkcheck.py` scores it with the
viewer's rules. Chapel: 0.01 → 28.1 m² reachable, capture positions 2.6% → 100%, places 6.8% → 100%. Open: publish
the new grids to the live twins; wall crossings are not yet measured against ground truth (phase 1). Replaced by 2.3
wherever a GPU pass exists.

**2.1 Viewer (free).** Free is the default on every twin (owner's decision, 2026-11-11, after trying Walk on the
rebuilt chapel grid: it moves too slowly and catches on things). The pipeline still writes its walk check into
`occupancy.json` (`walk.pass`); the viewer no longer reads it. Both complaints are in the engine: Walk moves at
1.4 m/s against 2.2 m/s in Free, and a blocked axis has its speed set to zero, so every 10 cm step in a grid edge
stops the walker dead where it should slide. Still to do, in this order: walking speed and sliding along edges,
then sub-step movement
so no step exceeds half a cell, and cap wheel speed in Walk. A failed `unstick()` returns to the spawn point. Put the
grid behind one collider interface (the engine calls it directly in four places today), so a mesh collider or a
navmesh can replace it later. Files: `app/src/walk/engine.ts:22,136-144,163,183-187`, `app/src/walk/scene.ts`. Tests
(Playwright): holding W into a wall for five seconds at maximum wheel speed keeps the camera inside the room, and a
guided route between two places finishes without getting stuck.

**2.2 Dense geometry pass** (`plypipe/dense.py`, run in `export`, where the splat is already on the GPU). Render
expected depth at every 5th training view at half resolution. Keep a pixel only if its 3D point lands within 2% of
the depth seen by at least two neighbouring views. Save the surviving points with normals. The crop already renders
those views in 1.3 s for 2M splats, so this adds seconds. Tests on S5: wall points within 2 cm; injected floaters
are removed by the consistency check.

**2.3 Occupancy from depth** (`stages/occupancy.py`). Each view's depth between 0.2 and 1.8 m above the floor is a
2D range scan: cells between the camera and the hit were seen free, the hit cell is blocked, and a cell needs three
views to decide. The camera path still wins. Unknown stays blocked. Free space is never closed across a cell that any
view saw blocked. Cells of 5 cm, written in the same `occupancy.png` and `occupancy.json`, so the viewer needs no
change. Without a GPU the sparse grid is used as today.

**2.4 Mesh** (`stages/mesh.py`). TSDF fusion of the same filtered depth, 2 to 4 cm voxels. Outputs `mesh.glb` (about
300k triangles, vertex colours) and `collision.glb` (about 20k triangles).

**2.5 Layout** (`shell.py`). Walls are lines fitted to the dense points of the wall band, snapped to the main axes
when within 5°. A door is a gap in a wall line that the camera path crosses, or a gap at least 0.7 m wide and 1.9 m
high that a SAM 3 "door" detection confirms. Windows are SAM 3 "window" detections lifted onto the nearest wall plane
and seen in at least three views. The shell is validated before it is written (walls inside the bounds, a ceiling
the cameras could have seen, no opening under 0.5 m); the viewer hides the measured layer when validation fails and
draws the openings when it passes.

| Variant | Idea | Retrain? | Licence |
|---|---|---|---|
| 5a | current grid from sparse points | no | ours |
| 5b | grid from depth rays (2.3) | no | ours |
| 5c | Recast navmesh from `collision.glb` (recast-navigation-js), the grid kept as a fallback | no | MIT |
| 3a | COLMAP MVS + Poisson (current `mesh` stage) | no | BSD |
| 3b | TSDF from the splat's own depth (2.4) | no | ours |
| 3c | train with depth and normal losses from the phase 5 priors, then 3b | yes | Apache 2.0 method |
| 3d | gsplat's 2DGS trainer as a separate geometry model, then 3b | yes, second train | Apache 2.0 |
| 4a | current `shell.py` on sparse points | no | ours |
| 4b | 2.5 on dense points, with the camera path and SAM 3 detections | no | ours |
| 4c | SpatialLM 1.1 on the dense points, one comparison run | no | weights **CC-BY-NC** |

3c and 3d run only if 3b fails wall flatness: depth from an unregularised splat is weakest exactly on plain walls. 5c
runs only if 5b fails its gate, or for a space with more than one floor level.

**Measure** on S3 and S1: the walking, mesh and layout metrics above; minutes; phone frame rate.

**Ships if:** wall crossings under 0.5% at maximum speed, camera path at least 99% walkable, every door passable,
a walkable route between every pair of labelled places in the same room, frame rate unchanged. Mesh: F-score at
5 cm beats 3a by at least 10 points on both rooms. Layout: median wall
endpoint error at most 10 cm, every real door found, at most one false door per room.

**Unit tests:** `test_dense.py`, `test_nav.py` (on the two-room scene the plain partition is blocked along its whole
length, the door is free, every camera position is free), `test_mesh_fusion.py` (walls within 2 cm, floor at y = 0),
`test_layout.py` (snapping never moves a wall more than 20 cm; a door drawn on a solid wall is rejected).

Cost: about $2 (the dense pass on the saved chapel twin and the two MuSHRoom twins, one SpatialLM run).

## 7. Phase 3: camera solve

All variants start from the chapel's saved matches on a CPU box. 1a ships in phase 0.

| Variant | Idea | Cost to test |
|---|---|---|
| 1a | Ordered video of 400 frames or more skips the global mapper | done in 0.8 |
| 1g | **Hierarchical mapping**: `pycolmap.hierarchical_mapping` solves clusters of up to 500 frames in parallel and merges them; then one global bundle adjustment. On 8 and 16 cores | ~$1 |
| 1b | **Keyframes first**: solve about 400 keyframes (every third frame plus every clip boundary), then register the rest with `image_names` and `fix_existing_frames`, and one final bundle adjustment | ~$0.60 |
| 1f | **GPU bundle adjustment**: on the GPU image, check whether `ba_use_gpu` or the `CASPAR` backend runs at all on a 60-frame model. Only if it does, time it on the chapel | ~$0.30 for the check |
| 1c | Core count (8, 16, 32) for the winner only | ~$0.60 |

**Measure:** solve minutes and $, registered frames, ATE against the chapel-high-2 poses, then a fixed 7k-step train
on the winning solve only.

**Ships if:** at least 30% faster than 1a, registered frames within 0.5%, ATE under 2 cm, held-out PSNR within the
noise floor.

**Unit tests** (`test_sfm_plan.py`): mapper choice by frame count and ordering; keyframe selection keeps every clip
boundary and caps the gaps; registering the remaining frames leaves none unposed on the synthetic model.

## 8. Phase 4: splat quality

Train-only runs from one saved chapel solve, one factor at a time against the baseline, at 15k steps on an L40S.
Since 2026-11-11 each variant runs first on the small scenario (Dr Johnson, 15k steps at a 600k cap), and the chapel
repeats only what wins there (decision 8).

| Variant | Change | Why |
|---|---|---|
| Q0 | baseline, twice | noise floor (phase 1) |
| Q1 | **Exposure correction, folded back.** Train with the bilateral grid, then fine-tune 2k steps with the grid replaced by its mean and fold that mean into the colours at export | 10–12% of the chapel is glare at the windows; correction is off today only because the viewer showed ×1.2 brightness |
| Q2 | pose refinement (`--pose-opt`, exists) | small solve errors blur detail |
| Q3 | **higher resolution**: re-extract the same frames at 2560 px, scale the intrinsics, keep the poses | the source is 4K and we train at 1728 px |
| Q4a | sparse depth loss (`--depth-loss`, exists) | free to try; fewer floaters |
| Q4b | dense depth and normal losses from the phase 5 priors (the DN-Splatter / AGS-Mesh method) | flatter walls; also variant 3c |
| Q5 | opacity regulariser 0.01 → 0.005 and 0.002 at the 2M cap. A 3M cap only if at least 85% of the splats then survive export | half of the High budget is deleted today |
| Q6 | **phone file fine-tuned, not decimated**: continue 3k steps with the cap at the phone size | the phone file is 2.15 dB below the desktop file |

Then one run with the winners combined, and one at full length (phase 7).

**Measure:** held-out cc-PSNR, SSIM and LPIPS on both hold-outs; hard-region LPIPS; viewer PSNR, brightness and
see-through for both files; file sizes; share of trained splats kept; train minutes and $.

**Ships if:** LPIPS improves by at least 0.005 or PSNR by at least 0.2 dB (or twice the noise floor, whichever is
larger), no other image metric is worse by more than the noise floor, viewer brightness is within ±5%, and the phone
file still loads on a mid-range phone. Q6 ships if the phone file comes within 0.7 dB of the desktop file at the same
size.

**Unit tests:** `test_train_config.py` gains a case for each new flag; `test_upres.py` checks that re-extracted frames
keep their pose and that the intrinsics scale exactly; a test that folding the mean correction leaves the rendered
brightness unchanged.

Cost: ten runs at about $1, plus the combined run: about $11.50.

## 9. Phase 5: scale

Scale sources, most trusted first. `scene.json` records which one was used and its spread across frames, and the
viewer shows measurements with that uncertainty.

| ID | Source | Notes |
|---|---|---|
| 2r | **A reference measurement**: the user enters one known length (a door width, a tape measurement) and taps its two ends in the viewer | exact; makes the final-day tape measurements usable in the product. `plypipe rescale` applies it to the splat, grid, shell and places without retraining |
| 2e | AR poses, when the capture has them | already in `align` |
| 2b | MoGe-2 ViT-L on about 150 frames: the median ratio of its metric depth to SfM depth at the track points | new `priors` stage; also saves depth and normals at half resolution for Q4b and 3c |
| 2c / 2d | MoGe-2 ViT-B; MapAnything (Apache 2.0) | comparisons |
| 2f | door height from SAM 3 door detections | a sanity check only |
| 2a | camera height assumed at 1.45 m | today's fallback |

A monocular model's metric scale carries a systematic error of a few percent, so 3% on every room is not assured.
That is why 2r is built whatever 2b scores.

**Measure:** scale error on S3 against laser distances; implied camera height on S1 and S4 (must fall between 1.2
and 1.8 m); minutes on an L4.

**Ships if:** 2b's scale error is at most 3% on both MuSHRoom rooms and better than 2a. It falls back to 2a, and says
so, when frames disagree by more than 5%.

**Unit tests** (`test_priors.py`, `test_rescale.py`): the scale solver recovers a known scale within 1% with 30%
outlier frames; it reports "inconsistent" and falls back when frames disagree; a rescale by 1.1 moves a known wall by
exactly 10% and leaves the rendered image unchanged.

Cost: about $0.20 per capture on an L4.

## 10. Phase 6: an existing LiDAR scan as a splat

Goal: a customer who already has a scan (E57) gets the same twin as a phone walk: `scene.spz` and `scene_lite.spz`
in the viewer, with the measured layer, occupancy and walking taken from the scan. `lidar-office-1` (184.6 M points)
is the test. Today `scripts/lidar/ingest_e57.py` reads only scan 0 into a 6 M-point coloured PLY, and
`office_shell.py` measures the shell from it. Nothing turns it into Gaussians yet.

**Step 0, inspect the E57 (free):** the number of scans, each scanner pose, and whether `images2D` holds the
scanner's photos with their poses. This decides whether 8b is possible.

| ID | Route | Needs | Expected |
|---|---|---|---|
| 8a | **Direct conversion, no training.** Voxel-downsample to about 1 cm (capped at the High splat count), one Gaussian per point: SH0 colour from the point, rotation from the local normal, a flat disk sized to the neighbour spacing, high opacity. Write a 3DGS PLY and hand it to the existing `export` stage | points only | exact geometry and scale; looks like a dense coloured cloud: no view-dependent colour, holes in the scanner's shadows |
| 8b | **Train on the scanner's photos.** Cut each panorama into six 90° pinhole faces, poses fixed from the E57 (no SfM), initialise from 8a, depth loss from the scan, densification only where the scan has holes | E57 images | real texture and highlights; weak between scanner stations |
| 8c | **Phone video plus scan.** Our SfM on the video, aligned to the scan, trained with 8a as initialisation and scan depth as supervision | a phone video of the same space | phone-walk texture on laser geometry |

Integration: a job input kind `e57` that runs `ingest → lidar_splat → export`, writing `shell.json` and
`occupancy.json` from the full-density scan through the phase 2 code.

**Ships:** 8a when the phone file loads on a phone, F-score at 1 cm from splat centres to the scan is at least 95%,
and wall crossings are under 0.5%. 8b replaces 8a only if held-out LPIPS improves by at least 0.02 and F-score at 1 cm
stays at or above 90%.

Cost: step 0 and 8a on a CPU box, about $0.20. 8b about $1.50 per 15k-step train. 8c only if a video appears.

## 11. Phase 7: combine, rehearse, freeze

1. The winners of phases 2 to 5 become the `best` preset. One full-length training run of it from the saved chapel
   solve, then one end-to-end run at Quick length, both scored by the harness (~$3.50).
2. A two-minute timing probe of the `best` preset on each GPU type that may be used on the final day. The stored
   wheels are built for sm_89 only (`TORCH_CUDA_ARCH_LIST: 8.9` in the build log), which covers L4 and L40S. An H100
   (sm_90) or an A100 (sm_80) most likely cannot load them. If the final-day runs should use either, rebuild first
   with `arch: 8.0;8.9;9.0` in the `wheels` workflow, then `modal run scripts/wheels.py::publish --tag <build>`.
3. The worker is deployed and smoke-tested on every account that will be used, each with its own secrets and its own
   copy of the inputs.
4. A real clip from the final day's camera goes through the phone upload path once (`PROJECT-REVIEW.md`, C1).
5. No code changes after this point, except fixes for a failed gate. `chapel-high-2` stays published as the
   fallback.

## 12. Final day

| When | What | Stop rule |
|---|---|---|
| Morning | Tape measurements: room length and width, two door widths, the altar step. Enter one as the reference measurement (2r); score the others | — |
| Morning | Capture the new space with the SOP: exposure and white balance locked, HDR off | re-shoot a clip the capture check flags |
| Then | Smoke capture on both accounts | no long run until both pass |
| Then | Job 1: the chapel at `best` on one account. Job 2: the new capture at `best` on the other. In parallel | each job has a $12 cap, enforced as the function timeout |
| During | Read the gates as they arrive: frames kept, registered frames at least 90% and no path jumps, the loss curve at 5k steps, `verify` at the end | a failed gate stops the job; one re-run from the saved stage |
| End | Publish the twins that pass. A job that fails twice is not retried; the fallback stays | $10 held back |

## 13. Rules for every paid run

The `MISTAKES.md` checklist, plus four rules from the review:

1. The run is named in this plan, or approved in writing first.
2. The smoke capture passed on that account since the last deploy.
3. The stop-loss is the function's timeout, sized to 1.5 times the estimate, not a note in a checklist.
4. One variable per run, and its row goes into `docs/EXPERIMENTS.md` with the cost from billing, whether it won or not.

## 14. Cut from the first version

| Item | Why |
|---|---|
| InstantSfM (1d), VGGT-Long or MapAnything poses as initialisation (1e) | Each needs a new image and has no record on 1,200 4K frames; 1g, 1b and 1f use what is already installed |
| PGSR or MILo as an upper bound (3e) | $3 for a number that cannot ship (non-commercial) |
| RoomFormer (4d) | 4b covers floor-plan polygons with signals we already have |
| Full-run hardware comparison (T1) | Replaced by a timing probe |
| Rolling-shutter rendering with 3DGUT (T6) | It needs each frame's motion during readout, which we do not record. Revisit only if Q2 shows pose error is the limit |
| 100k steps (T8), LOD chunks | Only relevant above 2M shipped splats, which Q5 has to earn first |

## 15. Licences

MILo, PGSR and SpatialLM's weights are non-commercial. That is fine for a demo and for a comparison. If Ply is sold,
the shipping path must use only the Apache/MIT/BSD options: 2DGS in gsplat, DN-Splatter-style losses, MoGe-2 (MIT
code; check the weights), MapAnything, recast.

## 16. Decisions

Made by the owner on 2026-11-10:

1. The wheels live in the `ply-data` dataset on Hugging Face (0.1, done).
2. The credentials posted in chat are not rotated. New secrets go in project variables.
3. The repository stays public for now, at the competition organiser's request.
4. Walk is the intended mode of the walkthrough. Free stays the default until Walk works (phase 2).

Made on 2026-11-11:

5. The rebuilt chapel walking grid replaces the published one on `chapel-high-2` (done; see `STATUS.md`).
6. Free is the default mode on every twin, whatever the walk check says. This replaces decision 4. Walk stays one
   click away; making it the default again is the owner's call once 2.1 is done.
7. Work continues down this plan with the third account's credit; quality and pipeline stability come first.
8. Experiments run on a small twin first, so the credit lasts until the work is done: Dr Johnson
   (`PLY_EVAL_SCENARIO=drjohnson`, about $0.20 a run by billing). The chapel repeats only a winner, and no full-length run is made
   before the rest of the plan is in. The credentials for the three Modal accounts were given in chat for this.

Open: the date of the final, so the phases can be given dates and the cut order applied.
