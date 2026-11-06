# Improvement plan: faster solve, better twins, real layout and walking

Written 2026-11-06. This plan says what to build next, how each idea is tested against the alternatives, and what
result lets it ship. Nothing ships on a paper's claim alone: each change is measured on our own captures and on one
public dataset with a laser-scanned ground truth, and must not make another metric worse.

Budget: the dev phase uses whatever credit is left now. The final day has about $60 ($30 on riccomarcelino, $30 on
ricco-mallick), so hardware is not the constraint that day. Follow the checklist in `MISTAKES.md` before every paid run.

Prices used below are Modal list prices per hour (2026-11): H100 $3.95, A100 80 GB $2.50, L40S $1.95, L4 $0.80,
CPU $0.047 per core plus $0.008 per GiB. Times come from the chapel runs in `IMPROVEMENT.md`.

## Where we are (chapel, 1,236 frames)

| Stage | Time | What it does now | Weakness |
|---|---|---|---|
| frames | 4 min | sharpest frame per window, 1728 px from 3840 px source | throws away resolution |
| sfm | 15 min match (GPU) + 102 min solve (CPU) | SIFT, global mapper, then incremental fallback | global mapper fails on every chapel run: 32 min wasted |
| align | 30 s | floor plane by RANSAC; scale from an assumed 1.45 m camera height | scale is a guess (`scaleSource: camera-height-prior`) |
| occupancy | 4 s | 10 cm grid from **sparse** SfM points | 37% of cells unknown; plain walls have no points, so the walker goes through them |
| label | 3–5 min | SAM 3 boxes lifted on SfM points | fine |
| shell | inside align/label | RANSAC wall lines on sparse points | misses walls, no reliable doors/windows |
| train | 14 min (Quick) / 29 min (High, L40S) | gsplat MCMC, 2M cap, SH3, no depth or normal supervision | soft detail (altar), 10–12% glare at windows |
| mesh | optional | COLMAP MVS + Poisson | fragile, slow, not used for walking |

## Phase 0: the evaluation harness (build first, everything else depends on it)

### Scenarios (test data)

| ID | Scenario | Source | Ground truth | Tests |
|---|---|---|---|---|
| S1 | **Chapel**: 8 DJI 4K clips, vaulted, glare, some blur, 1,236 frames | `ply-bench:chapel/in` (riccomarcelino) | held-out photos; tape measurements if you can revisit | solve speed, splat quality, layout on a non-box room |
| S2 | **Dr Johnson** (Deep Blending photo set) | `ply-bench:drjohnson/prep` (ricco-mallick) | held-out photos; published scores | splat quality against public numbers |
| S3 | **MuSHRoom**, 2 rooms (iPhone video + Faro laser scan) | Zenodo, open ([records](https://zenodo.org/communities/mushroom/records)) | laser mesh, aligned poses | scale, mesh, floor plan, walking |
| S4 | **Classroom** (phone walk, box room) | our earlier capture, if the clips are still stored | tape, if revisitable | everyday case; regression check |
| S5 | **Synthetic room** (`tests/synth.py`, extended with a door and depth renders) | generated | exact | CPU unit tests in CI |
| S6 | **LiDAR office** (`lidar-office-1`) | E57 on HF | scanner cloud | only if there is a phone video of the same office (question below) |

S3 is the key addition: our captures have no geometric ground truth, so a mesh or floor plan can only be judged by
eye. MuSHRoom is the dataset AGS-Mesh and DN-Splatter report on, which also lets us check our numbers against theirs.

### Metrics

| Area | Metric | How |
|---|---|---|
| Image | cc-PSNR, SSIM, LPIPS on the held-out views (every 30th frame, already in `report.json`) | existing |
| Image | **hard-region LPIPS**: the held-out views that show the altar and the windows, masked to that region | new: pick view ids once, store in `eval/regions.json` |
| Image | viewer PSNR and brightness (`viewer_eval.py`) | existing: catches "trainer looks better, viewer looks worse" |
| Poses | registered frames, reprojection error, path jumps; ATE/rotation error against a reference solve after a Sim3 fit | new `eval/poses.py` |
| Scale | relative error of known distances (MuSHRoom: laser; chapel: tape) | new `eval/scale.py` |
| Mesh | Chamfer distance, F-score at 5 and 10 cm, normal consistency against the laser mesh | new `eval/geometry.py` (scipy cKDTree, no new deps) |
| Mesh | wall flatness: RMS distance to the fitted plane, per detected wall | new |
| Layout | wall endpoint error (cm), room polygon IoU, door/window precision and recall | new `eval/layout.py`; ground truth drawn by hand from the laser mesh (about 1 h per room) |
| Walking | free-space IoU against the laser floor plan | new `eval/nav.py` |
| Walking | wall-crossing rate: 10,000 random agent steps; % that cross a ground-truth wall | new |
| Walking | camera-path walkability: % of capture positions the agent can stand on (target 100%: someone walked there) | new |
| Walking | door passability: every doorway ≥ 0.7 m wide is passable | new |
| Cost | minutes and $ per stage, from `report.json` and `modal billing report` | existing |

### Tooling

- `pipeline/plypipe/eval/` with the modules above, each taking a job folder (and a ground-truth folder) and writing
  `metrics.json`. Pure numpy/scipy, so it runs on the CPU and in CI.
- `pipeline/modal/ply_eval.py`: runs a list of variants against one scenario. Like `ply_bench.py`, it saves to the
  `ply-bench` Volume after every stage and resumes. **Every variant starts from the same saved stage**
  (`--from train`, `--from align`, `--reuse-matches`), so a training experiment never pays for a new camera solve,
  and differences come from the change, not from a different solve.
- `scripts/eval_table.py`: collects every `metrics.json` into `docs/EXPERIMENTS.md`, one table per workstream, with
  the cost of each run.

**Done when:** the harness scores the current pipeline on S1, S2 and S3 (the baseline row of every table), and its
unit tests pass on synthetic data with known answers.

Cost: about $3 (MuSHRoom: SfM on CPU, plus one Quick train per room to get a baseline mesh and floor plan).

## Workstream 1: camera solve, faster at equal quality

| Variant | Idea | Cost to test |
|---|---|---|
| 1a | Skip the global mapper on long ordered video (it failed on every chapel run); go straight to incremental | $0.80: CPU box from saved matches |
| 1b | Solve ~400 keyframes, then register the rest with COLMAP's image registrator and one final bundle adjustment | $0.60 |
| 1c | Core count: 8 vs 16 vs 32 cores on the CPU box (incremental mapping is partly serial; measure, don't assume) | 3 × $0.60 |
| 1d | InstantSfM (GPU global SfM, claims up to 40× COLMAP) on an A100 | $1.50; experimental |
| 1e | VGGT-Long or MapAnything poses as initialisation, refined by COLMAP BA | $1.50; experimental |

**Measure:** solve minutes and $, registered frames, ATE against the chapel-high-2 poses, then a fixed 7k-step train on
each solve to check that held-out PSNR doesn't drop.

**Ships if:** ≥ 30% faster, registered frames within 0.5%, ATE < 2 cm, held-out PSNR within 0.1 dB.

**Unit tests** (`test_sfm_plan.py`): mapper choice by frame count and ordering; keyframe selection keeps every clip
boundary and caps the gaps; registering the remaining frames leaves none unposed on the synthetic model.

## Workstream 2: real-world scale and gravity from MoGe-2

New stage `priors` after `sfm`: run MoGe-2 (ViT-L, NeurIPS 2025) on about 150 evenly spread frames. Each frame gives
metric depth and normals.
- **Scale:** per frame, the median ratio of MoGe depth to SfM depth at the frame's track points; across frames, a robust
  median. `align` uses it as `scaleSource: "moge"` when the frames agree (spread under 5%). Otherwise it falls back to the
  camera-height prior and says so in the report.
- **Gravity:** the floor-normal vote from MoGe normals cross-checks the RANSAC floor.
- **For W3/W6:** depth and normal maps are saved at half resolution.

| Variant | Source |
|---|---|
| 2a | camera-height prior (baseline) |
| 2b | MoGe-2 ViT-L |
| 2c | MoGe-2 ViT-B (faster) |
| 2d | MapAnything (Apache 2.0) |

**Measure:** scale error on S3, using laser distances; camera height implied on S1/S4 (must land between 1.2 and 1.8 m);
minutes on an L4.

**Ships if:** scale error ≤ 3% on both MuSHRoom rooms and better than the baseline. Never worse than the baseline on
S1/S4 plausibility.

**Unit tests** (`test_priors.py`): the scale solver recovers a known scale within 1% with 30% outlier frames; it reports
"inconsistent" and falls back when frames disagree; the y-up conversion of normals is right.

Cost: about $0.20 per capture on an L4.

## Workstream 3: a real surface (mesh)

| Variant | Idea | Retrain? | Licence |
|---|---|---|---|
| 3a | COLMAP MVS + Poisson (current `mesh` stage) | no | BSD |
| 3b | **Render depth from the trained splat** at every training view (median depth), fuse with TSDF, extract the mesh | no: runs on any finished twin, chapel-high-2 included | ours |
| 3c | Train with **depth and normal losses** from the W2 priors (the DN-Splatter / AGS-Mesh method, added to our gsplat trainer), then 3b | yes | Apache 2.0 method |
| 3d | gsplat's **2DGS** trainer as a separate geometry model, then 3b; the 3DGS twin stays for looks | yes, second train | Apache 2.0 |
| 3e | PGSR or MILo as an upper bound, separate environment | yes | **non-commercial** |

**Measure:** Chamfer, F-score at 5/10 cm and normal consistency on S3; wall flatness and hole ratio on S1; minutes. For
3c, also held-out PSNR/LPIPS, because the losses must not cost image quality.

**Ships if:** F-score at 5 cm beats 3a by ≥ 10 points on both rooms. 3c also needs held-out PSNR within −0.2 dB and LPIPS
within +0.005 of the same run without the losses. Otherwise it is only used for the geometry model (as 3d).

**Outputs:** `mesh.glb` (decimated, about 300k triangles, vertex colours) and `collision.glb` (about 20k triangles,
walls/floor/furniture only, for W5).

**Unit tests** (`test_mesh_fusion.py`): TSDF fusion of synthetic depth renders recovers the walls within 2 cm and the
floor at y = 0; the depth loss is zero at the true depth and its gradient points the right way; `build_config` adds the
losses only when asked.

Cost: 3b about $0.30 (render + fuse). 3c/3d about $1 per Quick-length train. 3e about $3.

## Workstream 4: layout (walls, doors, windows, floor plan)

The input is a uniform point sample of the W3 mesh, not the sparse SfM points.

| Variant | Idea | Licence |
|---|---|---|
| 4a | current `shell.py` (RANSAC on sparse points) | ours |
| 4b | `shell.py` on dense mesh points, with Manhattan snapping where walls are within 5° of the main axes | ours |
| 4c | **SpatialLM 1.1** (NeurIPS 2025): point cloud in, walls/doors/windows/object boxes out. Qwen-0.5B and Llama-1B | weights **CC-BY-NC** |
| 4d | RoomFormer on a top-down density map (floor-plan polygons only) | MIT |
| 4e | 4c snapped to 4b's planes; a door kept only where the mesh has a real gap; objects merged with SAM 3 places | mixed |

Expected trouble: SpatialLM was trained on box-like homes and offices. The chapel (vaulted, apse) is the stress test,
and 4e exists for that case.

**Measure:** wall endpoint error, room IoU, door/window precision and recall on S3 (hand-drawn ground truth) and S1
(drawn from the High twin's top view).

**Ships if:** median wall endpoint error ≤ 10 cm, every real door found on the test rooms, and at most one false door per
room.

**Unit tests** (`test_layout.py`): y-up/z-up round trip; SpatialLM's text output parses into the `shell.json` schema
(walls, openings, objects); snapping moves a wall onto the nearest plane but never by more than 20 cm; door validation
keeps the synthetic room's real door and rejects a fake one drawn on a solid wall.

Cost: under $0.50 per run (inference on an L4).

## Workstream 5: walking (collision and navmesh)

| Variant | Idea |
|---|---|
| 5a | current sparse-point grid (10 cm) |
| 5b | grid from **mesh slices**: blocked = surface between 0.2 and 1.8 m; free = floor surface; 5 cm cells. No viewer change |
| 5c | **Recast navmesh** built from `collision.glb` in the pipeline (recast-navigation-js in Node, already in the image for splat-transform); viewer moves along it with `@recast-navigation/three`; grid kept as a fallback |
| 5d | 5c plus W4 walls as hard edges and doors carved open |

**Measure** on S3 and S1: free-space IoU, wall-crossing rate, camera-path walkability, door passability. Phone frame rate
must not drop.

**Ships if:** wall-crossing rate under 0.5%, camera-path walkability ≥ 99%, every door passable, frame rate unchanged.

**Unit tests:** in Python (`test_nav.py`), a synthetic room with a table and a door: the agent can't cross a wall or the
table, can pass the door, and can stand on every camera position. A viewer test (Playwright, like the review-page
check): holding W into a wall keeps the camera inside the room, and walking through the doorway works.

Cost: CPU only.

## Workstream 6: splat quality ("much better", no trade-offs)

Train-only runs from one saved chapel solve, one factor at a time against the baseline, at Quick-plus length (15k
steps). The winners are then combined at full length on the final day.

| Variant | Change | Why |
|---|---|---|
| T0 | baseline: chapel-high-2 settings (60k steps, 2M cap, SH3) | reference |
| T1 | same config on L40S / A100 80 GB / H100 | speed per $ only; output identical |
| T2 | **higher resolution**: re-extract the same frames at 2560 and 3840 px, scale the intrinsics, keep the poses | source is 4K; we train at 1728. No new solve needed |
| T3 | splat cap 2M → 4M → 6M (A100/H100 memory) | altar detail |
| T4 | depth and normal losses (W3 variant 3c) | flatter walls, fewer floaters |
| T5 | pose refinement in training (`--pose-opt`) | small solve errors blur detail |
| T6 | rolling-shutter-aware rendering (gsplat 3DGUT, `with_ut`) | DJI footage; fast pans |
| T7 | exposure model (`--bilateral-grid` / `--ppisp`) with a viewer brightness fix | 10–12% glare at the windows. Off today because the viewer showed ×1.2 brightness |
| T8 | steps 60k → 100k at the larger cap | more splats need more steps |

**Measure:** held-out cc-PSNR, SSIM, LPIPS; hard-region LPIPS (altar, windows); viewer PSNR and brightness; file sizes
(full and phone); train minutes and $.

**Ships if:** LPIPS improves by ≥ 0.005 or PSNR by ≥ 0.2 dB, no other image metric is worse by more than 0.1 dB /
0.003, viewer brightness is within ±5%, and the phone file still loads on a mid-range phone.

The pinned gsplat (`512d366`) already has the 2DGS trainer (W3d) and 3DGUT (`with_ut`, T6), so neither needs a
dependency change.

**Unit tests:** `test_train_config.py` gains a case for each new flag. A new `test_upres.py` checks that re-extracted
frames keep their pose and that the intrinsics scale exactly.

Cost: about $1 per variant on an L40S (15k steps), about $10 for the set.

## Workstream 7: viewer and delivery

Load `mesh.glb` / `collision.glb` / `navmesh.bin` from `scene.json` (new optional keys, old scenes keep working). Draw
walls, doors and windows from the layout in the measured layer. Above 2M splats, use LOD chunks, so phones get the lite
file and desktops the full one. Tested with the existing build and a Playwright walk test.

## Schedule and spend

| Phase | Work | Spend |
|---|---|---|
| 0 | harness, MuSHRoom download and baselines, regions file | ~$3 |
| 1 | W1 solve variants (CPU, saved matches); W2 priors; W3b mesh from chapel-high-2; W4 and W5 on that mesh | ~$6 |
| 2 | W3c/3d and W6 train-only variants at 15k steps | ~$12 |
| 3 | combine the winners into a `best` preset; one Quick end-to-end run per scenario to catch breakage | ~$5 |
| Final day | 2–3 full `best` runs on H100 (chapel plus one new capture), each ~$6–10 with the bigger cap and resolution; a second worker deployed on ricco-mallick so two jobs run at once; $10 held back for a failed run | ~$50 of $60 |

The order matters: W2 feeds W3, W3 feeds W4 and W5, and W1/W6 are independent. Results go into `docs/EXPERIMENTS.md`
as they come in, and every shipped change gets an entry in `IMPROVEMENT.md` with its numbers.

## Licences

MILo, PGSR and SpatialLM's weights are non-commercial. That is fine for a demo and for measuring an upper bound. If Ply
is sold, the shipping path must use only the Apache/MIT/BSD options: 2DGS in gsplat, DN-Splatter-style losses, MoGe-2
(MIT code; check the weights), MapAnything, RoomFormer, recast.

## Open questions for the team

1. Can someone revisit the chapel or the classroom with a tape measure (room length and width, two door widths, altar
   step height)? Five numbers make the scale and layout scores real on our own captures.
2. Is there a phone video of the LiDAR office? With it, S6 becomes our best geometry test.
3. How much credit is left on each account now, for phases 0–3?
