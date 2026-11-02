# Pipeline review, 2026-11

A code read of `pipeline/plypipe`, a check of the pinned gsplat (`512d366`) and pycolmap 4.2.1, and web research on
production 3DGS services. Part 1 lists the faults this review fixed in code. Part 2 lists ranked changes still to
make; each needs a GPU A/B on a real capture before it ships. Part 3 covers Real Horizons Studio, for the product roadmap.

Claims marked *(unverified)* come from public docs or papers and have not been reproduced on a Ply capture.

## 1. Fixed in this change

| Fault | Effect | Fix |
|---|---|---|
| `c0e7cf2` passed the export opacity threshold as a logit (`opacity,gt,-3.476`). splat-transform 3.10.1 converts `opacity`/`scale_*` from user units to raw itself, and it rejects an opacity outside (0, 1). | **Every cleaned export crashed.** The probe failure was caught, but the real pass reused the same flags. The worker's export retry (`--min-opacity 0.002` becomes logit -6.2) crashed the same way. `tests/test_export.py` caught it. | `value_filter_flags()` passes a probability. A test pins the units. |
| `--fps 2.5` and `--long-side 1600` were argparse defaults. `ctx.opt(key, planned)` only falls back to the plan on `None`. | The measured plan (`plan.input_plan`, docs/PROCESSING.md) **never applied** to CLI or worker runs. Every video trained at 2.5 fps and 1600 px, whatever the walk or source. | No argparse defaults, so the plan decides and explicit flags still win. |
| The floater pass cut every splat under 5 % opacity *after* the probe had chosen 3 %, 2 % or 1 % to keep 85 % of the model. | The second, stricter cut undid the probe, so translucent MCMC splats were deleted twice. | The floater pass reuses the opacity the probe chose. |
| The floater pass counted splats in a splat's *own* 12 cm cell only. | Plain walls and ceilings, which MCMC covers with few large splats, and any surface straddling a cell edge read as "isolated". The pass punched holes in them. | It now counts the splat's own cell plus the 26 around it. That takes 0.9 s for 4M splats, and a sparse 15 cm-spaced wall is kept whole (`tests/test_clean_verify.py`). |
| The views gate failed anything under 1.8 views/s, while the plan picks ~1.5/s for a slow walk. | Once the plan applied, a slow, careful walk would have **failed for following the plan**. | With `plan.json` present, the floor is 80 % of the planned rate, clamped to [1.2, 1.8]. The classroom (1.12/s vs 2.5 planned) and the chapel (0.53/s) still fail. Worker, CLI and `python -m plypipe.verify` all pass the plan, and the retry logic now reads it too. Before, it got `report["plan"]`, which never existed. |
| MCMC `refine_stop_iter` was fixed at 25k (gsplat stops relocating at 25k of 30k steps). The plan trains for 18k–55k steps. | Runs under 25k steps (the plan's 18k floor and the 7k quick preset) relocated splats until the last refine, with no settling phase. | Set to 0.83 × steps, upstream's ratio. The PPISP controller already uses that ratio. |

## 2. Ranked roadmap

**Status, 2026-11-02:** items 1 and 2 are done and item 3 is partly done. Section 4 covers them, plus two faults found
while doing them. Items 4–7 are still open.

### 1. Measure image quality, then gate and retry on it
The pipeline cannot see sharpness today: `test_every=10**9` and `eval_steps=[]`. Every gate is a proxy (views/s, kept
fraction, file size). gsplat already scores held-out views with PSNR/SSIM/LPIPS and writes `stats/val_step*.json`.
- Hold out ~3 % of views (`test_every≈30`) and evaluate once at the last step. Set `use_color_correction_metric=True`, because
  PPISP renders novel views without per-frame exposure.
- Copy the scores into `report.json`, gate on them, and A/B every change below against them.
  ([simple_trainer](https://github.com/nerfstudio-project/gsplat/blob/512d366b67073d77ca099ede742683c165dfc23b/examples/simple_trainer.py))

### 2. Pose robustness: loop closure, QC and fallback
Bad poses produce doubled walls and softness that no training setting can recover.
- Above `exhaustive_max` frames, matching is sequential with **no loop closure**. pycolmap 4.2.1 has
  `SequentialPairingOptions.loop_detection` and a `vocab_tree_path`. COLMAP 4.0 added ALIKED vocab trees
  ([release](https://github.com/colmap/colmap/releases/tag/4.0.0)).
- Global SfM with ALIKED+LightGlue on sequential video pairs has produced catastrophic outlier poses
  ([glomap#145](https://github.com/colmap/glomap/issues/145): ~11 vs ~20 PSNR against incremental) *(unverified on Ply)*.
  Today `trajectory_jumps` only logs a warning; it should trigger the incremental re-run that a crash already triggers.
- AR captures: pycolmap 4.2.1 exposes `PosePrior`, `match_spatial` and incremental `use_prior_position`. Global rotation
  averaging has `use_gravity`. These would let phone poses constrain SfM instead of only aligning it afterwards.
- Feed-forward poses (VGGT, MASt3R) are not a drop-in replacement: VGGT-X trails COLMAP+MCMC on Mip-NeRF360
  ([2509.25191](https://arxiv.org/html/2509.25191v2)). Use them only as a fallback for texture-poor failures. GlueMap
  ([code](https://github.com/colmap/gluemap)) is the candidate, but its runtime on room-sized captures is unknown.

### 3. Training A/Bs (scored by item 1)
- `pose_opt=True`: with imperfect poses, MCMC+pose-opt went from 22.4 to 26.3 PSNR in the VGGT-X ablation *(unverified on COLMAP-quality poses)*.
- `depth_loss=True`: sparse COLMAP-track disparity, already in gsplat, no new model.
  Dense depth or normal priors for plain walls need custom code. AGS-Mesh reports +0.5–0.9 dB
  ([2411.19271](https://arxiv.org/html/2411.19271v2)), and LighthouseGS reports more on rotation-heavy indoor captures ([2507.06109](https://arxiv.org/html/2507.06109v2)).
- Gaussian budget: gsplat's own table shows MCMC 1M → 3M at +0.47 dB on Mip-NeRF360
  ([eval](https://docs.gsplat.studio/main/tests/eval.html)). Keep training to the size that ships (the plan does this),
  and test 1.5× the planned cap once held-out scores exist.
- Rolling shutter: gsplat's rasterizer accepts `rolling_shutter`, but simple_trainer never passes it.

### 4. Frames and resolution
- The plan now applies, so a 1080p clip trains at 1280 px and a 4K clip at 1728 px. The evidence is mixed. Ply's
  classroom comparison favoured more views at lower resolution. Teleport's lowest training resolution is 1600, and
  Aholo's Pro mode starts at 2K. Run a 1280 vs 1600 vs 1920 A/B on held-out scores before trusting the 45 % rule.
- Add a parallax/overlap gate on top of sharpest-per-window, so near-duplicate frames from standing still don't count as views.
  ([Postshot guide](https://www.jawset.com/docs/d/Postshot+User+Guide/Capturing+Guidelines))

### 5. Speed
gsplat's 30k-step benchmark averages ~36 min on one GPU. Ply's classroom took 164 min, and Aholo took 32. Profile step
time on the Modal L4 before buying a bigger GPU: image decode, PPISP, `packed`, `visible_adam`.

### 6. Delivery
Aholo's own docs warn that SPZ's low SH precision shifts colour and that SOG looks softer
([aholojs](https://aholojs.dev/en-US/manual/3dgs-preset-config/)). Their LOD streams 400k-splat octree chunks with 5 %/1%
coarse levels ([chunk-lod](https://aholojs.dev/en-US/manual/chunk-lod/)). Revisit `?lod=1` with that layout.

### 7. Capture coaching in the app
Every vendor agrees on the capture discipline:
- lock exposure and white balance
- shutter 1/240 s or faster
- walk at 0.5–1 m/s with ≥ 70 % overlap
- make passes at two or three heights

([Aholo](https://aholo.featurebase.app/help/articles/7063834-space-capture-optimization),
[XGRIDS](https://docs.xgrids.com/en-us/06-lixel-cybercolor/04-capture-guide/10-video-reconstruction-capture.html)).
The capture app already records `quality` samples, so it can coach these live.

### Constraints still worth revisiting
- the 3 MB "tiny" failure, which a small bathroom could hit
- `LITE_MAX` 800k
- the +1 m crop box, which removes what is seen through windows
- `--max-frames` being accepted at all

## 4. Second pass, 2026-11-02 (Dr Johnson benchmark)

| Change | Why |
|---|---|
| Every 30th view is held out and scored after training (`report.train.eval`, renders in `eval/`). The gate warns below 20 dB. | Item 1: the pipeline can now see sharpness. |
| `sfm` probes ALIKED in a throwaway process. | **ALIKED never ran in production.** The pycolmap-cuda12 4.2.1 wheel has no ONNX, and COLMAP *aborts the process* instead of raising, so the `except` fallback never ran. Every job crashed in `sfm` and survived only through the worker's crash retry (SIFT + incremental + sequential). |
| `sfm` drops views with fewer than 20 points, or more than 20× the median camera distance. | The global mapper keeps views registered with 0 points wherever global positioning left them, one of them 4 million median distances away. gsplat sizes its learning rates from the farthest camera, so training diverged (CUDA error in MCMC relocation). `train` now refuses such a solve instead of burning GPU time. |
| Sequential matching adds retrieval loop closure (COLMAP's vocab trees, fetched and sha256-checked by us), and photo sets use exhaustive matching, then retrieval. | Item 2. COLMAP's own downloader aborts on a TLS error, so it is never used. |
| A path that jumps, or a solve under 90 % registered, is re-solved with the incremental mapper, and the better model is kept. | Item 2. Before, it only logged a warning. |
| Plan: per-preset step floors (7k / 30k / 40k), the view cap counted in pixels and scaled with the preset, and `high` at 2× the budget. | The presets were indistinguishable on small captures (see `docs/BENCHMARK.md`). |
| Worker: checkpoints after `label` and `train` too. Training's 120-min limit stretches to the projected finish while the machine's deadline allows, and a stopped training run exports its newest snapshot (now taken at 10/30/60/85 %) instead of failing. | Jobs lost hours of training to a fixed limit, with nothing saved. |
| `pipeline/modal/ply_bench.py` runs a dataset through the pipeline on the worker's image, saving to a Volume after every stage and every 3 minutes of training. | Benchmarks never lose progress and never touch the job queue. |
| A/B flags `--pose-opt`, `--depth-loss` and `--scale-mcmc` (all off). | Item 3. The `--scale-mcmc` A/B on Dr Johnson was a tie. |

## 5. Third pass: score what the viewer shows (2026-11-02)

The gaps and brightness the owner saw in the Dr Johnson twins were invisible to every metric the pipeline had.
`plypipe/viewer_eval.py` renders the exported file at held-out cameras without any appearance model and reports
PSNR, brightness and see-through pixels. With it:

- **Crop and floater pass:** export now picks its cleanup by measurement. The crop box was hiding 43 % of every view.
- **Floater pass:** splats at least a cell across are no longer removed.
- **Appearance models:** PPISP and the bilateral grid are off by default, because their twins rendered ×1.2–1.3 too bright.
- **Gate:** warns on brightness off by more than 8 % and on more than 3 % see-through pixels.

Results are in `docs/BENCHMARK.md`.

## 3. Real Horizons Studio (studio.realhorizons.ai)

The app requires login. The findings below come from marketing pages and the app's public JS bundles, read on
2026-11-01. Bundle-only features may be gated.

**Company.** Real Horizons (GastroSpectra Pvt Ltd) is based in Bengaluru, was founded in 2024, and lists two people. Spatial Studio launched around January 2026.
It is positioned as the "last mile" of splats, publishing and sharing, for real estate, hospitality and retail. No funding is public.

**Feature set**
- **Input:** phone video, photo ZIPs, 360 video, drone, DSLR, XGRIDS PortalCam, and existing splats. The Spatial Lens iPhone app provides guided capture.
- **Processing:**
  - cloud quality tiers (draft to ultra), metric scale, streaming LOD, collision voxels and 360 operator masking
  - phases: Prepare → Align cameras → Build → Optimise → Label objects
  - a Mac beta trains locally with COLMAP and Brush
- **Editor:**
  - hotspots, portals, saved views, CTAs, measurements, Color Studio, galleries and floor plans
  - walk mode with collision authoring, skyboxes, Google 3D Maps and 360-pano tours, multi-splat tours
  - a setup wizard: orientation → scale and ground → centre and range → opening view
- **AI:**
  - Spatial Sense: VLM labels anchored to geometry
  - Smart Hotspots and auto-translate
  - 4K Reframe: restyles a view while keeping its geometry
  - Spatial Props: lasso an object and generate an object splat
  - "Dawn", an agent that builds a whole tour from a brief *(bundle only)*
  - a realtime voice visitor guide *(bundle only)*
- **Publish:** stable snapshots, links and embeds, an Explore listing, and analytics (views, sessions, sources, CTA clicks, devices).
  Exports cover SOG, SPZ, PLY and experimental GLB.
- **Pricing (from the bundle):**

  | Plan | Monthly | Credits/mo |
  |---|---|---|
  | Free | $0 | 0 (watermark) |
  | Starter | $12 | 200 |
  | Pro | $39 | 1,000 |
  | Studio | $149 | 3,000 |

  A 4-minute phone scan at Balanced costs about 25 credits. Failed jobs are refunded.

**Onboarding**
1. Two questions:
   - "What would you like to make?" offers a scene from photos/video, or a tour from an existing file.
   - "What are you capturing with?" offers phone, drone, 360 handheld or 360 drone. The subtitle reads "We will show you the right capture route and the mistakes to avoid for this device".
2. A plan screen with a 7-day trial, plus three escape hatches: an editable sample tour (no upload), showcases, and skip.
3. Device-specific capture lessons:
   - source footage plays beside the resulting 3D scene
   - the camera route is colour-coded: green forward pass, blue return pass
   - red "don't" segments show mistakes such as standing still and panning
   - lessons unlock in order
4. Before upload:
   - files are checked in the browser and a credit estimate is shown
   - per-frame reports flag dark, low-detail, fast-turn or disconnected stretches, plus a "% aligned" warning
   - uploads are resumable
   - processing shows phases with an ETA, offers an email when done, and says the tab is safe to close
5. Then "let Dawn set up the tour while you watch".

**What to copy first**
1. **The two-question router into device-specific capture lessons.** Ply already has `capture.json` quality samples and the
   coverage preview (`--until align`), so the lesson can end with a real check of the user's own clip.
2. **A no-upload sample project** that opens in the editor. The `room-hq-classroom-1` twin can be that sample.
3. **Cost and ETA before upload, phases and ETA after.** The worker already reports stages and progress.
4. **Hotspots, saved views, measurements and publish/embed with analytics.** This is UI and data-model work. The pipeline already
   provides metric scale, floor, occupancy and the shell.
5. **Geometry-anchored labels and an agent-built tour.** SAM 3 labels (`stages/label.py`) and `tour.py` keyframes are
   the starting point. Hand the agent the measured shell and labels, as `.agents/skills/ply-room-model` does.

**Gap list.** Most of Studio is UI on top of a splat, metric scale and a floor, which Ply already produces. The new
pipeline capabilities are:
- 360/fisheye input (`OPENCV_FISHEYE` exists for SfM; training would need 3DGUT)
- operator masking
- generative object splats (Spatial Props)
- view restyling (Reframe)
- streamed LOD that reads well while loading
