# Improvements

What changed in the pipeline, why, and how it was measured. The newest work is first. Each entry names the defect,
the evidence for it, the change, and what the change was checked against. Costs are from `modal billing report`.

## 2026-11-04: the chapel end-to-end test and its fixes

The chapel's 8 DJI clips (9.7 min of 4K 10-bit HEVC) went through the whole pipeline from scratch, at the quick
preset, to find what breaks or wastes time on a large real capture. The test cost $9.46: a first attempt was lost
to a client disconnect ($4.23), and the rerun cost $5.22 for 119 min on an L40S. What it showed, and what was
done about each finding:

| # | Finding (quick test, before) | Change | After |
|---|---|---|---|
| 1 | The camera solve was 103 of 119 min, on the CPU, with the GPU idle | Lighter bundle-adjustment schedule for long captures; `--reuse-matches`; `modal/sfm_cpu.py` | 58.8 min on a CPU-only box, $0.68, same solve quality |
| 2 | 3 frames posed 22–66 median steps off the walk (rows of identical pews) | `drop_spikes` removes a single view that is out and back between two neighbours | Exactly those 3 dropped; 0 path jumps left |
| 3 | Path jumps counted across clip boundaries | `trajectory_jumps` only compares frames of the same clip | Clip changes no longer send a good solve to the slow fallback |
| 4 | Floor sloped 0.86 m over the 16 m room; floor fit 16° off gravity | `geometry.gravity_guess`: gravity from the cameras' right axes, not the mean up vector | Slope 0.11 m; floor fit 2° off; floor support 5.6% → 18.7% |
| 5 | Frame extraction was 9 min of CPU video decoding on a GPU box | NVDEC (`-hwaccel cuda`) where it works, probed per codec, CPU fallback | 546 s → 129 s for all 8 clips; frames pixel-identical |
| 6 | A blurry clip keeps nearly all its frames (blur is judged within each clip) | `blurry_clips` names clips under 30% of the sharpest clip's sharpness | Reported in the frames log and report (`blurryClips`) |

The result, on the same held-out photo, which neither model trained on:

![The same held-out chapel photo next to the quick test and the high run](report-assets/chapel-compare.jpg)

| Chapel | Quick test (7k steps, before) | High (60k steps, after) |
|---|---|---|
| Held-out PSNR / SSIM / LPIPS (41 views) | 23.37 / 0.820 / 0.323 | 25.59 / 0.872 / 0.212 |
| As the viewer shows it (12 views) | 24.00 dB, 0.66% see-through | 26.11 dB, 0.01% see-through |
| Desktop file (`scene.spz`) | 23.97 dB, 17.7 MB | 26.07 dB, 39.0 MB |
| Phone file (`scene_lite.spz`) | 23.97 dB, 17.7 MB | 23.92 dB, 21.9 MB (800k splats) |
| Splats shipped | 752,661 | 1,058,149 |
| GPU time and cost | 119 min, $5.22 (camera solve included) | 33.5 min, $1.49 (frames and camera solve reused) |

Most of the score gain is the high preset (60k steps, 2M splats against 7k and 1M), not the fixes. The fixes show
in the solve and the floor (rows 2–4), which a score on 12 held-out views barely registers. Published as
`scenes/chapel-high-2` (labelled "Latest pipeline" in the viewer); the library card still points at the older
`hq-chapel-1`.

### 1. The camera solve: from the GPU box to a CPU box

**Evidence.** The first attempt's global mapper ran 23 min and was rejected (57 path jumps), so the rerun went
straight to the incremental mapper: extraction and matching took 10 min (GPU), then incremental mapping took 93 min
(07:02 to 08:35, undistortion included). COLMAP's bundle adjustment runs on the
CPU: the pycolmap-cuda12 wheel's Ceres has no CUDA ("Ceres was compiled without CUDA support"). The GPU box bills
about $2.58/h on an L40S (GPU, 8 cores, 32 GB) and $1.43/h on an L4. A CPU-only box with the same cores and memory
bills $0.63/h.

**Change.**
- `sfm.incremental_options`: above 400 frames, global bundle adjustment runs after 25% growth, not 10%
  (`ba_global_frames_ratio` and `ba_global_points_ratio` 1.25), and skips redundant 3D points
  (`ba_global_ignore_redundant_points3D`).
- `--reuse-matches`: re-solves from `colmap/database.db` without extracting or matching again. Matching writes
  `colmap/matching.json` so a reused solve still reports which features and matcher it used.
- `pipeline/modal/sfm_cpu.py`: runs that re-solve on a CPU-only Modal container and saves the solved job for
  `plypipe run --from align`.

**Measured** (chapel, the same 1,236 frames and matches): 58.8 min on the CPU box for $0.68. It registered 1,223
frames with 0 path jumps, 431k points and 0.582 px error, against 1,226 frames, 433k points and 0.581 px on the GPU
box before the spike filter. The GPU box's incremental mapping took 93 min on the same 8 cores and memory, so the
lighter schedule is the likely cause of the difference, though the two containers' CPUs may differ.

**Not done yet.** The live worker still runs a whole job in one GPU container. Moving the solve there to a CPU
container means splitting a job across containers (GPU: frames, features, matching; CPU: mapping, align, occupancy;
GPU: labels, training, export) with the job folder handed over between them, which touches the worker and the API
and needs a real test job. Saving per long video: about $1.40 (L4) to $2.50 (L40S) at the chapel's size.

### 2. Mismatched poses dropped (`sfm.drop_spikes`)

**Evidence.** Frames 310, 760 and 770 sat 66, 24 and 25 median steps from both neighbours, and those neighbours
sat one step apart. Frame 310 looks down rows of identical pews, so its pose was most likely matched to the wrong
pew. Training would bake each one in as a ghost.

**Change.** After the solve, a view is deregistered when both of its steps exceed 8 median steps and its two
neighbours sit within a quarter of the shorter step. A path that breaks and carries on elsewhere (a folded model)
is left alone and still reported as a jump. Dropped views are listed in `report.sfm.droppedSpikes`.

**Checked** on the chapel's real solve: exactly those 3 frames are dropped, and jumps go from 6 to 0. Tests
cover a spike, a real break, and clip boundaries.

### 3. Path jumps counted within a clip (`sfm.trajectory_jumps`)

**Evidence.** The jump counter compared consecutive frame numbers, and the next clip's first frame follows the
previous clip's last one, wherever in the room that clip starts. Any jump sends a global solve to the slow
incremental fallback.

**Change.** Steps are only measured between frames of the same clip (`frames_index.json`). Photo sets and older
jobs without the index behave as before.

### 4. Gravity from level cameras (`geometry.gravity_guess`)

**Evidence.** Alignment took "up" as the mean of the cameras' up vectors. On the chapel most upward shots faced the
altar, so the mean tilted 4.8° toward it. RANSAC found no floor (2.5% support, 16° off), the gravity fallback
inherited the tilt, and the floor ran 0.86 m downhill along the 16 m room. The room crop was then refused,
because cropping a tilted room would have hidden 24% of every view.

**Change.** A walk keeps the camera level but pitches it freely, so the cameras' right axes stay horizontal. The
least-spread direction of their scatter is gravity. When the camera never turned, or the answer is more than 25°
from the mean up vector (a sideways video), the mean up vector is used as before. `report.align.gravityFrom` says
which was used.

**Checked.** On the chapel's solve the floor slope fell from 0.86 m to 0.11 m end to end. In the high run the floor
fit landed 2° off gravity, and 18.7% of points lie on the floor (5.6% before). Tests cover pitch-biased cameras and
the fallback.

### 5. GPU video decoding (`frames.decode_args`)

**Evidence.** The frames stage decodes every source frame on the CPU to keep the sharpest per window: 546 s for the
chapel's 9.7 min of 4K HEVC.

**Change.** Where `nvidia-smi` exists, one frame is test-decoded with `-hwaccel cuda` per codec. If that works, the
clip is decoded on NVDEC; if decoding fails part-way, the clip is redone on the CPU. `PLY_NO_HWDEC=1` turns it off.
`report.frames.clips[].decoder` records which was used.

**Measured.** A 30 s clip: 25.7 s on the CPU, 4.8 s on NVDEC, with frames identical to the CPU path at
138 dB PSNR (no difference). All 8 clips: 546 s → 129 s.

### 6. Blurry clips reported (`frames.blurry_clips`)

**Evidence.** Frames are dropped only against their own clip's median sharpness, so a clip that is soft throughout
keeps nearly every frame: the chapel's 0053 sat at 27% of the sharpest clip and lost 3 of 240.

**Change.** Clips under 30% of the sharpest clip's median are named in a warning and in `report.frames.blurryClips`.
They are not cut: frames that are the only view of a place still beat a gap in the solve. The threshold is 30%, not
40%, because darkness lowers the measure too (the chapel's dim but sharp 0064 sat at 37%).

### Still open

- **The room crop still costs detail.** On the high chapel, cropping would cost 3.4 dB and 9.8% see-through pixels,
  so the export gate leaves it off (and keeps the floater pass). A wider crop box (0.5–99.5th percentile instead of
  2–98th) only brought that to 2.8 dB and 7.3%, so it was not kept. The cause is not found yet.
- **ALIKED + LightGlue never run.** The pycolmap-cuda12 4.2.1 wheel has no ONNX runtime, so every job uses SIFT.
  Adding it means building COLMAP with ONNX or matching in PyTorch and importing the matches. SIFT registered
  99% of the chapel, so the gain is unproven.
- **The global mapper fails on the chapel**, even with clip-aware jump counting (27 genuine jumps), and costs
  ~23 min before the fallback. Skipping it for long videos was not done: one capture is not enough evidence, and it
  succeeds on the classroom.
- **About 10% of the chapel's splats are blown out** at the windows. That is the footage's exposure, not the pipeline.

### Tests and tools

- 117 fast tests pass (`pytest -m "not slow"`). `test_export_decimates_the_phone_file` needs a WebGPU adapter and
  fails without one, as on `main`.
- `pipeline/modal/chapel_e2e.py`: fetches the chapel clips from Drive onto the `ply-bench` Volume and runs the
  pipeline resumably, saving each finished stage. `pipeline/modal/sfm_cpu.py`: the CPU re-solve.

## 2026-11-03: export precision (PR #7)

**Evidence.** On Dr Johnson high, the trained model scored 31.0 dB on held-out photos, the desktop viewer showed
29.8 dB and phones 26.4 dB. splat-transform rounds spherical harmonics to 4–5 bits when it writes SPZ (the format
holds 8). Spark's default in-memory packing stores positions at half precision, and the phone file dropped SH
entirely.

**Change.** `plypipe/spz.py` rewrites the SH block at 6 bits from the cleaned PLY, after checking both files hold the
same splats. The phone file keeps SH up to degree 2, and the export scores the shipped files (`shippedEval`).
Desktop loads with Spark's `extSplats` (float32 positions).

**Measured** (Dr Johnson high, simulated through Spark's encoders): desktop 29.8 → 30.9 dB, phone 26.4 → 29.0 dB.
Files grew from 22 to 29 MB (desktop) and 13 to 19 MB (phone). Training with 2M splats instead of 1.47M gave
31.06 against 31.02 dB, so more training was not the headroom.

## 2026-11-02: export cleanup and brightness (PR #4)

**Evidence.** The Dr Johnson twins had see-through gaps (the room-box crop removed walls and doorways; the
floater filter removed large distant splats) and opened up to 25% too bright (PPISP's exposure correction is
trained in but never applied by the viewer).

**Change.** Each cleanup step is scored the way the viewer shows the twin (`viewer_eval.py`) and dropped if it
costs more than 0.3 dB or opens 1% more see-through pixels. PPISP is off by default. Twins are capped at 2M splats
after two 3M-splat runs went unstable.

**Measured.** Low 20.2 → 29.0 dB, medium 20.4 → 30.8 dB, high 18.6 → 31.0 dB, with brightness ×1.00 and no gaps.
