# plypipe: phone walkthrough → navigable 3D twin

```
video ─► frames ─► sfm ─► align ─► occupancy ─► train ─► export
         sharp     COLMAP  floor y=0  walkable    gsplat   scene.spz
         frames    4.2.1   +Y up, m   grid (R1)   MCMC     scene_lite.spz
```

One command, resumable, JSON-lines progress:

```bash
python -m plypipe run walk.mp4 --out work/lab2 --preset standard
```

From the capture app (`app/capture.html`): "Save files to this phone" gives you the clips plus `capture.json` (per-frame
quality and sensor log). Put them in one folder and pass the folder; the app shows the exact command, including any
Advanced settings (`--iterations`, `--sh-degree`, `--cap-max`, `--mapper`, `--features`, `--fps`, `--height-cam`):

```bash
python -m plypipe run captures/lab-2/ --out work/lab-2 --preset standard --height-cam 1.45
```

| Stage | Needs | What it does |
|---|---|---|
| `frames` | ffmpeg, OpenCV | Samples the planned rate (~2.2 frames/s for a normal walk, 1.5–3.2 by motion; `--fps` overrides), oversampled ×4; keeps the sharpest of each window (variance of Laplacian), drops very blurry ones without opening gaps; long side planned from the source (45 %, 1280–2048 px; 1600 px for photos; `--long-side` overrides). A folder of clips (one space, name order) is numbered straight through and keeps `capture.json`; a folder of photos also works. |
| `sfm` | pycolmap 4.2.1 (GPU wheel) | One shared SIMPLE_RADIAL camera; ALIKED + LightGlue when the wheel has ONNX (probed in a throwaway process: the cuda12 4.2.1 wheel has none, and ALIKED aborts instead of raising), else SIFT. Pairs: exhaustive up to 250 frames (700 multi-clip, 500 photos), then sequential + quadratic overlap **with retrieval loop closure** for video, vocab-tree retrieval for photos. **Global SfM**, except ordered video of 400 frames or more (`--incremental-min`), which starts incremental because the global solve was rejected on every chapel run (`--mapper` overrides either way); views it could not anchor (< 20 points, or > 20× the median camera distance) are dropped, and a path that jumps or a solve under 90 % registered is re-solved incrementally, keeping the better model. Undistort to PINHOLE. |
| `align` | numpy | Gravity from the cameras' up vectors → floor = lowest plane covering ≥ 50 % of the best horizontal area (beats textured table tops) → walls on X/Z (Manhattan histogram sweep) → metric scale from `--height-cam` (1.45 m). Writes gsplat's `data/` already in the scene frame. |
| `occupancy` | numpy, OpenCV | 10 cm grid: blocked = points 0.2–1.8 m above the floor, free = floor points ∪ the walked path, closed over 30 cm gaps. The walked line is never blocked. |
| `train` | torch 2.11 cu128, gsplat @ `512d366` | gsplat `simple_trainer`, MCMC strategy, no appearance model by default (`--ppisp` / `--bilateral-grid` opt in; both made the exported twin render ×1.2 too bright). `--exposure-fold STEPS` fits one colour map from the model trained with an appearance model to the photos, folds it into the splats' colours and trains STEPS more without it, so the twin keeps the photos' brightness (`report.train.exposureFold`); `--exposure-affine` trains with one affine colour transform per photo (the bilateral grid at 1×1×1), exactly what the fold can undo. Both are A/B options (`docs/EXPERIMENTS.md`, Q1). Trained **in the scene frame** (`normalize_world_space=False`), so the PLY needs no transform. Every 30th view is held out and the twin is scored on it (PSNR/SSIM/LPIPS, `report.train.eval`; `--holdout 0` disables). |
| `export` | Node ≥ 20 | splat-transform 3.10.1: NaN filter, opacity filter, crop to the room box (+1 m), floater pass (splats under 12 cm with no neighbours), `scene.spz` (full SH) and `scene_lite.spz` (SH2, for phones), with the SH coefficients rewritten at 6 bits (`plypipe/spz.py`; splat-transform rounds them to 4–5). On a GPU each cleanup step is scored as the viewer shows the twin (`plypipe/viewer_eval.py`: held-out photos, no appearance model) and dropped if it costs more than 0.3 dB or opens 1 % more see-through pixels; `report.export.viewerEval` and `cleanupChoice` record it, and `shippedEval` scores the two SPZ files themselves. `--dense-occupancy` (opt-in, needs a GPU) rebuilds the walking grid from the cleaned model's own depth in 5 cm cells and keeps it only when it walks at least as well as the sparse grid (`report.export.denseOccupancy`). |

Presets (`--preset`) scale the plan (`plypipe/plan.py`): `quick` ≥ 7k steps, 0.5× the gaussian budget, SH2 · `standard` ≥ 30k steps, 1× budget, SH3 · `high` ≥ 40k steps, 2× budget. Steps grow with views past the floor (50 per view); the budget follows surface area and is capped by the views' pixels (2400 gaussians per view-megapixel × the preset). Measured on Dr Johnson: see `docs/BENCHMARK.md`.

A/B options (off by default): `--pose-opt`, `--depth-loss` (sparse SfM depth), `--scale-mcmc` (MCMC noise scaled to the room), `--exposure-fold` and `--exposure-affine` (train), `--dense-occupancy` (export).

Image-quality experiments run one variant at a time from a saved camera solve (`pipeline/modal/ply_eval.py`; `python -m plypipe.evalrun plan --variant q1b` prints what a run would do and cost). `--export-from VARIANT` with a new name exports again the model a variant already trained, with export options only (`--args=--dense-occupancy`), and stops before anything runs if that training is not on the Volume.

Useful flags: `--until occupancy` (CPU box: everything but training), `--force sfm` (re-run a stage and all after it), `--focal-35mm 24` (phone lens prior when SfM struggles; the S24 FE main lens is 24 mm-equivalent, about 26 with video stabilisation), `--max-frames 400`, `--features sift`, `--no-ppisp`.

## Output contract (what the web viewer reads)

All coordinates are in the **scene frame**: right-handed, **+Y up, floor plane at y = 0, metres**, walls along X / Z, origin on the floor under the median camera. This is three.js's own convention: load the SPZ with no rotation.

| File | Contents |
|---|---|
| `scene.spz` | Gaussian splats, full SH. Positions stored as-is (splat-transform writes SPZ with `CoordinateSystem.UNSPECIFIED`, i.e. no axis conversion). |
| `scene_lite.spz` | Same splats up to SH degree 2, decimated to 800k above that (smaller; use on phones). Where that costs it splats or SH bands, the phone model is then trained on for 3,000 steps at its own splat count and SH degree, and shipped when the viewer shows it better (`--lite-finetune N`, 0 = off; needs a GPU and held-out views; `report.export.liteFinetune`; +0.80 dB on the chapel, +0.73 and +0.52 dB on Dr Johnson, `docs/EXPERIMENTS.md`). |
| `scene.json` | Scene metadata, below. |
| `occupancy.png` | 8-bit grey grid: **255 free, 0 blocked, 127 unknown**. Pixel (row r, col c) = cell r along +Z, c along +X; the top-left pixel is the cell at `origin`. |
| `occupancy.json` | Grid metadata, below. |
| `topview.webp` | The twin from straight above with the ceiling cut away, transparent outside the room (`plypipe.topview`, CPU, a few seconds). Laid out like `occupancy.png`: row = +Z, column = +X. The viewer's minimap. `python -m plypipe.topview <folder> --write-scene` makes it for a twin that has none. |
| `report.json` | Per-stage stats (frames kept, registered ratio, reprojection error, floor tilt, scale, splat counts, timings). |

`scene.json`
```jsonc
{
  "version": 1,
  "units": "meters",
  "up": [0, 1, 0],
  "handedness": "right",
  "floorY": 0.0,
  "scale": 0.349,                    // SfM units -> metres
  "scaleSource": "camera-height-prior",   // or "none" (--height-cam 0: SfM units kept)
  "cameraHeight": 1.45,
  "transform": [[...4x4 row-major...]],    // COLMAP world -> scene frame (similarity)
  "bounds": { "min": [x, y, z], "max": [x, y, z] },   // robust (2-98 %) + 0.25 m
  "cameraPath": [[x, y, z], ...],    // capture trajectory, <= ~200 points, in capture order
  "spawn": { "position": [x, 1.6, z], "lookAt": [x, 1.6, z] },  // first camera, eye height
  "occupancy": { "image": "occupancy.png", "meta": "occupancy.json" },
  "splat": { "file": "scene.spz", "lite": "scene_lite.spz", "format": "spz" },
  "topview": { "version": 1, "image": "topview.webp", "origin": [x, z], "metersPerPixel": 0.02,
               "width": 1010, "height": 441, "cutY": 1.8 }   // pixel (c, r) covers x = origin[0] + (c + .5) * metersPerPixel, z likewise
}
```

`occupancy.json`
```jsonc
{
  "version": 1, "image": "occupancy.png",
  "cellSize": 0.1,                   // metres
  "origin": [x0, z0],                // scene x, z of the corner of cell (row 0, col 0)
  "width": 71, "height": 51,         // columns (+x), rows (+z)
  "colAxis": "+x", "rowAxis": "+z",
  "values": { "free": 255, "blocked": 0, "unknown": 127 },
  "obstacleBand": [0.2, 1.8], "floorBand": [-0.08, 0.08],
  "stats": { "free": 1976, "blocked": 333, "unknown": 1312 }
}
```
Lookup for a position `(x, z)`: `col = floor((x - origin[0]) / cellSize)`, `row = floor((z - origin[1]) / cellSize)`. Treat out-of-grid as blocked. A sensible R1 walker collides a 0.25 m circle against `blocked` cells and lets `unknown` through (sparse floors leave gaps). R2 replaces this with a recast navmesh.

## Run it

Before any paid GPU run (Modal, RunPod), go through the checklist in [docs/MISTAKES.md](../docs/MISTAKES.md).
What has changed in the pipeline, and how each change was measured, is in [docs/IMPROVEMENT.md](../docs/IMPROVEMENT.md).

### Locally (CPU: frames → occupancy, and the tests)
```bash
cd pipeline
python3 -m venv .venv && .venv/bin/pip install -e '.[test]' 'pycolmap==4.2.1'
.venv/bin/python -m pytest -q                      # 20 tests; includes a real CPU SfM run on a rendered room
.venv/bin/python -m plypipe run walk.mp4 --out work/test --until occupancy --features sift --device cpu
```
CPU SfM is fine for ~50 frames of tests and far too slow for a real capture.

### Kaggle (free T4)
`kaggle/ply_pipeline.ipynb`: accelerator **GPU T4** (P100 is unsupported by torch cu128), internet on, add the video plus this `pipeline/` folder as a Dataset named `ply-pipeline`. The first run builds the CUDA wheels (~15 min) and saves them in `/kaggle/working/wheels`; publish that folder as a Dataset `ply-wheels` and later runs skip the build. Outputs land in `/kaggle/working/output` and `output.zip`.

### RunPod (or any NVIDIA box with Docker)
```bash
docker build -t ply-pipeline --build-arg TORCH_CUDA_ARCH_LIST="8.9" pipeline/   # 8.9 = 4090 / L40S, 8.0 = A100
docker run --gpus all --rm -v $PWD/work:/work ply-pipeline run /work/walk.mp4 --out /work/lab2 --preset high
```
On a pod without Docker-in-Docker (most RunPod templates), run `scripts/install_gpu.sh` in a CUDA 12.8 devel image instead; it is the same script the Dockerfile uses.

## Pinned stack
| Component | Version |
|---|---|
| COLMAP | 4.2.1 via `pycolmap-cuda12==4.2.1` (prebuilt; global SfM, ALIKED/LightGlue ONNX) |
| gsplat | `nerfstudio-project/gsplat@512d366b67073d77ca099ede742683c165dfc23b` (main, 2026-09-19, v1.6.0: MCMC + PPISP) |
| PPISP / fused-ssim / fused-bilagrid | `nv-tlabs/ppisp@v1.2.1`, `fused-ssim@a7c48d6`, `fused-bilagrid@49f0ef0` (pins from gsplat's examples) |
| torch | 2.11.0+cu128, torchvision 0.26.0 (sm_75 T4 … sm_90) |
| splat-transform | `@playcanvas/splat-transform@3.10.1` |
| Python | 3.12 (3.10–3.14 work) |

## Verified vs not yet
- **Verified here (CPU):** frames (real ffmpeg), alignment math (30 random similarity frames, 0.05–30 scale), occupancy, the CLI's resume/force logic, splat-transform export (real tool: crop, SPZ round-trip, axis convention), and a **real pycolmap SfM run** on 48 ray-traced frames of a synthetic room → 48/48 registered, 0.24 px, focal 700.0 px (true 700), floor tilt 0.01°, scale exact.
- **Needs a GPU, not run yet:** ALIKED/LightGlue on CUDA, `install_gpu.sh` (CUDA builds of gsplat / ppisp / fused-ssim), `train`, the Dockerfile and the Kaggle notebook end to end. First real run: Kaggle, `--preset quick`.
