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

| Stage | Needs | What it does |
|---|---|---|
| `frames` | ffmpeg, OpenCV | Samples `--fps` (2.5) frames/s, oversampled ×4; keeps the sharpest of each window (variance of Laplacian), drops very blurry ones without opening gaps; long side ≤ 1600 px. A folder of photos also works. |
| `sfm` | pycolmap 4.2.1 (GPU wheel) | One shared SIMPLE_RADIAL camera; ALIKED + LightGlue (falls back to SIFT); exhaustive pairs up to 250 frames, sequential + quadratic overlap above; **global SfM** (`--mapper incremental` as plan B); undistort to PINHOLE. Flags camera-path discontinuities (a folded model can still have a low reprojection error). |
| `align` | numpy | Gravity from the cameras' up vectors → floor = lowest plane covering ≥ 50 % of the best horizontal area (beats textured table tops) → walls on X/Z (Manhattan histogram sweep) → metric scale from `--height-cam` (1.45 m). Writes gsplat's `data/` already in the scene frame. |
| `occupancy` | numpy, OpenCV | 10 cm grid: blocked = points 0.2–1.8 m above the floor, free = floor points ∪ the walked path, closed over 30 cm gaps. The walked line is never blocked. |
| `train` | torch 2.11 cu128, gsplat @ `512d366` | gsplat `simple_trainer`, MCMC strategy, PPISP appearance (exposure/vignetting compensation), trained **in the scene frame** (`normalize_world_space=False`), so the PLY needs no transform. |
| `export` | Node ≥ 20 | splat-transform 3.10.1: NaN filter, crop to the room box (+1 m), `scene.spz` (full SH) and `scene_lite.spz` (SH0, for phones). |

Presets (`--preset`): `quick` 7k steps / SH2 / ≤ 0.6 M splats / no PPISP (~25 min on a T4) · `standard` 30k / SH3 / ≤ 1.5 M / PPISP (~1.5–2 h on a T4, ~25 min on a 4090) · `high` 40k / SH3 / ≤ 3 M / PPISP (needs > 16 GB: RunPod).

Useful flags: `--until occupancy` (CPU box: everything but training), `--force sfm` (re-run a stage and all after it), `--focal-35mm 24` (phone lens prior when SfM struggles; the S24 FE main lens is 24 mm-equivalent, about 26 with video stabilisation), `--max-frames 400`, `--features sift`, `--no-ppisp`.

## Output contract (what the web viewer reads)

All coordinates are in the **scene frame**: right-handed, **+Y up, floor plane at y = 0, metres**, walls along X / Z, origin on the floor under the median camera. This is three.js's own convention: load the SPZ with no rotation.

| File | Contents |
|---|---|
| `scene.spz` | Gaussian splats, full SH. Positions stored as-is (splat-transform writes SPZ with `CoordinateSystem.UNSPECIFIED`, i.e. no axis conversion). |
| `scene_lite.spz` | Same splats, SH degree 0 (smaller; use on phones). |
| `scene.json` | Scene metadata, below. |
| `occupancy.png` | 8-bit grey grid: **255 free, 0 blocked, 127 unknown**. Pixel (row r, col c) = cell r along +Z, c along +X; the top-left pixel is the cell at `origin`. |
| `occupancy.json` | Grid metadata, below. |
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
  "splat": { "file": "scene.spz", "lite": "scene_lite.spz", "format": "spz" }
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
