# test/

Sample data for the pipeline, and the shortest path from a folder of clips to a walkable twin.

## What is in here

| Path | What |
|---|---|
| `sample-frames/` | four frames from the classroom capture this repository's screenshots come from (a 542-frame solve, sampled here so the folder stays small) |
| `run.sh` | runs the pipeline on any folder of clips and prints the viewer URL for the result |

The full 8-minute capture, the solved scene and the trained splat are too large for a repository. They are in the
Round 1 Google Drive folder (link open to anyone) next to the abstracts, in `capture/` and `scene/`.

## Run it on your own capture

Requirements: Python 3.12, ffmpeg, a CUDA GPU (torch 2.11 + gsplat) for the training stage, Node 20 for export.
`pipeline/README.md` has the full setup; the short version:

```bash
cd pipeline
python -m venv .venv && . .venv/bin/activate
pip install -e .            # plus the CUDA wheels listed in README.md
cd ..
./test/run.sh /path/to/folder-of-clips work/my-room
```

`run.sh` is a two-line wrapper around:

```bash
python -m plypipe run <folder> --out <dir> --preset standard
```

A folder of `.mp4` files of one space, in name order, is one capture. If the clips came from the capture app, the
folder also holds `capture.json`, which the pipeline uses for scale and for the stretches the operator marked to skip.

## What it writes

| File | Read by |
|---|---|
| `scene.json` | the viewer: bounds, floor, spawn, camera path, splat references |
| `shell.json` | measurement overlay, AR alignment, semantic model |
| `places.json` | search: every detected place with class, confidence and box |
| `occupancy.png` / `.json` | the router and the minimap |
| `report.json` | stage timings, frame statistics, splat counts, the settings plan |
| `scene.spz`, `scene_lite.spz` | the twin, full and phone-sized |
| `plan.json` | what settings this capture was given, and why |

## What to expect

A 10 × 6 m room from 8 minutes of video takes about 2.5 hours of GPU on one L40S and lands at roughly 1.3 M Gaussians.
CPU-only stages (`--until occupancy`) run in minutes on a laptop and are enough to check that the capture solves:
`python -m plypipe run <folder> --out <dir> --until occupancy`.

## Checking a run

Read `report.json` rather than the UI. Useful thresholds, and the reasons behind them, are in
`docs/PROCESSING.md`: kept frames per second at or above 1.8, `train.antialiased` true, and
`export.gaussiansOut / gaussiansIn` at or above 0.85.
