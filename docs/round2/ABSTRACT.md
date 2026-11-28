# Ply: one phone walkthrough, one navigable twin

Technical abstract, Techfest IIT Bombay "Namma Space", Round 2: points of interest, search and navigation.
Team Ply, 14 November 2026. Every number below comes from the pipeline's `report.json` for the twin it names.

## What Ply is and what this round adds

Ply turns one handheld video of a space into a metric 3D Gaussian splat you can walk in a browser, with labelled
places, a search over them, and a walking route that goes around the furniture instead of through it. There is no
scanner and no floor plan behind it: poses, gravity, floor, scale, walls and walkable floor all come from the video.

Round 1 delivered the capture, the twin and the walkthrough. Round 2 adds the features this brief asks for, and each
is marked below as shipped, partial or not built.

The twin shown for this round is `chapel-q1b-1`, a 1,236-frame capture of an 18.5 m chapel, trained for 15,000 steps
with one affine colour map per photo. `chapel-high-2`, the same solve at 60,000 steps with no appearance model, is the
fallback. The laser path is `lidar-office-8b`, an office scanned as 184.6 million points.

## Capture to twin, stage by stage

| Stage | What it does | Measured, `chapel-q1b-1` |
|---|---|---|
| Frames | keeps the sharpest frame every 0.4 s, at a rate planned from how fast the operator walked | 1,236 frames from eight handheld clips |
| SfM | COLMAP features, sequential matching with loop detection, incremental mapper | 1,223 registered (98.9%), 0.582 px mean reprojection, 431,447 points, 3,530 s |
| Align | gravity from the phone, floor as the largest horizontal surface, walls to the axes, metric scale from the 1.45 m camera height | floor tilt 0.0°, yaw 76.5°, scale 1.5827, floor residual 70 mm median |
| Occupancy | sparse shell merged with a depth grid rendered from the trained splat, at 10 cm cells | 206 × 87 cells: 5,835 free, 5,505 blocked, 6,582 unknown |
| Label | SAM 3 over 112 frames with 25 indoor prompts, lifted to 3D through the solve's 2D-3D matches, merged into objects | 3,993 detections into 281 places, 16 classes, 157 s |
| Train | gsplat with MCMC densification, antialiasing, one affine exposure map per photo, 2M splat cap | 15,000 steps, 1,598 s, held-out 25.52 dB cc-PSNR on 40 views |
| Export | opacity and floater filters, room crop, SPZ, plus a phone file fine-tuned at its own size | 1,229,190 splats kept (61.5%), 38.59 MB, phone file 800k at 20.35 MB, 403 s |

The phone file is trained at its own size: the export cuts the twin to 800k splats, trains that model for 3,000 more
steps and ships it only if the viewer scores it better. On the chapel that is +0.80 dB over the cut-down file, for
210 s of GPU inside the export.

## The three Round 2 features

**Points of interest.** The labelling stage writes `places.json` next to the scene: 281 places on the chapel, each
with an id, a name, a category, a 3D point, a detection class, a confidence and a half-size box. Nine classes are
pinned by rule (doors, stairs, boards, the altar, stained glass); the rest are searchable. A person can accept or
dismiss each detection in the design lab's review panel; the walkthrough draws the pinned ones from that file, merging
same-kind neighbours within 2.5 m into one badge ("Bench ×6") so 281 places do not become 281 floating labels.

Free-form tagging, where a visitor clicks the twin and names a spot, is not shipped: places are read from
`places.json`, the walkthrough cannot add, rename or move one, and no tag is written back to the scene.

**Search.** The palette (`Ctrl`/`Cmd`+`K`) fetches that catalogue and filters it on name, category and detection class
as you type, pinned places first and then by distance, with each result's distance in metres and a confidence for
detections. This is a filter over an array of at most 400 entries, not an index: no inverted index, no synonyms, no
fuzzy match, no per-floor namespace, because one twin is one captured space.

**Obstacle-aware routing.** The viewer plans on the exported occupancy grid with A\*: 8-connected cells, no corner
cutting, a cost that prefers 60 cm off the walls, and clearance from a two-pass chamfer transform. Five clearance
tiers are tried widest first, from 0.23 m beyond the walker's 0.22 m radius down to cell centres, so a route is
refused only when the walker genuinely does not fit. Routes are string-pulled to straight runs and read out as turns
with distances; "Take me there" walks the route at 1.25 m/s. Unknown cells count as blocked, which is why the merged
depth grid matters: on `chapel-high-2` it makes the pews solid and keeps the aisles and side walkways clear.

Evidence for the routing is on `chapel-high-2`: 16 headings held into walls at 9.7 m/s for 5 s each never put the
walker more than 1 cm inside its radius, and routes from the spawn point to 20 places along the aisles were all found
and followed to the end without stopping. That test also found a defect: on a 5 cm grid routes passed 5 cm from a wall,
inside the walker's width, which the clearance tiers and `app/test/walk.test.mts` now cover.

Not built: routing across floors or across twins; routes are 2D on one grid.

## What was measured

| | `chapel-q1b-1` | `chapel-high-2` | `lidar-office-8b` |
|---|---|---|---|
| Input | 8 clips, 1,236 frames | same solve | 184.6 M point E57, no photos |
| Registered, reprojection | 1,223 (98.9%), 0.582 px | 1,223 (98.9%), 0.582 px | not applicable |
| Training | 15,000 steps, affine exposure | 60,000 steps, no appearance model | none, splats from the scan |
| Splats, twin on the wire | 1,229,190; 38.59 MB, phone 20.35 MB | 1,058,149; 39.04 MB, phone 21.91 MB | 1,921,505; 21.8 MB, phone 8.6 MB |
| Places | 281 | 281 | 0 |
| Occupancy | 10 cm cells, 5,835 free | 10 cm cells, 7,465 free | 5 cm cells, 9,750 free |
| Viewer PSNR, 12 views | 23.81 dB | 26.11 dB | not measured |

Levers tested: an opacity regulariser of 0.005 is the best result on the small scene at 15k and 30k steps, while
0.002 loses SSIM at 30k and was dropped; the phone fine-tune buys 0.80 dB on the chapel for about 0.11 dollars a job;
sparse depth loss and pose refinement buy nothing; a 16 × 16 × 8 bilateral grid loses 0.50 dB and does not ship.
Against a laser reference on two MuSHRoom rooms the shipped scale path is 5.16% and 3.12% long; MoGe-2 depth reaches
1.4% on honka and falls back on coffee_room (6.9% spread against a 5% rule).

The viewer scores `chapel-q1b-1` at 23.81 dB: 2.30 dB under `chapel-high-2` and 1.4 dB under the two 15,000-step
baselines it was trained against (25.21 dB mean). That is the price of one colour map per photo, which the owner chose
after a blind look on `ab.html` because the twin then carries one exposure (brightness 1.028 of the photos) rather than
each clip's own. It is the twin shown for this round; `chapel-high-2` is the metric fallback.

## What is not shipped, and why

The TSDF mesh reaches its triangle counts on the chapel (298,319 and 19,976) but is a soup of crossing faces at 5.7
triangles per vertex, so export refuses to write it. The dense wall layout is worse than the sparse shell there: both
long sides come out in fragments and neither end wall is found. The 4K resolution lever has not run, the depth prior
is opt-in, and AR navigation is built behind a flag and switched off until it is signed off on a device.

## Open-source components and licences

Ply itself is proprietary. Versions, pins and evidence for each licence are in `ATTRIBUTION.md`.

BSD-3-Clause: COLMAP, pycolmap-cuda12 4.2.1, PyTorch 2.11.0, torchvision 0.26.0, numpy, scipy. Apache-2.0: gsplat 1.6.0
at pin `512d366`, ppisp 1.2.1, nerfview, transformers 5.19.0, accelerate, OpenCV headless, TypeScript 7.0.2. MIT:
fused-ssim at `a7c48d6`, fused-bilagrid at `49f0ef0`, MoGe-2, utils3d, three.js 0.186.1, `@sparkjsdev/spark` 2.3.1,
React and react-dom 19.3.0, marked 18.1.1, qrcode 1.5.4, `@playcanvas/splat-transform` 3.10.1, SuperSplat v2.32.5
(vendored at `/editor/` with its licence text), Vite 8.3.4, Tailwind 4.3.3, Node.js 22. HPND: Pillow.
LGPL-2.1-or-later: FFmpeg, build variant unverified. Unverified: trimesh, in the worker image for the mesh stage.
SAM 3 weights (`facebook/sam3`): Meta's SAM License, downloaded at build for labelling and not redistributed.
MuSHRoom rooms: CC-BY-4.0, Zenodo 10230733 and 10222321. Cloudflare Workers, D1 and R2, Modal, Backblaze B2,
Hugging Face Hub and Kaggle are services under their own terms.

Page budget: 2 pages here, 3 in `SOP.md`, the five allowed together. Every link is in `MASTER-PDF.md`.
