# Demonstration video: detailed shooting script

Round 1 asks for **3 to 5 minutes** covering scope, capture process and the 3D web walkthrough.
Target **4:00**. 1080p, 30 fps, cursor visible, one browser window, notifications off.

This version is written against the state of the system on **1 November 2026, 21:45 IST**, and against the
screenshots and captures you sent during the session. Every number below was read from the live system, not
remembered. Where something is not finished, the script says so on camera rather than hiding it.

---

## 0. Ground truth, right now

| Twin | Job id | Pipeline | State |
|---|---|---|---|
| Classroom | `hq-classroom-1` | one version behind current | done, 1.35 M splats, walkable |
| Chapel | `hq-chapel-1` | older pipeline | done, 680k splats, walkable |
| Chapel second pass | `c09a4c82-…` | **current** | running, SfM ~1400/1703 |
| Kitchen (Nov 1) | `989e2ca0-…` | current | done, **fragmentary**: 109 of 407 cameras |
| AR scan (`scan-01.tar`) | `4660d1e4-…` | current | stopped, waits for a machine |
| LiDAR office | `lidar-office-1` | separate path | done, under development |
| Classroom dense mesh | `mesh-classroom-1` | — | failed, no mesh |

Each twin now carries a chip on the page saying which pipeline produced it, and the library marks the
legacy runs. Do not speak against those chips; they are the honest read and they protect you from the
obvious question, "is this your best work".

---

## 1. Review kit, without storing media

You do not have the disk for clips, and neither does this machine. Everything below is under a megabyte.

**Contact sheet per capture** (the one I reviewed the kitchen from):

```bash
source cloud/.secrets
curl -sL -H "Authorization: Bearer $HF_TOKEN" \
  -o /tmp/frames.jpg \
  "https://huggingface.co/datasets/Ricco-Marcelino/ply-data/resolve/main/scenes/<job-id>/live-frames.jpg"
```

It is a grid of every kept frame, so it shows exactly what the operator filmed: coverage, blur, blown
highlights. 754 KB for 407 frames.

**Camera islands** (`live-sparse.json`, about 19 KB): lists every frame with its time, whether it registered,
and its solved centre. That is how we know the kitchen solve kept three islands and lost the rest; run the
same check on any job before you decide whether to show it.

**Your own screenshots** are b-roll. The ones worth reusing as stills:

- walkthrough with clustered place labels (`Chair ×6`, `Table ×6`, distances) — the "searchable" shot;
- the route card with turn-by-turn and the orange path on the floor — the "navigation" shot;
- the library grid with a running job and ready twins — the "team" shot;
- the editor with a twin framed at eye height — the "edit it" shot;
- the phone header after the fix, three controls — the "it works on a phone" shot;
- the loading gate with just a title and one button — the "no install" shot.

Do **not** put the kitchen capture in the video as a hero shot. It is the honest answer to "what happens when
the walk is bad", which is worth ten seconds and nothing more.

---

## 2. The capture itself, in one paragraph you can read on camera

> "This is a kitchen, filmed in one 142 second walk on a phone. The pipeline kept 407 frames from 1,709 and
> solved 109 of them, in three separate stretches: the first eleven seconds through to thirty-three, eight
> seconds around minute one, and five seconds near the end. The other two thirds never registered, because the
> operator panned quickly past a window and a tube light and the frames are blurred and blown out. That is not
> a bug to hide: it is exactly what the SOP exists to prevent, and it is why the app coaches the walk."

That paragraph is the credibility of the whole video. It shows you know which frames are missing and why.

---

## 3. Shot list

Ten shots, each with the picture, the action, and the narration. Times are cumulative.

### 0:00 – 0:14 — The problem (landing film, no scrolling)

Picture: the landing film hero, point cloud hanging in black. Cursor still.

Say: *"Indoor maps are still made by hand. Outdoors was solved by driving every street; inside, nobody drives,
so a floor plan costs a survey and a week."*

Caption: none. Let the picture sit.

### 0:14 – 0:30 — The promise (scroll one notch)

Picture: the cloud resolves into the room, then the closing card. Stop on the card.

Say: *"Ply turns one phone walkthrough into a photoreal twin you can walk, search and navigate, from any
browser, with no install and no plugin."*

Caption: `one walk · one GPU · no app`

### 0:30 – 1:05 — Capture is a procedure (phone recording + capture app)

Picture: your phone recording of the walk, cut against `capture.html` on desktop: the start screen, then the
review screen with coverage.

Say: *"Capture is not luck. Main lens, 4K, shutter fixed at a hundredth of a second so the lights do not band,
exposure and white balance locked, half a metre per second, two loops and a cross pass. The web app checks
sharpness, exposure, motion and coverage while there is still time to redo a pass."*

Caption: `4K · 1/100 s · ISO & WB locked · 0.5 m/s`

### 1:05 – 1:25 — What went wrong on the kitchen (contact sheet on screen)

Picture: the contact sheet from §1, full width. Point at the blown-out rows.

Say: *"Here is what a rushed walk looks like to the solver. Four hundred frames offered, one hundred and nine
placed, in three islands. The rest are blurred or blown out, so there is no camera to give them. Training can
only use frames that have a pose, which is why this twin has holes."*

Caption: `407 frames offered · 109 placed · 3 islands`

This shot buys you enormous credibility. Do not skip it.

### 1:25 – 1:55 — The classroom, and the numbers (report page, then the abstract's table)

Picture: `capture.html?job=hq-classroom-1` figures, then the abstract's stage table.

Say: *"The classroom walk is what the procedure produces. Eight minutes of video, five clips, 2,167 frames
extracted, 542 kept at one point one per second, zero point eight two pixels of reprojection error, and a floor
that lands within two point five millimetres of real metres. One hundred and sixty four minutes on a single
GPU, unattended."*

Caption: `542 poses · 0.82 px · floor ±2.5 mm · 164 min`

### 1:55 – 2:35 — Walkthrough (desktop, classroom twin)

Picture: `walk.html` on the classroom. Click to enter, walk forward, look around, `F` to free, back to walk,
open **Views** → plan, then dollhouse, then back.

Say: *"This is the twin in a browser. WASD to walk, mouse to look, F for a free camera when you want to fly.
Two views for orientation: the occupancy plan, and the dollhouse cut away from above."*

Caption: `WASD · no install · 1.35 M splats`

### 2:35 – 3:05 — Search and route (the shot from your screenshot)

Picture: press `⌘K`, type "blackboard", pick it. The route card appears with distance, time, turns. The orange
path bends around desks on the floor. Walk it; the minimap tracks.

Say: *"Search any of the hundred and ninety four places. The route is planned on the occupancy grid, so it
bends around the desks instead of through them, and it stays on the floor. Six point two metres, five seconds,
three turns."*

Caption: `194 places · route on the occupancy grid`

### 3:05 – 3:20 — The editor

Picture: click **Edit**. SuperSplat opens with the twin already framed at eye height inside the room.

Say: *"The same twin opens in an editor, seeded with the capture's own camera so you start inside the room
rather than outside it. Drag a gaussian out of the way and export a new SPZ."*

Caption: `SuperSplat · MIT · runs on WebGL2`

### 3:20 – 3:40 — Library, and the team story

Picture: `library.html` with the team code entered. Show a running job with a progress bar, a ready twin, and
the provenance chips.

Say: *"Every capture the team has run lives in one library, not in one browser. A job that is still on the GPU
shows here with its stage and progress, and a finished one opens in the viewer with one click."*

Caption: `phone captures appear on the laptop`

### 3:40 – 4:00 — Close: LiDAR, model, and the honest line

Picture: LiDAR viewer orbiting the office cloud, cut to the semantic model with one object selected, cut to
the closing card.

Say: *"Where a scanner exists, the same contract takes LiDAR: a hundred and eighty four million points in, six
million streamed to the browser, six walls and a four point one three metre ceiling measured. Ply: walk it
once, navigate it forever."*

Caption: `184.6 M points → 6.15 M in the browser`

---

## 4. What you can and cannot show right now

**Show freely:** walkthrough, search and route, plan and dollhouse, editor with the seeded camera, library with
the team code, LiDAR viewer, semantic model with the mesh toggle untouched.

**Do not show, or show with the caveat spoken:**

- **AR.** It is disabled in the UI behind a flag because it has not been signed off on a device. If a judge
  asks, say exactly that: built, not yet tested, switched off rather than shipped half-tested.
- **The kitchen as a hero shot.** Show it only in the contact-sheet moment at 1:05.
- **The classroom dense mesh.** The Poisson stage failed (`dense/images missing`); the mesh toggle in the model
  viewer has nothing behind it.
- **The ceiling estimate from the automatic stage.** It returns 7.19 m for a 3.1 m room. The semantic model
  uses the corrected envelope; do not quote the raw estimate.
- **An E57 download.** The 3.3 GB scanner file is not hosted. What is hosted is the 6.15 M point decimated PLY.
- **A Kaggle run.** This machine has no Kaggle token, so nothing in the video should claim a Kaggle path.

**Bugs fixed this session, worth mentioning if a judge asks about polish:** the loader drew a band across half
the screen (radial gradient sized to the closest side); the editor once served a black page (its module script
was missing); the closing card's links were clickable while invisible; every safe-area inset now has a plain
fallback, which is what had moved the minimap, the skip button and the joystick.

---

## 5. Numbers you may be asked for

| Figure | Value | Where it comes from |
|---|---|---|
| Classroom kept frames | 542 of 2,167 | `report.json` |
| Classroom reprojection | 0.82 px | `report.json` |
| Classroom splats | 1,352,799 (from a 4 M cap) | `report.json` |
| Classroom SPZ / lite | 37.1 MB / 12.9 MB | bucket |
| Classroom places | 194 | `places.json` |
| Classroom GPU time | 164 min, L40S | Modal |
| Chapel frames | 275, 271 registered, 0.78 px | `report.json` |
| Chapel splats | 679,517 | `report.json` |
| Chapel SPZ / lite | 20.7 MB / 11.0 MB | bucket |
| Kitchen frames | 407 kept, 109 registered | `report.json` |
| Kitchen splats | 556,207 → 409,362 after the opacity/scale filter | `report.json` |
| Kitchen SPZ / lite | 7.25 MB / 4.76 MB | bucket |
| LiDAR | 184.6 M E57 → 6.15 M PLY (89 MB), 6 walls, 4.13 m ceiling, 42 volumes | `shell.json` |
| aHolo reference | 973 keyframes at 1024 px, 692,267 splats, 14.5 MB, 32 min | their published sample |

Planned settings come from `plan.json`, which the pipeline writes next to every result: kept frames per second
from the motion log, resolution from the source pixels, gaussian cap from surface area and view count, steps
from views at about fifty per view, and export thresholds probed against the model's own values. If a judge
asks "why these numbers", the answer is a file, not an opinion.

---

## 6. Licences, and how to answer the licence question

Ply itself is proprietary. What it stands on:

| Component | Licence | Used for |
|---|---|---|
| three.js, Spark, React, Vite, marked, qrcode | MIT | viewer, UI, tooling |
| SuperSplat v2.32.5 | MIT (PlayCanvas) | the editor at `/editor/` |
| gsplat | Apache-2.0 | the trainer |
| COLMAP / pycolmap | BSD-3-Clause | structure from motion |
| PyTorch, torchvision, numpy | BSD-3-Clause | runtime |
| OpenCV headless | Apache-2.0 | image work |
| Hugging Face transformers | Apache-2.0 | SAM 3 interface |
| SAM 3 weights (facebook/sam3) | **SAM License** (Meta) | labelling only, downloaded at image build, not redistributed |
| `@playcanvas/splat-transform` | MIT | writing SPZ |
| Cloudflare, Modal, Backblaze, Hugging Face, Kaggle | services, own terms | hosting, GPU, storage |

Say this if asked: *"The editor is SuperSplat, MIT, built from tag v2.32.5 because v3 dropped the WebGL2
fallback. The only changes are a newer splat-transform so it reads SPZ v4, and a camera parameter we added so
it opens inside the room. SAM 3 weights are downloaded at build time and never redistributed; the SAM licence
permits commercial use but must travel with any redistribution, which is why they are not in the repo."*

Full text: `NOTICE` at the repo root, and `app/public/editor/LICENSE.supersplat.txt`.

---

## 7. Recording checklist

1. One window, tabs preloaded in this order: landing, classroom walk, capture report, library, editor, LiDAR,
   model.
2. Load the classroom twin once before recording so the splats are warm.
3. Hide the bookmarks bar, silence notifications, freeze the clock if you can.
4. Record picture at 1080p30; record voice separately and lay it underneath. It hides cursor jitter.
5. Keep the cursor still while talking; move it only when you click.
6. Burn the captions from §3 lower-left, one line each, timed to the shot.
7. Export H.264, upload unlisted, put the link in `Ply_Additional_Links_and_Prototypes.pdf` before the folder
   goes out.

Runtime check: the ten shots total 4:00 at the spoken lengths above. If you run long, cut the classroom
numbers shot to five seconds and the editor shot to ten; never cut the kitchen contact sheet.
