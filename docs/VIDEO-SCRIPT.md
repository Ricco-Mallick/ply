# Demonstration video: shot list

Round 1 asks for 3 to 5 minutes covering the project scope, the capture process, and the 3D web walkthrough. Target
**4 minutes**. Record at 1080p, 30 fps, cursor visible, notifications off, one browser window, one tab per shot.

Everything below is already live. Nothing needs to be rebuilt for the video.

## Before you hit record

Open these tabs, in this order, and leave them loaded:

1. `https://ricco.codes/ply/` (landing film)
2. `https://ricco.codes/ply/walk.html?scene=https%3A%2F%2Fply-api.ricco-ply.workers.dev%2Fv1%2Fscenes%2Fhq-classroom-1%2Fscene.json` (the twin; let it finish loading once)
3. `https://ricco.codes/ply/capture.html` (capture app)
4. `https://ricco.codes/ply/lidar.html?ply=https%3A%2F%2Fply-api.ricco-ply.workers.dev%2Fv1%2Fscenes%2Flidar-office-1%2Foffice.ply` (LiDAR room)
5. `https://ricco.codes/ply/model.html?scene=https%3A%2F%2Fply-api.ricco-ply.workers.dev%2Fv1%2Fscenes%2Fhq-classroom-1%2Fscene.json&module=/ply/models/classroom2.js` (semantic model)

Also have on the desktop: a phone screen recording of the capture walk, and one of the AR session.

## Shot list

| Time | Picture | Say (or caption) |
|---|---|---|
| 0:00–0:12 | Landing film hero, not scrolling | "Every indoor space has a map problem. Outdoors was solved by driving every street; indoors nobody drives." |
| 0:12–0:25 | Scroll a little: the point cloud morphs into the room; then the closing card, hold on "Now map your own" | "Ply maps a building from one phone walkthrough, and turns the result into something you can walk, search and navigate." |
| 0:25–0:55 | Phone screen recording: the capture walk. Overlay the SOP settings as captions: 4K 30, 1/100 s, ISO and WB locked, 0.5 m/s, two loops and a cross pass | "Capture is a procedure, not luck. Main lens, 4K, shutter fixed at a hundredth of a second so the lights do not band, exposure and white balance locked, a slow half-metre-per-second walk, two loops and a cross pass." |
| 0:55–1:15 | Cut to the desktop on `capture.html`, show the start screen (the iPhone-only LiDAR row is visible), then the review screen with coverage | "The web app coaches the walk: sharpness, exposure, motion and coverage, while there is still time to redo a pass. Eight minutes of video produced the room you are about to see." |
| 1:15–1:40 | Status or report figures on screen: 542 poses, 0.82 px, 2.5 mm floor, 164 min, then a still of `plan.json` or the abstract's stage table | "Camera solving keeps five hundred and forty two frames at zero point eight two pixels of reprojection error. The floor lands within two and a half millimetres, so the twin is in real metres." |
| 1:40–2:20 | Walkthrough: click to lock the pointer, walk forward, look around, open Plan, then Dollhouse, back to Walk | "This is the twin in a browser, no install. WASD to walk, mouse to look, and two views for orientation: the plan and the dollhouse." |
| 2:20–2:50 | Search a place by name, select it, show the route drawn, then walk along it while the minimap follows | "Search any of the hundred and ninety four places. The route is computed on the occupancy grid, so it bends around the desks instead of through them, and it stays on the floor." |
| 2:50–3:15 | Phone screen recording of AR: two-tap alignment, then arrows on the floor, walking | "On Android, the same route is projected onto the real floor through the camera. AR navigation is the bonus round, and it runs now." |
| 3:15–3:40 | LiDAR viewer, orbit the office cloud; then the measured shell numbers on screen (6 walls, 4.13 m ceiling) | "Where a scanner is available, the same contract takes LiDAR: a hundred and eighty four million points in, six million streamed to the browser, six walls and a four point one three metre ceiling measured." |
| 3:40–4:00 | Semantic model in the model viewer, one object selected, then cut to the closing card | "Structure, objects and places come out of the same solve, and the next capture path is the depth sensor on an iPhone Pro, recorded beside the video. Ply: walk it once, navigate it forever." |

## Captions to burn in

Short, lower-left, one line each, timed to the shot above:

- `Capture: 4K · 1/100 s · ISO & WB locked · 0.5 m/s`
- `542 poses · 0.82 px reprojection · floor ±2.5 mm`
- `164 min on one GPU, unattended`
- `194 places · route planned on the occupancy grid`
- `AR navigation · Android Chrome`
- `LiDAR: 184.6 M points → 6.15 M in the browser`

## Do not show

- **The Mesh toggle** in the model viewer. The Poisson stage has not produced a mesh worth showing.
- **`model.html?lidar=…`**. That path still renders black; use `lidar.html` as above.
- **AR on an iPhone.** It is hidden there by design; do not imply it works.
- **Any claim about the ceiling height of the classroom from the automatic estimate.** The semantic model uses the
  corrected envelope; the measurement stage is the one that is wrong.

## Recording

`gnome-screenshot` and OBS both work; OBS with a display capture at 1080p30 is the simplest. Record voice separately if
you can, then lay it under the picture: it hides bumps in the screen recording. Export H.264, upload unlisted to
YouTube or Drive with link sharing on, and paste the link into
`Ply_Additional_Links_and_Prototypes.pdf` before the folder goes out.

If a live demo has to be shorter: cut the pipeline shot and the LiDAR shot, keep capture, walkthrough, search and AR.
That is 2:15 and still covers the brief.
