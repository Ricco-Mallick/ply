# Ply capture SOP: one phone, one walk, one twin

Round 2 edition, 14 November 2026, in the order the organisers asked for: device, camera settings, lighting, movement,
angles, distance, capture sequence, corner coverage, floor plan, capture examples with raw-video references, and the
multi-room and multi-floor sequence.

Page budget: this SOP is 3 pages of A4 and the abstract is 2, the five allowed together. Both are rendered from the
Markdown by `docs/round2/render.py`, which prints the page count quoted here.

Reference device: Samsung Galaxy S24 FE, 50 MP main camera, 24 mm equivalent, OIS, in Pro Video mode. The chapel used a
DJI Osmo Action 5 Pro for the same reasons: one fixed lens, manual exposure, no electronic stabilisation.

## 1. Quick card

| Item | Do | Don't |
|---|---|---|
| Lens | main 1× camera | ultra-wide 0.5×, zoom |
| Video | 4K 30 fps, HEVC, HDR off | 8K, slow motion, "Super steady" |
| Exposure | Pro Video, 1/100 s (India, 50 Hz), ISO and WB fixed | auto exposure, auto WB |
| Hold | upright, chest height about 1.45 m, tilted 10 to 15 degrees down | overhead, waist-level swings |
| Move | 0.5 m/s, one step per second, turn while walking | pivoting on the spot, fast pans |
| Path | two loops in opposite directions, one cross pass, close on the start | one pass along one wall |

A small room takes 1 to 2 minutes, a four-room floor 5 to 8 minutes. The pipeline keeps the sharpest frame every
0.4 s, so walking slowly costs nothing.

## 2. Device and camera settings

| Setting | Value | Why |
|---|---|---|
| Resolution | 4K, 30 fps, HEVC, HDR10+ off | 4K downsampled to 1600 px beats native 1080p; per-frame tone mapping does not |
| Shutter | 1/100 s indoors, 1/200 in bright rooms, 1/120 in 60 Hz countries | 50 Hz mains flicker at 100 Hz; 1/60 s or slower is where blur starts |
| ISO | fixed, lowest that exposes at 1/100, usually 200 to 800 | exposure must not move mid-take |
| White balance | fixed: 4000 K for warm LED, 5000 to 5500 K near daylight | auto WB shifts as you turn |
| Focus | manual at 1.5 m, or AE/AF lock | hunting autofocus changes the focal length the solver assumes |
| Stabilisation | OIS in hardware, electronic off | electronic stabilisation warps each frame differently |

In a room too dim for 1/100 s at ISO 1600, use 1080p 60 fps: more frames, less noise, less detail. On a Pixel use
"Lock" in the video exposure menu; on an iPhone use Blackmagic Camera or ProCamera for shutter, ISO and WB, with
Action Mode off.

**Lighting.** Every light on. Blinds half closed against direct sun, and never point the lens at a window with the sun in it: the
windows clip and the room goes dark. Turn monitors off, and keep one colour temperature per room. Then record 5
seconds and play it back paused on a turn: a smeared frame means walk slower or add light, bands mean the shutter is
wrong for the mains frequency.

## 3. Movement, angles and distance

- 0.5 m/s, about one step per second, arms in, heel to toe. The phone should move as if on a rail.
- Turn while walking, never on the spot, at 30 degrees per second at most. A pure rotation gives no parallax, so those
  frames cannot be placed and the mapper drops them.
- Upright at chest height, tilted 10 to 15 degrees down in loop 1, so the floor fills the lower third and the
  floor-wall line stays in frame. Portrait or landscape both work; do not rotate mid-take.
- Loop 1 runs 0.8 to 1 m from the walls, camera across the room; loop 2 runs 1.5 m in, the other way round, facing
  walls and furniture. Distance means distance to the surface being photographed, not distance walked. Keep something
  textured in every frame: the floor-wall line, a door frame, a table edge. Blank wall or ceiling is refused.

```
 ┌──────────────────────────┐  ●  start and end, marker in view
 │ →loop 1 (0.8-1 m in)     │  loop 1: camera across the room
 │  ┌───────────────┐   ↑   │  loop 2: reversed, 1.5 m in,
 │  │ ←loop 2       │   │   │          camera at the walls
 │  └───────────────┘   │   │  pass 3: once through the middle
 │ ●▣  ──── pass 3 ──────→  │
 └──────────────────────────┘
```

## 4. Capture sequence, and corner coverage

1. Start at the entrance with the A4 Ply marker in frame for 3 seconds, holding still.
2. Loop 1 inward, looking across the room.
3. Loop 2 outward, reversed, looking at the walls and furniture.
4. Cross pass once through the middle.
5. Corners: every inside corner gets a pass within 1.5 m with the corner in frame for 2 seconds. A corner seen from
   one side comes out as a smear, and it is the first place a walker clips a wall.
6. Close on the start view for 3 seconds, facing the marker. That is what lets the solver remove the drift, and the
   pair of loops is what gives every surface views from several angles, which is where the occupancy grid's depth
   comes from.

## 5. Doorways, multi-room and multi-floor

A doorway is the joint between two rooms, so film both sides: approach slowly, stop for 1 second in the opening with
the next room in view, walk through, then turn back and show the doorway from the new side for 2 seconds.

For a floor of several rooms, one take, in this order.

1. Entrance room: the six steps above.
2. Walk to the first doorway, stop in it, cross, show it from the other side.
3. Next room: loop 1, loop 2, cross pass, then back through the same doorway showing the previous room for 2 seconds.
4. Repeat room by room, always leaving through the doorway you came in by, so every joint is filmed from both sides.
5. Corridor: down the middle facing forward, then back facing forward, weaving gently for side-wall parallax.

Keep a take under about 8 minutes. A larger floor is split by wing: each take starts and ends at a shared doorway or
junction with 5 to 10 seconds of overlap, and opens with the marker in view. Stairs get one take per floor, walked
slowly with the handrail and step edges in frame. Floors are separate takes and separate twins, and the capture record
has no floor field, so put the floor and wing in the name ("library-floor-2-west"). Linking floors into one model is
not built; it is a Round 3 feature.

## 6. Floor plan: what Ply derives, and how a supplied plan would combine

No floor plan goes in. Ply builds one from the capture. The occupancy stage merges the sparse points the solve placed between
0.2 and 1.8 m above the floor with a depth grid rendered from the trained splat: depth supplies floor and solid
furniture, sparse supplies floor the depth left undecided. On the chapel that is 206 by 87 cells at 10 cm, 5,835 free,
5,505 blocked, 6,582 unknown, exported as `occupancy.png` (free 255, blocked 0, unknown 127) and `occupancy.json`
(cell size, origin, axes, counts). The viewer draws `topview.webp` over it, 0.02 m per pixel cut at 1.8 m, and falls
back to the grid when a scene has no top view. Routes and place pins are drawn on top of whichever is there.

A plan you bring is a tool for the operator today, not an input to the pipeline: use it to check that every room was
walked, and to check scale by measuring a known dimension (a 0.9 m door) against the twin. A door that reads 0.85 m
means the scale is short, and the height prior is the first thing to check.

Reading a supplied plan is the next step and is not shipped. The intended procedure, so it is judged on a description
rather than a guess: photograph the plan flat and square, mark two points a known distance apart (the same two-tap the
viewer already uses), project it onto the twin's floor plane, snap its wall lines to `shell.json`, close regions the
depth never saw, and take room names from its labels. Until then the plan is a checklist and a ruler, and the occupancy
grid is the floor plan the viewer draws.

## 7. Capture examples and raw video

| Capture | Device, size | What happened | Lesson |
|---|---|---|---|
| Chapel, `chapel-q1b-1` | Osmo Action 5 Pro, 8 clips, 1,236 frames | 1,223 registered (98.9%), 0.582 px, 281 places | the reference: slow walk, two loops, doors from both sides |
| Kitchen, `989e2ca0` | phone, 142 s, one clip | 407 kept, 109 registered, three islands | the failure to avoid: a fast pan past a window and a tube light |
| honka, MuSHRoom | iPhone, 335 photos, plus a Faro scan | 334 registered, scale 5.16% long against the laser | phone metres drift; check a known dimension |
| coffee_room, MuSHRoom | iPhone, 391 photos, plus a Faro scan | 363 registered, scale 3.12% long | the same, with 28 photos lost in the solve |

Raw video, so the procedure can be checked against the footage it came from.

| Clip | What it is | Where | Public |
|---|---|---|---|
| `DJI_20260912135353_*` and 7 more | the chapel capture, one file per pass | Modal volume `ply-bench`, `chapel/high/job/clips/` | not yet; to publish under `captures/chapel/` |
| classroom, 5 clips | 8 minutes of 4K, the 542-frame solve | not recorded in this repository | not yet; no link recorded |
| kitchen clip | 142 s, the failed walk | Modal volume, job `989e2ca0` | not yet; published with the failure write-up |
| `smoke-chapel-30s.mp4` | 30 s at 1080p, 15 MB, cut from a chapel clip | Modal volumes, `smoke/` on both accounts | not yet; it is the deploy test, not a demo |
| honka, coffee_room | MuSHRoom iPhone walks with laser scans | Zenodo 10230733 and 10222321 | yes, CC-BY-4.0 |

Only the MuSHRoom archives have a public link today. Publishing the other four under `captures/` is on the
`MASTER-PDF.md` to-do list, which is where the links appear once they work.

## 8. After the take, and hand-off

1. Play the clip at 2×: smeared frames at turns, flicker, people walking, any moment pointed at blank wall or
   ceiling. Fix it now, because a re-shoot costs 2 minutes and a return visit costs a day.
2. Confirm it ends on the start view, and copy the file over USB or Drive unedited: no trimming apps, no
   re-encoding, no stabilisation.
3. Measure the operator's camera height once, floor to lens, and pass it:
   `python -m plypipe run walk.mp4 --out work/lab2 --preset standard --height-cam 1.45`. A 5 cm error is a 3.5%
   scale error. Nothing else is dialled in: frame rate, resolution and gaussian budget are planned from the clips
   (`docs/PROCESSING.md`). Do not pass `--max-frames` to save time; it only throws views away.

## 9. When something goes wrong

| Symptom in `report.json` or the viewer | Cause | Fix |
|---|---|---|
| `registeredRatio` under 0.8, rooms missing | fast turns, blank walls, dark | re-shoot slower with texture in frame, add light |
| Several models, or path discontinuities | pivot on the spot, a door crossed too fast | `--force sfm --mapper incremental`, re-shoot the doorway |
| Walls blotchy, ghost floaters | auto exposure or WB, 50 Hz banding; people; mirrors | lock ISO and WB; empty the space |
| Floor tilted, `floorTiltDeg` over 5 | phone sideways or rotated mid-take | upright, 10 to 15 degrees down in loop 1 |
| Scale wrong, a door is not about 2.1 m | wrong height prior | re-run from `align` with the measured `--height-cam` |
