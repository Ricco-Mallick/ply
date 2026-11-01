# Ply capture SOP: one phone, one walk, one twin

**Goal:** a single smartphone video from which the pipeline (`pipeline/`) recovers every camera pose, a gravity-aligned metric floor, and a photoreal Gaussian-splat twin you can walk with WASD and navigate.

**Reference device:** Samsung Galaxy S24 FE (main camera: 50 MP, f/1.8, 24 mm equivalent, OIS). Any recent Android or iPhone works with the equivalent settings listed in §2.

---

## 0. Quick card (print this)

| | Do | Don't |
|---|---|---|
| **Lens** | Main 1× camera | Ultra-wide 0.5× or zoom |
| **Video** | 4K 30 fps | 8K, slow-mo, "Super steady" |
| **Exposure** | Pro Video: shutter **1/100 s** (India, 50 Hz lights) · ISO fixed · WB fixed | Auto exposure, auto WB |
| **Focus** | Manual, about 1.5 m (or AE/AF lock) | Continuous AF hunting |
| **Hold** | Upright, chest height (about 1.45 m), tilted 10–15° down | Overhead, waist-level swings |
| **Move** | Slow walk, about 0.5 m/s (one step per second); turn while walking | Pivoting on the spot, fast pans |
| **Path** | 2 perimeter loops (inward, then outward) + 1 cross pass, end where you started | One pass along one wall |
| **Scene** | All lights on, doors open, room empty, Ply marker at the entrance | People walking, mirrors in close-up |

Duration: a small room takes 1–2 min, a 4-room floor 5–8 min. The pipeline keeps the sharpest frame of every 0.4 s, so a slow walk costs nothing extra.

---

## 1. Before you press record (pre-flight)

- [ ] **Phone:** battery > 50 %, free storage > 5 GB (4K30 HEVC is about 350 MB/min), lens wiped with a microfibre cloth.
- [ ] **Space:** every light on; blinds half-closed against direct sun; doors you will walk through propped open; chairs pushed in; cables off the floor.
- [ ] **People:** ask everyone to leave the area, or to stand still out of the walking path for the whole take. A moving person becomes a ghost ("floater") in the twin.
- [ ] **Marker:** stick the A4 Ply QR marker (printed at 100 %, no scaling) on a wall at the entrance, about 1.2 m high. It gives a known size (A4 = 297 mm) to check scale against, and it is the anchor AR navigation uses later.
- [ ] **Settings** (§2), then a **5-second test clip**: play it back paused at a turn. If the frame is smeared, slow down or raise the light; if it flickers or shows bands, fix the shutter (§2.3).

## 2. Camera settings

### 2.1 Mode and lens
- Samsung Camera → **More → Pro Video**. Stay on the **1×** main lens. The ultra-wide has strong distortion and soft corners; zoom lenses narrow the view and break overlap.
- **Resolution 4K (UHD), 30 fps.** 4K downsampled to 1600 px gives crisper features than native 1080p.
  - Use **1080p 60 fps** instead only in a dim space where 1/100 s at 4K forces ISO above about 1600 (more frames, less noise per frame, less detail).
- Format: HEVC is fine (the pipeline decodes it with ffmpeg). Turn off HDR10+ video: tone-mapping changes frame to frame.

### 2.2 Stabilisation
- Turn **Super steady OFF** (it switches to the ultra-wide and crops).
- Turn **Video stabilisation OFF** for the best geometry: the S24 FE's OIS still works in hardware, while electronic stabilisation warps each frame differently, which breaks the "one camera" model SfM relies on.
- If you cannot walk smoothly without it, ON is better than blur: tell the operator and add `--focal-35mm 26` when processing (the EIS crop makes the lens behave like about 26 mm).

### 2.3 Exposure, white balance, focus
- **Shutter 1/100 s.** India's mains are 50 Hz, so lights flicker at 100 Hz: 1/100 s (or 1/200 s in a bright space) avoids banding and is fast enough to freeze a slow walk.
  - In 60 Hz countries use 1/120 or 1/240.
  - Never go slower than 1/60: motion blur is the most common cause of a soft twin.
- **ISO:** fixed, lowest that gives a correct exposure at 1/100 (typically 200–800 indoors). Exposure must not change during the take; auto exposure makes walls change brightness between frames.
- **White balance:** fixed, about 4000 K for warm LED/office light, 5000–5500 K near daylight windows. Auto WB drifts as you face different walls.
- **Focus:** manual, at about 1.5 m. On phones without manual focus, use AE/AF lock (tap and hold the subject 1.5 m away).

**Other phones:**
- **Pixel:** use "Lock" in the video exposure and focus menu, and turn stabilisation to "Standard/Off".
- **iPhone:** a third-party app (Blackmagic Camera or ProCamera) for shutter, ISO and WB lock; turn Action Mode off; 4K 30.

## 2.5 Hand-off to the pipeline

Nothing to dial in. `plypipe` plans each run from the clips: the kept frame rate follows how fast you walked, the working resolution follows the pixels you shot, and the gaussian budget follows the solved room. Do not pass `--max-frames` to make it quicker, it only throws views away. See [PROCESSING.md](PROCESSING.md) for the rules and the post-run checklist.

## 3. How to walk

```
   ┌──────────────────────────────┐
   │  ←───── loop 1 (look in) ──── │       ● start / end (facing the marker)
   │ │  ┌──────────────────────┐ ↑ │       loop 1: 0.8–1 m from the walls, camera across the room
   │ │  │ ←── loop 2 (look out)│ │ │       loop 2: reversed, 1.5 m in, camera towards walls & furniture
   │ ↓  │                      │ │ │       pass 3: once through the middle
   │ │  └──────────────────────┘ │ │
   │  ──────── pass 3 ─────────→   │
   │ ●▣ marker / door              │
   └──────────────────────────────┘
```

1. **Start** at the entrance with the marker in view for 3 seconds (hold still).
2. **Loop 1 (inward):** walk the perimeter 0.8–1 m from the walls with the camera looking **across the room**, tilted 10–15° down so the floor fills the lower third. This loop builds the floor, the far walls and the layout.
3. **Loop 2 (outward):** walk a smaller loop in the opposite direction, about 1.5 m from the walls, with the camera facing the **walls and furniture**. It adds the detail behind tables, under counters and around sofas.
4. **Cross pass:** walk once through the middle of the room.
5. **Close the loop:** finish where you started, facing the marker again for 3 seconds. Returning to the start view lets SfM remove drift.

**Rules while walking**
- About **0.5 m/s**, roughly one step per second, with the phone moving like it is on a rail. Keep your elbows in and walk heel to toe.
- **Turn while walking, never pivot on the spot.** A pure rotation gives no depth (parallax), so those frames cannot be placed, and global SfM can fold the path there. Turn at most about 30°/s.
- Keep **something textured** in every frame: the floor–wall line, a door frame, furniture. Never fill the frame with a blank wall, window or ceiling.
- Hold the phone **upright** (portrait or landscape; just don't rotate it mid-take). The pipeline uses "phone up" as its first guess for gravity.

### 3.1 Doorways and multi-room floors
- **Doorways:** approach slowly, stop for 1 second in the door with the next room in view, walk through, then turn back and show the doorway frame from the new side for 2 seconds. The two sides of a door are what stitch rooms together.
- **Corridors:** walk down the middle facing forward, then walk back facing forward again (both directions), weaving gently left and right so the side walls get some parallax.
- **One take per floor** while it stays under about 8 minutes. For larger floors, split by wing: each take starts and ends at a shared doorway or junction, with 5–10 s of overlap, and each starts with a marker in view.
- **Stairs / other floors:** a separate take per floor. Each floor is its own twin; linking floors is an R2/R3 feature.

### 3.2 Hard surfaces
| Surface | Problem | What to do |
|---|---|---|
| Mirrors, glass doors, glossy screens | Reflections look like rooms that move | Pass at an angle, don't linger; cover large mirrors with paper if allowed; turn screens off |
| Plain white walls, plain floors | No features to match | Keep a corner, skirting or furniture edge in frame; tilt so the floor–wall junction shows |
| Shiny polished floors | Reflected lights and ceilings | Less downward tilt; avoid shooting straight at bright fixtures |
| Windows with daylight | Blown highlights, dark room | Half-close blinds; expose for the room, not the window; don't point at the sun |
| Thin things (chair legs, railings, plants) | Become fuzzy | Walk past them slowly at 1–2 m in loop 2 |

## 4. After the take (on the spot, 1 minute)
1. Play the clip at 2× speed. Look for: smeared frames at turns, flicker or bands, people walking, a moment pointed at a blank wall or the ceiling.
2. Confirm it ends at the start view.
3. If anything failed, re-shoot now; going back later costs far more than 2 minutes.
4. Copy the file over USB/Drive unedited, with no trimming or re-encoding apps (they re-compress and may stabilise).

## 5. Processing
```bash
python -m plypipe run walk.mp4 --out work/lab2 --preset standard            # Kaggle T4 or RunPod
python -m plypipe run walk.mp4 --out work/lab2 --preset standard --focal-35mm 24   # if SfM struggles (S24 FE main lens)
```
`--height-cam` (default 1.45 m) is how high you held the phone; it sets the metric scale. Measure the operator once (phone at chest height, floor to lens) and pass the real value: a 5 cm error is a 3.5 % scale error.

## 6. When things go wrong

| Symptom (report.json / viewer) | Likely cause | Fix |
|---|---|---|
| `registeredRatio` < 0.8, rooms missing | Fast turns, blank walls, dark | Re-shoot slower with texture in frame; add light; `--features sift` as a check |
| Several separate models / "camera path has discontinuities" warning | Pivoting on the spot; a doorway passed too fast | `--force sfm --mapper incremental`; re-shoot the doorway (§3.1) |
| Soft, blurry splats | Shutter too slow, walking too fast | 1/100–1/200 s; slow down; more light |
| Walls change brightness / blotchy colour | Auto exposure or WB, or flicker | Lock ISO/shutter/WB; 1/100 s under 50 Hz lights (PPISP compensates mild changes, not 2-stop jumps) |
| Ghosts and floaters in mid-air | People moving; mirrors | Empty the space; avoid mirrors; export crops floaters outside the room box |
| Floor tilted or bent, `floorTiltDeg` > 5 | Phone held sideways or rotated mid-take; floor almost never in view | Keep the phone upright; tilt 10–15° down in loop 1 |
| Scale wrong (doors not about 2.1 m) | Operator held the phone higher or lower than 1.45 m | Re-run from `align` with the measured `--height-cam` (`--force align`) |
| Holes behind furniture | Loop 2 skipped | Add loop 2; walk around big furniture |
| Occupancy marks a free aisle as blocked | Sparse points from overhanging shelves | Expected in R1 (sparse grid); the R2 navmesh fixes it |

## 7. Why these rules (for judges)
- **Locked exposure, WB, focus and one lens** keep one camera model, which is what SfM and splat training assume.
- **A fast, flicker-safe shutter** removes blur, the most common quality killer, and avoids 50 Hz banding.
- **Two loops plus a cross pass** give every surface views from several angles: parallax for geometry and coverage behind furniture.
- **Turning while walking, never on the spot** keeps a baseline between every pair of frames; pure rotations are what break global SfM.
- **Returning to the start** closes the loop and removes accumulated drift.
- **An upright phone at a known height** gives the first gravity estimate and the metric scale; the A4 marker gives a check and the AR anchor.

The pipeline does the rest automatically:
- **Frames:** it keeps the sharpest frame every 0.4 s.
- **Floor:** it fits the floor as the most extended horizontal surface, not the most textured one.
- **Axes:** it aligns walls to the axes.
- **Training:** it trains in metres with per-frame exposure compensation (PPISP).
- **Export:** it crops floaters outside the room.
