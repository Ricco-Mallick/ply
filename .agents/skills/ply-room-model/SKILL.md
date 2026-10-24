---
name: ply-room-model
description: Rebuild a captured Ply room as editable Three.js geometry from a phone walk — no LiDAR. Use for turning a Ply job (posed frames + measured shell + splat) into a parametric, modellable scene: walls, openings, furniture, materials, exports.
---

# Ply room model

A Ply capture gives you three kinds of evidence about one room. Use them as evidence, never as the model:

| Evidence | Where | What it is good for |
|---|---|---|
| **Posed photographs** | `evidence/frames/*.jpg` + `poses.json` | what things *look* like, colour, proportions, what is where |
| **Measured shell** | `shell.json` | wall lines and lengths, openings, object boxes and their class |
| **Photoreal twin** | `scene.spz` / `mesh.ply` | answering "what was actually there" when the photos disagree |

The job is to **author** a small, clean Three.js scene — walls, slabs, openings, furniture — not to re-project the evidence. The output is editable because *you* placed every element; that is the whole point, and it is why this cannot be fully automatic.

There is **no LiDAR** here. Every measurement comes from the photogrammetric solve, so treat measurements as ±5 cm and **state uncertainty rather than inventing precision**.

## Inputs

A bundle from `python scripts/agent_bundle.py <job-id>` (or `--out <dir>` for a local run) containing:

```
room/
  shell.json          walls (a, b, length, normal), openings (kind, wall, u, width, sill, height), objects (id, name, kind, center, half)
  places.json         the same objects with SAM 3 confidence
  scene.json          bounds, floorY, spawn, cameraPath, splat/mesh refs
  evidence/poses.json [{ index, file, position:[x,y,z], yaw, pitch, fov }]
  evidence/frames/*.jpg
  report.md           a readable measurement sheet (wall lengths, openings, the 30 largest objects)
  model/              where you write the room module (index.ts)
```

Coordinates: **metres, +Y up, floor at y = 0, scene frame** (the same frame the splat and mesh are in). Never invent your own frame; the viewer loads your module into it.

## Workflow

1. **Read the measurement sheet first.** `report.md` lists every wall, opening and object with numbers. Print it, note the room's dimensions, and note which walls are inferred (short, odd angles) versus certain (long, axis-aligned).
2. **Fix the shell before anything else.** The hull gives you walls; check each against the frames. If a wall is a chamfer that is really a corner, correct it. If a door or window is missing, add it from the photo, with its measured span. Get floor height and ceiling height right (the shell's `ceilingY` is a splat percentile — clamp it to what the photos show).
3. **Pick the hero objects** — the ten to twenty that make the room recognisable (desks, benches, altar, blackboards, sofas). For each, look at the two or three frames where it is closest and least occluded, then model it:
   - build from primitives with **real proportions in metres** (seat 0.45, desk 0.74, door 2.05, sill 0.9);
   - keep it parametric (width, height, depth as parameters), because the editor will resize it;
   - name it with its class and instance (`desk-1`, `chair-12`) and set `userData.semantic = { kind, confidence }`.
4. **Everything else gets a proxy.** A box at the measured size and place is correct, cheap and honest. Do not spend an hour on a bin.
5. **Materials from the photographs.** Sample colour from the frames: crop the object, take the median of the middle 50 % so highlights and shadows do not dominate. Roughness high (0.7–0.95) for fabric, wood and plaster; low only for glass and metal.
6. **Compare, then fix the biggest error.** Render your scene from a captured pose and diff it against that photograph:
   ```sh
   node scripts/render_pose.mjs room 3        # screenshot from pose 3
   python scripts/compare.py room 3           # per-region difference + worst areas
   ```
   The comparison never says "good enough"; it points at the region that is most wrong. Fix that one thing, then compare the next pose. Three or four rounds is normal.
7. **Write the module.** `model/index.ts` exports:
   ```ts
   export function buildRoom(shell: Shell): THREE.Group   // walls, floor, ceiling, openings
   export function buildObjects(): THREE.Group            // your modelled objects
   ```
   Keep geometry static and precomputed; merge small parts; no per-frame rebuilds.

## Rules that keep this honest

- **Measure, then model.** Every number comes from `shell.json`, a frame you can point at, or a standard object size. If you are guessing, write `// estimated` beside it.
- **Do not crop the evidence.** Empty regions in the solve are information.
- **Do not scale the room to make the furniture fit.** Fix the furniture.
- **Do not add objects the capture did not see.** A clean empty room beats an invented one.
- **Report what is measured, what is modelled, and what is approximate** in `model/README.md` — the editor shows that note to the user.

## Deliverables

- `model/index.ts` — the room, as above.
- `model/README.md` — measured vs modelled vs approximate, with the frames used.
- `compare/` — the render/photo pairs from step 6, and the score that came with them.

## What "done" looks like

A person who has seen the room recognises it immediately; every wall and opening matches the photographs; the hero objects read as themselves at 3 m; and the whole module builds in under 200 ms in the viewer with the splat hidden.
