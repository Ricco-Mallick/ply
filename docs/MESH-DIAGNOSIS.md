# Why the mesh is a soup

Measured 2026-11-13 on `chapel/eval/q0a-meshraw/mesh_raw.npz` from the experiments account's `ply-bench` Volume:
the extraction the export refused to ship, saved by `--mesh-keep-raw`. 274 MB, the fused surface **before** any
simplification: 7,981,407 vertices, 13,894,426 faces.

## The fused surface is sound where it is sound, and shredded elsewhere

| Measure | Value | Reading |
|---|---|---|
| V/F ratio | 0.57 | a normal closed surface, not a soup |
| degenerate faces (< 1e-9 m²) | 0.118% | negligible |
| median face area | 6.6e-05 m² (0.66 cm²) | fine detail, as expected from a 2 cm TSDF |
| edges used by more than two faces | **0.00%**, max 2 | **the fused surface is manifold; nothing crosses** |
| interior edges (two faces) | 95.15% | mostly closed |
| boundary edges (one face) | 4.85% | holes, not crossings |
| **connected components** | **84,443** | the real problem: a giant sheet plus 80,590 specks |
| largest component | 4,021,028 vertices (50.4%) | half the surface is one piece |
| components under 100 vertices | 80,590 | 59% of the pieces are crumbs |

So the soup does not come from faces crossing in the fusion. It comes from **fragmentation**: the TSDF leaves
tens of thousands of small disconnected sheets, and whatever simplifies afterwards has to reconcile them.

## The simplifier makes it worse

Filtering to components of 2,000 vertices or more keeps **136 pieces and 73.9% of the faces**, which is a sane
starting mesh. Clustering those vertices on a 2 cm grid then produces a mesh whose used-vertex-to-face ratio is
**0.79** against the raw 0.57, and the piece count does not improve. That is the stage that turns a fragmented but
manifold surface into something export refuses to write.

## What to do

1. **Filter components before simplifying**, not after: drop pieces under a few thousand vertices. It costs a
   quarter of the faces and removes 99.8% of the pieces.
2. **Do not use vertex clustering as the simplifier for this data.** On a fragmented sheet it collapses each
   fragment independently. Quadric edge collapse on the filtered mesh, or meshing only the largest components, is
   the path that preserves a surface.
3. Keep the export's refusal: a mesh with more than three triangles per vertex, or hundreds of thousands of
   pieces, should not ship. The gate is doing its job; the geometry is not ready.

## Decision

Mesh stays **opt-in and off by default** for phase 7. The fix above is small, but it is a pipeline change that
needs its own run to validate, and phase 7's value is in the splat quality, not in a mesh nobody has scored
against a laser yet.
