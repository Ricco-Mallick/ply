// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* What the walker cannot walk through. The engine asks every collision question through `Collider`, as circles in
   world metres on the floor plane (x, z), so a mesh or a navmesh can answer them later in place of the occupancy
   grid. No three.js and no DOM here: the tests run this file in plain Node. */

export interface Point { x: number; z: number; }
/** Where a move ended, and whether anything was touched on the way. */
export interface Move extends Point { hit: boolean; }

export interface Collider {
  /** A circle of radius r at (x, z) touches nothing. */
  free(x: number, z: number, r: number): boolean;
  /** Move the circle by (dx, dz), sliding along what it meets. It ends where `free` holds and never on the far side
      of a wall, whatever the distance asked for. From a place that is not free it does not move. */
  move(x: number, z: number, dx: number, dz: number, r: number): Move;
  /** The nearest place to (x, z), no further than `reach`, where the circle is free; null when there is none. */
  nearestFree(x: number, z: number, r: number, reach?: number): Point | null;
  /** The most open place to stand (the furthest from anything), or null when nothing is open. */
  openSpot(): Point | null;
}

/** The occupancy grid as scene.ts loads it: 1 = blocked, row j along +z, column i along +x, `cell` metres a side. */
export interface Grid { data: ArrayLike<number>; meta: { origin: [number, number]; cell: number; width: number; height: number }; }

/** An overlap under a micron is contact, not penetration: it absorbs the rounding of a push to exactly r. */
const EPS = 1e-6;
/** Pushes tried per sub-step (two free a circle from a wall and a corner; more made no difference on cluttered
    grids), and sub-steps per move. */
const PUSHES = 4, STEPS = 64;

// scratch: the push that frees the circle from the contact `deepest` found, and where `settle` left the circle
let PX = 0, PZ = 0, SX = 0, SZ = 0;

/** How far the circle reaches into what it overlaps most, a blocked cell's square or the outside of the grid
    (0 = it overlaps nothing), with the shortest way out of that one left in (PX, PZ). Only the cells under the
    circle's bounding square are read. */
function deepest(g: Grid, x: number, z: number, r: number): number {
  const m = g.meta, ox = m.origin[0], oz = m.origin[1], c = m.cell, w = m.width, h = m.height, d = g.data;
  PX = PZ = 0; if (!Number.isFinite(x) || !Number.isFinite(z)) return Infinity;
  // the four sides of the grid are walls
  let best = 0, p = r - (x - ox); if (p > best) { best = p; PX = p; PZ = 0; }
  p = r - (ox + w * c - x); if (p > best) { best = p; PX = -p; PZ = 0; }
  p = r - (z - oz); if (p > best) { best = p; PX = 0; PZ = p; }
  p = r - (oz + h * c - z); if (p > best) { best = p; PX = 0; PZ = -p; }
  const i0 = Math.max(0, Math.floor((x - r - ox) / c)), i1 = Math.min(w - 1, Math.floor((x + r - ox) / c));
  const j0 = Math.max(0, Math.floor((z - r - oz) / c)), j1 = Math.min(h - 1, Math.floor((z + r - oz) / c));
  for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
    if (!d[j * w + i]) continue;
    const x0 = ox + i * c, z0 = oz + j * c, x1 = x0 + c, z1 = z0 + c;
    // from the closest point of the cell's square to the centre
    const ex = x < x0 ? x - x0 : x > x1 ? x - x1 : 0, ez = z < z0 ? z - z0 : z > z1 ? z - z1 : 0, e = Math.sqrt(ex * ex + ez * ez);
    if (e > 0) { p = r - e; if (p > best) { best = p; PX = ex / e * p; PZ = ez / e * p; } continue; }
    // the centre is inside the square: out through its nearest side, the first of -x, +x, -z, +z on a tie
    const l = x - x0, rt = x1 - x, t = z - z0, n = Math.min(l, rt, t, z1 - z); p = r + n;
    if (p > best) { best = p; PX = n === l ? -p : n === rt ? p : 0; PZ = n === l || n === rt ? 0 : n === t ? -p : p; }
  }
  return best;
}

/** Push the circle out of what it overlaps, the deepest first, at most n times, and leave it at (SX, SZ). Returns the
    pushes it took, or -1 when the circle still overlaps something. The deepest first because a wall is many cells:
    its nearest cell pushes straight out, a neighbour's corner would push sideways too. */
function settle(g: Grid, x: number, z: number, r: number, n: number): number {
  let p = deepest(g, x, z, r), k = 0;
  for (; k < n && p > EPS; k++) { x += PX; z += PZ; p = deepest(g, x, z, r); }
  SX = x; SZ = z; return p > EPS ? -1 : k;
}

/** Move a circle over the grid by (dx, dz): in sub-steps of at most half a cell and half the radius, each one pushed
    out of the cells it lands in, so the circle slides along walls and rounds corners. A sub-step that cannot be
    freed (a corner too tight to enter) is dropped and the move ends at the last free place. Every place it passes
    through is free and within r of the one before, and two such places cannot have a blocked cell between them:
    that is what stops it going through a wall at any speed. A move is at most STEPS sub-steps long; the rest of a
    longer one is dropped. */
export function moveCircle(g: Grid, x: number, z: number, dx: number, dz: number, r: number): Move {
  const len = Math.hypot(dx, dz);
  if (!(len > 0 && len < Infinity)) return { x, z, hit: false };
  if (deepest(g, x, z, r) > EPS) return { x, z, hit: true };
  const h = Math.min(g.meta.cell, r) / 2, n = Math.min(STEPS, Math.ceil(len / h)), k = Math.min(h, len / n) / len;
  let hit = false;
  for (let s = 0; s < n; s++) {
    const pushes = settle(g, x + dx * k, z + dz * k, r, PUSHES), far = (SX - x) ** 2 + (SZ - z) ** 2;
    if (pushes) hit = true;
    // not freed, or pressed against something and going nowhere: the sub-steps left would end the same way
    if (pushes < 0 || far > r * r || (pushes && far < EPS * EPS)) break;
    x = SX; z = SZ;
  }
  return { x, z, hit };
}

export class GridCollider implements Collider {
  grid: Grid;
  /** the smallest radius known to fit at no cell centre, so the search of the whole grid is made once */
  private none = Infinity;
  constructor(grid: Grid) { this.grid = grid; }

  free(x: number, z: number, r: number) { return deepest(this.grid, x, z, r) <= EPS; }
  move(x: number, z: number, dx: number, dz: number, r: number) { return moveCircle(this.grid, x, z, dx, dz, r); }

  /** Pushed out of the wall when that is enough (the exact nearest place for a shallow overlap), otherwise the nearest
      cell centre where the circle fits, searched in rings of cells outwards. */
  nearestFree(x: number, z: number, r: number, reach = Infinity): Point | null {
    const g = this.grid, m = g.meta, ox = m.origin[0], oz = m.origin[1], c = m.cell, w = m.width, h = m.height;
    if (!Number.isFinite(x) || !Number.isFinite(z)) return null;
    if (settle(g, x, z, r, 16) >= 0 && Math.hypot(SX - x, SZ - z) <= reach) return { x: SX, z: SZ };
    const ci = Math.floor((x - ox) / c), cj = Math.floor((z - oz) / c);
    if (r >= this.none || !(Math.abs(ci) < 1e9 && Math.abs(cj) < 1e9)) return null;  // 1e9: so far out that ring numbers stop counting
    const k0 = Math.max(0, -ci, ci - w + 1, -cj, cj - h + 1), k1 = Math.min(Math.ceil(reach / c) + 1, Math.max(ci, w - 1 - ci, cj, h - 1 - cj));
    let best: Point | null = null, bd = reach;
    const at = (i: number, j: number) => {
      if (g.data[j * w + i]) return;
      const px = ox + (i + .5) * c, pz = oz + (j + .5) * c, d = Math.hypot(px - x, pz - z);
      if (d <= bd && (d < bd || !best) && this.free(px, pz, r)) { bd = d; best = { x: px, z: pz }; }
    };
    // ring k holds the cells k away along the further axis, none nearer than (k - 1) cells
    for (let k = k0; k <= k1 && (k - 1) * c <= bd; k++) {
      const a = Math.max(0, ci - k), b = Math.min(w - 1, ci + k);
      for (const j of k ? [cj - k, cj + k] : [cj]) if (j >= 0 && j < h) for (let i = a; i <= b; i++) at(i, j);
      for (const i of k ? [ci - k, ci + k] : []) if (i >= 0 && i < w) for (let j = Math.max(0, cj - k + 1); j <= Math.min(h - 1, cj + k - 1); j++) at(i, j);
    }
    if (!best && reach === Infinity) this.none = r;
    return best;
  }

  /** The free cell furthest from any blocked one: a two-pass chamfer distance transform. */
  openSpot(): Point | null {
    const { width: w, height: h, cell, origin } = this.grid.meta, data = this.grid.data, d = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) d[i] = data[i] ? 0 : 1e6;
    const pass = (y0: number, y1: number, dy: number, x0: number, x1: number, dx: number) => {
      for (let y = y0; y !== y1; y += dy) for (let x = x0; x !== x1; x += dx) {
        const i = y * w + x; if (!d[i]) continue;
        for (const [ox, oy, c] of [[-dx, 0, 1], [0, -dy, 1], [-dx, -dy, 1.414], [dx, -dy, 1.414]]) { const X = x + ox, Y = y + oy; if (X >= 0 && Y >= 0 && X < w && Y < h) d[i] = Math.min(d[i], d[Y * w + X] + c); }
      }
    };
    pass(0, h, 1, 0, w, 1); pass(h - 1, -1, -1, w - 1, -1, -1);
    let best = 0, bi = -1; for (let i = 0; i < w * h; i++) if (d[i] < 1e6 && d[i] > best) { best = d[i]; bi = i; }
    return bi < 0 ? null : { x: origin[0] + (bi % w + .5) * cell, z: origin[1] + (Math.floor(bi / w) + .5) * cell };
  }
}
