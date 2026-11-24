// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Several floors in one twin (pipeline plypipe/levels.py): each floor has its own walking grid, and the staircases
   are polylines of (x, floor y, z) taken from the frames filmed on them. Pure (no three.js, no DOM) so
   test/levels.test.mts runs it in Node. */
import type { Collider, Move, Point } from './collider';

export type P3 = [number, number, number];
export interface Stair { from: number; to: number; path: P3[] }
export interface LevelInfo { index: number; name: string; floorY: number; spawn?: P3 }

/** How close to a staircase's line (metres, across the floor) still counts as on the stairs. */
export const STAIR_REACH = 0.6;

/** The floor height under (x, z) if it is on a staircase: the nearest point of the nearest stair line, its height
    interpolated. null off the stairs. */
export function stairAt(stairs: readonly Stair[], x: number, z: number, reach = STAIR_REACH): { y: number; stair: Stair; d: number } | null {
  let best: { y: number; stair: Stair; d: number } | null = null;
  for (const s of stairs) {
    const p = s.path;
    for (let k = 1; k < p.length; k++) {
      const ax = p[k - 1][0], az = p[k - 1][2], bx = p[k][0], bz = p[k][2], dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz;
      const t = L2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / L2)) : 0;
      const d = Math.hypot(x - (ax + dx * t), z - (az + dz * t));
      if (d <= reach && (!best || d < best.d)) best = { y: p[k - 1][1] + (p[k][1] - p[k - 1][1]) * t, stair: s, d };
    }
  }
  return best;
}

/** The floor whose height is nearest below a standing height (y of the feet), with a little tolerance. */
export function levelAt(levels: readonly LevelInfo[], y: number): number {
  let best = 0, bd = Infinity;
  for (let i = 0; i < levels.length; i++) { const d = y - levels[i].floorY; if (d > -0.4 && d < bd) { bd = d; best = i; } }
  return bd === Infinity ? 0 : best;
}

/** A floor's collider that also lets the walker onto its staircases (the grid has no cells on the steps). */
export class LevelCollider implements Collider {
  base: Collider; stairs: readonly Stair[]; level: number;
  constructor(base: Collider, stairs: readonly Stair[], level: number) { this.base = base; this.stairs = stairs; this.level = level; }
  private mine() { return this.stairs.filter(s => s.from === this.level || s.to === this.level); }
  free(x: number, z: number, r: number) { return this.base.free(x, z, r) || !!stairAt(this.mine(), x, z); }
  move(x: number, z: number, dx: number, dz: number, r: number): Move {
    const st = this.mine(), to = stairAt(st, x + dx, z + dz);
    if (to && (stairAt(st, x, z) || this.base.free(x, z, r))) return { x: x + dx, z: z + dz, hit: false };
    return this.base.move(x, z, dx, dz, r);
  }
  nearestFree(x: number, z: number, r: number, reach?: number): Point | null { return stairAt(this.mine(), x, z) ? { x, z } : this.base.nearestFree(x, z, r, reach); }
  openSpot() { return this.base.openSpot(); }
}

/** The staircases to take from one floor to another (fewest flights), as [stair, going up?] in order; [] on the same
    floor, null when no stairs join them. */
export function stairChain(stairs: readonly Stair[], from: number, to: number): [Stair, boolean][] | null {
  if (from === to) return [];
  const prev = new Map<number, [number, Stair, boolean]>(), q = [from], seen = new Set([from]);
  while (q.length) {
    const a = q.shift()!;
    for (const s of stairs) {
      const b = s.from === a ? s.to : s.to === a ? s.from : null;
      if (b === null || seen.has(b)) continue;
      seen.add(b); prev.set(b, [a, s, s.from === a]); q.push(b);
      if (b === to) {
        const out: [Stair, boolean][] = []; let c = to;
        while (c !== from) { const [p, st, up] = prev.get(c)!; out.unshift([st, up]); c = p; }
        return out;
      }
    }
  }
  return null;
}

type Route = (from: [number, number], to: [number, number]) => [number, number][] | null;

/** A walk across floors: on each floor the grid's route, joined by the stair lines. Each point keeps its floor so the
    caller can draw only the part on the floor in view. null when any leg has no route. */
export function routeAcross(route: (level: number) => Route | null, stairs: readonly Stair[], from: { level: number; at: [number, number] },
  to: { level: number; at: [number, number] }): { x: number; z: number; level: number; y?: number }[] | null {
  const chain = stairChain(stairs, from.level, to.level); if (!chain) return null;
  const out: { x: number; z: number; level: number; y?: number }[] = [];
  let level = from.level, at = from.at;
  for (const [s, up] of chain) {
    const steps = up ? s.path : [...s.path].reverse(), entry: [number, number] = [steps[0][0], steps[0][2]];
    const leg = route(level)?.(at, entry); if (!leg) return null;
    for (const [x, z] of leg) out.push({ x, z, level });
    for (const p of steps) out.push({ x: p[0], z: p[2], level: -1, y: p[1] });
    level = up ? s.to : s.from; at = [steps[steps.length - 1][0], steps[steps.length - 1][2]];
  }
  const last = route(level)?.(at, to.at); if (!last) return null;
  for (const [x, z] of last) out.push({ x, z, level });
  return out;
}
