/* Places in a twin, and routes between them on the occupancy grid.
   The lab ships hand-placed places for the first real capture (job 0ffc07fb, found by hit-testing the splat);
   in the product they come from people pinning spots and from the AI labelling pass. */
import type { OccupancyGrid } from '../walk/scene';

export type Category = 'Furniture' | 'Storage' | 'Exits' | 'Things';
export interface Place {
  id: string; name: string; cat: Category; p: [number, number, number];
  /** AI detection: class, confidence, half-size of the box around it */
  ai?: { cls: string; conf: number; half: [number, number, number] };
  pinned?: boolean;
}

export const PLACES: Place[] = [
  { id: 'desk', name: 'Study desk', cat: 'Furniture', p: [0.62, 0.86, 1.62], pinned: true, ai: { cls: 'desk', conf: .93, half: [.55, .42, .32] } },
  { id: 'laptop', name: 'Laptop', cat: 'Things', p: [0.56, 1.07, 1.83], ai: { cls: 'laptop', conf: .97, half: [.2, .14, .16] } },
  { id: 'flowers', name: 'Flower shelf', cat: 'Storage', p: [1.46, 1.72, 1.52], pinned: true, ai: { cls: 'potted plant', conf: .88, half: [.26, .24, .2] } },
  { id: 'door', name: 'Main door', cat: 'Exits', p: [-0.98, 1.2, 2.19], pinned: true, ai: { cls: 'door', conf: .91, half: [.45, 1.0, .08] } },
  { id: 'helmets', name: 'Helmet hooks', cat: 'Storage', p: [-3.11, 1.43, 1.56], pinned: true, ai: { cls: 'helmet', conf: .84, half: [.12, .32, .5] } },
  { id: 'clothes', name: 'Clothes rack', cat: 'Storage', p: [-1.2, 1.15, -1.75], pinned: true, ai: { cls: 'clothing', conf: .79, half: [.55, .45, .18] } },
  { id: 'shelf', name: 'Wall shelf', cat: 'Storage', p: [-0.33, 1.75, -2.1], ai: { cls: 'shelf', conf: .72, half: [.3, .2, .14] } },
  { id: 'curtain', name: 'Curtained doorway', cat: 'Exits', p: [2.2, 1.44, -1.1], pinned: true, ai: { cls: 'door', conf: .67, half: [.1, .9, .5] } },
  { id: 'chair', name: 'Chair', cat: 'Furniture', p: [1.4, 0.55, 0.06], ai: { cls: 'chair', conf: .95, half: [.3, .45, .3] } },
];

/** Distance (in cells) from every cell to the nearest blocked one: two-pass chamfer transform. */
export function clearance(g: OccupancyGrid) {
  const { width: w, height: h } = g.meta, d = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) d[i] = g.data[i] ? 0 : 1e6;
  const pass = (y0: number, y1: number, dy: number, x0: number, x1: number, dx: number) => {
    for (let y = y0; y !== y1; y += dy) for (let x = x0; x !== x1; x += dx) {
      const i = y * w + x; if (!d[i]) continue;
      for (const [ox, oy, c] of [[-dx, 0, 1], [0, -dy, 1], [-dx, -dy, 1.414], [dx, -dy, 1.414]]) {
        const X = x + ox, Y = y + oy; if (X >= 0 && Y >= 0 && X < w && Y < h) d[i] = Math.min(d[i], d[Y * w + X] + c);
      }
    }
  };
  pass(0, h, 1, 0, w, 1); pass(h - 1, -1, -1, w - 1, -1, -1);
  return d;
}

export class Navigator {
  clr: Float32Array; w: number; h: number; cell: number; ox: number; oz: number;
  constructor(public grid: OccupancyGrid) {
    this.clr = clearance(grid); const m = grid.meta; this.w = m.width; this.h = m.height; this.cell = m.cell; this.ox = m.origin[0]; this.oz = m.origin[1];
  }
  idx(x: number, z: number) { return [Math.floor((x - this.ox) / this.cell), Math.floor((z - this.oz) / this.cell)] as const; }
  pos(i: number, j: number): [number, number] { return [this.ox + (i + .5) * this.cell, this.oz + (j + .5) * this.cell]; }
  ok(i: number, j: number, need: number) { return i >= 0 && j >= 0 && i < this.w && j < this.h && this.clr[j * this.w + i] >= need; }

  /** Nearest cell with at least `need` cells of clearance (a place on a wall becomes the floor spot in front of it). */
  nearestFree(x: number, z: number, need = 2.5): [number, number] | null {
    const [ci, cj] = this.idx(x, z);
    for (let r = 0; r < 40; r++) {
      let best: [number, number] | null = null, bd = Infinity;
      for (let j = cj - r; j <= cj + r; j++) for (let i = ci - r; i <= ci + r; i++) {
        if (Math.max(Math.abs(i - ci), Math.abs(j - cj)) !== r || !this.ok(i, j, need)) continue;
        const d = (i - ci) ** 2 + (j - cj) ** 2; if (d < bd) { bd = d; best = [i, j]; }
      }
      if (best) return best;
    }
    return null;
  }

  /** A* over 8-connected cells that keep `need` cells from walls, preferring the middle of open floor; then
      string-pulled to straight runs. Returns floor points (x, z) or null. */
  route(from: [number, number], to: [number, number]): [number, number][] | null {
    for (const need of [2.5, 1.8, 1.2]) {
      const a = this.nearestFree(from[0], from[1], need), b = this.nearestFree(to[0], to[1], need); if (!a || !b) continue;
      const cells = this.astar(a, b, need); if (!cells) continue;
      const pts = cells.map(([i, j]) => this.pos(i, j));
      return this.pull(pts, need);
    }
    return null;
  }
  private astar(a: readonly [number, number], b: readonly [number, number], need: number) {
    const { w } = this, N = w * this.h, g = new Float32Array(N).fill(Infinity), came = new Int32Array(N).fill(-1), closed = new Uint8Array(N);
    const start = a[1] * w + a[0], goal = b[1] * w + b[0]; g[start] = 0;
    const open: number[] = [start], f = new Float32Array(N).fill(Infinity);
    const hh = (k: number) => Math.hypot(k % w - b[0], Math.floor(k / w) - b[1]); f[start] = hh(start);
    while (open.length) {
      let bi = 0; for (let k = 1; k < open.length; k++) if (f[open[k]] < f[open[bi]]) bi = k;
      const cur = open[bi]; open[bi] = open[open.length - 1]; open.pop();
      if (cur === goal) { const out: [number, number][] = []; for (let k = cur; k !== -1; k = came[k]) out.push([k % w, Math.floor(k / w)]); return out.reverse(); }
      if (closed[cur]) continue; closed[cur] = 1;
      const ci = cur % w, cj = Math.floor(cur / w);
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        if (!di && !dj) continue; const ni = ci + di, nj = cj + dj; if (!this.ok(ni, nj, need)) continue;
        if (di && dj && (!this.ok(ci + di, cj, need) || !this.ok(ci, cj + dj, need))) continue; // no corner cutting
        const k = nj * w + ni, near = Math.max(0, 6 - this.clr[k]) * .35; // keep off the walls when there is room
        const ng = g[cur] + (di && dj ? 1.414 : 1) + near;
        if (ng < g[k]) { g[k] = ng; came[k] = cur; f[k] = ng + hh(k); open.push(k); }
      }
    }
    return null;
  }
  /** Line of sight on the grid at the given clearance. */
  see(a: [number, number], b: [number, number], need: number) {
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]), n = Math.ceil(L / (this.cell * .5));
    for (let k = 0; k <= n; k++) { const t = k / Math.max(1, n), [i, j] = this.idx(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t); if (!this.ok(i, j, need * .8)) return false; }
    return true;
  }
  private pull(pts: [number, number][], need: number) {
    if (pts.length < 3) return pts;
    const out = [pts[0]]; let k = 0;
    while (k < pts.length - 1) { let j = pts.length - 1; while (j > k + 1 && !this.see(pts[k], pts[j], need)) j--; out.push(pts[j]); k = j; }
    return out;
  }
}

export const pathLength = (p: [number, number][]) => p.slice(1).reduce((s, q, i) => s + Math.hypot(q[0] - p[i][0], q[1] - p[i][1]), 0);

/** Turn-by-turn from a pulled path: one step per corner. Angles use the viewer's yaw convention (forward = -z). */
export function steps(p: [number, number][], dest: string) {
  const out: { text: string; at: number }[] = []; let run = 0;
  for (let i = 1; i < p.length; i++) {
    const seg = Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]); run += seg;
    if (i === p.length - 1) { out.push({ text: `Arrive at ${dest}`, at: run }); break; }
    const a1 = Math.atan2(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]), a2 = Math.atan2(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]);
    let d = a2 - a1; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
    const deg = d * 180 / Math.PI; if (Math.abs(deg) < 20) continue;
    out.push({ text: `${Math.abs(deg) > 120 ? 'Turn around' : Math.abs(deg) > 55 ? (deg > 0 ? 'Turn left' : 'Turn right') : (deg > 0 ? 'Bear left' : 'Bear right')} in ${run.toFixed(1)} m`, at: run });
  }
  return out;
}
