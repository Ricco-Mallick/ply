// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Rooms from a floor's walking grid, with no learned model (the idea of multi-room layout estimation, done as plain
   geometry):
   1. the building's main wall direction: the orientation of the blocked cells' edges, folded to 90 degrees;
   2. room cores: free cells far from any wall (a doorway is narrow, a room is not), at the largest distance that still
      leaves a core, then grown back over all the free floor in step (geodesic), so a room ends at its doorways;
   3. each room's outline: its cells in the building's own axes, traced and straightened (Manhattan walls), notches
      under 30 cm dropped;
   4. doors: where two grown rooms touch. Rooms under 1.5 m² join the neighbour they share most border with.
   Pure (no DOM): test/rooms.test.mts runs it in Node. */

export interface RoomGrid { data: ArrayLike<number>; meta: { origin: [number, number]; cell: number; width: number; height: number } }
export interface Room { id: number; name: string; areaM2: number; centroid: [number, number]; outline: [number, number][]; doors: { to: number; at: [number, number] }[] }

const MIN_ROOM_M2 = 1.5, NOTCH_M = 0.3, WIDE_OPENING_M = 1.5, SMOOTH_M = 0.5;

/** Chamfer distance (in cells) from every free cell to the nearest blocked one or the grid's edge. */
function distance(free: Uint8Array, w: number, h: number): Float32Array {
  const d = new Float32Array(w * h), INF = 1e9, a = 1, b = Math.SQRT2;
  for (let k = 0; k < w * h; k++) d[k] = free[k] ? INF : 0;
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const k = j * w + i; if (!d[k]) continue;
    let v = d[k];
    v = Math.min(v, i > 0 ? d[k - 1] + a : 1, j > 0 ? d[k - w] + a : 1, i > 0 && j > 0 ? d[k - w - 1] + b : 1, i < w - 1 && j > 0 ? d[k - w + 1] + b : 1);
    d[k] = v;
  }
  for (let j = h - 1; j >= 0; j--) for (let i = w - 1; i >= 0; i--) {
    const k = j * w + i; if (!d[k]) continue;
    let v = d[k];
    v = Math.min(v, i < w - 1 ? d[k + 1] + a : 1, j < h - 1 ? d[k + w] + a : 1, i < w - 1 && j < h - 1 ? d[k + w + 1] + b : 1, i > 0 && j < h - 1 ? d[k + w - 1] + b : 1);
    d[k] = v;
  }
  return d;
}

/** The main wall direction in radians (0..pi/2): the angle at which the wall cells (blocked cells next to floor),
    projected on the two axes, pile up most sharply (walls along an axis fall into few bins). Half-degree steps. */
export function mainAxis(g: RoomGrid): number {
  const { width: w, height: h, cell } = g.meta, xs: number[] = [], zs: number[] = [];
  for (let j = 1; j < h - 1; j++) for (let i = 1; i < w - 1; i++) {
    const k = j * w + i; if (!g.data[k]) continue;
    if (!g.data[k - 1] || !g.data[k + 1] || !g.data[k - w] || !g.data[k + w]) { xs.push(i * cell); zs.push(j * cell); }
  }
  if (xs.length < 20) return 0;
  let best = 0, bv = -1; const bins = new Map<number, number>();
  for (let deg = 0; deg < 90; deg += 0.5) {
    const a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a); let v = 0;
    for (const proj of [0, 1]) {
      bins.clear();
      for (let t = 0; t < xs.length; t++) { const u = proj ? -xs[t] * s + zs[t] * c : xs[t] * c + zs[t] * s, key = Math.round(u / cell); bins.set(key, (bins.get(key) ?? 0) + 1); }
      for (const n of bins.values()) v += n * n;
    }
    if (v > bv) { bv = v; best = a; }
  }
  return best < 0.035 || best > Math.PI / 2 - 0.035 ? 0 : best;   // within 2 degrees of the grid: the grid's own axes
}

export function findRooms(g: RoomGrid): { axis: number; rooms: Room[]; label: Int32Array } {
  const { width: w, height: h, cell, origin } = g.meta, n = w * h;
  // furniture is not a wall: blocked patches that do not reach the outside (the grid's edge) are floor for the rooms
  const outside = new Uint8Array(n), oq = new Int32Array(n); let oh = 0, ot = 0;
  for (let i = 0; i < w; i++) for (const k of [i, (h - 1) * w + i]) if (g.data[k] && !outside[k]) { outside[k] = 1; oq[ot++] = k; }
  for (let j = 0; j < h; j++) for (const k of [j * w, j * w + w - 1]) if (g.data[k] && !outside[k]) { outside[k] = 1; oq[ot++] = k; }
  while (oh < ot) { const k = oq[oh++], i = k % w, j = (k / w) | 0;
    for (const nk of [i > 0 ? k - 1 : -1, i < w - 1 ? k + 1 : -1, j > 0 ? k - w : -1, j < h - 1 ? k + w : -1]) if (nk >= 0 && g.data[nk] && !outside[nk]) { outside[nk] = 1; oq[ot++] = nk; } }
  const free = new Uint8Array(n); for (let k = 0; k < n; k++) free[k] = outside[k] ? 0 : 1;
  const d = distance(free, w, h);
  const minCells = Math.max(4, Math.round(MIN_ROOM_M2 / (cell * cell)));
  // cores: the largest clearance (from 1.2 m down to 0.35 m) at which the floor still splits into cores; one big room
  // with no narrow doorway keeps a single core
  let label = new Int32Array(n).fill(-1), cores = 0;
  for (const m of [1.2, 1.0, 0.85, 0.7, 0.55, 0.45, 0.35]) {
    const t = m / cell, lab = new Int32Array(n).fill(-1); let c = 0;
    const q = new Int32Array(n);
    for (let s = 0; s < n; s++) {
      if (lab[s] >= 0 || d[s] < t) continue;
      let qh = 0, qt = 0, size = 0; q[qt++] = s; lab[s] = c;
      while (qh < qt) { const k = q[qh++]; size++; const i = k % w, j = (k / w) | 0;
        for (const nk of [i > 0 ? k - 1 : -1, i < w - 1 ? k + 1 : -1, j > 0 ? k - w : -1, j < h - 1 ? k + w : -1]) if (nk >= 0 && lab[nk] < 0 && d[nk] >= t) { lab[nk] = c; q[qt++] = nk; } }
      if (size * cell * cell < 0.25) { for (let k = 0; k < qt; k++) lab[q[k]] = -2; continue; }  // a speck of clearance is not a room
      c++;
    }
    for (let k = 0; k < n; k++) if (lab[k] === -2) lab[k] = -1;
    if (c >= 1) { label = lab; cores = c; if (c >= 2 || m <= 0.55) break; }
  }
  if (!cores) return { axis: mainAxis(g), rooms: [], label };
  // grow the cores over the free floor together, nearest core first (breadth-first, so a room stops at a doorway)
  const q = new Int32Array(n); let qh = 0, qt = 0;
  for (let k = 0; k < n; k++) if (label[k] >= 0) q[qt++] = k;
  while (qh < qt) { const k = q[qh++], i = k % w, j = (k / w) | 0;
    for (const nk of [i > 0 ? k - 1 : -1, i < w - 1 ? k + 1 : -1, j > 0 ? k - w : -1, j < h - 1 ? k + w : -1]) if (nk >= 0 && free[nk] && label[nk] < 0) { label[nk] = label[k]; q[qt++] = nk; } }
  // two cores joined by more open floor than a doorway (a pillar or a row of pews split them) are one room
  for (let pass = 0; pass < 4; pass++) {
    const border = new Map<string, number>();
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) { const k = j * w + i, a = label[k]; if (a < 0) continue;
      for (const nk of [i < w - 1 ? k + 1 : -1, j < h - 1 ? k + w : -1]) { const b = nk >= 0 ? label[nk] : -1; if (b < 0 || b === a) continue;
        const key = a < b ? `${a},${b}` : `${b},${a}`; border.set(key, (border.get(key) ?? 0) + 1); } }
    let merged = false;
    for (const [key, len] of border) if (len * cell > WIDE_OPENING_M) {
      const [a, b] = key.split(',').map(Number); for (let k = 0; k < n; k++) if (label[k] === b) label[k] = a; merged = true; break;
    }
    if (!merged) break;
  }
  // small rooms join the neighbour they touch most
  for (let pass = 0; pass < 3; pass++) {
    const size = new Int32Array(cores); for (let k = 0; k < n; k++) if (label[k] >= 0) size[label[k]]++;
    let changed = false;
    for (let r = 0; r < cores; r++) {
      if (!size[r] || size[r] >= minCells) continue;
      const touch = new Map<number, number>();
      for (let k = 0; k < n; k++) if (label[k] === r) { const i = k % w, j = (k / w) | 0;
        for (const nk of [i > 0 ? k - 1 : -1, i < w - 1 ? k + 1 : -1, j > 0 ? k - w : -1, j < h - 1 ? k + w : -1]) if (nk >= 0 && label[nk] >= 0 && label[nk] !== r) touch.set(label[nk], (touch.get(label[nk]) ?? 0) + 1); }
      const to = [...touch.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
      for (let k = 0; k < n; k++) if (label[k] === r) label[k] = to ?? -1;
      size[r] = 0; changed = true;
    }
    if (!changed) break;
  }
  // number what is left by size, largest first
  const size = new Int32Array(cores); for (let k = 0; k < n; k++) if (label[k] >= 0) size[label[k]]++;
  const order = [...Array(cores).keys()].filter(r => size[r] >= minCells).sort((a, b) => size[b] - size[a]), re = new Int32Array(cores).fill(-1);
  order.forEach((r, i) => { re[r] = i; });
  for (let k = 0; k < n; k++) label[k] = label[k] >= 0 ? re[label[k]] : -1;
  const axis = mainAxis(g), world = (i: number, j: number): [number, number] => [origin[0] + i * cell, origin[1] + j * cell];
  const rooms: Room[] = order.map((_, id) => {
    let sx = 0, sz = 0, c = 0; for (let k = 0; k < n; k++) if (label[k] === id) { sx += k % w + .5; sz += ((k / w) | 0) + .5; c++; }
    return { id, name: `Room ${id + 1}`, areaM2: Math.round(c * cell * cell * 10) / 10, centroid: world(sx / c, sz / c), outline: outline(label, id, w, h, cell, origin, axis), doors: [] };
  });
  // doors: the middle of each shared border between two rooms
  const shared = new Map<string, number[]>();
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) { const k = j * w + i, a = label[k]; if (a < 0) continue;
    for (const nk of [i < w - 1 ? k + 1 : -1, j < h - 1 ? k + w : -1]) { const b = nk >= 0 ? label[nk] : -1; if (b < 0 || b === a) continue;
      const key = a < b ? `${a},${b}` : `${b},${a}`; const v = shared.get(key) ?? []; v.push(i + (nk === k + 1 ? 1 : .5), j + (nk === k + w ? 1 : .5)); shared.set(key, v); } }
  for (const [key, v] of shared) {
    const [a, b] = key.split(',').map(Number); let x = 0, z = 0; for (let t = 0; t < v.length; t += 2) { x += v[t]; z += v[t + 1]; }
    const at = world(x / (v.length / 2), z / (v.length / 2)); rooms[a].doors.push({ to: b, at }); rooms[b].doors.push({ to: a, at });
  }
  return { axis, rooms, label };
}

/** A room's outline in world metres: its cells resampled in the building's axes, the outer boundary traced, then
    straightened (every edge along one of the two axes, notches under NOTCH_M dropped). */
function outline(label: Int32Array, id: number, w: number, _h: number, cell: number, origin: [number, number], axis: number): [number, number][] {
  const ca = Math.cos(axis), sa = Math.sin(axis), pts: [number, number][] = [];
  for (let k = 0; k < label.length; k++) if (label[k] === id) { const x = origin[0] + (k % w + .5) * cell, z = origin[1] + (((k / w) | 0) + .5) * cell; pts.push([x * ca + z * sa, -x * sa + z * ca]); }
  if (!pts.length) return [];
  let u0 = Infinity, v0 = Infinity, u1 = -Infinity, v1 = -Infinity; for (const [u, v] of pts) { u0 = Math.min(u0, u); v0 = Math.min(v0, v); u1 = Math.max(u1, u); v1 = Math.max(v1, v); }
  const W = Math.ceil((u1 - u0) / cell) + 3, H = Math.ceil((v1 - v0) / cell) + 3, m = new Uint8Array(W * H);
  for (const [u, v] of pts) m[(Math.floor((v - v0) / cell) + 1) * W + Math.floor((u - u0) / cell) + 1] = 1;
  // smooth the room's raster before tracing: a closing fills notches and the gaps the rotation leaves, an opening
  // drops spurs; both at NOTCH_M, so what is traced is already the straightened shape
  const r = Math.max(1, Math.round(SMOOTH_M / cell / 2));
  const morph = (src: Uint8Array, grow: boolean) => { const o = new Uint8Array(W * H);
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) { let v = grow ? 0 : 1;
      for (let dj = -r; dj <= r && (grow ? !v : v); dj++) for (let di = -r; di <= r; di++) { const ii = i + di, jj = j + dj, x = ii >= 0 && jj >= 0 && ii < W && jj < H ? src[jj * W + ii] : 0; if (grow ? x : !x) { v = grow ? 1 : 0; break; } }
      o[j * W + i] = v; } return o; };
  const mc = morph(morph(morph(morph(m, true), false), false), true);
  // trace: boundary edges between filled and empty cells, chained; the longest loop is the outside
  const edges = new Map<string, [number, number]>(), key = (x: number, y: number) => `${x},${y}`;
  const F = (i: number, j: number) => i >= 0 && j >= 0 && i < W && j < H && mc[j * W + i] === 1;
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) { if (!F(i, j)) continue;
    if (!F(i, j - 1)) edges.set(key(i, j), [i + 1, j]); if (!F(i + 1, j)) edges.set(key(i + 1, j), [i + 1, j + 1]);
    if (!F(i, j + 1)) edges.set(key(i + 1, j + 1), [i, j + 1]); if (!F(i - 1, j)) edges.set(key(i, j + 1), [i, j]); }
  let best: [number, number][] = [];
  const used = new Set<string>();
  for (const start of edges.keys()) {
    if (used.has(start)) continue;
    const loop: [number, number][] = []; let k = start, guard = 0;
    while (!used.has(k) && guard++ < 200000) { used.add(k); const [x, y] = k.split(',').map(Number); loop.push([x, y]); const nx = edges.get(k); if (!nx) break; k = key(nx[0], nx[1]); }
    if (loop.length > best.length) best = loop;
  }
  // keep the corners only, then flatten bumps and dents: out by less than NOTCH_M and straight back (a doorway's lip)
  const corners = (pts: [number, number][]) => pts.filter((p, i) => { const a = pts[(i - 1 + pts.length) % pts.length], b = pts[(i + 1) % pts.length]; return !((a[0] === p[0] && p[0] === b[0]) || (a[1] === p[1] && p[1] === b[1])); });
  let poly = corners(best);
  const T = NOTCH_M / cell, len = (a: [number, number], b: [number, number]) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
  for (let guard = 0; guard < 50 && poly.length > 4; guard++) {
    let hit = -1;
    for (let k = 0; k < poly.length; k++) {
      const a = poly[k], b = poly[(k + 1) % poly.length], c = poly[(k + 2) % poly.length], d = poly[(k + 3) % poly.length];
      const out = [b[0] - a[0], b[1] - a[1]], back = [d[0] - c[0], d[1] - c[1]];
      if (len(a, b) < T && len(c, d) < T && out[0] === -back[0] && out[1] === -back[1]) { hit = k; break; }
    }
    if (hit < 0) break;
    const drop = new Set([(hit + 1) % poly.length, (hit + 2) % poly.length]);
    poly = corners(poly.filter((_, k) => !drop.has(k)));
  }
  return poly.map(([i, j]) => { const u = u0 + (i - 1) * cell, v = v0 + (j - 1) * cell; return [Math.round((u * ca - v * sa) * 100) / 100, Math.round((u * sa + v * ca) * 100) / 100]; });
}
