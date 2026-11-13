/* The walker on the real grid: the scan twin lidar-office-8b, whose occupancy comes from 184.6 M laser returns and no
   camera pose at all (docs/EXPERIMENTS.md, phase 6 of IMPROVEMENT-PLAN). The fixture is that twin's occupancy as the
   viewer reads it (scripts/lidar/grid_fixture.py), with the SHA-256 of the bytes and the counts the twin published.
   Run: node --experimental-strip-types test/walk-lidar.test.mts */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { GridCollider, moveCircle, type Grid } from '../src/walk/collider.ts';
import { RADIUS as R, SPEED, WHEEL, pace, recover, walkFrame, type Walker } from '../src/walk/walker.ts';
import { Navigator } from '../src/lab/places.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
const TOL = 2e-6;
const fixture = './fixtures/';
const F = JSON.parse(readFileSync(new URL(fixture + 'lidar-office-8b.json', import.meta.url), 'utf8'));
const raw = readFileSync(new URL(fixture + 'lidar-office-8b.bin', import.meta.url));
const O: [number, number] = [F.origin[0], F.origin[1]];

/** The grid as walk/scene.ts builds it from the twin's PNG: a byte a cell, darker than 128 is blocked. */
const grid = (): Grid => {
  const data = new Uint8Array(F.width * F.height);
  for (let i = 0; i < data.length; i++) data[i] = raw[i] < 128 ? 1 : 0;
  return { data, meta: { origin: O, cell: F.cell, width: F.width, height: F.height } };
};
const G = grid();
const blockedCells = G.data.reduce((n, v) => n + v, 0);

/** Distance from (x, z) to the nearest blocked cell or to the outside of the grid, worked out the slow way. */
const gap = (g: Grid, x: number, z: number) => {
  const { cell: c, width: w, height: h } = g.meta, lx = x - O[0], lz = z - O[1], ci = Math.floor(lx / c), cj = Math.floor(lz / c), n = Math.ceil(R / c) + 2;
  let d = Math.min(lx, w * c - lx, lz, h * c - lz);
  for (let j = Math.max(0, cj - n); j <= Math.min(h - 1, cj + n); j++) for (let i = Math.max(0, ci - n); i <= Math.min(w - 1, ci + n); i++) {
    if (g.data[j * w + i]) d = Math.min(d, Math.hypot(Math.max(O[0] + i * c - x, 0, x - (O[0] + i * c + c)), Math.max(O[1] + j * c - z, 0, z - (O[1] + j * c + c))));
  }
  return d;
};
const fits = (g: Grid, x: number, z: number) => Number.isFinite(x) && Number.isFinite(z) && gap(g, x, z) >= R - TOL;
/** The cell the viewer would read as blocked at that point, if any. */
const inBlocked = (g: Grid, x: number, z: number) => {
  const { cell: c, width: w, height: h } = g.meta, i = Math.floor((x - O[0]) / c), j = Math.floor((z - O[1]) / c);
  return i >= 0 && j >= 0 && i < w && j < h && g.data[j * w + i] === 1;
};
const frames = (dt: number, s = 5) => new Array<number>(Math.round(s / dt)).fill(dt);
const at = (x: number, z: number): Walker => ({ x, z, vx: 0, vz: 0 });
const TOP = pace(true, true, WHEEL);

// ---- the fixture is the twin the results dataset publishes, not a copy that drifted
const sha = createHash('sha256').update(raw).digest('hex');
check(`the fixture is the published grid (${F.width} x ${F.height} cells of ${F.cell} m, ${F.counts.free} free, ${F.counts.blocked} blocked, ${F.counts.unknown} unknown)`,
  sha === F.sha256 && raw.length === F.width * F.height && blockedCells === F.counts.blocked + F.counts.unknown);
check(`it came from ${F.source}`, /lidar-office-8b/.test(`${F.source}${F.scene}`) && F.walk?.pass === true);

// ---- starts: the twin's own spawn, then the standable cells furthest from it and from each other
const lattice: [number, number][] = [];
for (let j = 0; j < F.height; j++) for (let i = 0; i < F.width; i++) {
  const x = O[0] + (i + .5) * F.cell, z = O[1] + (j + .5) * F.cell;
  if (fits(G, x, z)) lattice.push([x, z]);
}
const pick = (from: [number, number]) => {
  let best = from, bd = -1;
  for (const p of lattice) { const d = Math.min(...starts.map(s => Math.hypot(p[0] - s[0], p[1] - s[1]))); if (d > bd) { bd = d; best = p; } }
  return best;
};
const starts: [number, number][] = [[F.spawn[0], F.spawn[1]]];
check(`the twin's spawn is a place the walker fits (${lattice.length} of ${F.width * F.height} cells fit it: ${(lattice.length * F.cell * F.cell).toFixed(2)} m2)`,
  lattice.length > 0 && fits(G, F.spawn[0], F.spawn[1]) && !inBlocked(G, F.spawn[0], F.spawn[1]));
while (starts.length < 5 && lattice.length) starts.push(pick(starts[starts.length - 1]));

// ---- the wall test: 16 headings, held hard into whatever is there for five seconds, from each start
{ const cc = new GridCollider(G), hitch = frames(.016); hitch[60] = 2;
  for (const [name, dts] of [['4 ms', frames(.004)], ['16 ms', frames(.016)], ['100 ms', frames(.1)], ['16 ms and one 2 s hitch', hitch]] as [string, number[]][]) {
    let bad = 0, pressed = 0, moved = 0, cellHits = 0, runs = 0;
    for (const [sx, sz] of starts) for (let k = 0; k < 16; k++) {
      const ang = k / 16 * 2 * Math.PI, w = at(sx, sz), spawn = { x: sx, z: sz }, x0 = sx, z0 = sz;
      let hit = false;
      for (const dt of dts) {
        const touched = walkFrame(cc, w, Math.cos(ang) * TOP, Math.sin(ang) * TOP, dt, spawn);
        hit = hit || touched;
        if (!fits(G, w.x, w.z)) bad++;
        if (inBlocked(G, w.x, w.z)) cellHits++;
      }
      runs++;
      if (hit) pressed++;
      if (Math.hypot(w.x - x0, w.z - z0) > .5) moved++;
    }
    check(`five seconds at ${TOP.toFixed(2)} m/s, frames of ${name}: ${runs} runs into the twin's walls, never in one (${bad} frames unfit, ${cellHits} in a blocked cell)`,
      bad === 0 && cellHits === 0);
    // A start is a cell the walker fits, but it can be a pocket in a desk row: some headings are blocked within a few
    // centimetres and the run ends where it began. What must hold is that the runs press hard into something, and that
    // enough of them travel to prove the walls were met at speed rather than from a standstill.
    check(`  and the runs are not trivial: ${pressed}/${runs} touched something, ${moved}/${runs} moved over half a metre`,
      pressed === runs && moved >= runs / 2);
  }
}

// ---- the move lengths Walk can ask for, on this grid: a room's width in one frame, and past the end of it
{ const cc = new GridCollider(G); let bad = 0, hit = 0;
  for (const len of [.01, .3, 1, 7, 100, 1e4, 1e9]) for (let k = 0; k < 64; k++) {
    const ang = k / 64 * 2 * Math.PI + .01; let [x, z] = starts[k % starts.length];
    for (let f = 0; f < 20; f++) { const m = moveCircle(G, x, z, Math.cos(ang) * len, Math.sin(ang) * len, R); x = m.x; z = m.z; if (m.hit) hit++; if (!fits(G, x, z) || inBlocked(G, x, z)) bad++; }
  }
  check(`448 moves of 1 cm to 1e9 m from the five starts all end where the walker fits (${hit} stopped by something)`, bad === 0 && hit > 0);
}

// ---- recover(): a walker put inside a wall or off the grid comes back to floor, never further into it
{ const cc = new GridCollider(G);
  let ok = true;
  for (const [sx, sz] of starts) {
    const [i, j] = [Math.floor((sx - O[0]) / F.cell), Math.floor((sz - O[1]) / F.cell)];
    const wallCell = (() => { for (let r = 1; r < 40; r++) for (let dj = -r; dj <= r; dj++) for (let di = -r; di <= r; di++) {
      const ii = i + di, jj = j + dj; if (ii >= 0 && jj >= 0 && ii < F.width && jj < F.height && G.data[jj * F.width + ii] === 1) return [ii, jj] as const; } return null; })();
    if (!wallCell) continue;
    const px = O[0] + (wallCell[0] + .5) * F.cell, pz = O[1] + (wallCell[1] + .5) * F.cell;
    const spawn = { x: sx, z: sz }, back = recover(cc, px, pz, spawn);
    ok = ok && !!back && fits(G, back.x, back.z) && !inBlocked(G, back.x, back.z);
    const still = at(px, pz); walkFrame(cc, still, 0, 0, .016, spawn);   // no input: a guide may be carrying it
    ok = ok && still.x === px && still.z === pz;
    const w = at(px, pz); walkFrame(cc, w, SPEED, 0, .016, spawn);      // a step: it goes back to floor first
    ok = ok && fits(G, w.x, w.z) && !inBlocked(G, w.x, w.z);
  }
  check('a walker put in a blocked cell is moved out to floor before it steps (and not while it stands still)', ok);
}

// ---- guided routes along the walk the twin ships, the way "Take me there" asks the Navigator: from the spawn to the
// far end of the published walk and to three points on the way there
{ const nav = new Navigator(G, R), cc = new GridCollider(G);
  const follow = (w: Walker, route: [number, number][]) => {
    let i = 0;
    for (let t = 0; t < 60; t += .016) {
      const tx = route[i][0] - w.x, tz = route[i][1] - w.z, d = Math.hypot(tx, tz);
      if (d < (i < route.length - 1 ? .35 : .15)) { if (++i === route.length) return true; continue; }
      const s = Math.min(SPEED, d / .016) / d; walkFrame(cc, w, tx * s, tz * s, .016, w);
      if (!fits(G, w.x, w.z)) return false;
    }
    return false;
  };
  const path = F.cameraPath.length ? F.cameraPath : [F.spawn];
  const dests = path.length > 1 ? [...path.slice(1).filter((_, i) => i % Math.max(1, Math.floor((path.length - 1) / 4)) === 0), path[path.length - 1]] : [];
  let found = 0, clear = 0, walked = 0, closest = Infinity;
  for (const to of dests) {
    const route = nav.route([F.spawn[0], F.spawn[1]], to);
    if (!route) continue;
    found++;
    let ok = true;
    for (let s = 1; s < route.length; s++) {
      const [ax, az] = route[s - 1], [bx, bz] = route[s], n = Math.ceil(Math.hypot(bx - ax, bz - az) / .01);
      for (let t = 0; t <= n; t++) {
        const x = ax + (bx - ax) * t / n, z = az + (bz - az) * t / n;
        closest = Math.min(closest, gap(G, x, z));
        if (!fits(G, x, z) || inBlocked(G, x, z)) ok = false;
      }
    }
    if (ok) clear++;
    const w = at(route[0][0], route[0][1]);
    if (follow(w, route) && Math.hypot(w.x - to[0], w.z - to[1]) < 1.6) walked++;
  }
  check(`a route from the spawn to each of ${dests.length} points along the twin's walk is found (${found}/${dests.length}), and the walk has ${path.length} points`,
    dests.length >= 2 && found === dests.length && path.length > 2);
  check(`every point of those routes fits the walker (closest ${closest.toFixed(3)} m to a wall, radius ${R})`, clear === found);
  check(`Walk follows each route to its end without getting stuck (${walked}/${found})`, walked === found);
}
process.exit(fail ? 1 : 0);
