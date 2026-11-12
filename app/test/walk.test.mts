/* The walker against the occupancy grid: the mover (collider.ts) and one frame of Walk (walker.ts), the code the engine
   runs every frame. Run: node --experimental-strip-types test/walk.test.mts */
import { GridCollider, moveCircle, type Grid } from '../src/walk/collider.ts';
import { RADIUS as R, REACH, SPEED, WHEEL, WHEEL_WALK, pace, recover, walkFrame, type Walker } from '../src/walk/walker.ts';
import { Navigator } from '../src/lab/places.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
let seed = 1; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
const TOL = 2e-6;  // the mover calls an overlap under a micron contact
const O: [number, number] = [-3.37, 2.11];  // every grid starts here, off the lattice on purpose; places below are metres from it

/** A grid from a rule over (column, row). */
const grid = (w: number, h: number, cell: number, blocked: (i: number, j: number) => boolean): Grid => {
  const data = new Uint8Array(w * h); for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) data[j * w + i] = blocked(i, j) ? 1 : 0;
  return { data, meta: { origin: O, cell, width: w, height: h } };
};
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
/** The same grid, counting the cells the collider reads. */
const counted = (g: Grid) => { const n = { reads: 0 }; return { n, g: { meta: g.meta, data: new Proxy(g.data as Uint8Array, { get: (t, k) => { if (typeof k === 'string' && k.charCodeAt(0) < 58) n.reads++; return Reflect.get(t, k); } }) } as Grid }; };
const frames = (dt: number, s = 5) => new Array<number>(Math.round(s / dt)).fill(dt);
const walker = (x: number, z: number): Walker => ({ x: O[0] + x, z: O[1] + z, vx: 0, vz: 0 });
/** Steer a walker through places in turn, as a guide would, slowing so it does not overshoot. Seconds taken, or -1. */
const follow = (c: GridCollider, w: Walker, route: [number, number][], v: number, dt: number, limit: number) => {
  let i = 0;
  for (let t = 0; t < limit; t += dt) {
    const tx = O[0] + route[i][0] - w.x, tz = O[1] + route[i][1] - w.z, d = Math.hypot(tx, tz);
    if (d < (i < route.length - 1 ? .35 : .15)) { if (++i === route.length) return t; continue; }
    const s = Math.min(v, d / dt) / d; walkFrame(c, w, tx * s, tz * s, dt, w);
    if (!fits(c.grid, w.x, w.z)) return -1;
  }
  return -1;
};

check('Walk is as fast as Free, with and without Shift', pace(true, false, 1) === SPEED && pace(false, false, 1) === SPEED && pace(true, true, 1) === pace(false, true, 1) && SPEED === 2.2);
check('the wheel stops at its Walk limit', pace(true, false, WHEEL) === SPEED * WHEEL_WALK && pace(true, false, 1e9) === SPEED * WHEEL_WALK && pace(false, false, WHEEL) === SPEED * WHEEL);
const TOP = pace(true, true, WHEEL);  // Shift held and the wheel all the way up

for (const cell of [.1, .05]) {
  const tag = `(${cell * 100} cm cells)`, n = (m: number) => Math.round(m / cell);
  check(`the fastest Walk frame (${(TOP * .05).toFixed(2)} m in 50 ms) fits a move's 64 sub-steps ${tag}`, TOP * .05 / (Math.min(cell, R) / 2) <= 64);

  // a 3 m room behind a wall one cell thick with open floor all round it, so only the wall can stop the walker
  const a = n(1), b = n(4), room = grid(n(5), n(5), cell, (i, j) => i >= a && i <= b && j >= a && j <= b && (i === a || i === b || j === a || j === b));
  const lo = (a + 1) * cell + R - TOL, hi = b * cell - R + TOL, mid = (a + 1 + b) * cell / 2;
  const inRoom = (x: number, z: number) => x - O[0] >= lo && x - O[0] <= hi && z - O[1] >= lo && z - O[1] <= hi && fits(room, x, z);
  let out = 0, moves = 0;
  for (const len of [.01, .3, 1, 7, 100, 1e4, 1e9, 1e300]) for (let k = 0; k < 64; k++) {
    const ang = k / 64 * 2 * Math.PI + .01; let x = O[0] + mid, z = O[1] + mid;
    for (let f = 0; f < 40; f++) { const m = moveCircle(room, x, z, Math.cos(ang) * len, Math.sin(ang) * len, R); x = m.x; z = m.z; moves++; if (!inRoom(x, z)) out++; }
  }
  check(`${moves} moves of 1 cm to 1e300 m never leave a room walled one cell thick ${tag}`, out === 0);

  // a wall of cells that only touch at their corners
  const diag = grid(n(5), n(5), cell, (i, j) => i === j); out = 0;
  for (const len of [.05, .5, 5, 1e6]) for (let k = 0; k < 64; k++) {
    const ang = k / 64 * 2 * Math.PI + .01; let x = O[0] + 3.5, z = O[1] + 1.5;
    for (let f = 0; f < 40; f++) { const m = moveCircle(diag, x, z, Math.cos(ang) * len, Math.sin(ang) * len, R); x = m.x; z = m.z; if (!fits(diag, x, z) || x - O[0] < z - O[1]) out++; }
  }
  check(`a diagonal wall one cell thick is not crossed between its cells ${tag}`, out === 0);

  // the plan's first walk test: forward into a wall for five seconds, Shift held, the wheel at its limit
  const roomC = new GridCollider(room), hitch = frames(.016); hitch[60] = 2;
  for (const [name, dts] of [['4 ms', frames(.004)], ['16 ms', frames(.016)], ['100 ms', frames(.1)], ['16 ms and one 2 s hitch', hitch]] as [string, number[]][]) {
    out = 0;
    for (let k = 0; k < 16; k++) {
      const ang = k / 16 * 2 * Math.PI, w = walker(mid, mid), spawn = { x: w.x, z: w.z };
      for (const dt of dts) { walkFrame(roomC, w, Math.cos(ang) * TOP, Math.sin(ang) * TOP, dt, spawn); if (!inRoom(w.x, w.z)) out++; }
    }
    check(`five seconds into the wall at ${TOP.toFixed(2)} m/s, frames of ${name}: still in the room ${tag}`, out === 0);
  }

  // clutter: every move from a place that fits ends in one, and the cells read per move stay bounded
  const N = n(8), blobs = new Uint8Array(N * N);
  for (let k = 0; k < 40; k++) { const bi = Math.floor(rnd() * N), bj = Math.floor(rnd() * N), bw = 1 + Math.floor(rnd() * n(.6)), bh = 1 + Math.floor(rnd() * n(.6)); for (let j = bj; j < Math.min(N, bj + bh); j++) for (let i = bi; i < Math.min(N, bi + bw); i++) if (rnd() < .9) blobs[j * N + i] = 1; }
  const clutter = grid(N, N, cell, (i, j) => blobs[j * N + i] === 1), cc = new GridCollider(clutter), open = cc.openSpot()!, { n: cnt, g: clutterN } = counted(clutter);
  let { x, z } = cc.nearestFree(open.x, open.z, R)!, bad = fits(clutter, x, z) ? 0 : 1, worst = 0, worstTop = 0;
  for (let leg = 0; leg < 400; leg++) {
    const ang = rnd() * 2 * Math.PI, len = leg % 4 ? .002 + rnd() * .3 : leg % 8 ? TOP * .05 : 10 ** (rnd() * 12);
    for (let f = 0; f < 20; f++) {
      cnt.reads = 0; const m = moveCircle(clutterN, x, z, Math.cos(ang) * len, Math.sin(ang) * len, R); x = m.x; z = m.z;
      if (!fits(clutter, x, z) || !cc.free(x, z, R)) bad++; worst = Math.max(worst, cnt.reads); if (len === TOP * .05) worstTop = Math.max(worstTop, cnt.reads);
    }
  }
  const square = (Math.floor(2 * R / cell) + 2) ** 2, bound = (1 + 64 * 5) * square;
  check(`8000 moves through clutter all end where the walker fits ${tag}`, bad === 0);
  check(`cells read per move: at most ${worstTop} at the fastest Walk frame, ${worst} at any length, under the bound of ${bound} ${tag}`, worstTop > 0 && worst <= bound);

  // a long wall on the +z side, met at 45 degrees
  const hall = grid(n(14), n(3), cell, (_i, j) => j < n(.5) || j >= n(2.5)), hallC = new GridCollider(hall), s45 = Math.SQRT1_2;
  { let x = O[0] + 1, z = O[1] + 1.5, x0 = NaN, f0 = 0; const step = SPEED * .016;
    for (let f = 0; f < 250; f++) { const m = moveCircle(hall, x, z, step * s45, step * s45, R); x = m.x; z = m.z; if (m.hit && !(x0 === x0)) { x0 = x; f0 = f; } }
    const raw = (x - x0) / ((249 - f0) * step), w = walker(1, 1.5); let x1 = 0;
    for (let f = 0; f < 250; f++) { walkFrame(hallC, w, SPEED * s45, SPEED * s45, .016, w); if (f === 124) x1 = w.x; }
    const eng = (w.x - x1) / (125 * .016) / SPEED;
    check(`along a wall met at 45 degrees: ${raw.toFixed(4)} of the asked speed from the mover, ${eng.toFixed(4)} in a Walk frame (cos 45 = 0.7071) ${tag}`, Math.abs(raw - s45) < .002 && Math.abs(eng - s45) < .002 && fits(hall, w.x, w.z)); }
  { const w = walker(1, 2.5 - R), clear = walker(1, 1.5); let hit = false; for (let f = 0; f < 120; f++) { hit = walkFrame(hallC, w, SPEED, 0, .016, w) || hit; walkFrame(hallC, clear, SPEED, 0, .016, clear); }
    check(`touching a wall and walking along it costs nothing ${tag}`, !hit && w.x - O[0] > 4 && Math.abs(w.x - clear.x) < 1e-9); }

  // a diagonal wall as the grid draws it, in steps: the old mover stopped dead at each one
  let slowest = Infinity, stalls = 0;
  for (const [run, dir] of [[1, 1], [1, -1], [2, 1], [2, -1], [3, 1], [3, -1]]) {
    const stairs = grid(n(14), n(14), cell, (i, j) => j >= Math.floor(i / run) + n(2)), sc = new GridCollider(stairs);
    const tl = Math.hypot(run, 1), tx = dir * run / tl, tz = dir / tl, nx = -1 / tl, nz = run / tl, s0 = dir > 0 ? 1.5 : 10;
    for (const deg of [10, 30, 45]) for (const dt of [.004, .016, .05]) {
      const ang = deg * Math.PI / 180, ux = tx * Math.cos(ang) + nx * Math.sin(ang), uz = tz * Math.cos(ang) + nz * Math.sin(ang);
      const w = walker(s0, s0 / run + 2 - .6); let t0 = -1, p0 = 0, still = 0, x = w.x, z = w.z, r0 = -1, q0 = 0, rstill = 0; const T = 3;
      for (let t = 0; t < T; t += dt) {
        const bx = w.x, bz = w.z, hit = walkFrame(sc, w, ux * SPEED, uz * SPEED, dt, w); if (hit && t0 < 0) { t0 = t; p0 = w.x * tx + w.z * tz; }
        if (t0 >= 0) { still = Math.hypot(w.x - bx, w.z - bz) < 1e-9 ? still + 1 : 0; if (still === 3) stalls++; }
        const m = moveCircle(stairs, x, z, ux * SPEED * dt, uz * SPEED * dt, R); if (m.hit && r0 < 0) { r0 = t; q0 = m.x * tx + m.z * tz; }
        if (r0 >= 0) { rstill = Math.hypot(m.x - x, m.z - z) < 1e-9 ? rstill + 1 : 0; if (rstill === 3) stalls++; } x = m.x; z = m.z;
      }
      const ideal = SPEED * Math.cos(ang);
      slowest = Math.min(slowest, (w.x * tx + w.z * tz - p0) / (T - t0) / ideal, (x * tx + z * tz - q0) / (T - r0) / ideal);
      if (t0 < 0 || r0 < 0 || !fits(stairs, w.x, w.z) || !fits(stairs, x, z)) stalls++;
    }
  }
  check(`along stepped diagonal walls (1, 2 and 3 cells a step, pushed 10 to 45 degrees into them): never stopped, slowest ${slowest.toFixed(2)} of the speed a straight wall would give ${tag}`, stalls === 0 && slowest > .5);

  // two rooms and a doorway 0.7 m wide in a wall 20 cm thick
  const two = (door: number) => grid(n(8), n(4), cell, (i, j) => i >= n(4) && i < n(4.2) && !(j >= n(1.6) && j < n(1.6) + n(door)));
  const doors = two(.7), dc = new GridCollider(doors), centre = 1.6 + .35;
  for (const off of [0, .1, -.1]) {
    const w = walker(2, centre + off); let hit = false, t = 0; for (; t < 3 && w.x < O[0] + 6; t += .016) hit = walkFrame(dc, w, SPEED, 0, .016, w) || hit;
    check(`straight through the doorway ${off ? `${Math.abs(off) * 100} cm off its middle` : 'down its middle'}: 4 m in ${t.toFixed(2)} s, nothing touched ${tag}`, !hit && w.x >= O[0] + 6 && t < 2);
  }
  { const w = walker(1, .6), t = follow(dc, w, [[7, 3.4]], SPEED, .016, 8), free = Math.hypot(6, 2.8) / SPEED;
    check(`through the doorway at an angle, round the jamb: ${t.toFixed(2)} s against ${free.toFixed(2)} s with no wall ${tag}`, t > 0 && t < free * 1.25 + .3); }
  { const narrow = new GridCollider(two(.4)), w = walker(2, 1.6 + .2); for (let f = 0; f < 300; f++) walkFrame(narrow, w, SPEED, 0, .016, w);
    check(`a doorway 0.4 m wide is too narrow for a walker ${2 * R} m wide ${tag}`, w.x < O[0] + 4 && fits(two(.4), w.x, w.z)); }

  // the plan's second walk test: a guided route between two places, through the doorway, aimed carelessly close to a jamb
  const route: [number, number][] = [[4.1, centre + .2], [7.4, .6]], length = Math.hypot(3.5, 3.4 - centre - .2) + Math.hypot(3.3, centre + .2 - .6);
  for (const [v, dt] of [[SPEED, .004], [SPEED, .016], [SPEED, .1], [TOP, .016], [TOP, .05]]) {
    const w = walker(.6, 3.4), t = follow(dc, w, route, v, dt, 20);
    check(`a guided route through the doorway at ${v.toFixed(1)} m/s, ${dt * 1000} ms frames: arrives in ${t.toFixed(2)} s (${(length / v).toFixed(2)} s with no wall) ${tag}`, t > 0 && t < length / v * 1.5 + 1 && Math.hypot(w.x - O[0] - 7.4, w.z - O[1] - .6) < .15);
  }

  // edge cases: on a cell edge, on and inside a blocked cell, lengths that are not lengths
  { const e = O[0] + n(2) * cell, m1 = moveCircle(room, e, O[1] + mid, .03, .02, R), m2 = moveCircle(room, e, O[1] + mid, .03, .02, R);
    check(`from exactly on a cell edge: a plain move, the same twice ${tag}`, m1.x === m2.x && m1.z === m2.z && !m1.hit && Math.abs(m1.x - e - .03) < 1e-12 && Math.abs(m1.z - O[1] - mid - .02) < 1e-12);
    const starts: [number, number][] = [[a * cell, mid], [a * cell, a * cell], [(a + .5) * cell, mid], [(a + 1) * cell, mid], [(a + 1) * cell + R - .01, mid], [-1, -1], [1e12, 0]]; let ok = true;
    for (const [sx, sz] of starts) {
      const px = O[0] + sx, pz = O[1] + sz, m = moveCircle(room, px, pz, .4, .3, R), p = roomC.nearestFree(px, pz, R), q = roomC.nearestFree(px, pz, R);
      ok = ok && m.x === px && m.z === pz && m.hit && !roomC.free(px, pz, R) && !!p && !!q && p.x === q.x && p.z === q.z && fits(room, p.x, p.z);
    }
    check(`from on or inside a blocked cell, or outside the grid: no move, and the nearest place that fits is the same every time ${tag}`, ok);
    const p = roomC.nearestFree(O[0] + (a + 1) * cell + R - .01, O[1] + mid, R)!;
    check(`a centimetre into a wall's reach, the nearest place is a centimetre away ${tag}`, Math.abs(p.x - O[0] - (a + 1) * cell - R) < 1e-9 && p.z === O[1] + mid);
    ok = true; for (const [dx, dz] of [[0, 0], [NaN, 1], [1, NaN], [Infinity, 0], [-Infinity, Infinity]]) { const m = moveCircle(room, O[0] + mid, O[1] + mid, dx, dz, R); ok = ok && m.x === O[0] + mid && m.z === O[1] + mid && !m.hit; }
    check(`a move of zero, NaN or infinity goes nowhere and returns no NaN ${tag}`, ok && !roomC.free(NaN, 0, R) && roomC.nearestFree(NaN, 0, R) === null && roomC.nearestFree(0, Infinity, R) === null); }

  // a walker inside a wall: the nearest place that fits, else the spawn point
  const pocket = grid(n(16), n(4), cell, (i, j) => !(i >= n(.5) && i < n(2.5) && j >= n(.5) && j < n(3.5))), pc = new GridCollider(pocket), spawn = { x: O[0] + 1.5, z: O[1] + 2 };
  { const near = recover(pc, O[0] + 3, O[1] + 2, spawn)!, far = recover(pc, O[0] + 12, O[1] + 2, spawn)!, lost = recover(pc, O[0] + 15, O[1] + 2, { x: O[0] + 9, z: O[1] + 2 })!;
    check(`inside a wall: back to the nearest place that fits (${Math.hypot(near.x - O[0] - 3, near.z - O[1] - 2).toFixed(2)} m away) ${tag}`, fits(pocket, near.x, near.z) && Math.hypot(near.x - O[0] - 3, near.z - O[1] - 2) < .5 + R + cell);
    check(`nowhere fits within ${REACH} m: back to the spawn point ${tag}`, pc.nearestFree(O[0] + 12, O[1] + 2, R, REACH) === null && far.x === spawn.x && far.z === spawn.z);
    check(`the spawn point is blocked too: the nearest place to it that fits ${tag}`, fits(pocket, lost.x, lost.z) && lost.x > O[0] + 2.2);
    const w = walker(3, 2), still = walker(3, 2); walkFrame(pc, w, -SPEED, 0, .016, spawn); walkFrame(pc, still, 0, 0, .016, spawn);
    check(`a frame that starts inside a wall puts the walker back before it steps; one with no input leaves it to its guide ${tag}`, fits(pocket, w.x, w.z) && w.x < O[0] + 2.5 && still.x === O[0] + 3 && still.z === O[1] + 2);
    const { n: reads, g: solid } = counted(grid(n(6), n(6), cell, () => true)), sc = new GridCollider(solid), s = { x: O[0] + 3, z: O[1] + 3 }, first = recover(sc, s.x, s.z, s); reads.reads = 0;
    const v = walker(3, 3); for (let f = 0; f < 10; f++) walkFrame(sc, v, SPEED, SPEED, .016, s);
    check(`a grid with no place for the walker: it stays put, and the grid is not searched again (${reads.reads / 10} cells read a frame) ${tag}`, first === null && v.x === s.x && v.z === s.z && reads.reads <= 10 * 40 * square); }

  { const big = grid(n(8), n(4), cell, (i, j) => i === 0 || j === 0 || i === n(8) - 1 || j === n(4) - 1 || (i >= n(2) && i < n(2.2))), p = new GridCollider(big).openSpot()!;
    check(`the most open place is in the larger room, ${gap(big, p.x, p.z).toFixed(2)} m from its walls ${tag}`, p.x - O[0] > 2.2 && gap(big, p.x, p.z) > 1.8 && new GridCollider(grid(4, 4, cell, () => true)).openSpot() === null); }
  // the plan's guided-route test: "Take me there" between places in two rooms finishes without getting stuck. The route
  // is the one Places.tsx asks the Navigator for; the guide moves the visitor along it directly, so every point of it
  // must be a place the walker fits, or Walk starts the next frame inside a wall
  { const W = n(9), H = n(5), wall = n(4.5), door = (lo: number, hi: number) => (j: number) => j >= n(lo) && j < n(hi);
    for (const [label, opening] of [['a 0.7 m door', door(2, 2.7)], ['a 1.2 m door', door(1.5, 2.7)]] as const) {
      const rooms = grid(W, H, cell, (i, j) => i === 0 || j === 0 || i === W - 1 || j === H - 1 || (i === wall && !opening(j)));
      const nav = new Navigator(rooms as never), gc = new GridCollider(rooms);
      // a place in the open, one against the far wall, and one in a corner, as the labelling stage pins them
      const from: [number, number] = [O[0] + 1.5, O[1] + 2.5], ends: [number, number][] = [[O[0] + 7, O[1] + 3], [O[0] + 8.95, O[1] + 2.5], [O[0] + 8.9, O[1] + .1]];
      let routed = 0, clear = 0, walked = 0, closest = Infinity;
      for (const to of ends) {
        const path = nav.route(from, to); if (!path) continue; routed++;
        let ok = true;
        for (let s = 1; s < path.length; s++) {
          const [ax, az] = path[s - 1], [bx, bz] = path[s], k = Math.ceil(Math.hypot(bx - ax, bz - az) / .01);
          for (let t = 0; t <= k; t++) { const x = ax + (bx - ax) * t / k, z = az + (bz - az) * t / k; closest = Math.min(closest, gap(rooms, x, z)); if (!fits(rooms, x, z)) ok = false; }
        }
        if (ok) clear++;
        const w: Walker = { x: path[0][0], z: path[0][1], vx: 0, vz: 0 };
        if (follow(gc, w, path.map(([x, z]) => [x - O[0], z - O[1]] as [number, number]), SPEED, .016, 20) >= 0) walked++;
      }
      check(`a guided route through ${label} to each of three places is found (${routed}/3) ${tag}`, routed === 3);
      check(`every point of those routes fits the walker (closest ${closest.toFixed(3)} m to a wall, radius ${R}) ${tag}`, clear === routed && routed === 3);
      check(`Walk follows each route to its end without getting stuck (${walked}/3) ${tag}`, walked === 3);
    }
  }
}
process.exit(fail ? 1 : 0);
