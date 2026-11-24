/* Rooms from a walking grid. Run: node --experimental-strip-types test/rooms.test.mts */
import { findRooms, mainAxis } from '../src/walk/rooms.ts';

let fail = 0; const check = (name: string, ok: boolean, extra = '') => { console.log(ok ? 'ok  ' : 'FAIL', name, ok ? '' : extra); if (!ok) fail++; };
/** A plan drawn in a 10 cm grid: '#' wall, '.' floor; returned as the viewer's grid (1 blocked). */
function plan(w: number, h: number, draw: (i: number, j: number) => boolean) {
  const data = new Uint8Array(w * h); for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) data[j * w + i] = draw(i, j) ? 1 : 0;
  return { data, meta: { origin: [0, 0] as [number, number], cell: 0.1, width: w, height: h } };
}
// two rooms, 5 x 4 m and 3 x 4 m, a 0.25 m wall between them with a 0.9 m door
const two = plan(84, 44, (i, j) => i < 2 || j < 2 || i > 81 || j > 41 || (i >= 52 && i < 55 && !(j >= 18 && j < 27)));
const r = findRooms(two);
check('two rooms', r.rooms.length === 2, JSON.stringify(r.rooms.map(x => x.areaM2)));
check('the bigger is first and about 20 m²', Math.abs(r.rooms[0].areaM2 - 20) < 1.5, String(r.rooms[0]?.areaM2));
check('the smaller about 10.8 m²', Math.abs(r.rooms[1]?.areaM2 - 10.8) < 1.5, String(r.rooms[1]?.areaM2));
check('one door between them, in the doorway', r.rooms[0].doors.length === 1 && Math.abs(r.rooms[0].doors[0].at[0] - 5.35) < 0.4 && Math.abs(r.rooms[0].doors[0].at[1] - 2.25) < 0.4, JSON.stringify(r.rooms[0].doors));
check('a rectangle room has a four-corner outline', r.rooms[1].outline.length === 4, JSON.stringify(r.rooms[1].outline));
check('axis-aligned walls read as 0 degrees', Math.min(r.axis, Math.PI / 2 - r.axis) < 0.05, String(r.axis));
// one open room with a pillar: still one room
const one = plan(60, 40, (i, j) => i < 2 || j < 2 || i > 57 || j > 37 || (i >= 28 && i < 32 && j >= 18 && j < 22));
check('one room with a pillar stays one room', findRooms(one).rooms.length === 1);
// a building turned 30 degrees: the walls' direction is found
const c = Math.cos(Math.PI / 6), s = Math.sin(Math.PI / 6);
const turned = plan(120, 120, (i, j) => { const x = (i - 60) * c + (j - 60) * s, z = -(i - 60) * s + (j - 60) * c; return Math.abs(x) > 35 || Math.abs(z) > 25; });
check('a building turned 30 degrees reads as 30 degrees', Math.abs(mainAxis(turned) - Math.PI / 6) < 0.05, String(mainAxis(turned) * 180 / Math.PI));
process.exit(fail ? 1 : 0);
