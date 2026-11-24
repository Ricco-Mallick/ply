/* Several floors: stair heights, the floor-aware collider, and routes across floors. Run:
   node --experimental-strip-types test/levels.test.mts */
import { LevelCollider, levelAt, routeAcross, stairAt, stairChain, type Stair } from '../src/walk/levels.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
const up: Stair = { from: 0, to: 1, path: [[0, 0, 0], [0, 1.6, 2], [0, 3.2, 4]] };
const up2: Stair = { from: 1, to: 2, path: [[5, 3.2, 4], [5, 6.4, 8]] };
const levels = [{ index: 0, name: 'Floor 1', floorY: 0 }, { index: 1, name: 'Floor 2', floorY: 3.2 }, { index: 2, name: 'Floor 3', floorY: 6.4 }];

check('half way up the stairs is half a storey up', Math.abs(stairAt([up], 0.1, 2)!.y - 1.6) < 1e-9);
check('a metre off the stairs is not on them', stairAt([up], 1.2, 2) === null);
check('feet at 3.25 m stand on the second floor', levelAt(levels, 3.25) === 1);
check('feet just under a floor (rounding) still count as on it', levelAt(levels, 3.0) === 1 && levelAt(levels, 0) === 0);
check('two flights from the ground to the third floor', JSON.stringify(stairChain([up, up2], 0, 2)?.map(([s, u]) => [s.from, s.to, u])) === '[[0,1,true],[1,2,true]]');
check('going down takes the flights the other way', JSON.stringify(stairChain([up, up2], 2, 0)?.map(([s, u]) => [s.from, s.to, u])) === '[[1,2,false],[0,1,false]]');
check('no stairs between floors is no chain', stairChain([up], 0, 2) === null);

const wall = { free: () => false, move: (x: number, z: number) => ({ x, z, hit: true }), nearestFree: () => null, openSpot: () => null };
const c = new LevelCollider(wall, [up], 0);
check('the grid blocks everything, but the stairs are walkable', c.free(0, 1, 0.2) && !c.free(3, 3, 0.2));
const m = c.move(0, 1, 0, 0.5, 0.2);
check('moving along the stairs is not blocked', !m.hit && Math.abs(m.z - 1.5) < 1e-9);
check('a floor with no stairs of its own does not use the others', !new LevelCollider(wall, [up2], 0).free(5, 5, 0.2));

const straight = () => (a: [number, number], b: [number, number]) => [a, b] as [number, number][];
const r = routeAcross(straight, [up], { level: 0, at: [3, -2] }, { level: 1, at: [2, 6] })!;
check('a route upstairs walks to the stairs, up them, then on', r[0].level === 0 && r.some(p => p.level === -1 && p.y === 3.2) && r[r.length - 1].level === 1 && r[r.length - 1].x === 2);
check('the same floor is one leg', routeAcross(straight, [up], { level: 0, at: [0, 0] }, { level: 0, at: [1, 1] })!.length === 2);
check('a floor with no route is no route', routeAcross(() => null, [up], { level: 0, at: [0, 0] }, { level: 1, at: [1, 1] }) === null);
process.exit(fail ? 1 : 0);
