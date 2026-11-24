/* The machine-readable twin summary. Run: node --experimental-strip-types test/summary.test.mts */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { summarize, summaryText } from '../src/summary.ts';

const input = {
  id: 'chapel-p7-1', site: 'https://ricco.codes/ply', api: 'https://api.test',
  scene: { floorY: 0, scaleSource: 'ar-poses', bounds: { min: [-8, -0.3, -3.7], max: [10.5, 6.6, 4.3] } },
  occupancy: { cellSize: 0.05, counts: { free: 40000 } },
  shell: { ceilingY: 6.37, walls: [{ length: 18.6 }, { length: 7.1 }] },
  places: { places: [
    { id: 'altar', name: 'Altar', cat: 'Things', p: [0, 0.9, -6], pinned: true, info: { note: 'Mass at 7', link: 'https://example.org' } },
    { id: 'door', name: 'Main door', cat: 'Exits', p: [3, 1, -2], pinned: true },
    { id: 'bad', name: 'Broken', cat: 'Things', p: [0, 'x', 0] },
  ] },
};

test('the summary carries size, walkable floor, places with links and distances', () => {
  const j = summarize(input as any);
  assert.deepEqual(j.sizeM, { x: 18.5, y: 6.9, z: 8 });
  assert.equal(j.ceilingM, 6.37);
  assert.equal(j.walkableM2, 100);
  assert.equal(j.places.length, 2, 'a place with a broken position is left out');
  assert.equal(j.places[0].note, 'Mass at 7');
  assert.match(j.places[0].open, /^https:\/\/ricco\.codes\/ply\/walk\.html\?scene=https%3A%2F%2Fapi\.test%2Fv1%2Fscenes%2Fchapel-p7-1%2Fscene\.json&place=altar$/);
  assert.deepEqual(j.distances, [{ from: 'Altar', to: 'Main door', straightM: 5 }]);
  assert.equal(j.scale.note, 'measured by the phone (ARKit / ARCore tracking)');
});

test('the text version reads as sentences and lists every place', () => {
  const t = summaryText(summarize(input as any));
  assert.match(t, /Walkable floor: about 100 m²\./);
  assert.match(t, /- Altar \(Things\) at x 0, z -6, 0\.9 m up/);
  assert.match(t, /Altar to Main door: 5 m/);
});

test('a twin with nothing but a scene still summarizes', () => {
  const j = summarize({ id: 'x', site: 's', api: 'a', scene: {}, occupancy: null, shell: null, places: null });
  assert.equal(j.places.length, 0); assert.equal(j.sizeM, null); assert.equal(j.walkableM2, null);
  assert.match(summaryText(j), /None tagged yet/);
});
