// Classroom room module — assembled from four authored parts against the capture frames.
// The agents who built the parts all reported the same thing, which the measurements had wrong: the shell's hull is
// 10.8 x 10.0 m (it was built from the 1st-99th percentile of the splat cloud, including far floaters), but the room
// the frames show is about 10.55 x 6.26 m — the door facade sits at z = +3.10, the window wall at z = -3.16, and the
// side walls at x = -5.17 / +5.38. The walls here use that corrected envelope, so the doors, whiteboard and furniture
// land on their walls instead of floating inside them.
import { buildWalls } from './parts/walls.js';
import { buildCeiling } from './parts/ceiling.js';
import { buildFurniture } from './parts/furniture.js';
import { chairDesk, benchTable } from './parts/chairdesk.js';

export const ENVELOPE = {
  floorY: 0,
  ceilingY: 3.1,
  walls: [
    { a: [-5.17, 3.10], b: [5.38, 3.10], length: 10.55, normal: [0, -1] },
    { a: [5.38, 3.10], b: [5.38, -3.16], length: 6.26, normal: [-1, 0] },
    { a: [5.38, -3.16], b: [-5.17, -3.16], length: 10.55, normal: [0, 1] },
    { a: [-5.17, -3.16], b: [-5.17, 3.10], length: 6.26, normal: [1, 0] },
  ],
  openings: [],
};

function withEnvelope(shell) {
  return Object.assign({}, shell, ENVELOPE, { objects: shell.objects || [] });
}

export function buildRoom(THREE, shell) {
  const s = withEnvelope(shell);
  const g = new THREE.Group();
  g.name = 'room';
  const walls = buildWalls(THREE, s);
  walls.name = 'walls';
  const ceiling = buildCeiling(THREE, s);
  ceiling.name = 'ceiling';
  g.add(walls, ceiling);
  return g;
}

export function buildObjects(THREE, shell) {
  const s = withEnvelope(shell);
  const g = new THREE.Group();
  g.name = 'objects';
  const fixed = buildFurniture(THREE, s);
  fixed.name = 'furniture';
  g.add(fixed);

  const objects = (s.objects || []).filter((o) => !['door', 'window'].includes((o.kind || '').toLowerCase()));
  const yaw0 = Math.atan2(ENVELOPE.walls[0].b[1] - ENVELOPE.walls[0].a[1], ENVELOPE.walls[0].b[0] - ENVELOPE.walls[0].a[0]);
  const snap = (y) => yaw0 + Math.round((y - yaw0) / (Math.PI / 2)) * (Math.PI / 2);

  // the room is a chair-desk room: every "table"/"chair" detection is one chair with its writing tablet
  const seats = objects.filter((o) => ['chair', 'table', 'desk'].includes((o.kind || '').toLowerCase()));
  const tables = objects.filter((o) => ['table', 'desk'].includes((o.kind || '').toLowerCase()));
  for (const o of seats) {
    const [cx, , cz] = o.center;
    let z = cz;
    for (const t of tables) {
      if (t.id === o.id) continue;
      if (Math.abs(cx - t.center[0]) < t.half[0] && Math.abs(z - t.center[2]) < t.half[2]) {
        z = t.center[2] + (t.half[2] + 0.3) * (z >= t.center[2] ? 1 : -1);
        break;
      }
    }
    const chair = chairDesk(THREE);
    chair.position.set(cx, ENVELOPE.floorY, z);
    // face the front of the room (+z end, where the blackboard wall is)
    chair.rotation.y = snap(z > 0 ? Math.PI : 0);
    chair.name = o.id;
    chair.userData = { kind: 'object', id: o.id, label: o.name, type: 'chairdesk' };
    g.add(chair);
  }
  return g;
}
