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
import { buildWindows } from './parts/windows.js';
import { buildLectern } from './parts/lectern.js';

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

function terrazzo(THREE) {
  // speckled terrazzo, drawn into a canvas so the floor is not a flat grey
  const size = 512;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#9a9992';
  ctx.fillRect(0, 0, size, size);
  const chips = ['#6f6e68', '#c9c7bd', '#7d7f76', '#b3b1a6', '#57564f', '#d8d5c9'];
  for (let i = 0; i < 5200; i++) {
    ctx.fillStyle = chips[(Math.random() * chips.length) | 0];
    ctx.globalAlpha = 0.25 + Math.random() * 0.5;
    const r = 0.8 + Math.random() * 3.2;
    ctx.beginPath();
    ctx.arc(Math.random() * size, Math.random() * size, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 3);
  return new THREE.MeshStandardMaterial({ map: tex, roughness: 0.72, metalness: 0.02 });
}

function longDesk(THREE, w, d) {
  // the rows in this room are long dark bench-desks, not single tables
  const g = new THREE.Group();
  const topMat = new THREE.MeshStandardMaterial({ color: 0x2f2b28, roughness: 0.55 });
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x24262b, roughness: 0.6, metalness: 0.35 });
  const top = new THREE.Mesh(new THREE.BoxGeometry(w, 0.05, d), topMat);
  top.position.y = 0.745;
  const apron = new THREE.Mesh(new THREE.BoxGeometry(w * 0.94, 0.08, d * 0.7), topMat);
  apron.position.y = 0.69;
  g.add(top, apron);
  for (const sx of [-1, 1]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.74, d * 0.8), frameMat);
    leg.position.set(sx * (w / 2 - 0.08), 0.37, 0);
    g.add(leg);
  }
  const rail = new THREE.Mesh(new THREE.BoxGeometry(w * 0.9, 0.04, 0.04), frameMat);
  rail.position.y = 0.18;
  g.add(rail);
  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return g;
}

export function buildRoom(THREE, shell) {
  const s = withEnvelope(shell);
  const g = new THREE.Group();
  g.name = 'room';
  // floor: terrazzo slab under the whole envelope, plus the yellow guide line painted near the walls
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(10.9, 6.6), terrazzo(THREE));
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0.1, 0.001, -0.03);
  floor.receiveShadow = true;
  floor.name = 'floor';
  g.add(floor);
  const line = new THREE.Mesh(new THREE.PlaneGeometry(10.4, 0.08), new THREE.MeshStandardMaterial({ color: 0xd8b23a, roughness: 0.8 }));
  line.rotation.x = -Math.PI / 2;
  line.position.set(0.1, 0.004, -2.5);
  line.name = 'floor-line';
  g.add(line);
  const walls = buildWalls(THREE, s);
  walls.name = 'walls';
  const ceiling = buildCeiling(THREE, s);
  ceiling.name = 'ceiling';
  // the ceiling is lit from above by our lights, so its underside rendered dark; the capture shows bright mottled
  // white plaster. Lift every material in the ceiling group with a little self-illumination rather than adding a
  // light that would wash the room out.
  ceiling.traverse((o) => {
    if (!o.isMesh || !o.material) return;
    const m = o.material.clone();
    if (m.color) {
      m.color.lerp(new THREE.Color(0xffffff), 0.35);
      m.emissive = (m.color ? m.color.clone() : new THREE.Color(0xffffff)).multiplyScalar(0.45);
      m.emissiveIntensity = 0.9;
      m.roughness = 1;
    }
    o.material = m;
  });
  const windows = buildWindows(THREE);
  windows.name = 'windows';
  const daylight = new THREE.MeshBasicMaterial({ color: 0xcfe4f2 });   // daylight behind the grille, not a black hole
  windows.traverse((o) => {
    if (o.isMesh && /glas|pane/i.test(o.name + ((o.material && o.material.name) || ''))) o.material = daylight;
  });
  g.add(walls, ceiling, windows);
  return g;
}

export function buildObjects(THREE, shell) {
  const s = withEnvelope(shell);
  const g = new THREE.Group();
  g.name = 'objects';
  const fixed = buildFurniture(THREE, s);
  // the door leaves and their lattice glass render near-black under our lighting; lift them the way a camera does
  fixed.traverse((o) => {
    if (!o.isMesh || !o.material || !o.material.color) return;
    const c = o.material.color;
    const dark = c.r < 0.22 && c.g < 0.22 && c.b < 0.22;
    if (dark && /door|leaf|panel|timber/i.test(o.name)) {
      o.material = o.material.clone();
      o.material.color.setHex(0x4a3527);
      o.material.roughness = 0.55;
    }
  });
  fixed.name = 'furniture';
  const lectern = buildLectern(THREE);
  lectern.name = 'lectern';
  g.add(fixed, lectern);

  const objects = (s.objects || []).filter((o) => !['door', 'window'].includes((o.kind || '').toLowerCase()));
  const yaw0 = Math.atan2(ENVELOPE.walls[0].b[1] - ENVELOPE.walls[0].a[1], ENVELOPE.walls[0].b[0] - ENVELOPE.walls[0].a[0]);
  const snap = (y) => yaw0 + Math.round((y - yaw0) / (Math.PI / 2)) * (Math.PI / 2);

  // the room is a chair-desk room: every "table"/"chair" detection is one chair with its writing tablet
  const seats = objects.filter((o) => ['chair', 'table', 'desk'].includes((o.kind || '').toLowerCase()));
  const tables = objects.filter((o) => ['table', 'desk'].includes((o.kind || '').toLowerCase()));
  const desks = objects.filter((o) => ['table', 'desk'].includes((o.kind || '').toLowerCase()));
  const rowKey = (o) => Math.round(o.center[2] / 0.85) * 0.85;
  const rows = new Map();
  for (const o of desks) {
    const k = rowKey(o);
    (rows.get(k) || rows.set(k, []).get(k)).push(o);
  }
  for (const [z, row] of rows) {
    row.sort((a, b) => a.center[0] - b.center[0]);
    const n = row.length;
    const span = Math.min(8.4, Math.max(1.2, n * 1.25));
    row.forEach((o, i) => {
      o.center = [n === 1 ? o.center[0] : -span / 2 + (span * i) / (n - 1), o.center[1], z];
    });
  }
  for (const o of objects) {
    if (!['table', 'desk'].includes((o.kind || '').toLowerCase())) continue;
    const w = Math.min(2.4, Math.max(0.9, o.half[0] * 2)), d = Math.min(0.6, Math.max(0.42, o.half[2] * 2));
    const desk = longDesk(THREE, w, d);
    desk.position.set(o.center[0], 0, o.center[2]);
    desk.rotation.y = snap(0);
    desk.name = o.id;
    desk.userData = { kind: 'object', id: o.id, label: o.name, type: 'desk' };
    g.add(desk);
  }
  for (const o of seats) {
    if (['table', 'desk'].includes((o.kind || '').toLowerCase())) continue;
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
    chair.traverse((m) => {
      if (!m.isMesh || !m.material || !m.material.color) return;
      const c = m.material.color;
      if (c.b > c.r * 1.6 && c.b > c.g * 1.4) m.material = m.material.clone(), (m.material.color = new THREE.Color(0x3f63a8));
    });
    chair.name = o.id;
    chair.userData = { kind: 'object', id: o.id, label: o.name, type: 'chairdesk' };
    g.add(chair);
  }
  return g;
}
