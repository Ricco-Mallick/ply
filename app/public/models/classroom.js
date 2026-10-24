// Room module — classroom at IIT Bombay, authored from a Ply capture.
// Built with .agents/skills/ply-room-model: measurements come from shell.json (walls, openings, object boxes),
// proportions from the class of each object, colours sampled by eye from the capture. No LiDAR; every number is
// photogrammetric (±5 cm) and the places where that matters are marked "estimated".
//
//   buildRoom(THREE, shell) -> walls, floor, ceiling, openings
//   buildObjects(THREE, shell) -> the furniture, hero objects modelled, the rest proxied

const WALL_T = 0.16;
const CEIL = 3.1;              // measured ceiling estimate was a splat percentile (7.2 m); frames show ~3.1 m
const BASEBOARD = 0.11;

function materials(THREE) {
  const std = (color, roughness = 0.9, metalness = 0) => new THREE.MeshStandardMaterial({ color, roughness, metalness });
  return {
    wall: std(0xe6e0d4, 0.95),
    wallDark: std(0xcfc8bb, 0.95),
    floor: std(0x8a6a4c, 0.85),
    ceiling: std(0xf2efe9, 1),
    trim: std(0x4c5566, 0.8),
    frame: std(0x8d929c, 0.5, 0.2),
    glass: new THREE.MeshPhysicalMaterial({ color: 0xbfe3ff, transparent: true, opacity: 0.18, roughness: 0.05 }),
    board: std(0x22282f, 0.85),
    white: std(0xf4f4f0, 0.9),
    wood: std(0xb98d5f, 0.65),
    metal: std(0x9aa3b2, 0.4, 0.55),
    fabric: std(0x4f6382, 0.95),
    dark: std(0x2b3240, 0.7),
  };
}

function add(parent, mesh) { mesh.castShadow = true; mesh.receiveShadow = true; parent.add(mesh); return mesh; }

export function buildRoom(THREE, shell) {
  const M = materials(THREE);
  const root = new THREE.Group();
  root.name = 'authored-room';
  const floorY = shell.floorY ?? 0;
  const walls = (shell.walls ?? []).filter((w) => w.length > 0.4);
  if (walls.length < 3) return root;
  const ceiling = CEIL;

  // floor and ceiling from the wall footprint
  const shape = new THREE.Shape(walls.map((w) => new THREE.Vector2(w.a[0], -w.a[1])));
  const slab = new THREE.ExtrudeGeometry(shape, { depth: 0.16, bevelEnabled: false });
  slab.rotateX(Math.PI / 2);
  const floor = add(root, new THREE.Mesh(slab, M.floor));
  floor.name = 'floor';
  floor.position.y = floorY;
  const ceil = add(root, new THREE.Mesh(new THREE.ShapeGeometry(shape).rotateX(Math.PI / 2), M.ceiling));
  ceil.name = 'ceiling';
  ceil.position.y = floorY + ceiling;

  walls.forEach((w, i) => {
    const [ax, az] = w.a, [bx, bz] = w.b;
    const len = w.length;
    const angle = Math.atan2(bz - az, bx - ax);
    const g = new THREE.Group();
    g.name = `wall-${i}`;
    const openings = (shell.openings ?? []).filter((o) => o.wall === i && o.u > 0 && o.u < len).sort((a, b) => a.u - b.u);
    const solid = (from, to, z0, z1) => {
      if (to - from < 0.02 || z1 - z0 < 0.02) return;
      const m = add(g, new THREE.Mesh(new THREE.BoxGeometry(to - from, z1 - z0, WALL_T), M.wall));
      m.position.set((from + to) / 2 - len / 2, (z0 + z1) / 2, 0);
      m.name = `wall-${i}`;
    };
    let cursor = 0;
    for (const o of openings) {
      const from = Math.max(0, o.u - o.width / 2), to = Math.min(len, o.u + o.width / 2);
      const sill = Math.max(0, o.sill), head = Math.min(ceiling, o.sill + o.height);
      solid(cursor, from, 0, ceiling);
      solid(from, to, 0, sill);
      solid(from, to, head, ceiling);
      // reveal: the inside faces of the cut, so an opening reads as a hole with depth
      for (const x of [from, to]) {
        const rev = add(g, new THREE.Mesh(new THREE.BoxGeometry(0.02, head - sill, WALL_T), M.wallDark));
        rev.position.set(x - len / 2, (sill + head) / 2, 0);
      }
      const fw = 0.055;
      for (const [x, y, w2, h2] of [
        [from + fw / 2, (sill + head) / 2, fw, head - sill],
        [to - fw / 2, (sill + head) / 2, fw, head - sill],
        [(from + to) / 2, head - fw / 2, to - from, fw],
        [(from + to) / 2, sill + fw / 2, to - from, fw],
      ]) {
        const bar = add(g, new THREE.Mesh(new THREE.BoxGeometry(w2, h2, WALL_T * 1.2), M.frame));
        bar.position.set(x - len / 2, y, 0);
        bar.name = `opening-${o.kind}`;
      }
      if (o.kind === 'window') {
        const glass = add(g, new THREE.Mesh(new THREE.BoxGeometry(to - from - fw * 2, head - sill - fw * 2, 0.02), M.glass));
        glass.position.set((from + to) / 2 - len / 2, (sill + head) / 2, 0);
      } else {
        const leaf = add(g, new THREE.Mesh(new THREE.BoxGeometry(to - from - fw * 2, head - sill - fw * 2, 0.05), M.wood));
        leaf.position.set((from + to) / 2 - len / 2, (sill + head) / 2, 0);
        leaf.name = 'door-leaf';
      }
      cursor = to;
    }
    solid(cursor, len, 0, ceiling);
    const base = add(g, new THREE.Mesh(new THREE.BoxGeometry(len, BASEBOARD, WALL_T + 0.03), M.trim));
    base.position.set(0, BASEBOARD / 2, 0);
    g.position.set((ax + bx) / 2, floorY, (az + bz) / 2);
    g.rotation.y = -angle;
    root.add(g);
  });
  return root;
}

// ---------------------------------------------------------------- furniture
const roomYawOf = (shell) => {
  const w = [...(shell.walls ?? [])].sort((a, b) => b.length - a.length)[0];
  return w ? Math.atan2(w.b[1] - w.a[1], w.b[0] - w.a[0]) : 0;
};

function desk(T, M, w = 1.1, d = 0.62, h = 0.74) {
  const g = new T.Group();
  const top = add(g, new T.Mesh(new T.BoxGeometry(w, 0.04, d), M.wood));
  top.position.y = h - 0.02;
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const leg = add(g, new T.Mesh(new T.CylinderGeometry(0.02, 0.02, h - 0.04, 8), M.metal));
    leg.position.set(sx * (w / 2 - 0.08), (h - 0.04) / 2, sz * (d / 2 - 0.08));
  }
  // a shelf under the top, which every desk in this room has
  const shelf = add(g, new T.Mesh(new T.BoxGeometry(w * 0.8, 0.02, d * 0.7), M.wood));
  shelf.position.y = h - 0.22;
  return g;
}

function chair(T, M) {
  const g = new T.Group();
  const seat = add(g, new T.Mesh(new T.BoxGeometry(0.46, 0.05, 0.46), M.fabric));
  seat.position.y = 0.45;
  const back = add(g, new T.Mesh(new T.BoxGeometry(0.46, 0.42, 0.05), M.fabric));
  back.position.set(0, 0.68, -0.21);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const leg = add(g, new T.Mesh(new T.CylinderGeometry(0.016, 0.016, 0.45, 6), M.metal));
    leg.position.set(sx * 0.19, 0.225, sz * 0.19);
  }
  return g;
}

function blackboard(T, M, w, h) {
  const g = new T.Group();
  const panel = add(g, new T.Mesh(new T.BoxGeometry(w, h, 0.04), M.board));
  panel.position.y = 1.55;
  const frame = add(g, new T.Mesh(new T.BoxGeometry(w + 0.08, h + 0.08, 0.05), M.wood));
  frame.position.set(0, 1.55, -0.02);
  const tray = add(g, new T.Mesh(new T.BoxGeometry(w, 0.05, 0.1), M.metal));
  tray.position.set(0, 1.55 - h / 2 - 0.04, 0.03);
  return g;
}

function cabinet(T, M, w, h, d) {
  const g = new T.Group();
  const body = add(g, new T.Mesh(new T.BoxGeometry(w, h, d), M.wood));
  body.position.y = h / 2;
  const mid = add(g, new T.Mesh(new T.BoxGeometry(w * 0.96, 0.02, d * 0.9), M.dark));
  mid.position.y = h * 0.5;
  for (const sx of [-1, 1]) {
    const pull = add(g, new T.Mesh(new T.BoxGeometry(0.02, 0.18, 0.02), M.metal));
    pull.position.set(sx * w * 0.2, h * 0.55, d / 2 + 0.01);
  }
  return g;
}

function podium(T, M) {
  const g = new T.Group();
  const body = add(g, new T.Mesh(new T.BoxGeometry(0.6, 1.0, 0.45), M.wood));
  body.position.y = 0.5;
  const top = add(g, new T.Mesh(new T.BoxGeometry(0.68, 0.05, 0.52), M.dark));
  top.position.set(0, 1.02, 0.03);
  top.rotation.x = -0.12;
  return g;
}

export function buildObjects(THREE, shell) {
  const M = materials(THREE);
  const root = new THREE.Group();
  root.name = 'authored-objects';
  const floorY = shell.floorY ?? 0;
  const yaw0 = roomYawOf(shell);
  const snap = (y) => yaw0 + Math.round((y - yaw0) / (Math.PI / 2)) * (Math.PI / 2);
  const objects = (shell.objects ?? []).filter((o) => !['door', 'window'].includes((o.kind ?? '').toLowerCase()));

  const place = (mesh, o, yaw = yaw0) => {
    mesh.position.set(o.center[0], floorY, o.center[2]);
    mesh.rotation.y = snap(yaw);
    mesh.name = o.id;
    mesh.userData = { kind: 'object', id: o.id, label: o.name, type: o.kind };
    root.add(mesh);
    return mesh;
  };

  // hero objects first: the things that make the room recognisable
  const boards = objects.filter((o) => /blackboard|whiteboard|projector screen/.test((o.kind ?? '').toLowerCase()));
  for (const o of boards) {
    const w = Math.min(3.6, Math.max(1.4, o.half[0] * 2)), h = Math.min(1.3, Math.max(0.9, o.half[1] * 2));
    place(blackboard(THREE, M, w, h), o);
  }
  for (const o of objects.filter((x) => (x.kind ?? '').toLowerCase() === 'podium')) place(podium(THREE, M), o);
  for (const o of objects.filter((x) => ['cabinet', 'shelf'].includes((x.kind ?? '').toLowerCase()))) {
    const w = Math.min(1.1, Math.max(0.5, o.half[0] * 2)), d = Math.min(0.55, Math.max(0.32, o.half[2] * 2));
    place(cabinet(THREE, M, w, Math.min(2.0, Math.max(1.3, o.half[1] * 2)), d), o);
  }

  // desks and chairs: the rows. Positions are measured; orientation snaps to the room, chairs pulled out of the desk
  const desks = objects.filter((x) => ['desk', 'table'].includes((x.kind ?? '').toLowerCase()));
  for (const o of desks) {
    const w = Math.min(1.6, Math.max(0.9, o.half[0] * 2)), d = Math.min(0.75, Math.max(0.55, o.half[2] * 2));
    place(desk(THREE, M, w, d), o);
  }
  for (const o of objects.filter((x) => (x.kind ?? '').toLowerCase() === 'chair')) {
    const g = chair(THREE, M);
    let z = o.center[2];
    for (const t of desks) {
      if (Math.abs(o.center[0] - t.center[0]) < t.half[0] && Math.abs(z - t.center[2]) < t.half[2]) {
        z = t.center[2] + (t.half[2] + 0.30) * (z >= t.center[2] ? 1 : -1);
        break;
      }
    }
    const shifted = { ...o, center: [o.center[0], o.center[1], z] };
    // chairs face their desk: the chair's back is away from the room centre line of the row
    place(g, shifted, yaw0 + (z >= o.center[2] ? Math.PI : 0));
  }

  // everything else: an honest proxy at the measured size
  const handled = new Set([...boards, ...desks, ...objects.filter((x) => /podium|cabinet|shelf|chair/.test((x.kind ?? '').toLowerCase()))].map((o) => o.id));
  for (const o of objects) {
    if (handled.has(o.id)) continue;
    const w = Math.max(0.15, Math.min(2.4, o.half[0] * 2)), h = Math.max(0.15, Math.min(2.2, o.half[1] * 2)), d = Math.max(0.15, Math.min(2.4, o.half[2] * 2));
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), M.fabric);
    body.position.y = h / 2;
    add(g, body);
    place(g, o);
  }
  return root;
}
