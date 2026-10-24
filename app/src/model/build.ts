// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Geometry for the semantic model: a wall is a solid with a sill and a head, an opening is a frame with glass or a
   door leaf, and each labelled object becomes a small proxy built to its measured size — a table with legs, a chair
   with a back, a shelf with boards. Proxies are boxes at heart: what they carry is the object's identity, size and
   position, which is what an editor and an export need. */
import * as THREE from 'three';
import { PALETTE, type Obj, type Shell } from './types';

const WALL_T = 0.14;   // wall thickness, metres
const mat = (color: number, opts: THREE.MeshStandardMaterialParameters = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.82, metalness: 0.02, ...opts });

const mesh = (geo: THREE.BufferGeometry, material: THREE.Material, name: string) => {
  const m = new THREE.Mesh(geo, material);
  m.name = name;
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
};

/** A box with its outline drawn, so a proxy reads as a modelled object rather than a blob. */
function outlined(m: THREE.Mesh, color = 0x10151f, opacity = 0.35) {
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry, 24), new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  edges.position.copy(m.position);
  edges.rotation.copy(m.rotation);
  edges.scale.copy(m.scale);
  edges.name = `${m.name}:edges`;
  return edges;
}

export function buildRoom(shell: Shell): THREE.Group {
  const root = new THREE.Group();
  root.name = 'room';
  const floorY = shell.floorY ?? 0;
  const ceiling = shell.ceilingY ?? 2.8;
  const walls = shell.walls ?? [];

  // floor: the walls' footprint, extruded a little, so the room reads as a solid even where the hull is a chamfer
  if (walls.length >= 3) {
    const shape = new THREE.Shape(walls.map((w) => new THREE.Vector2(w.a[0], -w.a[1])));
    const geo = new THREE.ExtrudeGeometry(shape, { depth: 0.12, bevelEnabled: false });
    geo.rotateX(Math.PI / 2);
    const floor = mesh(geo, mat(PALETTE.floor, { roughness: 0.95 }), 'floor');
    floor.position.y = floorY;
    root.add(mesh(geo, mat(PALETTE.floor, { roughness: 0.95, side: THREE.DoubleSide }), 'floor'));
    root.children[root.children.length - 1].position.y = floorY;
    const ceil = mesh(new THREE.ShapeGeometry(shape).rotateX(Math.PI / 2), mat(PALETTE.ceiling, { side: THREE.BackSide }), 'ceiling');
    ceil.position.y = floorY + ceiling;
    ceiling && root.add(ceil);
  }

  walls.forEach((w, i) => {
    const [ax, az] = w.a, [bx, bz] = w.b;
    const len = Math.hypot(bx - ax, bz - az);
    if (len < 0.3) return;
    const angle = Math.atan2(bz - az, bx - ax);
    const group = new THREE.Group();
    group.name = `wall-${i}`;
    group.userData = { kind: 'wall', length: len };

    // the wall solid, split only where an opening really cuts through it
    const openings = (shell.openings ?? []).filter((o) => o.wall === i && Math.abs(o.u) < len);
    const cuts = openings
      .map((o) => ({ from: Math.max(0, o.u - o.width / 2), to: Math.min(len, o.u + o.width / 2), ...o }))
      .sort((p, q) => p.from - q.from);
    const solid = (from: number, to: number, z0: number, z1: number) => {
      if (to - from < 0.02 || z1 - z0 < 0.02) return null;
      const g = new THREE.BoxGeometry(to - from, z1 - z0, WALL_T);
      const m = mesh(g, mat(PALETTE.wall), `wall-${i}`);
      m.position.set((from + to) / 2 - len / 2, (z0 + z1) / 2, 0);
      return m;
    };

    let cursor = 0;
    for (const c of cuts) {
      const pier = solid(cursor, c.from, 0, ceiling);
      if (pier) group.add(pier);
      const lintel = solid(c.from, c.to, c.sill + c.height, ceiling);
      if (lintel) group.add(lintel);
      const sill = solid(c.from, c.to, 0, Math.max(0, c.sill));
      if (sill) group.add(sill);
      // the frame and its leaf or glass
      const fw = 0.06;
      const frameMat = mat(PALETTE.frame, { roughness: 0.6 });
      for (const [x, y, w, h] of [
        [c.from + fw / 2, c.sill + c.height / 2, fw, c.height],
        [c.to - fw / 2, c.sill + c.height / 2, fw, c.height],
        [(c.from + c.to) / 2, c.sill + c.height - fw / 2, c.to - c.from, fw],
        [(c.from + c.to) / 2, c.sill + fw / 2, c.to - c.from, fw],
      ] as const) {
        const bar = mesh(new THREE.BoxGeometry(w, h, WALL_T * 1.2), frameMat, `opening-${c.kind}`);
        bar.position.set(x - len / 2, y, 0);
        group.add(bar);
      }
      if (c.kind === 'window') {
        const glass = mesh(new THREE.BoxGeometry(c.to - c.from - fw * 2, c.height - fw * 2, 0.02),
          new THREE.MeshPhysicalMaterial({ color: PALETTE.glass, transparent: true, opacity: 0.28, roughness: 0.05, metalness: 0 }), 'window-glass');
        glass.position.set((c.from + c.to) / 2 - len / 2, c.sill + c.height / 2, 0);
        group.add(glass);
      } else {
        const leaf = mesh(new THREE.BoxGeometry(c.to - c.from - fw * 2, c.height - fw * 2, 0.04), mat(PALETTE.wood), 'door-leaf');
        leaf.position.set((c.from + c.to) / 2 - len / 2, c.sill + c.height / 2, 0);
        group.add(leaf);
      }
      cursor = c.to;
    }
    const last = solid(cursor, len, 0, ceiling);
    if (last) group.add(last);

    group.position.set((ax + bx) / 2, floorY, (az + bz) / 2);
    group.rotation.y = -angle;
    group.traverse((o) => { if (o instanceof THREE.Mesh) { o.castShadow = true; o.receiveShadow = true; } });
    root.add(group);
  });
  return root;
}

/** One proxy per labelled object, sized to its measured box. */
export function buildObjects(shell: Shell): THREE.Group {
  const root = new THREE.Group();
  root.name = 'objects';
  const floorY = shell.floorY ?? 0;
  for (const o of shell.objects ?? []) root.add(proxy(o, floorY));
  return root;
}

export function proxy(o: Obj, floorY = 0): THREE.Group {
  const g = new THREE.Group();
  const [cx, , cz] = o.center;
  const [hx, hy, hz] = o.half.map((v) => Math.max(v, 0.05)) as [number, number, number];
  const kind = (o.kind ?? '').toLowerCase();
  const w = hx * 2, h = hy * 2, d = hz * 2;
  g.name = o.id;
  g.userData = { kind: 'object', id: o.id, label: o.name, type: kind, size: [w, h, d] };
  g.position.set(cx, floorY, cz);

  const wood = mat(PALETTE.wood, { roughness: 0.7 });
  const fabric = mat(PALETTE.fabric, { roughness: 0.95 });
  const metal = mat(PALETTE.metal, { roughness: 0.45, metalness: 0.6 });
  const dark = mat(PALETTE.dark, { roughness: 0.6 });

  const add = (geo: THREE.BufferGeometry, material: THREE.Material, name: string, x = 0, y = 0, z = 0) => {
    const m = mesh(geo, material, `${o.id}-${name}`);
    m.position.set(x, y - hy, z);
    g.add(m, outlined(m));
    return m;
  };

  if (kind === 'table' || kind === 'desk' || kind === 'bench' || kind === 'podium') {
    add(new THREE.BoxGeometry(w, 0.06, d), wood, 'top', 0, h, 0);
    const legR = Math.min(0.05, w * 0.05);
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      add(new THREE.CylinderGeometry(legR, legR, h - 0.06, 8), metal, 'leg', sx * (w / 2 - legR * 2), (h - 0.06) / 2, sz * (d / 2 - legR * 2));
    }
  } else if (kind === 'chair') {
    const seatY = Math.min(0.45, h * 0.55);
    add(new THREE.BoxGeometry(w, 0.06, d * 0.85), fabric, 'seat', 0, seatY);
    add(new THREE.BoxGeometry(w, Math.max(0.2, h - seatY - 0.05), 0.05), fabric, 'back', 0, (h + seatY) / 2, -d / 2 + 0.03);
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) add(new THREE.CylinderGeometry(0.018, 0.018, seatY, 6), metal, 'leg', sx * (w / 2 - 0.05), seatY / 2, sz * (d / 2 - 0.05));
  } else if (kind === 'cabinet' || kind === 'shelf' || kind === 'bookcase') {
    add(new THREE.BoxGeometry(w, h, d), wood, 'body');
    const rows = Math.max(2, Math.round(h / 0.4));
    for (let i = 1; i < rows; i++) {
      const board = mesh(new THREE.BoxGeometry(w * 0.96, 0.02, d * 0.96), dark, `${o.id}-board`);
      board.position.set(0, -hy + (h / rows) * i, 0);
      g.add(board);
    }
  } else if (kind === 'blackboard' || kind === 'whiteboard' || kind === 'projector screen') {
    add(new THREE.BoxGeometry(w, h, 0.05), dark, 'panel');
    add(new THREE.BoxGeometry(w, 0.04, 0.09), metal, 'rail', 0, h, 0);
  } else if (kind === 'ceiling fan') {
    add(new THREE.CylinderGeometry(0.09, 0.09, 0.1, 12), metal, 'hub', 0, h / 2);
    for (let i = 0; i < 3; i++) {
      const blade = mesh(new THREE.BoxGeometry(w * 0.9, 0.02, 0.12), dark, `${o.id}-blade`);
      blade.position.set(Math.cos((i / 3) * Math.PI * 2) * w * 0.45, 0, Math.sin((i / 3) * Math.PI * 2) * w * 0.45);
      blade.rotation.y = (i / 3) * Math.PI * 2;
      g.add(blade);
    }
  } else if (kind === 'door' || kind === 'window') {
    add(new THREE.BoxGeometry(w, h, Math.max(0.04, d)), kind === 'window' ? mat(PALETTE.glass, { transparent: true, opacity: 0.3, roughness: 0.1 }) : wood, 'leaf');
  } else if (kind === 'lamp') {
    add(new THREE.CylinderGeometry(0.02, 0.02, Math.max(h, 0.4), 6), metal, 'stem', 0, h * 0.5);
    add(new THREE.ConeGeometry(w * 0.5, 0.18, 14), mat(PALETTE.ceiling), 'shade', 0, h);
  } else {
    add(new THREE.BoxGeometry(w, h, d), fabric, 'body');
  }
  return g;
}
