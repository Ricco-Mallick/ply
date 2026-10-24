// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Geometry for the semantic model.
   Two rules learned the hard way:
   1. Furniture is built to *plausible furniture proportions* for its class, not to the raw detector box. A SAM box
      around a chair is 41 x 38 x 66 cm, and a desk often comes back 10 cm tall — extruding those gives slabs, not a
      room. The box decides where the object stands and roughly how wide it is; the class decides what it is.
   2. The room is a shell you can see into: floor, walls with baseboards, real openings cut through, and a ceiling
      at a believable height (the 97th percentile of splat heights is 7.2 m in a classroom, which is the void above
      the ceiling grid, not the ceiling). */
import * as THREE from 'three';
import { PALETTE, type Obj, type Shell } from './types';

const WALL_T = 0.16;                       // wall thickness, metres
const CEIL_MIN = 2.5, CEIL_MAX = 3.6;
const mat = (color: number, opts: THREE.MeshStandardMaterialParameters = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.8, metalness: 0.02, ...opts });

const mesh = (geo: THREE.BufferGeometry, material: THREE.Material, name: string) => {
  const m = new THREE.Mesh(geo, material);
  m.name = name; m.castShadow = true; m.receiveShadow = true;
  return m;
};

/** What each class is allowed to measure, in metres. The detector's box picks a value inside the range. */
const SIZES: Record<string, { h: [number, number]; w: [number, number]; d: [number, number] }> = {
  chair: { h: [0.82, 1.0], w: [0.42, 0.56], d: [0.44, 0.58] },
  bench: { h: [0.8, 0.95], w: [1.1, 2.0], d: [0.45, 0.6] },
  table: { h: [0.68, 0.78], w: [0.7, 2.2], d: [0.6, 0.95] },
  desk: { h: [0.68, 0.78], w: [1.0, 1.6], d: [0.55, 0.8] },
  podium: { h: [0.95, 1.15], w: [0.45, 0.7], d: [0.4, 0.6] },
  cabinet: { h: [1.4, 2.1], w: [0.5, 1.2], d: [0.35, 0.6] },
  shelf: { h: [1.4, 2.1], w: [0.6, 1.4], d: [0.28, 0.45] },
  sofa: { h: [0.75, 0.95], w: [1.4, 2.4], d: [0.8, 1.0] },
  blackboard: { h: [0.85, 1.2], w: [1.2, 4.0], d: [0.04, 0.08] },
  whiteboard: { h: [0.85, 1.2], w: [1.2, 4.0], d: [0.04, 0.08] },
  'projector screen': { h: [0.9, 1.6], w: [1.4, 2.2], d: [0.04, 0.08] },
  door: { h: [1.95, 2.1], w: [0.8, 1.0], d: [0.04, 0.06] },
  window: { h: [1.0, 1.6], w: [0.7, 1.6], d: [0.04, 0.06] },
  'ceiling fan': { h: [0.12, 0.2], w: [1.0, 1.4], d: [1.0, 1.4] },
  lamp: { h: [0.4, 1.6], w: [0.2, 0.5], d: [0.2, 0.5] },
  'potted plant': { h: [0.35, 1.1], w: [0.3, 0.7], d: [0.3, 0.7] },
  statue: { h: [0.5, 1.8], w: [0.25, 0.7], d: [0.25, 0.7] },
};
const fallback = { h: [0.6, 1.1] as [number, number], w: [0.5, 1.1] as [number, number], d: [0.5, 1.1] as [number, number] };

const clampTo = (v: number, [lo, hi]: [number, number]) => Math.min(hi, Math.max(lo, v));

/** Size from the measured box, clamped into the class's plausible range. */
export function sizeFor(o: Obj) {
  const s = SIZES[(o.kind ?? '').toLowerCase()] ?? fallback;
  const raw = [o.half[0] * 2, o.half[1] * 2, o.half[2] * 2];
  const w0 = clampTo(raw[0], s.w), h = clampTo(raw[1], s.h), d0 = clampTo(raw[2], s.d);
  const longIsZ = w0 < d0;                     // put the long side on X, then turn it to the object's long axis
  return { w: longIsZ ? d0 : w0, h, d: longIsZ ? w0 : d0, yaw: longIsZ ? Math.PI / 2 : 0 };
}

function outlined(m: THREE.Mesh, color = 0x0d1420, opacity = 0.22) {
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry, 30), new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  e.position.copy(m.position); e.rotation.copy(m.rotation); e.scale.copy(m.scale);
  e.name = `${m.name}:edges`;
  return e;
}
const box = (w: number, h: number, d: number, material: THREE.Material, name: string) => mesh(new THREE.BoxGeometry(w, h, d), material, name);

export function buildRoom(shell: Shell): THREE.Group {
  const root = new THREE.Group();
  root.name = 'room';
  const floorY = shell.floorY ?? 0;
  const measured = shell.ceilingY ?? 2.9;
  const ceiling = Math.min(CEIL_MAX, Math.max(CEIL_MIN, measured > 4.2 ? 3.1 : measured));
  const walls = shell.walls ?? [];
  if (walls.length < 3) return root;

  const shape = new THREE.Shape(walls.map((w) => new THREE.Vector2(w.a[0], -w.a[1])));
  const slab = new THREE.ExtrudeGeometry(shape, { depth: 0.14, bevelEnabled: false });
  slab.rotateX(Math.PI / 2);
  const floor = mesh(slab, mat(PALETTE.floor, { roughness: 0.95 }), 'floor');
  floor.position.y = floorY;
  root.add(floor);
  const ceil = mesh(new THREE.ShapeGeometry(shape).rotateX(Math.PI / 2), mat(PALETTE.ceiling, { side: THREE.DoubleSide }), 'ceiling');
  ceil.position.y = floorY + ceiling;
  root.add(ceil);

  const wallMat = mat(PALETTE.wall, { roughness: 0.92 });
  const trimMat = mat(PALETTE.wallSide, { roughness: 0.8 });
  const frameMat = mat(PALETTE.frame, { roughness: 0.55, metalness: 0.15 });

  walls.forEach((w, i) => {
    const [ax, az] = w.a, [bx, bz] = w.b;
    const len = Math.hypot(bx - ax, bz - az);
    if (len < 0.3) return;
    const g = new THREE.Group();
    g.name = `wall-${i}`;
    g.userData = { kind: 'wall', length: len };
    const solid = (from: number, to: number, z0: number, z1: number) => {
      if (to - from < 0.02 || z1 - z0 < 0.02) return;
      const m = box(to - from, z1 - z0, WALL_T, wallMat, `wall-${i}`);
      m.position.set((from + to) / 2 - len / 2, (z0 + z1) / 2, 0);
      g.add(m);
    };
    const openings = (shell.openings ?? []).filter((o) => o.wall === i && o.u > 0 && o.u < len).sort((p, q) => p.u - q.u);
    let cursor = 0;
    for (const o of openings) {
      const from = Math.max(0, o.u - o.width / 2), to = Math.min(len, o.u + o.width / 2);
      const sill = Math.min(Math.max(o.sill, 0), ceiling - 0.4);
      const head = Math.min(ceiling, sill + o.height);
      solid(cursor, from, 0, ceiling);
      solid(from, to, 0, sill);
      solid(from, to, head, ceiling);
      const fw = 0.05;
      for (const [x, y, w2, h2] of [
        [from + fw / 2, (sill + head) / 2, fw, head - sill],
        [to - fw / 2, (sill + head) / 2, fw, head - sill],
        [(from + to) / 2, head - fw / 2, to - from, fw],
        [(from + to) / 2, sill + fw / 2, to - from, fw],
      ] as const) {
        const bar = box(w2, h2, WALL_T * 1.15, frameMat, `opening-${o.kind}`);
        bar.position.set(x - len / 2, y, 0);
        g.add(bar);
      }
      if (o.kind === 'window') {
        const glass = mesh(new THREE.BoxGeometry(to - from - fw * 2, head - sill - fw * 2, 0.02),
          new THREE.MeshPhysicalMaterial({ color: PALETTE.glass, transparent: true, opacity: 0.22, roughness: 0.04 }), 'window-glass');
        glass.position.set((from + to) / 2 - len / 2, (sill + head) / 2, 0);
        g.add(glass);
      } else {
        const leaf = box(to - from - fw * 2, head - sill - fw * 2, 0.045, mat(PALETTE.wood), 'door-leaf');
        leaf.position.set((from + to) / 2 - len / 2, (sill + head) / 2, 0);
        g.add(leaf);
      }
      cursor = to;
    }
    solid(cursor, len, 0, ceiling);
    const base = box(len, 0.1, WALL_T + 0.02, trimMat, `wall-${i}-base`);
    base.position.set(0, 0.05, 0);
    g.add(base);
    g.position.set((ax + bx) / 2, floorY, (az + bz) / 2);
    g.rotation.y = -Math.atan2(bz - az, bx - ax);
    root.add(g);
  });
  return root;
}

export function buildObjects(shell: Shell): THREE.Group {
  const root = new THREE.Group();
  root.name = 'objects';
  for (const o of shell.objects ?? []) root.add(proxy(o, shell.floorY ?? 0, shell.ceilingY ?? 3));
  return root;
}

/** One piece of furniture, at a believable size for what it is, standing where the capture found it. */
export function proxy(o: Obj, floorY = 0, ceiling = 3): THREE.Group {
  const g = new THREE.Group();
  const kind = (o.kind ?? '').toLowerCase();
  const { w, h, d, yaw } = sizeFor(o);
  const [cx, , cz] = o.center;
  g.name = o.id;
  g.userData = { kind: 'object', id: o.id, label: o.name, type: kind, size: [w, h, d] };
  g.position.set(cx, floorY, cz);
  g.rotation.y = yaw;

  const wood = mat(PALETTE.wood, { roughness: 0.62 });
  const fabric = mat(PALETTE.fabric, { roughness: 0.95 });
  const metal = mat(PALETTE.metal, { roughness: 0.4, metalness: 0.55 });
  const dark = mat(PALETTE.dark, { roughness: 0.55 });
  const panel = mat(PALETTE.ceiling, { roughness: 0.9 });
  const add = (m: THREE.Mesh, x = 0, y = 0, z = 0) => { m.position.set(x, y, z); g.add(m, outlined(m)); return m; };
  const leg = (x: number, z: number, height: number, r = 0.022) => add(mesh(new THREE.CylinderGeometry(r, r, height, 8), metal, `${o.id}-leg`), x, height / 2, z);

  if (kind === 'table' || kind === 'desk') {
    add(box(w, 0.04, d, wood, `${o.id}-top`), 0, h - 0.02, 0);
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) leg(sx * (w / 2 - 0.07), sz * (d / 2 - 0.07), h - 0.04);
  } else if (kind === 'chair') {
    const seatY = 0.45;
    add(box(w, 0.05, d, fabric, `${o.id}-seat`), 0, seatY, 0);
    add(box(w, h - seatY, 0.05, fabric, `${o.id}-back`), 0, (h + seatY) / 2, -d / 2 + 0.04);
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) leg(sx * (w / 2 - 0.04), sz * (d / 2 - 0.04), seatY, 0.018);
  } else if (kind === 'bench') {
    const seatY = 0.45;
    add(box(w, 0.06, d, wood, `${o.id}-seat`), 0, seatY, 0);
    add(box(w, h - seatY - 0.06, 0.05, wood, `${o.id}-back`), 0, (h + seatY) / 2, -d / 2 + 0.04);
    for (const sx of [-1, 1]) add(box(0.06, seatY, d * 0.9, metal, `${o.id}-frame`), sx * (w / 2 - 0.05), seatY / 2, 0);
  } else if (kind === 'sofa') {
    const seatY = 0.42;
    add(box(w, 0.16, d, fabric, `${o.id}-seat`), 0, seatY, 0);
    add(box(w, h - seatY, 0.18, fabric, `${o.id}-back`), 0, (h + seatY) / 2, -d / 2 + 0.09);
    for (const sx of [-1, 1]) add(box(0.16, h - 0.1, d, fabric, `${o.id}-arm`), sx * (w / 2 - 0.08), (h - 0.1) / 2, 0);
  } else if (kind === 'cabinet' || kind === 'shelf') {
    add(box(w, h, d, wood, `${o.id}-body`), 0, h / 2, 0);
    const rows = Math.max(2, Math.round(h / 0.38));
    for (let i = 1; i < rows; i++) add(box(w * 0.94, 0.018, d * 0.92, dark, `${o.id}-board`), 0, (h / rows) * i, 0);
    add(box(0.02, h * 0.35, 0.02, metal, `${o.id}-pull`), w * 0.3, h * 0.55, d / 2 + 0.01);
  } else if (kind === 'blackboard' || kind === 'whiteboard' || kind === 'projector screen') {
    const y = Math.min(1.55, ceiling - h / 2 - 0.2);
    add(box(w, h, 0.04, kind === 'whiteboard' ? panel : dark, `${o.id}-panel`), 0, y, 0);
    add(box(w + 0.08, 0.04, 0.1, metal, `${o.id}-rail`), 0, y + h / 2, 0.02);
  } else if (kind === 'ceiling fan') {
    const y = Math.min(2.65, ceiling - 0.25);
    add(mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.08, 12), metal, `${o.id}-hub`), 0, y, 0);
    for (let i = 0; i < 3; i++) {
      const blade = box(w * 0.42, 0.015, 0.16, dark, `${o.id}-blade`);
      blade.position.set(0, y - 0.05, 0);
      blade.rotation.y = (i / 3) * Math.PI * 2;
      blade.translateX(w * 0.26);
      g.add(blade);
    }
  } else if (kind === 'potted plant') {
    add(mesh(new THREE.CylinderGeometry(w * 0.3, w * 0.24, h * 0.35, 12), mat(0x8a5a44), `${o.id}-pot`), 0, h * 0.175, 0);
    add(mesh(new THREE.SphereGeometry(w * 0.42, 12, 10), mat(0x4e7a52, { roughness: 0.9 }), `${o.id}-leaves`), 0, h * 0.7, 0);
  } else if (kind === 'door' || kind === 'window') {
    add(box(w, h, Math.max(0.04, d), kind === 'window' ? mat(PALETTE.glass, { transparent: true, opacity: 0.25 }) : wood, `${o.id}-leaf`), 0, h / 2 + (kind === 'window' ? 0.9 : 0), 0);
  } else if (kind === 'lamp') {
    add(mesh(new THREE.CylinderGeometry(0.015, 0.015, h, 6), metal, `${o.id}-stem`), 0, h / 2, 0);
    add(mesh(new THREE.ConeGeometry(w * 0.5, 0.16, 14), panel, `${o.id}-shade`), 0, h, 0);
  } else {
    add(box(w, h, d, fabric, `${o.id}-body`), 0, h / 2, 0);
  }
  return g;
}
