// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The measured layer: walls, floor outline and object boxes recovered from the solve (shell.json next to the scene).
   The splats show what the room looks like; this shows what it measures — the layer an editor selects from, and the
   structure AR occlusion and pathfinding want. Toggle it from the HUD; W A S D stays live underneath. */
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import type { WalkEngine } from './engine';

export type Shell = {
  floorY?: number; ceilingY?: number;
  walls?: { a: [number, number]; b: [number, number]; length: number; normal?: [number, number] }[];
  openings?: { kind: string; wall: number; u: number; width: number; sill: number; height: number }[];
  objects?: { id: string; name: string; kind?: string; center: [number, number, number]; half: [number, number, number] }[];
};

export function useShell(base: URL | null) {
  const [shell, setShell] = useState<Shell | null>(null);
  useEffect(() => {
    if (!base) return;
    fetch(new URL('shell.json', base)).then((r) => (r.ok ? r.json() : null)).then(setShell).catch(() => setShell(null));
  }, [base]);
  return shell;
}

export function MeasureLayer({ engine, shell, on, showObjects }: { engine: WalkEngine; shell: Shell | null; on: boolean; showObjects: boolean }) {
  const group = useRef<THREE.Group | null>(null);
  useEffect(() => {
    if (!engine.scene || !shell) return;
    const g = new THREE.Group();
    g.name = 'ply-measure';
    const ceiling = Math.max(shell.ceilingY ?? 2.7, 2.2);
    const wallMat = new THREE.MeshBasicMaterial({ color: 0x7fb2ff, transparent: true, opacity: 0.13, side: THREE.DoubleSide, depthWrite: false });
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x9ec8ff, transparent: true, opacity: 0.65 });

    for (const w of shell.walls ?? []) {
      const [ax, az] = w.a, [bx, bz] = w.b;
      const len = Math.hypot(bx - ax, bz - az);
      if (len < 0.4) continue;
      // a thin wall panel, standing on the floor
      const geo = new THREE.PlaneGeometry(len, ceiling);
      const mesh = new THREE.Mesh(geo, wallMat);
      mesh.position.set((ax + bx) / 2, (shell.floorY ?? 0) + ceiling / 2, (az + bz) / 2);
      mesh.rotation.y = -Math.atan2(bz - az, bx - ax);
      mesh.renderOrder = 20;
      g.add(mesh);
      // its outline, base and top rules
      for (const y of [0, ceiling]) {
        const line = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(ax, (shell.floorY ?? 0) + y, az), new THREE.Vector3(bx, (shell.floorY ?? 0) + y, bz)]);
        g.add(new THREE.Line(line, edgeMat));
      }
      for (const [x, z] of [[ax, az], [bx, bz]]) {
        const line = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, shell.floorY ?? 0, z), new THREE.Vector3(x, (shell.floorY ?? 0) + ceiling, z)]);
        g.add(new THREE.Line(line, edgeMat));
      }
    }

    for (const o of shell.objects ?? []) {
      if (!showObjects) continue;
      const [cx, cy, cz] = o.center;
      const [hx, hy, hz] = o.half;
      const box = new THREE.Box3(new THREE.Vector3(cx - hx, cy - hy, cz - hz), new THREE.Vector3(cx + hx, cy + hy, cz + hz));
      g.add(new THREE.Box3Helper(box, 0xffd58a));
    }

    g.visible = on;
    engine.scene.add(g);
    group.current = g;
    return () => { engine.scene.remove(g); g.traverse((o) => { (o as THREE.Mesh).geometry?.dispose?.(); }); group.current = null; };
  }, [engine, shell, showObjects]);

  useEffect(() => { if (group.current) group.current.visible = on; }, [on]);
  return null;
}
