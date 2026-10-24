// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Spin-around view of an AR scan's coverage cloud (green / orange / red dots) with the walked path and keyframe
   cameras. Used right after a scan and in Review. Drag to orbit, pinch or scroll to zoom. */
import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { dotsMaterial, type ArKeyframe } from '../lib/ar';

export function CloudView({ pts, col, keyframes, path, height = 360, dot = .05 }: { pts: Float32Array; col: Float32Array; keyframes?: ArKeyframe[]; path?: number[][]; height?: number; dot?: number }) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = host.current!; const w = el.clientWidth, h = height;
    const r = new THREE.WebGLRenderer({ antialias: true, alpha: true }); r.setPixelRatio(Math.min(2, devicePixelRatio)); r.setSize(w, h); el.append(r.domElement);
    const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(50, w / h, .02, 200);
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pts, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const mat = dotsMaterial(dot); mat.uniforms.uAlpha.value = .95; scene.add(new THREE.Points(g, mat));
    // frame the bulk of the points (SfM leaves a few far outliers): 5th..95th percentile per axis
    const pc = (ax: number, q: number) => { const a: number[] = []; for (let i = ax; i < pts.length; i += 3 * Math.max(1, Math.floor(pts.length / 6000))) a.push(pts[i]); a.sort((x, y) => x - y); return a.length ? a[Math.floor(q * (a.length - 1))] : 0; };
    const b = new THREE.Box3(new THREE.Vector3(pc(0, .05), pc(1, .05), pc(2, .05)), new THREE.Vector3(pc(0, .95), pc(1, .95), pc(2, .95))), c = b.getCenter(new THREE.Vector3()), size = b.getSize(new THREE.Vector3()).length() || 3;
    // floor grid at y = 0 (AR 'local-floor'), the walked path and small frustums for the keyframes
    const grid = new THREE.GridHelper(Math.ceil(size * 1.4), Math.ceil(size * 1.4 * 2), 0x2a3448, 0x161c28); grid.position.set(c.x, 0, c.z); scene.add(grid);
    if (keyframes?.length) {
      const line: number[] = [], fr: number[] = [];
      for (const k of keyframes) {
        const m = new THREE.Matrix4().fromArray(k.m), o = new THREE.Vector3().setFromMatrixPosition(m); line.push(o.x, o.y, o.z);
        const s = .06, corners = [[-s * .6, -s, -s * 1.4], [s * .6, -s, -s * 1.4], [s * .6, s, -s * 1.4], [-s * .6, s, -s * 1.4]].map(p => new THREE.Vector3(...p).applyMatrix4(m));
        for (const q of corners) fr.push(o.x, o.y, o.z, q.x, q.y, q.z); for (let i = 0; i < 4; i++) { const a = corners[i], z = corners[(i + 1) % 4]; fr.push(a.x, a.y, a.z, z.x, z.y, z.z); }
      }
      const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.Float32BufferAttribute(line, 3)); scene.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0xFF6B00 })));
      const fg = new THREE.BufferGeometry(); fg.setAttribute('position', new THREE.Float32BufferAttribute(fr, 3)); scene.add(new THREE.LineSegments(fg, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: .35 })));
    }
    if (path && path.length > 1) { const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.Float32BufferAttribute(path.flat(), 3)); scene.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0xFF6B00 })));
      const cg = new THREE.BufferGeometry(); cg.setAttribute('position', new THREE.Float32BufferAttribute(path.flat(), 3)); const cm = dotsMaterial(.09); cm.uniforms.uAlpha.value = 1; cg.setAttribute('color', new THREE.Float32BufferAttribute(path.flatMap(() => [1, .42, 0]), 3)); scene.add(new THREE.Points(cg, cm)); }
    cam.position.set(c.x + size * .55, c.y + size * .6, c.z + size * .7);
    const ctl = new OrbitControls(cam, r.domElement); ctl.target.copy(c); ctl.enableDamping = true; ctl.autoRotate = true; ctl.autoRotateSpeed = .8; ctl.update();
    r.domElement.addEventListener('pointerdown', () => { ctl.autoRotate = false; }, { once: true });
    let raf = 0; const loop = () => { raf = requestAnimationFrame(loop); mat.uniforms.uScale.value = h * .9; ctl.update(); r.render(scene, cam); }; loop();
    const ro = new ResizeObserver(() => { const W = el.clientWidth; r.setSize(W, h); cam.aspect = W / h; cam.updateProjectionMatrix(); }); ro.observe(el);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); ctl.dispose(); r.dispose(); g.dispose(); r.domElement.remove(); };
  }, [pts, col, keyframes, path, height, dot]);
  return <div ref={host} className="cloud-view" style={{ height }} role="img" aria-label="3D coverage of the scan: green covered well, orange once, red only glimpsed. Drag to turn." />;
}
