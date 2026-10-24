// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* A Gaussian splat on a turntable (Spark): the live view's training snapshots and the finished twin. Drag to orbit,
   scroll or pinch to zoom; it turns slowly until touched. */
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SparkRenderer, SplatEdit, SplatEditRgbaBlendMode, SplatEditSdf, SplatEditSdfType, SplatMesh } from '@sparkjsdev/spark';

/** cut: hide everything above this height (metres), dollhouse style, so an indoor twin reads from outside. */
export function SplatOrbit({ url, height = 420, target = [0, .9, 0], radius = 6, cut = 2.3 }: { url: string; height?: number; target?: number[]; radius?: number; cut?: number | null }) {
  const host = useRef<HTMLDivElement>(null); const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  useEffect(() => {
    const el = host.current!; const w = el.clientWidth, h = height; setState('loading');
    const r = new THREE.WebGLRenderer({ antialias: false, alpha: true }); r.setPixelRatio(Math.min(1.5, devicePixelRatio)); r.setSize(w, h); el.append(r.domElement);
    const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(55, w / h, .05, 300);
    const spark = new SparkRenderer({ renderer: r }); scene.add(spark);
    const splat = new SplatMesh({ url }); scene.add(splat);
    if (cut != null) {
      const edit = new SplatEdit({ rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY, softEdge: .06 });
      const box = new SplatEditSdf({ type: SplatEditSdfType.BOX, opacity: 0, color: new THREE.Color(1, 1, 1) }); box.position.set(0, cut + 50, 0); box.scale.set(500, 50, 500);
      edit.addSdf(box); splat.add(edit);
    }
    const t = new THREE.Vector3(...(target as [number, number, number]));
    cam.position.set(t.x + radius * .62, t.y + radius * .78, t.z + radius * .62);
    const ctl = new OrbitControls(cam, r.domElement); ctl.target.copy(t); ctl.enableDamping = true; ctl.autoRotate = true; ctl.autoRotateSpeed = .6; ctl.update();
    r.domElement.addEventListener('pointerdown', () => { ctl.autoRotate = false; }, { once: true });
    splat.initialized.then(() => setState('ready')).catch(() => setState('error'));
    let raf = 0; const loop = () => { raf = requestAnimationFrame(loop); ctl.update(); r.render(scene, cam); }; loop();
    const ro = new ResizeObserver(() => { const W = el.clientWidth; r.setSize(W, h); cam.aspect = W / h; cam.updateProjectionMatrix(); }); ro.observe(el);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); ctl.dispose(); splat.dispose(); r.dispose(); r.domElement.remove(); };
  }, [url, height, cut, radius, target.join()]);
  return <div ref={host} className="splat-orbit" style={{ height }} role="img" aria-label="The 3D twin; drag to turn around it">
    {state !== 'ready' && <span className="mono so-state">{state === 'error' ? 'Could not load this snapshot' : 'Streaming splats…'}</span>}
  </div>;
}
