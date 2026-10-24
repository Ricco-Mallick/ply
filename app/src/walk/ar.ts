// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Real AR on Android Chrome (WebXR immersive-ar, 'local-floor', hit-test, DOM overlay).

   Navigation: the phone and the twin are lined up with two taps on the floor at two known places (both spaces are in
   metres with the floor level, so two points fix rotation and position; the scale is checked, not assumed). Then the
   route from where the phone is to the chosen place is drawn as chevrons on the real floor, re-planned as you walk.
   Dollhouse: tap a surface and the twin stands on it at about 40 cm across, ceiling cut away; drag to turn it. */
import * as THREE from 'three';
import { SparkRenderer, SplatEdit, SplatEditRgbaBlendMode, SplatEditSdf, SplatEditSdfType, SplatMesh } from '@sparkjsdev/spark';
import type { Navigator } from '../lab/places';

export async function arSupported() { try { return !!(await (navigator as any).xr?.isSessionSupported('immersive-ar')); } catch { return false; } }

type V2 = [number, number];
/** 2D similarity (x, z): twin = s R ar + t, from two matched floor points. */
export function fitTwoPoints(arA: V2, arB: V2, twA: V2, twB: V2) {
  const da = [arB[0] - arA[0], arB[1] - arA[1]], dt = [twB[0] - twA[0], twB[1] - twA[1]];
  const la = Math.hypot(da[0], da[1]), lt = Math.hypot(dt[0], dt[1]);
  const th = Math.atan2(dt[1], dt[0]) - Math.atan2(da[1], da[0]), s = lt / Math.max(1e-6, la), c = Math.cos(th) * s, n = Math.sin(th) * s;
  const tx = twA[0] - (c * arA[0] - n * arA[1]), tz = twA[1] - (n * arA[0] + c * arA[1]);
  const toTwin = (p: V2): V2 => [c * p[0] - n * p[1] + tx, n * p[0] + c * p[1] + tz];
  const toAr = (q: V2): V2 => { const x = q[0] - tx, z = q[1] - tz, k = 1 / (s * s); return [k * (c * x + n * z), k * (-n * x + c * z)]; };
  return { scale: s, toTwin, toAr };
}

abstract class ArBase {
  renderer!: THREE.WebGLRenderer; scene = new THREE.Scene(); cam = new THREE.PerspectiveCamera(); session: any = null;
  hitSource: any = null; reticle: THREE.Mesh; hit: THREE.Vector3 | null = null; listeners = new Set<() => void>();
  constructor() {
    this.reticle = new THREE.Mesh(new THREE.RingGeometry(.07, .09, 40).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: .9 }));
    this.reticle.visible = false; this.scene.add(this.reticle);
  }
  on(f: () => void) { this.listeners.add(f); return () => { this.listeners.delete(f); }; }
  emit() { this.listeners.forEach(f => f()); }
  async start(overlay: HTMLElement) {
    this.session = await (navigator as any).xr.requestSession('immersive-ar', { requiredFeatures: ['local-floor', 'hit-test'], optionalFeatures: ['dom-overlay'], domOverlay: { root: overlay } });
    this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true }); this.renderer.xr.enabled = true; this.renderer.xr.setReferenceSpaceType('local-floor');
    await this.renderer.xr.setSession(this.session);
    const viewer = await this.session.requestReferenceSpace('viewer'); this.hitSource = await this.session.requestHitTestSource({ space: viewer });
    this.session.addEventListener('end', () => { this.renderer.setAnimationLoop(null); this.renderer.dispose(); this.ended(); this.emit(); });
    this.setup();
    this.renderer.setAnimationLoop((t: number, frame: any) => {
      if (!frame) return; const ref = this.renderer.xr.getReferenceSpace();
      const hits = frame.getHitTestResults(this.hitSource); this.hit = null;
      if (hits.length) { const m = hits[0].getPose(ref)?.transform.matrix; if (m) { this.hit = new THREE.Vector3(m[12], m[13], m[14]); this.reticle.position.copy(this.hit); } }
      this.reticle.visible = !!this.hit && this.wantsReticle();
      const pose = frame.getViewerPose(ref); if (pose) this.tick(t / 1000, pose.transform.position);
      this.renderer.render(this.scene, this.cam);
    });
  }
  stop() { try { this.session?.end(); } catch { /* already ended */ } }
  protected setup() {} protected ended() {}
  protected abstract tick(t: number, p: DOMPointReadOnly): void; protected abstract wantsReticle(): boolean;
}

export interface NavPlace { id: string; name: string; x: number; z: number }
export class ArNavigator extends ArBase {
  step: 'align1' | 'align2' | 'pick' | 'go' = 'align1'; anchors: NavPlace[]; tapsAr: V2[] = []; fit: ReturnType<typeof fitTwoPoints> | null = null;
  dest: NavPlace | null = null; left = 0; next = ''; floorY = 0; warning = '';
  private chev: THREE.InstancedMesh; private beacon: THREE.Mesh; private path: V2[] | null = null; private lastPlan: V2 = [1e9, 1e9];
  constructor(public places: NavPlace[], public nav: Navigator) {
    super();
    // align on the two pinned places furthest apart (a long baseline makes the fit accurate)
    let best: [NavPlace, NavPlace] = [places[0], places[1]], bd = -1;
    for (const a of places) for (const b of places) { const d = Math.hypot(a.x - b.x, a.z - b.z); if (d > bd) { bd = d; best = [a, b]; } }
    this.anchors = best;
    const shape = new THREE.Shape(); shape.moveTo(-.18, -.07); shape.lineTo(0, .11); shape.lineTo(.18, -.07); shape.lineTo(.18, -.16); shape.lineTo(0, .02); shape.lineTo(-.18, -.16); shape.closePath();
    this.chev = new THREE.InstancedMesh(new THREE.ShapeGeometry(shape).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0xFF6B00, transparent: true, opacity: .92, side: THREE.DoubleSide }), 24);
    this.chev.count = 0; this.chev.frustumCulled = false; this.scene.add(this.chev);
    this.beacon = new THREE.Mesh(new THREE.CylinderGeometry(.18, .18, 2.2, 32, 1, true), new THREE.MeshBasicMaterial({ color: 0xFF8A3D, transparent: true, opacity: .35, side: THREE.DoubleSide, depthWrite: false }));
    this.beacon.visible = false; this.scene.add(this.beacon);
  }
  protected wantsReticle() { return this.step === 'align1' || this.step === 'align2'; }
  /** The user tapped "here": record the floor point for the current alignment step. */
  tap() {
    if (!this.hit || !(this.step === 'align1' || this.step === 'align2')) return;
    this.tapsAr.push([this.hit.x, this.hit.z]); this.floorY = this.hit.y;
    if (this.step === 'align1') { this.step = 'align2'; this.emit(); return; }
    const [a, b] = this.anchors; this.fit = fitTwoPoints(this.tapsAr[0], this.tapsAr[1], [a.x, a.z], [b.x, b.z]);
    this.warning = Math.abs(this.fit.scale - 1) > .15 ? `The two taps were ${Math.round(Math.abs(1 - this.fit.scale) * 100)}% off the twin's distance: aim at the floor right in front of each place, then realign if arrows drift.` : '';
    this.step = 'pick'; this.emit();
  }
  realign() { this.tapsAr = []; this.fit = null; this.step = 'align1'; this.chev.count = 0; this.beacon.visible = false; this.emit(); }
  go(p: NavPlace) { this.dest = p; this.step = 'go'; this.lastPlan = [1e9, 1e9]; this.emit(); }
  protected tick(t: number, pos: DOMPointReadOnly) {
    if (this.step !== 'go' || !this.fit || !this.dest) return;
    const me = this.fit.toTwin([pos.x, pos.z]);
    if (Math.hypot(me[0] - this.lastPlan[0], me[1] - this.lastPlan[1]) > .35) { // re-plan as you walk
      this.lastPlan = me; this.path = this.nav.route(me, [this.dest.x, this.dest.z]);
      const end = this.fit.toAr([this.dest.x, this.dest.z]); this.beacon.position.set(end[0], this.floorY + 1.1, end[1]); this.beacon.visible = true;
      let L = 0; if (this.path) for (let i = 1; i < this.path.length; i++) L += Math.hypot(this.path[i][0] - this.path[i - 1][0], this.path[i][1] - this.path[i - 1][1]); this.left = L;
      this.next = !this.path ? 'No walkable route from here' : L < .8 ? `You have arrived at ${this.dest.name}` : `${this.dest.name} · ${L.toFixed(1)} m`; this.emit();
    }
    // chevrons every 50 cm along the next 6 m, a wave running toward the destination
    const ps = this.path; if (!ps || ps.length < 2) { this.chev.count = 0; return; }
    const o = new THREE.Object3D(); let n = 0, d = .6, acc = 0;
    for (let i = 1; i < ps.length && n < 24; i++) {
      const a = this.fit.toAr(ps[i - 1]), b = this.fit.toAr(ps[i]), seg = Math.hypot(b[0] - a[0], b[1] - a[1]), yaw = Math.atan2(-(b[0] - a[0]), -(b[1] - a[1]));
      while (d <= acc + seg && n < 24 && d < 6) { const k = (d - acc) / seg; o.position.set(a[0] + (b[0] - a[0]) * k, this.floorY + .01, a[1] + (b[1] - a[1]) * k); o.rotation.set(0, yaw, 0);
        o.scale.setScalar(1 + .25 * Math.max(0, Math.sin(t * 5 - n * .8))); o.updateMatrix(); this.chev.setMatrixAt(n++, o.matrix); d += .5; }
      acc += seg;
    }
    this.chev.count = n; this.chev.instanceMatrix.needsUpdate = true;
  }
}

export class ArDollhouse extends ArBase {
  placed = false; private model: THREE.Group = new THREE.Group(); private splat: SplatMesh | null = null; spin = 0;
  constructor(public url: string, public bounds: { min: number[]; max: number[] } | null, public floorY = 0) { super(); }
  protected setup() { this.scene.add(new SparkRenderer({ renderer: this.renderer })); }
  protected wantsReticle() { return !this.placed; }
  /** Tap: stand the twin on the surface under the reticle, ~40 cm across, ceiling (2.2 m) cut away. */
  tap() {
    if (!this.hit || this.placed) return;
    const b = this.bounds ?? { min: [-3, 0, -3], max: [3, 2.6, 3] }, size = Math.max(b.max[0] - b.min[0], b.max[2] - b.min[2]), s = .4 / Math.max(1, size);
    const cx = (b.min[0] + b.max[0]) / 2, cz = (b.min[2] + b.max[2]) / 2;
    this.splat = new SplatMesh({ url: this.url }); this.splat.position.set(-cx, -this.floorY, -cz);
    const edit = new SplatEdit({ rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY, softEdge: .05 }); const cut = new SplatEditSdf({ type: SplatEditSdfType.BOX, opacity: 0, color: new THREE.Color(1, 1, 1) });
    cut.position.set(cx, this.floorY + 2.2 + 50, cz); cut.scale.set(500, 50, 500); edit.addSdf(cut); this.splat.add(edit);
    this.model.add(this.splat); this.model.scale.setScalar(s); this.model.position.copy(this.hit); this.scene.add(this.model);
    this.placed = true; this.emit();
  }
  turn(dx: number) { this.model.rotation.y += dx * .01; }
  reset() { if (this.splat) { this.model.remove(this.splat); this.splat.dispose(); this.splat = null; } this.placed = false; this.emit(); }
  protected tick() { /* static once placed */ }
  protected ended() { this.splat?.dispose(); }
}
