/* Cinematic tour: camera keyframes (from the pipeline's scene.json "tour", or made here from the walkable floor when a
   scene has none) played as one smooth move. Positions and look-at targets follow centripetal Catmull-Rom splines,
   timed by each keyframe's duration with ease-in-out per shot. Any key, click or touch hands control back. */
import * as THREE from 'three';
import type { WalkEngine } from './engine';
import type { OccupancyGrid } from './scene';

export interface TourKey { pos: [number, number, number]; target: [number, number, number]; fov: number; dur: number; }
export const SHOT_NAMES = ['Establishing', 'Push in', 'Walk 1', 'Walk 2', 'Walk 3', 'Walk 4', 'Reveal', 'Overview'];

/** No tour in the manifest: corners of the walkable floor, a loop through it, a high wide. */
export function autoTour(grid: OccupancyGrid, floorY: number, eye = 1.55): TourKey[] {
  const { width: w, height: h, cell, origin } = grid.meta; const free: [number, number][] = [];
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) if (!grid.data[j * w + i]) free.push([origin[0] + (i + .5) * cell, origin[1] + (j + .5) * cell]);
  if (free.length < 20) return [];
  const pc = (k: 0 | 1, f: number) => { const a = free.map(p => p[k]).sort((x, y) => x - y); return a[Math.floor(f * (a.length - 1))]; };
  const lo = [pc(0, .08), pc(1, .08)], hi = [pc(0, .92), pc(1, .92)], c: [number, number, number] = [(lo[0] + hi[0]) / 2, floorY + 1.1, (lo[1] + hi[1]) / 2];
  const high = floorY + eye + .7, k = (x: number, z: number, y = high): [number, number, number] => [x, y, z];
  const walk = [.15, .4, .65, .9].map(f => { const a = f * Math.PI * 2, r = .32; return k(c[0] + Math.cos(a) * (hi[0] - lo[0]) * r, c[2] + Math.sin(a) * (hi[1] - lo[1]) * r, floorY + eye); });
  return [
    { pos: k(lo[0], lo[1]), target: c, fov: 62, dur: 0 }, { pos: k(lo[0] + (c[0] - lo[0]) * .2, lo[1] + (c[2] - lo[1]) * .2), target: c, fov: 58, dur: 5 },
    ...walk.map((p, i) => ({ pos: p, target: walk[(i + 1) % walk.length].map((v, j) => j === 1 ? floorY + eye - .15 : v * .6 + c[j] * .4) as [number, number, number], fov: 66, dur: 4.5 })),
    { pos: k(hi[0], hi[1]), target: c, fov: 60, dur: 5 }, { pos: k(c[0] + (hi[0] - c[0]) * .35, c[2] + (hi[1] - c[2]) * .35, high + .4), target: [c[0], floorY, c[2]], fov: 70, dur: 5 },
  ];
}

export class Tour {
  playing = false; private hook: ((dt: number) => void) | null = null; private stopHandler = () => this.stop();
  constructor(public engine: WalkEngine, public keys: TourKey[], public onChange: () => void) {}

  play() {
    const e = this.engine, ks = this.keys; if (ks.length < 2 || this.playing) return;
    const P = new THREE.CatmullRomCurve3(ks.map(k => new THREE.Vector3(...k.pos)), false, 'centripetal');
    const T = new THREE.CatmullRomCurve3(ks.map(k => new THREE.Vector3(...k.target)), false, 'centripetal');
    const cum = ks.map((_, i) => ks.slice(1, i + 1).reduce((s, k) => s + k.dur, 0)), total = cum[cum.length - 1];
    let t = 0; e.manual = true; this.playing = true; this.onChange();
    const p = new THREE.Vector3(), q = new THREE.Vector3();
    this.hook = (dt: number) => {
      t = Math.min(total, t + dt);
      let i = 0; while (i < cum.length - 2 && t > cum[i + 1]) i++;
      const seg = (t - cum[i]) / Math.max(1e-6, cum[i + 1] - cum[i]), e2 = seg < .5 ? 2 * seg * seg : 1 - (-2 * seg + 2) ** 2 / 2; // ease each shot
      const u = (i + e2) / (ks.length - 1);
      P.getPoint(u, p); T.getPoint(u, q); e.camera.position.copy(p); e.camera.lookAt(q);
      e.camera.fov = ks[i].fov + (ks[i + 1].fov - ks[i].fov) * e2; e.camera.updateProjectionMatrix();
      if (t >= total) this.stop();
    };
    e.frameHooks.add(this.hook);
    setTimeout(() => { addEventListener('keydown', this.stopHandler, { once: true }); e.canvas.addEventListener('pointerdown', this.stopHandler, { once: true }); }, 300);
  }
  /** Hand control back where the camera is: fly mode, facing the same way. */
  stop() {
    const e = this.engine; if (!this.playing) return; this.playing = false;
    if (this.hook) e.frameHooks.delete(this.hook); this.hook = null;
    removeEventListener('keydown', this.stopHandler); e.canvas.removeEventListener('pointerdown', this.stopHandler);
    const eul = new THREE.Euler().setFromQuaternion(e.camera.quaternion, 'YXZ');
    e.state.yaw = eul.y; e.pitch = eul.x; e.camera.fov = 70; e.camera.updateProjectionMatrix();
    e.state.pos.set(e.camera.position.x, e.camera.position.y - e.eye, e.camera.position.z); e.state.mode = 'fly'; e.manual = false; e.emit(); this.onChange();
  }
  /** Jump (smoothly) to one keyframe and stop there. */
  goTo(i: number) {
    const e = this.engine, k = this.keys[i]; if (!k) return; this.stop();
    const from = e.camera.position.clone(), to = new THREE.Vector3(...k.pos), look0 = new THREE.Vector3(0, 0, -1).applyQuaternion(e.camera.quaternion).add(from), look1 = new THREE.Vector3(...k.target);
    let t = 0; e.manual = true; const p = new THREE.Vector3(), q = new THREE.Vector3();
    const hook = (dt: number) => { t = Math.min(1, t + dt / 1.6); const s = 1 - (1 - t) ** 3; p.lerpVectors(from, to, s); q.lerpVectors(look0, look1, s); e.camera.position.copy(p); e.camera.lookAt(q);
      if (t >= 1) { e.frameHooks.delete(hook); this.playing = true; this.stop(); } };
    e.frameHooks.add(hook);
  }
}
