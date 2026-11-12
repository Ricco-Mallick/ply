// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Walkthrough engine: three.js + Spark splat renderer, a walk/fly controller with WASD, mouse look, touch joystick,
   and collision through a Collider (collider.ts: the pipeline's occupancy grid today). The React HUD reads `state`
   and calls the methods. */
import * as THREE from 'three';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { OccupancyGrid, type SceneManifest } from './scene';
import { GridCollider, type Collider, type Point } from './collider';
import { RADIUS, REACH, SPEED, WHEEL, WHEEL_MIN, WHEEL_WALK, ease, pace, recover, walkFrame, type Walker } from './walker';
import { glassFrame, setGlassSource } from '../shared/glass';
import { PROFILES, deviceProfile, loadMode, saveLearned, saveMode, type PerfProfile, type QualityMode } from './perf';

export type Mode = 'walk' | 'fly';
/** Live renderer numbers for the HUD, refreshed twice a second. */
export interface PerfStats { fps: number; ms: number; worstMs: number; splats: number; dpr: number; }
export interface EngineState { progress: number; loaded: boolean; error: string | null; mode: Mode; locked: boolean; moved: boolean; pos: THREE.Vector3; yaw: number; speed: number; }

const RM = matchMedia('(prefers-reduced-motion: reduce)').matches;
const FW = new THREE.Vector3(), RT = new THREE.Vector3();

export class WalkEngine {
  renderer: THREE.WebGLRenderer; scene = new THREE.Scene(); camera: THREE.PerspectiveCamera; spark: SparkRenderer; splat: SplatMesh | null = null;
  grid: OccupancyGrid | null = null; floorY = 0; eye = 1.55;
  /** Every question about walls goes through this; `grid` is kept for the map, the tours and the routes. */
  collider: Collider | null = null;
  /** Where the twin opened: a walker with no place near it goes back here. */
  spawn: Point = { x: 0, z: 0 }; private body: Walker = { x: 0, z: 0, vx: 0, vz: 0 };
  state: EngineState = { progress: 0, loaded: false, error: null, mode: 'fly', locked: false, moved: false, pos: new THREE.Vector3(), yaw: 0, speed: 1 };
  pitch = 0; vel = new THREE.Vector3(); keys = new Set<string>(); joy = { x: 0, y: 0 }; bob = 0; flyUp = 0;
  listeners = new Set<() => void>(); raf = 0; last = performance.now(); drag: { x: number; y: number; id: number } | null = null;
  /** Called every frame before rendering (overlays, animations). */
  frameHooks = new Set<(dt: number, t: number) => void>();
  /** Someone else drives the camera (a guided walk, an orbit view): skip the walk controller. */
  manual = false;
  /** adaptive resolution: a 1.4M-splat twin on a laptop GPU drops frames at retina density */
  dprCap = 1.5; private fpsAcc = 0; private fpsN = 0; private lastDrop = 0;
  readonly perf: PerfProfile = deviceProfile(); private born = performance.now();
  /** fps before the last step down, the budget it was taken at, and how many steps in a row bought nothing */
  private dropFps = 0; private dropBudget = Infinity; private idleDrops = 0;
  /** auto adapts; a tier holds that tier's budget and resolution */
  quality: QualityMode = 'auto';
  stats: PerfStats = { fps: 0, ms: 0, worstMs: 0, splats: 0, dpr: 1 };
  private statT = 0; private statN = 0; private statWorst = 0;

  constructor(public canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    this.dprCap = Math.min(devicePixelRatio, this.perf.dpr);
    this.renderer.setPixelRatio(this.dprCap); this.renderer.setSize(innerWidth, innerHeight, false);
    this.scene.background = new THREE.Color('#070B14');
    this.camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, .03, 500); this.camera.rotation.order = 'YXZ';
    // No foveation and no fixed budget by default: the twin is drawn at full detail from the first frame, because a
    // budget that re-selects splats as the camera settles reads as "blurry, then it sharpens when you stare" — which
    // makes the whole model feel cheap at a glance. The budget only comes down if the frame rate actually suffers
    // (see the fps block below), so a fast machine never trades detail for a rate it already has. Older devices start
    // lower (perf.ts), because every step down on the way there is seconds of stutter.
    this.spark = new SparkRenderer({
      renderer: this.renderer,
      lodSplatCount: this.perf.budget,
      coneFov0: 0, coneFov: 0, coneFoveate: 1, behindFoveate: 1,
    }); this.scene.add(this.spark);
    setGlassSource(canvas);
    const q = new URLSearchParams(location.search).has('quality') ? 'auto' : loadMode();  // ?quality= already picked the tier
    if (q !== 'auto') this.setQuality(q, false);
    this.bind(); this.loop();
  }

  /** 0.5 = fewer splats, 1 = Spark's platform default, 2 = double */
  setSplatScale(k: number) { (this.spark as unknown as { lodSplatScale?: number }).lodSplatScale = k; this.emit(); }
  get splatScale() { return (this.spark as unknown as { lodSplatScale?: number }).lodSplatScale ?? 1; }

  /** Hold a tier's splat budget and resolution, or hand control back to the adaptive loop. */
  setQuality(q: QualityMode, remember = true) {
    this.quality = q; if (remember) saveMode(q);
    const spark = this.spark as unknown as { lodSplatCount?: number };
    const t = q === 'auto' ? this.perf : PROFILES[q](this.perf.mobile);
    spark.lodSplatCount = t.budget;
    this.dprCap = Math.min(devicePixelRatio, t.dpr); this.renderer.setPixelRatio(this.dprCap);
    // a fresh start for the adaptive loop, quick to react again
    this.dropFps = 0; this.dropBudget = Infinity; this.idleDrops = 0; this.born = performance.now(); this.lastDrop = 0;
    this.emit();
  }

  on(fn: () => void) { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; }
  emit() { this.listeners.forEach(f => f()); }

  async load(m: SceneManifest, base: URL) {
    this.floorY = m.floorY ?? 0; this.eye = m.eyeHeight ?? 1.55;
    const p = m.start?.position ?? [0, 0, 0]; this.state.pos.set(p[0], this.floorY, p[2]); this.state.yaw = m.start?.yaw ?? 0;
    if (m.occupancy) { try { this.grid = (await OccupancyGrid.load(base, m.occupancy)).grid; } catch { this.grid = null; } }
    // a prebuilt LOD tree streams: the small header arrives first, then only the chunks this camera needs. That is
    // what keeps a 1.3M-splat twin smooth and fast to open; without one, Spark builds the tree in a worker.
    const lodUrl = m.lod ? new URL(m.lod, base).href : null;
    const url = lodUrl ?? new URL(m.splat, base).href;
    this.splat = lodUrl
      ? new SplatMesh({ url, paged: true, onProgress: (e: ProgressEvent) => { if (e.lengthComputable) { this.state.progress = e.loaded / e.total; this.emit(); } } })
      // desktop keeps float32 centres and half-float colour/scale/SH (Spark's 16-byte packing costs ~0.3 dB of the
      // twin's held-out score); phones and weaker devices keep the packed layout, which halves GPU memory
      : new SplatMesh({ url, lod: true, extSplats: this.perf.extSplats, onProgress: (e: ProgressEvent) => { if (e.lengthComputable) { this.state.progress = e.loaded / e.total; this.emit(); } } });
    if (m.splatQuaternion) this.splat.quaternion.set(...m.splatQuaternion);
    if (m.splatPosition) this.splat.position.set(...m.splatPosition);
    this.scene.add(this.splat);
    try { await this.splat.initialized; this.state.loaded = true; this.state.progress = 1; } catch (e) { console.error('[ply] splat failed to load', url, e); this.state.error = (e as Error)?.message || String(e) || 'The scene failed to load'; this.emit(); return; }
    if (!this.grid || m.floorY === undefined) this.deriveFromSplat(m);
    this.collider = this.grid && new GridCollider(this.grid);
    // no start in the manifest: begin in the middle of the walkable floor
    if (!m.start && this.collider) this.openSpawn();
    // a start inside a wall moves to the nearest place the walker fits, when one is near
    const at = this.state.pos, q = this.collider?.nearestFree(at.x, at.z, RADIUS, REACH); if (q) { at.x = q.x; at.z = q.z; }
    this.spawn = { x: at.x, z: at.z };
    this.emit();
  }

  /** No grid or no floor in the manifest: work them out from the splat centres (world space). */
  deriveFromSplat(m: SceneManifest) {
    const sp = this.splat!; sp.updateMatrixWorld(true); const v = new THREE.Vector3(), pts: number[] = [];
    // with LOD on, Spark keeps the splats in the LOD tree, not the mesh's packed array
    const src: { numSplats: number; forEachSplat: SplatMesh['forEachSplat'] } = (sp as any).splats?.lodSplats?.numSplats ? (sp as any).splats.lodSplats : sp;
    src.forEachSplat((_i, c, _s, _q, o) => { if (o > .25) { v.copy(c).applyMatrix4(sp.matrixWorld); pts.push(v.x, v.y, v.z); } });
    if (!pts.length) return;
    const arr = new Float32Array(pts);
    if (m.floorY === undefined) { this.floorY = OccupancyGrid.estimateFloor(arr); this.state.pos.y = this.floorY; }
    if (!this.grid) this.grid = OccupancyGrid.fromPoints(arr, this.floorY);
  }
  /** Spawn where the floor is most open (largest distance to any obstacle), facing the longest clear sightline. */
  openSpawn() {
    const c = this.collider!, o = c.openSpot(); if (!o) return;
    let yaw = 0, far = 0;
    // a sightline 10 cm wide, its samples touching, so no wall is thin enough to pass between two of them
    for (let k = 0; k < 24; k++) { const a = k / 24 * Math.PI * 2, sx = -Math.sin(a), sz = -Math.cos(a); let r = 0; while (r < 30 && c.free(o.x + sx * r, o.z + sz * r, .05)) r += .1; if (r > far) { far = r; yaw = a; } }
    this.state.pos.x = o.x; this.state.pos.z = o.z; this.state.yaw = yaw;
  }
  setMode(mode: Mode) {
    const s = this.state; s.mode = mode;
    if (mode === 'walk') { s.pos.y = this.floorY; s.speed = Math.min(s.speed, WHEEL_WALK); this.unstick(); }
    else s.pos.y = this.floorY + Math.max(0, this.camera.position.y - this.floorY - this.eye);
    this.emit();
  }
  /** If a mode switch or a jump put the walker inside a wall: to the nearest place it fits, or back to the spawn
      point when there is none near. */
  unstick() {
    const c = this.collider, p = this.state.pos; if (!c || c.free(p.x, p.z, RADIUS)) return;
    const q = recover(c, p.x, p.z, this.spawn); if (q) { p.x = q.x; p.z = q.z; }
  }
  lock() { if (matchMedia('(pointer: fine)').matches) this.canvas.requestPointerLock?.(); }

  bind() {
    const kd = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest?.('input,textarea')) return;
      const k = e.code; this.keys.add(k);
      if (k === 'KeyF') this.setMode(this.state.mode === 'walk' ? 'fly' : 'walk');
      if (['Space', 'ArrowUp', 'ArrowDown'].includes(k)) e.preventDefault();
    };
    const ku = (e: KeyboardEvent) => this.keys.delete(e.code);
    addEventListener('keydown', kd); addEventListener('keyup', ku); addEventListener('blur', () => this.keys.clear());
    document.addEventListener('pointerlockchange', () => { this.state.locked = document.pointerLockElement === this.canvas; this.emit(); });
    addEventListener('mousemove', e => { if (this.state.locked) this.look(e.movementX, e.movementY, .0022); });
    // drag to look when the pointer is not locked (trackpads, touch on the right half)
    this.canvas.addEventListener('pointerdown', e => { if (this.state.locked) return; if (e.pointerType === 'touch' && e.clientX < innerWidth * .45) return; this.drag = { x: e.clientX, y: e.clientY, id: e.pointerId }; this.canvas.setPointerCapture(e.pointerId); });
    this.canvas.addEventListener('pointermove', e => { if (!this.drag || e.pointerId !== this.drag.id) return; this.look(e.clientX - this.drag.x, e.clientY - this.drag.y, e.pointerType === 'touch' ? .005 : .004); this.drag.x = e.clientX; this.drag.y = e.clientY; });
    const end = (e: PointerEvent) => { if (this.drag && e.pointerId === this.drag.id) this.drag = null; };
    this.canvas.addEventListener('pointerup', end); this.canvas.addEventListener('pointercancel', end);
    this.canvas.addEventListener('wheel', e => { e.preventDefault(); const s = this.state; s.speed = Math.min(s.mode === 'walk' ? WHEEL_WALK : WHEEL, Math.max(WHEEL_MIN, s.speed * (e.deltaY < 0 ? 1.12 : 1 / 1.12))); this.emit(); }, { passive: false });
    addEventListener('resize', () => { this.renderer.setSize(innerWidth, innerHeight, false); this.camera.aspect = innerWidth / innerHeight; this.camera.updateProjectionMatrix(); });
  }
  look(dx: number, dy: number, s: number) { this.state.yaw -= dx * s; this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch - dy * s)); this.touched(); }
  touched() { if (!this.state.moved) { this.state.moved = true; this.emit(); } }

  step(dt: number) {
    const k = this.keys, s = this.state;
    const f = (k.has('KeyW') || k.has('ArrowUp') ? 1 : 0) - (k.has('KeyS') || k.has('ArrowDown') ? 1 : 0) - this.joy.y;
    const r = (k.has('KeyD') || k.has('ArrowRight') ? 1 : 0) - (k.has('KeyA') || k.has('ArrowLeft') ? 1 : 0) + this.joy.x;
    const run = k.has('ShiftLeft') || k.has('ShiftRight');
    const up = s.mode === 'fly' ? (k.has('Space') || k.has('KeyE') ? 1 : 0) - (k.has('KeyC') || k.has('KeyQ') ? 1 : 0) : 0;
    const base = pace(s.mode === 'walk', run, s.speed);
    // move relative to where we look (fly follows pitch; walk stays on the floor)
    const fw = FW.set(-Math.sin(s.yaw), 0, -Math.cos(s.yaw)), rt = RT.set(Math.cos(s.yaw), 0, -Math.sin(s.yaw));
    if (s.mode === 'fly') fw.set(-Math.sin(s.yaw) * Math.cos(this.pitch), Math.sin(this.pitch), -Math.cos(s.yaw) * Math.cos(this.pitch));
    const want = fw.multiplyScalar(f).add(rt.multiplyScalar(r)); if (want.lengthSq() > 1) want.normalize(); want.multiplyScalar(base); want.y += up * base;
    if (f || r || up) this.touched();
    const p = s.pos, c = s.mode === 'walk' ? this.collider : null;
    if (c) {
      const b = this.body; b.x = p.x; b.z = p.z; b.vx = this.vel.x; b.vz = this.vel.z;
      walkFrame(c, b, want.x, want.z, dt, this.spawn); p.x = b.x; p.z = b.z; this.vel.set(b.vx, 0, b.vz);
    } else { this.vel.lerp(want, ease(dt, want.lengthSq() > 0)); p.x += this.vel.x * dt; p.z += this.vel.z * dt; }
    if (s.mode === 'fly') p.y += this.vel.y * dt; else p.y = this.floorY;
    // one bob every metre and a half: the cadence the walk had at its old 1.4 m/s
    const spd = Math.hypot(this.vel.x, this.vel.z); this.bob += spd * dt * 4.1;
    const bobY = s.mode === 'walk' && !RM ? Math.sin(this.bob) * .018 * Math.min(1, spd / SPEED) : 0;
    this.camera.position.set(p.x, (s.mode === 'walk' ? this.floorY + this.eye : p.y + this.eye) + bobY, p.z);
    this.camera.rotation.set(this.pitch, s.yaw, 0);
  }

  loop = () => {
    this.raf = requestAnimationFrame(this.loop); const now = performance.now(), raw = now - this.last, dt = Math.min(.05, raw / 1000); this.last = now;
    if (document.hidden) { this.statT = this.statN = this.statWorst = 0; return; }
    if (!this.manual) this.step(dt);
    for (const f of this.frameHooks) f(dt, now / 1000);
    this.renderer.render(this.scene, this.camera); glassFrame(dt);
    // a gap over a second is a hidden tab or a breakpoint, not a frame
    if (raw < 1000) { this.statT += raw; this.statN++; this.statWorst = Math.max(this.statWorst, raw); }
    if (this.statT >= 500) {
      const spark = this.spark as unknown as { lodSplatCount?: number }, n = this.splat?.numSplats || 0;
      this.stats = { fps: this.statN * 1000 / this.statT, ms: this.statT / this.statN, worstMs: this.statWorst,
        splats: n ? Math.min(n, spark.lodSplatCount ?? n) : 0, dpr: this.renderer.getPixelRatio() };
      this.statT = this.statN = this.statWorst = 0; this.emit();
    }
    this.fpsAcc += dt; this.fpsN++;
    if (this.fpsAcc >= 1) {
      const fps = this.fpsN / this.fpsAcc; this.fpsAcc = 0; this.fpsN = 0;
      this.adapt(fps, performance.now());
    }
  };

  /** Below 38 fps give back resolution, then splats, then sub-native resolution; above 55 fps restore splats. */
  private adapt(fps: number, now: number) {
    if (this.quality !== 'auto') return;
    const P = this.perf, spark = this.spark as unknown as { lodSplatCount?: number };
    const budget = spark.lodSplatCount ?? P.budget;
    // react quickly while the scene opens (that is when an old device stutters), then settle
    if (now - this.lastDrop < (now - this.born < 10_000 ? 1200 : 2500)) return;
    if (fps >= 38) this.idleDrops = 0;
    if (fps < 38) {
      // two steps in a row that bought nothing: the limit is elsewhere (a 30 Hz low-power cap, the CPU), so stop
      // throwing away detail for it
      if (this.dropFps) this.idleDrops = fps < this.dropFps * 1.05 ? this.idleDrops + 1 : 0;
      if (this.idleDrops >= 2) return;
      this.lastDrop = now; this.dropFps = fps;
      const setDpr = (v: number) => { this.dprCap = v; this.renderer.setPixelRatio(Math.min(devicePixelRatio, v)); };
      if (this.dprCap > 1.0) setDpr(Math.max(1.0, this.dprCap - 0.25));
      else if (budget > P.floor) { this.dropBudget = budget; spark.lodSplatCount = Math.max(P.floor, Math.round(budget * 0.75)); }
      else if (this.dprCap > P.dprMin) setDpr(Math.max(P.dprMin, this.dprCap - 0.25));
      else return;
    } else if (fps > 55 && budget < P.max) {
      // stay under the budget that last dropped frames, so it does not oscillate around it
      const cap = Math.min(P.max, this.dropBudget === Infinity ? P.max : Math.round(this.dropBudget * 0.9));
      if (budget >= cap) return;
      this.lastDrop = now; this.dropFps = 0; this.idleDrops = 0;
      spark.lodSplatCount = Math.min(cap, Math.round(budget * 1.15));
    } else return;
    saveLearned(spark.lodSplatCount ?? budget, this.dprCap);
    this.emit();
  }
  dispose() { cancelAnimationFrame(this.raf); this.splat?.dispose(); this.renderer.dispose(); }
}
