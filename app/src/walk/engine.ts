/* Walkthrough engine: three.js + Spark splat renderer, a walk/fly controller with WASD, mouse look, touch joystick,
   and collision against the pipeline's occupancy grid. The React HUD reads `state` and calls the methods. */
import * as THREE from 'three';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { OccupancyGrid, type SceneManifest } from './scene';
import { glassFrame, setGlassSource } from '../shared/glass';

export type Mode = 'walk' | 'fly';
export interface EngineState { progress: number; loaded: boolean; error: string | null; mode: Mode; locked: boolean; moved: boolean; pos: THREE.Vector3; yaw: number; speed: number; }

const RADIUS = .22, WALK = 1.4, RUN = 2.6, FLY = 2.2;
const RM = matchMedia('(prefers-reduced-motion: reduce)').matches;

export class WalkEngine {
  renderer: THREE.WebGLRenderer; scene = new THREE.Scene(); camera: THREE.PerspectiveCamera; spark: SparkRenderer; splat: SplatMesh | null = null;
  grid: OccupancyGrid | null = null; floorY = 0; eye = 1.55;
  state: EngineState = { progress: 0, loaded: false, error: null, mode: 'walk', locked: false, moved: false, pos: new THREE.Vector3(), yaw: 0, speed: 1 };
  pitch = 0; vel = new THREE.Vector3(); keys = new Set<string>(); joy = { x: 0, y: 0 }; bob = 0; flyUp = 0;
  listeners = new Set<() => void>(); raf = 0; last = performance.now(); drag: { x: number; y: number; id: number } | null = null;
  /** Called every frame before rendering (overlays, animations). */
  frameHooks = new Set<(dt: number, t: number) => void>();
  /** Someone else drives the camera (a guided walk, an orbit view): skip the walk controller. */
  manual = false;
  /** adaptive resolution: a 1.4M-splat twin on a laptop GPU drops frames at retina density */
  dprCap = 1.5; private fpsAcc = 0; private fpsN = 0; private lastDrop = 0;

  constructor(public canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5)); this.renderer.setSize(innerWidth, innerHeight, false);
    this.dprCap = Math.min(devicePixelRatio, 1.5);
    this.scene.background = new THREE.Color('#070B14');
    this.camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, .03, 500); this.camera.rotation.order = 'YXZ';
    // Spark's LOD budget defaults to 2.5M splats on desktop, i.e. it renders our whole 1.35M twin every frame and
    // there is nothing to stream down. Pin the budget lower and shape it: detail in front, coarse behind and at the
    // edges (this is the part that makes the twin feel smooth while walking). lodSplatScale can be raised again from
    // the Views menu on a fast machine.
    const mobile = matchMedia('(pointer: coarse), (max-width: 720px)').matches;
    this.spark = new SparkRenderer({
      renderer: this.renderer,
      lodSplatCount: mobile ? 500_000 : 900_000,
      coneFov0: 55, coneFov: 110, coneFoveate: 0.55, behindFoveate: 0.15,
    }); this.scene.add(this.spark);
    setGlassSource(canvas);
    this.bind(); this.loop();
  }

  /** 0.5 = fewer splats, 1 = Spark's platform default, 2 = double */
  setSplatScale(k: number) { (this.spark as unknown as { lodSplatScale?: number }).lodSplatScale = k; this.emit(); }
  get splatScale() { return (this.spark as unknown as { lodSplatScale?: number }).lodSplatScale ?? 1; }

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
      : new SplatMesh({ url, lod: true, onProgress: (e: ProgressEvent) => { if (e.lengthComputable) { this.state.progress = e.loaded / e.total; this.emit(); } } });
    if (m.splatQuaternion) this.splat.quaternion.set(...m.splatQuaternion);
    if (m.splatPosition) this.splat.position.set(...m.splatPosition);
    this.scene.add(this.splat);
    try { await this.splat.initialized; this.state.loaded = true; this.state.progress = 1; } catch (e) { console.error('[ply] splat failed to load', url, e); this.state.error = (e as Error)?.message || String(e) || 'The scene failed to load'; this.emit(); return; }
    if (!this.grid || m.floorY === undefined) this.deriveFromSplat(m);
    // no start in the manifest: begin in the middle of the walkable floor
    if (!m.start && this.grid) this.openSpawn();
    if (this.grid && !this.grid.clear(this.state.pos.x, this.state.pos.z, RADIUS)) this.unstick();
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
    const g = this.grid!, { width: w, height: h, cell, origin } = g.meta, d = new Float32Array(w * h);
    // two-pass chamfer distance transform over the blocked cells
    for (let i = 0; i < w * h; i++) d[i] = g.data[i] ? 0 : 1e6;
    const pass = (y0: number, y1: number, dy: number, x0: number, x1: number, dx: number) => {
      for (let y = y0; y !== y1; y += dy) for (let x = x0; x !== x1; x += dx) {
        const i = y * w + x; if (!d[i]) continue;
        const nb = [[-dx, 0, 1], [0, -dy, 1], [-dx, -dy, 1.414], [dx, -dy, 1.414]];
        for (const [ox, oy, c] of nb) { const X = x + ox, Y = y + oy; if (X >= 0 && Y >= 0 && X < w && Y < h) d[i] = Math.min(d[i], d[Y * w + X] + c); }
      }
    };
    pass(0, h, 1, 0, w, 1); pass(h - 1, -1, -1, w - 1, -1, -1);
    let best = -1, bi = 0; for (let i = 0; i < w * h; i++) if (d[i] < 1e6 && d[i] > best) { best = d[i]; bi = i; }
    const x = origin[0] + (bi % w + .5) * cell, z = origin[1] + (Math.floor(bi / w) + .5) * cell;
    let yaw = 0, far = 0;
    for (let k = 0; k < 24; k++) { const a = k / 24 * Math.PI * 2, sx = -Math.sin(a), sz = -Math.cos(a); let r = 0; while (r < 30 && !g.blocked(x + sx * r, z + sz * r)) r += cell; if (r > far) { far = r; yaw = a; } }
    this.state.pos.x = x; this.state.pos.z = z; this.state.yaw = yaw;
  }
  setMode(mode: Mode) {
    this.state.mode = mode;
    if (mode === 'walk') { this.state.pos.y = this.floorY; if (this.grid && !this.grid.clear(this.state.pos.x, this.state.pos.z, RADIUS)) this.unstick(); }
    else this.state.pos.y = this.floorY + Math.max(0, this.camera.position.y - this.floorY - this.eye);
    this.emit();
  }
  /** Nearest free cell, if a mode switch or a bad start put us inside a wall. */
  unstick() {
    const g = this.grid!, p = this.state.pos;
    for (let r = .1; r < 6; r += .1) for (let k = 0; k < 16; k++) { const a = k * Math.PI / 8, x = p.x + Math.cos(a) * r, z = p.z + Math.sin(a) * r; if (g.clear(x, z, RADIUS)) { p.x = x; p.z = z; return; } }
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
    this.canvas.addEventListener('wheel', e => { e.preventDefault(); this.state.speed = Math.min(4, Math.max(.25, this.state.speed * (e.deltaY < 0 ? 1.12 : 1 / 1.12))); this.emit(); }, { passive: false });
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
    const base = (s.mode === 'walk' ? (run ? RUN : WALK) : FLY * (run ? 2.2 : 1)) * s.speed;
    // move relative to where we look (fly follows pitch; walk stays on the floor)
    const fw = new THREE.Vector3(-Math.sin(s.yaw), 0, -Math.cos(s.yaw)), rt = new THREE.Vector3(Math.cos(s.yaw), 0, -Math.sin(s.yaw));
    if (s.mode === 'fly') fw.set(-Math.sin(s.yaw) * Math.cos(this.pitch), Math.sin(this.pitch), -Math.cos(s.yaw) * Math.cos(this.pitch));
    const want = fw.multiplyScalar(f).add(rt.multiplyScalar(r)); if (want.lengthSq() > 1) want.normalize(); want.multiplyScalar(base); want.y += up * base;
    if (f || r || up) this.touched();
    const a = 1 - Math.exp(-dt * (want.lengthSq() > 0 ? 10 : 7)); this.vel.lerp(want, a);
    const p = s.pos, dx = this.vel.x * dt, dz = this.vel.z * dt;
    if (s.mode === 'walk' && this.grid) { // axis-separated so we slide along walls
      if (this.grid.clear(p.x + dx, p.z, RADIUS)) p.x += dx; else this.vel.x = 0;
      if (this.grid.clear(p.x, p.z + dz, RADIUS)) p.z += dz; else this.vel.z = 0;
    } else { p.x += dx; p.z += dz; }
    if (s.mode === 'fly') p.y += this.vel.y * dt; else p.y = this.floorY;
    const spd = Math.hypot(this.vel.x, this.vel.z); this.bob += spd * dt * 6.5;
    const bobY = s.mode === 'walk' && !RM ? Math.sin(this.bob) * .018 * Math.min(1, spd / WALK) : 0;
    this.camera.position.set(p.x, (s.mode === 'walk' ? this.floorY + this.eye : p.y + this.eye) + bobY, p.z);
    this.camera.rotation.set(this.pitch, s.yaw, 0);
  }

  loop = () => {
    this.raf = requestAnimationFrame(this.loop); const now = performance.now(), dt = Math.min(.05, (now - this.last) / 1000); this.last = now;
    if (document.hidden) return;
    if (!this.manual) this.step(dt);
    for (const f of this.frameHooks) f(dt, now / 1000);
    this.renderer.render(this.scene, this.camera); glassFrame(dt);
    // drop the render resolution (never below 0.75) when the last second averaged under 42 fps
    this.fpsAcc += dt; this.fpsN++;
    if (this.fpsAcc >= 1) {
      const fps = this.fpsN / this.fpsAcc; this.fpsAcc = 0; this.fpsN = 0;
      const now = performance.now();
      if (fps < 42 && this.dprCap > 0.76 && now - this.lastDrop > 2500) {
        this.lastDrop = now; this.dprCap = Math.max(0.75, this.dprCap - 0.25);
        this.renderer.setPixelRatio(Math.min(devicePixelRatio, this.dprCap));
        this.emit();
      }
    }
  };
  dispose() { cancelAnimationFrame(this.raf); this.splat?.dispose(); this.renderer.dispose(); }
}
