/* AR capture (Android Chrome WebXR): Polycam / RealityScan-style live coverage.
   - ARCore tracks the phone (metres, gravity-aligned, floor at y = 0 with 'local-floor').
   - Depth (160x90 on the S24 FE) is unprojected every other frame into a 5 cm voxel map; each voxel remembers how many
     keyframes saw it and from how many directions, and is drawn as a dot coloured green / orange / red.
   - Keyframes: the camera image (886x1920 on the S24 FE) is read back when the phone has moved or turned enough and is
     moving slowly enough to be sharp; JPEG + pose + lens go into one tar file streamed to the phone's storage.
   The pipeline reads the tar: frames go straight to SfM and the poses give real scale and a level floor. */
import * as THREE from 'three';
import { openSink } from './store';
import { tarHeader, tarPad } from './tar';

export const GREEN = '#3DDC97', ORANGE = '#FFB84D', RED = '#F2555A';
const VOX = .05, CAP = 400_000, KF_MOVE = .1, KF_TURN = 7 * Math.PI / 180, KF_GAP = 220, MAX_TURN = 70, MAX_SPEED = .8;

export interface ArKeyframe { file: string; t: number; m: number[]; fx: number; fy: number; cx: number; cy: number; w: number; h: number; }
export interface ArStats { tracking: 'starting' | 'ok' | 'lost'; keyframes: number; seconds: number; green: number; orange: number; red: number; voxels: number;
  tooFast: boolean; walked: number; guide: { angle: number; dist: number } | null; depth: boolean; camera: string | null; bytes: number; }
export interface ArResult { tar: string; bytes: number; keyframes: ArKeyframe[]; seconds: number; coverage: { green: number; orange: number; red: number; voxels: number };
  points: Float32Array; colors: Float32Array; path: { x: number; z: number; t: number }[]; camera: string | null; }

export async function arSupported() { try { return !!(await (navigator as any).xr?.isSessionSupported('immersive-ar')); } catch { return false; } }

/** 5 cm voxels from depth. kf = number of keyframes that saw the voxel, dirs = 12 azimuth bins it was seen from. */
export class VoxelMap {
  index = new Map<number, number>(); n = 0;
  pos = new Float32Array(CAP * 3); col = new Float32Array(CAP * 3); kf = new Uint16Array(CAP); dirs = new Uint16Array(CAP);
  near = new Float32Array(CAP).fill(99); lastKf = new Int32Array(CAP).fill(-1); grade = new Uint8Array(CAP); counts = [0, 0, 0];
  dirtyLo = Infinity; dirtyHi = -1;
  private c = [new THREE.Color(GREEN), new THREE.Color(ORANGE), new THREE.Color(RED)];
  add(x: number, y: number, z: number, camX: number, camZ: number, dist: number, kfId: number) {
    const ix = Math.floor(x / VOX) + 512, iy = Math.floor(y / VOX) + 512, iz = Math.floor(z / VOX) + 512;
    if (ix < 0 || iy < 0 || iz < 0 || ix > 1023 || iy > 1023 || iz > 1023) return;
    const key = (ix << 20) | (iy << 10) | iz; let i = this.index.get(key);
    if (i === undefined) {
      if (this.n >= CAP) return; i = this.n++; this.index.set(key, i);
      this.pos[i * 3] = (ix - 512 + .5) * VOX; this.pos[i * 3 + 1] = (iy - 512 + .5) * VOX; this.pos[i * 3 + 2] = (iz - 512 + .5) * VOX; this.grade[i] = 255;
    }
    if (dist < this.near[i]) this.near[i] = dist;
    if (kfId >= 0 && this.lastKf[i] !== kfId) {
      this.lastKf[i] = kfId; if (this.kf[i] < 65535) this.kf[i]++;
      const az = Math.atan2(camX - x, camZ - z); this.dirs[i] |= 1 << (Math.floor((az + Math.PI) / (Math.PI * 2) * 12) % 12);
    }
    this.regrade(i);
  }
  /** green: 3+ keyframes from 2+ directions within 3 m; orange: in at least one keyframe; red: only glimpsed. */
  private regrade(i: number) {
    let bins = 0; for (let d = this.dirs[i]; d; d &= d - 1) bins++;
    const g = this.kf[i] >= 3 && bins >= 2 && this.near[i] <= 3 ? 0 : this.kf[i] >= 1 ? 1 : 2;
    if (g === this.grade[i]) return;
    if (this.grade[i] !== 255) this.counts[this.grade[i]]--; this.counts[g]++; this.grade[i] = g;
    const c = this.c[g]; this.col[i * 3] = c.r; this.col[i * 3 + 1] = c.g; this.col[i * 3 + 2] = c.b;
    this.dirtyLo = Math.min(this.dirtyLo, i); this.dirtyHi = Math.max(this.dirtyHi, i);
  }
  /** Direction (world x, z) to the nearest patch that still needs work, for the on-screen guide. */
  nearestRed(x: number, y: number, z: number) {
    let best = -1, bd = Infinity;
    for (let i = 0; i < this.n; i += 3) { if (this.grade[i] !== 2) continue; const dx = this.pos[i * 3] - x, dy = this.pos[i * 3 + 1] - y, dz = this.pos[i * 3 + 2] - z, d = dx * dx + dy * dy * .3 + dz * dz; if (d > .5 && d < bd) { bd = d; best = i; } }
    return best < 0 ? null : [this.pos[best * 3], this.pos[best * 3 + 1], this.pos[best * 3 + 2]] as const;
  }
}

/** Round dots sized in metres, so the cloud reads as a surface up close and as a point cloud from afar. */
export function dotsMaterial(size = .042) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, vertexColors: true, uniforms: { uSize: { value: size }, uScale: { value: 800 }, uAlpha: { value: .8 } },
    vertexShader: `uniform float uSize; uniform float uScale; varying vec3 vC; void main(){ vC = color; vec4 mv = modelViewMatrix * vec4(position,1.); gl_Position = projectionMatrix * mv; gl_PointSize = clamp(uSize * uScale / -mv.z, 1.5, 28.); }`,
    fragmentShader: `uniform float uAlpha; varying vec3 vC; void main(){ vec2 p = gl_PointCoord - .5; float r = dot(p,p); if (r > .25) discard; gl_FragColor = vec4(vC, uAlpha * smoothstep(.25, .12, r)); }`,
  });
}


/** One AR capture session. start() needs a user gesture; the overlay element becomes the DOM overlay in AR. */
export class ArCapture {
  map = new VoxelMap(); keyframes: ArKeyframe[] = []; stats: ArStats = { tracking: 'starting', keyframes: 0, seconds: 0, green: 0, orange: 0, red: 0, voxels: 0, tooFast: false, walked: 0, guide: null, depth: false, camera: null, bytes: 0 };
  path: { x: number; z: number; t: number }[] = [];
  private renderer!: THREE.WebGLRenderer; private scene = new THREE.Scene(); private cam = new THREE.PerspectiveCamera();
  private geo = new THREE.BufferGeometry(); private session: any = null; private t0 = 0; private frameNo = 0;
  private last: { p: THREE.Vector3; q: THREE.Quaternion; t: number } | null = null; private lastKf: { p: THREE.Vector3; q: THREE.Quaternion; t: number } | null = null;
  private sink: Awaited<ReturnType<typeof openSink>> | null = null; private pending = 0; private fb: WebGLFramebuffer | null = null; private pixels: Uint8Array | null = null;
  private listeners = new Set<() => void>(); private ended: ((r: ArResult) => void) | null = null; recording = false;
  constructor(public id: string) {}
  on(f: () => void) { this.listeners.add(f); return () => { this.listeners.delete(f); }; }
  private emit() { this.listeners.forEach(f => f()); }

  async start(overlay: HTMLElement) {
    const xr = (navigator as any).xr;
    this.session = await xr.requestSession('immersive-ar', {
      requiredFeatures: ['local-floor'], optionalFeatures: ['depth-sensing', 'camera-access', 'dom-overlay', 'hit-test'],
      depthSensing: { usagePreference: ['cpu-optimized'], dataFormatPreference: ['luminance-alpha', 'float32'] }, domOverlay: { root: overlay },
    });
    this.stats.depth = this.session.depthUsage === 'cpu-optimized';
    this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: false }); this.renderer.setPixelRatio(1);
    this.renderer.xr.enabled = true; this.renderer.xr.setReferenceSpaceType('local-floor'); await this.renderer.xr.setSession(this.session);
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.map.pos, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('color', new THREE.BufferAttribute(this.map.col, 3).setUsage(THREE.DynamicDrawUsage)); this.geo.setDrawRange(0, 0);
    const dots = new THREE.Points(this.geo, dotsMaterial()); dots.frustumCulled = false; this.scene.add(dots);
    this.sink = await openSink(this.id); this.t0 = performance.now();
    this.session.addEventListener('end', () => this.finish());
    this.renderer.setAnimationLoop((t: number, frame: any) => this.onFrame(t, frame));
  }
  stop() { this.recording = false; try { this.session?.end(); } catch { this.finish(); } }
  result() { return new Promise<ArResult>(res => { this.ended = res; }); }

  private onFrame(_t: number, frame: any) {
    if (!frame) return; this.frameNo++;
    const ref = this.renderer.xr.getReferenceSpace(), pose = frame.getViewerPose(ref);
    if (!pose) { this.stats.tracking = this.stats.keyframes ? 'lost' : 'starting'; this.emit(); return; }
    this.stats.tracking = pose.emulatedPosition ? 'lost' : 'ok';
    const view = pose.views[0], m = view.transform.matrix as Float32Array, now = performance.now();
    const p = new THREE.Vector3(m[12], m[13], m[14]), q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().fromArray(m));
    // speed: sharp frames need slow, steady motion
    let tooFast = false;
    if (this.last) { const dt = Math.max(1e-3, (now - this.last.t) / 1000), turn = this.last.q.angleTo(q) / dt * 180 / Math.PI, speed = this.last.p.distanceTo(p) / dt; tooFast = turn > MAX_TURN || speed > MAX_SPEED; this.stats.walked += this.last.p.distanceTo(p); }
    this.stats.tooFast = tooFast; this.last = { p, q, t: now };
    if (this.recording) { this.stats.seconds = (now - this.t0) / 1000; if (!this.path.length || Math.hypot(p.x - this.path[this.path.length - 1].x, p.z - this.path[this.path.length - 1].z) > .1) this.path.push({ x: p.x, z: p.z, t: now - this.t0 }); }
    // keyframe? moved or turned enough since the last one, steady, tracked, encoder not backed up
    const want = this.recording && this.stats.tracking === 'ok' && !tooFast && this.pending < 3 && !!view.camera &&
      (!this.lastKf || (now - this.lastKf.t > KF_GAP && (this.lastKf.p.distanceTo(p) > KF_MOVE || this.lastKf.q.angleTo(q) > KF_TURN)));
    let kfId = -1;
    if (want && this.grab(view, m)) { kfId = this.keyframes.length - 1; this.lastKf = { p: p.clone(), q: q.clone(), t: now }; }
    // depth into the voxel map: every other frame, and always on keyframes so their views count
    if (this.stats.depth && (kfId >= 0 || this.frameNo % 2 === 0) && this.stats.tracking === 'ok') {
      try { const d = frame.getDepthInformation(view); if (d) this.integrate(d, view.projectionMatrix, m, p, kfId); } catch { /* depth not ready yet */ }
    }
    const mp = this.map; this.stats.green = mp.counts[0]; this.stats.orange = mp.counts[1]; this.stats.red = mp.counts[2]; this.stats.voxels = mp.n;
    if (this.frameNo % 15 === 0) { const r = mp.nearestRed(p.x, p.y, p.z); if (r) { const want = Math.atan2(-(r[0] - p.x), -(r[2] - p.z)), yaw = new THREE.Euler().setFromQuaternion(q, 'YXZ').y; let a = want - yaw; while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; this.stats.guide = { angle: a, dist: Math.hypot(r[0] - p.x, r[2] - p.z) }; } else this.stats.guide = null; }
    // upload only what changed in the dot buffers
    if (mp.dirtyHi >= 0) { const c = this.geo.attributes.color as THREE.BufferAttribute, ps = this.geo.attributes.position as THREE.BufferAttribute;
      c.clearUpdateRanges(); ps.clearUpdateRanges(); c.addUpdateRange(mp.dirtyLo * 3, (mp.dirtyHi - mp.dirtyLo + 1) * 3); ps.addUpdateRange(mp.dirtyLo * 3, (mp.dirtyHi - mp.dirtyLo + 1) * 3);
      c.needsUpdate = ps.needsUpdate = true; mp.dirtyLo = Infinity; mp.dirtyHi = -1; this.geo.setDrawRange(0, mp.n); }
    this.renderer.render(this.scene, this.cam);
    if (this.frameNo % 6 === 0) this.emit();
  }

  /** Depth samples -> world points (view space from the projection, then the camera pose). */
  private integrate(d: any, P: Float32Array, m: Float32Array, cam: THREE.Vector3, kfId: number) {
    const NX = 24, NY = 44, v = new THREE.Vector3(), M = new THREE.Matrix4().fromArray(m);
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      const u = (i + .5) / NX, w = (j + .5) / NY; let z: number;
      try { z = d.getDepthInMeters(u, w); } catch { continue; }
      if (!(z > .2 && z < 5)) continue;
      const xn = u * 2 - 1, yn = 1 - w * 2;
      v.set((xn + P[8]) * z / P[0], (yn + P[9]) * z / P[5], -z).applyMatrix4(M);
      this.map.add(v.x, v.y, v.z, cam.x, cam.z, z, kfId);
    }
  }

  /** Read the camera image back, flip it upright, encode JPEG off the frame loop, append to the tar. */
  private grab(view: any, m: Float32Array) {
    const binding = this.renderer.xr.getBinding?.() as any; if (!binding) return false;
    const gl = this.renderer.getContext() as WebGL2RenderingContext, cam = view.camera, w = cam.width, h = cam.height;
    let tex: WebGLTexture | null = null; try { tex = binding.getCameraImage(cam); } catch { return false; } if (!tex) return false;
    this.stats.camera = `${w}x${h}`;
    if (!this.fb) this.fb = gl.createFramebuffer(); if (!this.pixels || this.pixels.length !== w * h * 4) this.pixels = new Uint8Array(w * h * 4);
    const prev = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    if (ok) gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, this.pixels);
    gl.bindFramebuffer(gl.FRAMEBUFFER, prev); this.renderer.resetState(); if (!ok) return false;
    const P = view.projectionMatrix as Float32Array, idx = this.keyframes.length, file = `frames/${String(idx).padStart(5, '0')}.jpg`;
    this.keyframes.push({ file, t: Math.round(performance.now() - this.t0), m: Array.from(m, x => +x.toFixed(6)), fx: P[0] * w / 2, fy: P[5] * h / 2, cx: w * (1 - P[8]) / 2, cy: h * (1 + P[9]) / 2, w, h });
    this.stats.keyframes = this.keyframes.length;
    // rows come bottom-up from GL; flip into an ImageData and encode asynchronously
    const img = new ImageData(w, h), src = this.pixels, dst = img.data, row = w * 4;
    for (let y = 0; y < h; y++) dst.set(src.subarray((h - 1 - y) * row, (h - y) * row), y * row);
    this.pending++;
    const oc = new OffscreenCanvas(w, h); oc.getContext('2d')!.putImageData(img, 0, 0);
    oc.convertToBlob({ type: 'image/jpeg', quality: .92 }).then(b => { this.sink!.write(new Blob([tarHeader(file, b.size), b, tarPad(b.size)])); this.stats.bytes += b.size + 1024; })
      .finally(() => { this.pending--; });
    return true;
  }

  private async finish() {
    if (!this.sink) return; const sink = this.sink; this.sink = null; this.renderer.setAnimationLoop(null);
    while (this.pending > 0) await new Promise(r => setTimeout(r, 50));
    const meta = new TextEncoder().encode(JSON.stringify({ version: 1, kind: 'ply-ar', space: 'local-floor (metres, +y up, floor y=0)', camera: 'OpenGL (x right, y up, looking -z); m = camera-to-world, column-major',
      device: navigator.userAgent, keyframes: this.keyframes, coverage: { green: this.map.counts[0], orange: this.map.counts[1], red: this.map.counts[2], voxels: this.map.n, voxelM: VOX } }));
    sink.write(new Blob([tarHeader('poses.json', meta.length), meta, tarPad(meta.length), new Uint8Array(1024)])); await sink.close('application/x-tar');
    const mp = this.map; const r: ArResult = { tar: this.id, bytes: this.stats.bytes + meta.length + 2048, keyframes: this.keyframes, seconds: this.stats.seconds,
      coverage: { green: mp.counts[0], orange: mp.counts[1], red: mp.counts[2], voxels: mp.n }, points: mp.pos.slice(0, mp.n * 3), colors: mp.col.slice(0, mp.n * 3), path: this.path, camera: this.stats.camera };
    this.renderer.dispose(); this.ended?.(r);
  }
}
