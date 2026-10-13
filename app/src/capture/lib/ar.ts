/* AR capture (Android Chrome WebXR): Polycam / RealityScan-style live coverage.
   - ARCore tracks the phone (metres, gravity-aligned, floor at y = 0 with 'local-floor').
   - Coverage from depth (coverage.ts): 8 cm cells on the surfaces depth hits, drawn as dots once confirmed, coloured by
     how many keyframes saw them and from how many directions.
   - Keyframes: the camera image (886x1920 on the S24 FE) is read back when the phone has moved or turned enough and is
     moving slowly enough to be sharp; JPEG + pose + lens go into one tar file streamed to the phone's storage.
   The pipeline reads the tar: frames go straight to SfM and the poses give real scale and a level floor. */
import * as THREE from 'three';
import { openSink } from './store';
import { tarHeader, tarPad } from './tar';
import { Coverage } from './coverage';

export const GREEN = '#3DDC97', ORANGE = '#FFB84D', RED = '#F2555A';
const KF_MOVE = .1, KF_TURN = 7 * Math.PI / 180, KF_GAP = 220, MAX_TURN = 70, MAX_SPEED = .8;

export interface ArKeyframe { file: string; t: number; m: number[]; fx: number; fy: number; cx: number; cy: number; w: number; h: number; }
export interface ArStats { tracking: 'starting' | 'ok' | 'lost'; keyframes: number; seconds: number; green: number; orange: number; red: number; voxels: number;
  tooFast: boolean; walked: number; guide: { angle: number; dist: number } | null; depth: boolean; camera: string | null; bytes: number; thumb: string | null; }
export interface ArResult { tar: string; bytes: number; keyframes: ArKeyframe[]; seconds: number; coverage: { green: number; orange: number; red: number; voxels: number };
  points: Float32Array; colors: Float32Array; path: { x: number; z: number; t: number }[]; camera: string | null; }

export async function arSupported() { try { return !!(await (navigator as any).xr?.isSessionSupported('immersive-ar')); } catch { return false; } }

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
  cov = new Coverage(dotsMaterial(.05)); keyframes: ArKeyframe[] = []; stats: ArStats = { tracking: 'starting', keyframes: 0, seconds: 0, green: 0, orange: 0, red: 0, voxels: 0, tooFast: false, walked: 0, guide: null, depth: false, camera: null, bytes: 0, thumb: null };
  path: { x: number; z: number; t: number }[] = [];
  private renderer!: THREE.WebGLRenderer; private scene = new THREE.Scene(); private cam = new THREE.PerspectiveCamera();
  private session: any = null; private t0 = 0; private frameNo = 0;
  private last: { p: THREE.Vector3; q: THREE.Quaternion; t: number } | null = null; private lastKf: { p: THREE.Vector3; q: THREE.Quaternion; t: number } | null = null;
  private sink: Awaited<ReturnType<typeof openSink>> | null = null; private pending = 0; private fb: WebGLFramebuffer | null = null; private pixels: Uint8Array | null = null;
  private listeners = new Set<() => void>(); private ended: ((r: ArResult) => void) | null = null; recording = false;
  /** 'scan': a keyframe each time the phone moves ~10 cm or turns 7 deg. 'video': a steady 4 frames a second while
      moving (video mode on AR phones: the browser cannot record normal video while AR runs, so AR records the frames). */
  constructor(public id: string, public mode: 'scan' | 'video' = 'scan') {}
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
    this.scene.add(this.cov.points);
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
      (!this.lastKf || (this.mode === 'video'
        ? now - this.lastKf.t > 250 && (this.lastKf.p.distanceTo(p) > .01 || this.lastKf.q.angleTo(q) > .017) // skip only when standing still
        : now - this.lastKf.t > KF_GAP && (this.lastKf.p.distanceTo(p) > KF_MOVE || this.lastKf.q.angleTo(q) > KF_TURN)));
    let kfId = -1;
    if (want && this.grab(view, m)) { kfId = this.keyframes.length - 1; this.lastKf = { p: p.clone(), q: q.clone(), t: now }; }
    // depth into the coverage map: every third frame, and always on keyframes so their views count
    if (this.stats.depth && this.stats.tracking === 'ok' && (kfId >= 0 || this.frameNo % 3 === 0)) {
      try { const d = frame.getDepthInformation(view); if (d) this.cov.integrate(d, view.projectionMatrix, m, kfId); } catch { /* depth not ready yet */ }
    }
    if (this.frameNo % 4 === 0) this.cov.flush();
    const sm = this.cov; this.stats.green = sm.counts[0]; this.stats.orange = sm.counts[1]; this.stats.red = sm.counts[2]; this.stats.voxels = sm.counts[0] + sm.counts[1] + sm.counts[2];
    if (this.frameNo % 15 === 0) { const r = sm.nearestRed(p.x, p.z); if (r) { const want = Math.atan2(-(r[0] - p.x), -(r[1] - p.z)), yaw = new THREE.Euler().setFromQuaternion(q, 'YXZ').y; let a = want - yaw; while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; this.stats.guide = { angle: a, dist: Math.hypot(r[0] - p.x, r[1] - p.z) }; } else this.stats.guide = null; }
    this.renderer.render(this.scene, this.cam);
    if (this.frameNo % 6 === 0) this.emit();
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
    oc.convertToBlob({ type: 'image/jpeg', quality: .92 }).then(b => { this.sink!.write(new Blob([tarHeader(file, b.size), b, tarPad(b.size)])); this.stats.bytes += b.size + 1024;
      if (this.stats.thumb) URL.revokeObjectURL(this.stats.thumb); this.stats.thumb = URL.createObjectURL(b); })
      .finally(() => { this.pending--; });
    return true;
  }

  private async finish() {
    if (!this.sink) return; const sink = this.sink; this.sink = null; this.renderer.setAnimationLoop(null);
    while (this.pending > 0) await new Promise(r => setTimeout(r, 50));
    const meta = new TextEncoder().encode(JSON.stringify({ version: 1, kind: 'ply-ar', space: 'local-floor (metres, +y up, floor y=0)', camera: 'OpenGL (x right, y up, looking -z); m = camera-to-world, column-major',
      device: navigator.userAgent, keyframes: this.keyframes, coverage: { green: this.cov.counts[0], orange: this.cov.counts[1], red: this.cov.counts[2], dots: this.stats.voxels, cellM: .08 } }));
    sink.write(new Blob([tarHeader('poses.json', meta.length), meta, tarPad(meta.length), new Uint8Array(1024)])); await sink.close('application/x-tar');
    this.cov.flush(); const ex = this.cov.export(), c = this.cov.counts; const r: ArResult = { tar: this.id, bytes: this.stats.bytes + meta.length + 2048, keyframes: this.keyframes, seconds: this.stats.seconds,
      coverage: { green: c[0], orange: c[1], red: c[2], voxels: c[0] + c[1] + c[2] }, points: ex.pts, colors: ex.col, path: this.path, camera: this.stats.camera };
    if (this.stats.thumb) URL.revokeObjectURL(this.stats.thumb); this.stats.thumb = null; this.renderer.dispose(); this.ended?.(r);
  }
}
