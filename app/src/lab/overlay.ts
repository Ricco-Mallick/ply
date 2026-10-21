/* Everything the lab draws on top of the splat: place pins (HTML, projected every frame), routes (three.js, three
   styles), AI detections (wire boxes or a tint inside the splat via Spark's SDF edits), the tape measure, and the AR
   mock-ups (floor arrows, path line, coverage paint, dollhouse cut). One instance per engine; Lab.tsx sets options. */
import * as THREE from 'three';
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { SplatEdit, SplatEditSdf, SplatEditSdfType, SplatEditRgbaBlendMode } from '@sparkjsdev/spark';
import type { WalkEngine } from '../walk/engine';
import type { Navigator, Place } from './places';

export type PinStyle = 'tag' | 'beacon' | 'dot';
export type RouteStyle = 'ribbon' | 'dots' | 'tube';
export type AiStyle = 'off' | 'boxes' | 'tint' | 'review';
export type ArMode = 'off' | 'nav' | 'capture' | 'doll';
export type ArNav = 'arrows' | 'line' | 'compass';
export type ArCap = 'floor' | 'trail';

const ORANGE = new THREE.Color('#FF6B00');
const CLASS_COL: Record<string, string> = { desk: '#FF8A3D', laptop: '#5CC8FF', 'potted plant': '#3DDC97', door: '#B79CFF', helmet: '#FFD24D', clothing: '#FF6FA8', shelf: '#7FE0D0', chair: '#FF6B00', bed: '#5CC8FF', wardrobe: '#FF8A3D', window: '#7FE0D0', fireplace: '#FF6FA8', lamp: '#FFD24D', nightstand: '#B79CFF', radiator: '#8FA3C0', boat: '#3DDC97' };
export const classColor = (c: string) => CLASS_COL[c] ?? '#FFB84D';

/** Smooth a polyline of floor points into evenly spaced samples (Catmull-Rom through the corners). */
function smooth(path: [number, number][], y: number, step = .08) {
  if (path.length < 2) return [] as THREE.Vector3[];
  const curve = new THREE.CatmullRomCurve3(path.map(([x, z]) => new THREE.Vector3(x, y, z)), false, 'centripetal', .2);
  const n = Math.max(2, Math.ceil(curve.getLength() / step)); return curve.getSpacedPoints(n);
}

const ribbonMat = () => new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
  uniforms: { uTime: { value: 0 }, uCol: { value: ORANGE }, uLen: { value: 1 } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.); }`,
  // uv.x = metres along the route, uv.y = across (0..1): chevrons that flow toward the destination
  fragmentShader: `uniform float uTime; uniform vec3 uCol; uniform float uLen; varying vec2 vUv;
    void main(){ float across = abs(vUv.y - .5) * 2.; float edge = smoothstep(1., .82, across);
      float s = fract((vUv.x - uTime * .9) * 2.2 - across * .35); float chev = smoothstep(.0, .12, s) * smoothstep(.5, .3, s);
      float endFade = smoothstep(0., .6, vUv.x) * smoothstep(uLen, uLen - .5, vUv.x);
      float a = (0.12 * edge + 0.66 * chev * edge + .3 * smoothstep(.86, .97, across) * (1. - smoothstep(.97, 1., across))) * endFade;
      gl_FragColor = vec4(uCol * (0.7 + chev * .6), a); }`,
});

const tubeMat = () => new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
  uniforms: { uTime: { value: 0 }, uCol: { value: ORANGE }, uLen: { value: 1 } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.); }`,
  fragmentShader: `uniform float uTime; uniform vec3 uCol; uniform float uLen; varying vec2 vUv;
    void main(){ float m = vUv.x * uLen; float pulse = pow(.5 + .5 * sin((m - uTime * 1.6) * 3.2), 6.);
      float a = (.28 + .7 * pulse) * smoothstep(0., .4, m) * smoothstep(uLen, uLen - .4, m);
      gl_FragColor = vec4(mix(uCol, vec3(1., .9, .7), pulse * .6), a); }`,
});

/** Screen-space thick line segments (WebGL lines are always 1 px). pairs = flat xyz list, two points per segment. */
function thick(pairs: number[], color: THREE.ColorRepresentation, width: number, opacity = 1, dashed = false) {
  const g = new LineSegmentsGeometry().setPositions(pairs);
  const m = new LineMaterial({ color: new THREE.Color(color).getHex(), linewidth: width, transparent: true, opacity, depthTest: false, dashed, dashSize: .05, gapSize: .035 });
  m.resolution.set(innerWidth, innerHeight); const l = new LineSegments2(g, m); if (dashed) l.computeLineDistances(); l.renderOrder = 3; return l;
}
/** Corner brackets of a box: 3 short ticks at each of the 8 corners (reads as a detection, lighter than full edges). */
function brackets(h: [number, number, number], k = .3) {
  const out: number[] = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    const c = [sx * h[0], sy * h[1], sz * h[2]];
    out.push(...c, c[0] - sx * h[0] * 2 * k, c[1], c[2], ...c, c[0], c[1] - sy * h[1] * 2 * k, c[2], ...c, c[0], c[1], c[2] - sz * h[2] * 2 * k);
  }
  return out;
}

export class Overlay {
  root = new THREE.Group(); labels = document.createElement('div');
  places: Place[] = []; pinStyle: PinStyle = 'tag'; showPins = true; dest: string | null = null;
  private pinEls = new Map<string, HTMLElement>(); private beacons = new THREE.Group();
  private route = new THREE.Group(); private routeMat: THREE.ShaderMaterial | null = null; private dots: THREE.InstancedMesh | null = null; private dotCount = 0;
  path: [number, number][] | null = null; routeStyle: RouteStyle = 'ribbon';
  private ai = new THREE.Group(); private aiEls = new Map<string, HTMLElement>(); aiStyle: AiStyle = 'off'; private tint: SplatEdit | null = null;
  private measure = new THREE.Group(); measurePts: THREE.Vector3[] = []; private measureEl: HTMLElement;
  ar: ArMode = 'off'; arNav: ArNav = 'arrows'; arCap: ArCap = 'floor';
  private arGroup = new THREE.Group(); private chevrons: THREE.InstancedMesh | null = null; private chevCount = 0; private arSamples: THREE.Vector3[] = [];
  private cover: THREE.InstancedMesh | null = null; private coverCells: [number, number][] = []; covered = new Uint8Array(0); coverage = 0;
  private doll: { edit: SplatEdit; table: THREE.Mesh; t: number; dist: number; target: THREE.Vector3 } | null = null; dollYaw = .6;
  private v = new THREE.Vector3();

  constructor(public engine: WalkEngine, public nav: Navigator | null) {
    engine.scene.add(this.root); this.root.add(this.beacons, this.route, this.ai, this.measure, this.arGroup);
    this.labels.className = 'lab-labels'; document.body.append(this.labels);
    this.measureEl = this.tag('measure-tag', ''); this.measureEl.hidden = true;
    engine.frameHooks.add(this.frame);
  }
  private tag(cls: string, html: string) { const d = document.createElement('div'); d.className = `lab-tag ${cls}`; d.innerHTML = html; this.labels.append(d); return d; }

  // ---------- pins ----------
  setPlaces(places: Place[], style: PinStyle, show: boolean) {
    this.places = places; this.pinStyle = style; this.showPins = show;
    for (const el of this.pinEls.values()) el.remove(); this.pinEls.clear(); this.beacons.clear();
    if (!show) return;
    for (const p of places) {
      const el = this.tag(`pin pin-${style}`, style === 'tag'
        ? `<span class="pin-chip"><i></i><b>${p.name}</b><em></em></span><span class="pin-stem"></span>`
        : style === 'beacon' ? `<span class="pin-beacon-l"><b>${p.name}</b><em></em></span>`
        : `<span class="pin-dot"></span><span class="pin-dot-l">${p.name}<em></em></span>`);
      el.dataset.id = p.id; this.pinEls.set(p.id, el);
      if (style === 'beacon') {
        const h = 2.2, geo = new THREE.CylinderGeometry(.035, .035, h, 12, 1, true);
        const mat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
          uniforms: { uTime: { value: 0 } }, vertexShader: `varying float vY; void main(){ vY = uv.y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.); }`,
          fragmentShader: `uniform float uTime; varying float vY; void main(){ float a = pow(1. - vY, 1.6) * .75 + .25 * pow(.5+.5*sin(vY*14. - uTime*3.), 8.) * (1.-vY); gl_FragColor = vec4(1., .42 + .3 * (1.-vY), 0., a); }` });
        const m = new THREE.Mesh(geo, mat); m.position.set(p.p[0], this.engine.floorY + h / 2, p.p[2]); m.userData.mat = mat;
        const ring = new THREE.Mesh(new THREE.RingGeometry(.12, .16, 40), new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: .8, depthTest: false, side: THREE.DoubleSide }));
        ring.rotation.x = -Math.PI / 2; ring.position.set(p.p[0], this.engine.floorY + .01, p.p[2]);
        this.beacons.add(m, ring);
      }
    }
  }

  // ---------- route ----------
  setRoute(path: [number, number][] | null, style: RouteStyle, dest: string | null) {
    this.path = path; this.routeStyle = style; this.dest = dest; this.route.clear(); this.routeMat = null; this.dots = null;
    this.rebuildAr();
    if (!path || this.ar === 'nav') return;
    const y = this.engine.floorY + .03, pts = smooth(path, y); if (pts.length < 2) return;
    const len = pts.reduce((s, p, i) => i ? s + p.distanceTo(pts[i - 1]) : 0, 0);
    if (style === 'ribbon') {
      const W = .2, pos: number[] = [], uv: number[] = [], idx: number[] = []; let m = 0;
      for (let i = 0; i < pts.length; i++) {
        const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)], t = this.v.subVectors(b, a).setY(0).normalize(), nx = -t.z, nz = t.x;
        if (i) m += pts[i].distanceTo(pts[i - 1]);
        pos.push(pts[i].x + nx * W, y, pts[i].z + nz * W, pts[i].x - nx * W, y, pts[i].z - nz * W); uv.push(m, 0, m, 1);
        if (i) { const k = i * 2; idx.push(k - 2, k - 1, k, k - 1, k + 1, k); }
      }
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); g.setIndex(idx);
      this.routeMat = ribbonMat(); this.routeMat.uniforms.uLen.value = len; this.route.add(new THREE.Mesh(g, this.routeMat));
    } else if (style === 'dots') {
      const curve = new THREE.CatmullRomCurve3(pts), n = Math.max(2, Math.floor(len / .32));
      this.dots = new THREE.InstancedMesh(new THREE.CircleGeometry(.07, 24), new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, depthTest: false }), n); this.dotCount = n;
      this.dots.userData.pts = curve.getSpacedPoints(n - 1); this.route.add(this.dots);
    } else {
      const curve = new THREE.CatmullRomCurve3(pts.map(p => p.clone().setY(this.engine.floorY + .95)), false, 'centripetal');
      this.routeMat = tubeMat(); this.routeMat.uniforms.uLen.value = len;
      this.route.add(new THREE.Mesh(new THREE.TubeGeometry(curve, Math.ceil(len * 30), .022, 10), this.routeMat));
      const glow = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.ceil(len * 30), .07, 10), new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, opacity: .1, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending }));
      this.route.add(glow);
    }
    // destination: pulsing rings on the floor
    const end = pts[pts.length - 1];
    for (let k = 0; k < 3; k++) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(.2, .23, 48), new THREE.MeshBasicMaterial({ color: ORANGE, transparent: true, depthTest: false, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2; ring.position.copy(end).setY(this.engine.floorY + .02); ring.userData.ring = k; this.route.add(ring);
    }
  }

  // ---------- AI ----------
  setAi(style: AiStyle, accepted: Set<string>) {
    this.aiStyle = style; this.ai.clear(); for (const el of this.aiEls.values()) el.remove(); this.aiEls.clear();
    if (this.tint) { this.engine.splat?.remove(this.tint); this.tint = null; }
    if (style === 'off') return;
    const found = this.allPlaces.filter(p => p.ai);
    if (style === 'tint') {
      // colour the splats inside each box: Spark evaluates the SDFs on the GPU, nothing is re-trained
      this.tint = new SplatEdit({ rgbaBlendMode: SplatEditRgbaBlendMode.ADD_RGBA, softEdge: .06 });
      for (const p of found) {
        const sdf = new SplatEditSdf({ type: SplatEditSdfType.BOX, color: new THREE.Color(classColor(p.ai!.cls)).multiplyScalar(.42), opacity: 0 });
        sdf.position.set(...p.p); sdf.scale.set(...p.ai!.half); this.tint.addSdf(sdf);
      }
      this.engine.splat?.add(this.tint);
    }
    for (const p of found) {
      const col = new THREE.Color(classColor(p.ai!.cls));
      if (style === 'boxes' || (style === 'review' && !accepted.has(p.id))) {
        const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(p.ai!.half[0] * 2, p.ai!.half[1] * 2, p.ai!.half[2] * 2)).attributes.position.array as Float32Array;
        const faint = thick([...edges], col, 1.2, style === 'review' ? .35 : .45), bold = thick(brackets(p.ai!.half), col, 3.2, style === 'review' ? .6 : 1);
        faint.position.set(...p.p); bold.position.set(...p.p); this.ai.add(faint, bold);
      }
      if (style !== 'review') {
        const el = this.tag(`ai-tag ai-${style}`, `<i style="background:${classColor(p.ai!.cls)}"></i>${p.ai!.cls}<em>${Math.round(p.ai!.conf * 100)}%</em>`);
        el.style.setProperty('--c', classColor(p.ai!.cls)); this.aiEls.set(p.id, el);
      }
    }
  }
  allPlaces: Place[] = [];

  // ---------- measure ----------
  addMeasure(p: THREE.Vector3) {
    if (this.measurePts.length >= 2) this.measurePts = [];
    this.measurePts.push(p.clone()); this.measure.clear();
    for (const q of this.measurePts) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(.03, 20, 14), new THREE.MeshBasicMaterial({ color: 0xFFB84D, depthTest: false })); s.position.copy(q); s.renderOrder = 4;
      const halo = new THREE.Mesh(new THREE.SphereGeometry(.06, 20, 14), new THREE.MeshBasicMaterial({ color: 0xFFB84D, transparent: true, opacity: .25, depthTest: false })); halo.position.copy(q);
      this.measure.add(s, halo);
    }
    if (this.measurePts.length === 2) { const [a, b] = this.measurePts; this.measure.add(thick([...a.toArray(), ...b.toArray()], 0xFFB84D, 3, 1, true)); }
  }
  clearMeasure() { this.measurePts = []; this.measure.clear(); }
  get measured() { return this.measurePts.length === 2 ? this.measurePts[0].distanceTo(this.measurePts[1]) : null; }

  // ---------- AR ----------
  setAr(mode: ArMode, nav: ArNav, cap: ArCap) {
    const was = this.ar; this.ar = mode; this.arNav = nav; this.arCap = cap;
    if (was === 'doll' && mode !== 'doll') this.exitDoll();
    if (mode === 'doll' && !this.doll) this.enterDoll();
    if (mode === 'capture' && !this.cover) this.buildCover();
    if (mode !== 'capture' && this.cover) { this.arGroup.remove(this.cover); this.cover = null; }
    this.setRoute(this.path, this.routeStyle, this.dest);
  }
  private rebuildAr() {
    for (const o of [...this.arGroup.children]) if (o !== this.cover) this.arGroup.remove(o);
    this.chevrons = null; this.arSamples = [];
    if (this.ar !== 'nav' || !this.path) return;
    const y = this.engine.floorY + .02, pts = smooth(this.path, y, .05); if (pts.length < 2) return;
    this.arSamples = pts;
    if (this.arNav === 'arrows') {
      const shape = new THREE.Shape(); shape.moveTo(-.16, -.06); shape.lineTo(0, .1); shape.lineTo(.16, -.06); shape.lineTo(.16, -.14); shape.lineTo(0, .02); shape.lineTo(-.16, -.14); shape.closePath();
      const n = 60; this.chevrons = new THREE.InstancedMesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, depthTest: false, side: THREE.DoubleSide, vertexColors: false }), n);
      this.chevrons.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3); this.chevCount = n; this.arGroup.add(this.chevrons);
    } else if (this.arNav === 'line') {
      const curve = new THREE.CatmullRomCurve3(pts.map(p => p.clone().setY(this.engine.floorY + 1.05)));
      const mat = tubeMat(); mat.uniforms.uLen.value = curve.getLength(); this.routeMat = mat;
      this.arGroup.add(new THREE.Mesh(new THREE.TubeGeometry(curve, Math.ceil(curve.getLength() * 30), .012, 8), mat));
    }
    if (this.arNav !== 'compass') {
      const end = pts[pts.length - 1], h = 1.8, mat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
        vertexShader: `varying float vY; void main(){ vY = uv.y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.); }`,
        fragmentShader: `varying float vY; void main(){ gl_FragColor = vec4(1., .5, .1, pow(1. - vY, 2.) * .55); }` });
      const beam = new THREE.Mesh(new THREE.CylinderGeometry(.22, .22, h, 32, 1, true), mat); beam.position.copy(end).setY(this.engine.floorY + h / 2); this.arGroup.add(beam);
    }
  }
  private buildCover() {
    if (!this.nav) return;
    const { w, h } = this.nav, cells: [number, number][] = [];
    for (let j = 0; j < h; j += 2) for (let i = 0; i < w; i += 2) if (this.nav.ok(i, j, 1)) cells.push([i, j]);
    this.coverCells = cells; this.covered = new Uint8Array(cells.length);
    const m = new THREE.InstancedMesh(new THREE.PlaneGeometry(.17, .17), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: .55, depthTest: false, side: THREE.DoubleSide }), cells.length);
    m.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(cells.length * 3), 3);
    const o = new THREE.Object3D();
    cells.forEach(([i, j], k) => { const [x, z] = this.nav!.pos(i, j); o.position.set(x + .05, this.engine.floorY + .015, z + .05); o.rotation.set(-Math.PI / 2, 0, 0); o.updateMatrix(); m.setMatrixAt(k, o.matrix); m.setColorAt(k, new THREE.Color(.25, .3, .4)); });
    this.cover = m; this.arGroup.add(m);
  }
  /** Nearest floor cell not yet covered, for the capture guide's "go here next". */
  nextGap(): [number, number] | null {
    const p = this.engine.state.pos; let best = -1, bd = Infinity;
    this.coverCells.forEach(([i, j], k) => { if (this.covered[k]) return; const [x, z] = this.nav!.pos(i, j), d = (x - p.x) ** 2 + (z - p.z) ** 2; if (d < bd && d > .8) { bd = d; best = k; } });
    return best < 0 ? null : this.nav!.pos(...this.coverCells[best]);
  }
  private enterDoll() {
    const e = this.engine, b = new THREE.Box3(); const pts = this.allPlaces.map(p => new THREE.Vector3(p.p[0], 0, p.p[2])); b.setFromPoints(pts);
    const target = b.getCenter(new THREE.Vector3()).setY(e.floorY + .6);
    // hide the ceiling: everything above 2.1 m is multiplied to zero opacity
    const edit = new SplatEdit({ rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY, softEdge: .08 });
    const cut = new SplatEditSdf({ type: SplatEditSdfType.BOX, opacity: 0, color: new THREE.Color(1, 1, 1) });
    cut.position.set(target.x, e.floorY + 2.1 + 5, target.z); cut.scale.set(20, 5, 20); edit.addSdf(cut); e.splat?.add(edit);
    const table = new THREE.Mesh(new THREE.CircleGeometry(4.6, 96), new THREE.ShaderMaterial({ transparent: true, depthWrite: false,
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.); }`,
      fragmentShader: `varying vec2 vUv; void main(){ float r = length(vUv - .5) * 2.; float ring = smoothstep(.985, .995, r) * (1. - smoothstep(.995, 1., r)); gl_FragColor = vec4(vec3(1., .55, .2) * ring + vec3(.04, .06, .1) * (1. - r), (1. - smoothstep(.6, 1., r)) * .55 + ring * .9); }` }));
    table.rotation.x = -Math.PI / 2; table.position.set(target.x, e.floorY - .02, target.z); this.root.add(table);
    this.doll = { edit, table, t: 0, dist: 7.5, target }; e.manual = true; this.saved = { pos: e.state.pos.clone(), yaw: e.state.yaw, pitch: e.pitch, fov: e.camera.fov };
  }
  private saved: { pos: THREE.Vector3; yaw: number; pitch: number; fov: number } | null = null;
  private exitDoll() {
    if (!this.doll) return; const e = this.engine; e.splat?.remove(this.doll.edit); this.root.remove(this.doll.table); this.doll = null; e.manual = false;
    if (this.saved) { e.state.pos.copy(this.saved.pos); e.state.yaw = this.saved.yaw; e.pitch = this.saved.pitch; }
  }
  dollDrag(dx: number, dy: number) { this.dollYaw -= dx * .006; if (this.doll) this.doll.dist = Math.max(4, Math.min(12, this.doll.dist + dy * .02)); }

  /** Floor point under the screen position (x, y in CSS px), or the splat surface for the tape measure. */
  pick(x: number, y: number, surface: boolean) {
    const e = this.engine, rc = new THREE.Raycaster(), ndc = new THREE.Vector2(x / innerWidth * 2 - 1, 1 - y / innerHeight * 2);
    rc.setFromCamera(ndc, e.camera);
    if (surface && e.splat) { const hit = rc.intersectObject(e.splat, false)[0]; if (hit) return hit.point; }
    const t = (e.floorY - rc.ray.origin.y) / rc.ray.direction.y; return t > 0 ? rc.ray.at(t, new THREE.Vector3()) : null;
  }

  // ---------- per frame ----------
  private acc = 0;
  /** orbit the dollhouse (runs every frame, it is the camera) */
  dollStep(dt: number) {
    if (!this.doll) return;
    const cam = this.engine.camera, d = this.doll;
    d.t = Math.min(1, d.t + dt * .8); const k = 1 - (1 - d.t) ** 3;
    this.dollYaw += dt * .05;
    const want = new THREE.Vector3(d.target.x + Math.sin(this.dollYaw) * d.dist, d.target.y + d.dist * .78, d.target.z + Math.cos(this.dollYaw) * d.dist);
    cam.position.lerp(want, k < 1 ? k * .2 : .12); cam.lookAt(d.target); cam.updateMatrixWorld();
  }
  frame = (dt: number, t: number) => {
    const e = this.engine, cam = e.camera, W = innerWidth, H = innerHeight, me = e.state.pos;
    // the pin pass projects and writes DOM for every place: at 194 places that is the frame budget on its own,
    // so it runs at 20 Hz while the splats keep the full frame rate
    this.acc += dt;
    if (this.acc < 0.05) { if (this.doll) this.dollStep(dt); return; }
    this.acc = 0;
    // draw after the splats: Spark composites the splat layer in the same pass, and depth-less overlays drawn
    // before it get painted over (this hid the route in some views)
    this.root.traverse(o => { if (o.renderOrder < 10) o.renderOrder = 10; });
    for (const g of [this.ai, this.measure]) for (const o of g.children) if ((o as LineSegments2).material instanceof LineMaterial) ((o as LineSegments2).material as LineMaterial).resolution.set(W, H);
    this.dollStep(dt);
    const proj = (p: THREE.Vector3 | [number, number, number]) => { this.v.set(...(Array.isArray(p) ? p : p.toArray()) as [number, number, number]).project(cam); return this.v.z < 1 ? [(this.v.x + 1) / 2 * W, (1 - this.v.y) / 2 * H] as const : null; };
    // pins: projected each frame; fade with distance, show distance in metres
    for (const p of this.places) {
      const el = this.pinEls.get(p.id); if (!el) continue;
      const dist = Math.hypot(p.p[0] - me.x, p.p[2] - me.z);
      // 200 labelled places is a lot of DOM: only the ones near you (or the destination) are laid out at all
      if (dist > 26 && this.dest !== p.id) { if (el.style.opacity !== '0') el.style.opacity = '0'; continue; }
      const anchor: [number, number, number] = this.pinStyle === 'beacon' ? [p.p[0], e.floorY + 2.25, p.p[2]] : p.p;
      const s = proj(anchor);
      if (!s || this.ar === 'doll' && this.pinStyle === 'dot') { el.style.opacity = '0'; continue; }
      const near = this.pinStyle === 'dot' ? Math.hypot(s[0] - W / 2, s[1] - H / 2) < Math.min(W, H) * .22 : true;
      el.style.transform = `translate(${s[0]}px, ${s[1]}px)`; el.style.opacity = String(Math.max(.25, Math.min(1, 1.3 - dist / 9)));
      el.classList.toggle('near', near); el.classList.toggle('dest', this.dest === p.id);
      const em = el.querySelector('em'); if (em) em.textContent = `${dist.toFixed(1)} m`;
    }
    for (const [id, el] of this.aiEls) {
      const p = this.allPlaces.find(q => q.id === id)!; const s = proj([p.p[0], p.p[1] + p.ai!.half[1], p.p[2]]);
      if (!s) { el.style.opacity = '0'; continue; } el.style.opacity = '1'; el.style.transform = `translate(${s[0]}px, ${s[1]}px)`;
    }
    if (this.measurePts.length === 2) { const m = this.measurePts[0].clone().add(this.measurePts[1]).multiplyScalar(.5), s = proj(m); this.measureEl.hidden = !s; if (s) { this.measureEl.style.transform = `translate(${s[0]}px, ${s[1]}px)`; this.measureEl.textContent = `${this.measured!.toFixed(2)} m`; } }
    else this.measureEl.hidden = true;
    // route animation
    if (this.routeMat) this.routeMat.uniforms.uTime.value = t;
    for (const o of this.beacons.children) if (o.userData.mat) o.userData.mat.uniforms.uTime.value = t;
    if (this.dots) {
      const o = new THREE.Object3D(), pts = this.dots.userData.pts as THREE.Vector3[];
      for (let i = 0; i < this.dotCount; i++) { const ph = (t * 2.2 - i * .35) % (Math.PI * 2), s = .65 + .5 * Math.max(0, Math.sin(ph)); o.position.copy(pts[i]); o.rotation.set(-Math.PI / 2, 0, 0); o.scale.setScalar(s); o.updateMatrix(); this.dots.setMatrixAt(i, o.matrix); }
      this.dots.instanceMatrix.needsUpdate = true;
    }
    for (const r of this.route.children) if (r.userData.ring !== undefined) { const ph = ((t * .7 + r.userData.ring / 3) % 1); r.scale.setScalar(1 + ph * 2.2); ((r as THREE.Mesh).material as THREE.MeshBasicMaterial).opacity = (1 - ph) * .9; }
    // AR chevrons: the next 6 m of route from where you stand, lit in sequence
    if (this.chevrons && this.arSamples.length) {
      let ci = 0, cd = Infinity; this.arSamples.forEach((p, i) => { const d = (p.x - me.x) ** 2 + (p.z - me.z) ** 2; if (d < cd) { cd = d; ci = i; } });
      const o = new THREE.Object3D(), col = new THREE.Color(); let n = 0;
      for (let i = ci + 6; i < this.arSamples.length - 2 && n < this.chevCount; i += 9) {
        const p = this.arSamples[i], q = this.arSamples[Math.min(this.arSamples.length - 1, i + 2)], dist = (i - ci) * .05; if (dist > 6) break;
        o.position.copy(p); o.rotation.set(-Math.PI / 2, 0, Math.atan2(-(q.x - p.x), -(q.z - p.z)) + Math.PI); o.scale.setScalar(1); o.updateMatrix(); this.chevrons.setMatrixAt(n, o.matrix);
        const wave = Math.max(0, Math.sin(t * 5 - n * .7)); col.setRGB(1, .45 + .4 * wave, .1 + .5 * wave).multiplyScalar(Math.max(.15, 1 - dist / 6)); this.chevrons.setColorAt(n, col); n++;
      }
      this.chevrons.count = n; this.chevrons.instanceMatrix.needsUpdate = true; if (this.chevrons.instanceColor) this.chevrons.instanceColor.needsUpdate = true;
    }
    // capture coverage: cells within ~1.6 m in front of the camera count as seen
    if (this.cover && this.nav) {
      const fx = -Math.sin(e.state.yaw), fz = -Math.cos(e.state.yaw); let seen = 0; const c = new THREE.Color();
      this.coverCells.forEach(([i, j], k) => {
        const [x, z] = this.nav!.pos(i, j), dx = x - me.x, dz = z - me.z, d = Math.hypot(dx, dz);
        if (!this.covered[k] && d < 1.8 && (d < .5 || (dx * fx + dz * fz) / d > .55)) this.covered[k] = 1;
        if (this.covered[k]) seen++;
        c.set(this.covered[k] ? '#3DDC97' : '#2A3448'); if (!this.covered[k] && this.arCap === 'trail') c.set('#3B465E'); this.cover!.setColorAt(k, c);
      });
      if (this.cover.instanceColor) this.cover.instanceColor.needsUpdate = true; this.coverage = seen / Math.max(1, this.coverCells.length);
    }
  };

  dispose() { this.engine.frameHooks.delete(this.frame); this.engine.scene.remove(this.root); this.labels.remove(); this.exitDoll(); if (this.tint) this.engine.splat?.remove(this.tint); }
}
