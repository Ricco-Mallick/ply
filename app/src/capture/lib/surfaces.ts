/* Coverage painted on surfaces, Polycam / RealityScan style: ARCore's detected planes (floor, walls, tabletops) are
   covered with a grid of flat dots; every saved keyframe marks the dots it really saw (in frame, facing the camera, not
   hidden behind something according to depth). Coverage is stored per 5 cm world cell, so it survives planes growing
   or merging and the dots being regenerated. Green: 3+ keyframes from 2+ directions within 3 m; orange: seen at least
   once; red: detected but not captured yet. */
import * as THREE from 'three';

const SPACING = .07, LIFT = .004, CAP = 60_000, CELL = .05, MAX_CELLS = 400_000;
export const GRADE_RGB = [new THREE.Color('#3DDC97'), new THREE.Color('#FFB84D'), new THREE.Color('#F2555A')];

export class SurfaceMap {
  mesh: THREE.InstancedMesh;
  private planes = new Map<any, { changed: number; ids: number[] }>();
  private pos = new Float32Array(CAP * 3); private nrm = new Float32Array(CAP * 3); private cellOf = new Int32Array(CAP).fill(-1);
  private alive = new Uint8Array(CAP); private free: number[] = []; private top = 0;
  private cells = new Map<number, number>(); private nc = 0;
  private views = new Uint16Array(MAX_CELLS); private dirs = new Uint16Array(MAX_CELLS); private near = new Float32Array(MAX_CELLS).fill(99); private lastKf = new Int32Array(MAX_CELLS).fill(-1);
  counts = [0, 0, 0]; private dirty = true;
  private o = new THREE.Object3D(); private v = new THREE.Vector3(); private z = new THREE.Vector3(0, 0, 1);

  constructor() {
    this.mesh = new THREE.InstancedMesh(new THREE.CircleGeometry(.021, 12), new THREE.MeshBasicMaterial({ transparent: true, opacity: .9, side: THREE.DoubleSide, depthWrite: false }), CAP);
    this.mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(CAP * 3), 3); this.mesh.count = 0; this.mesh.frustumCulled = false;
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  }

  /** Regenerate dots for planes that are new or changed; drop planes ARCore merged away. */
  syncPlanes(frame: any, ref: any) {
    const set: Set<any> | undefined = frame.detectedPlanes; if (!set) return;
    for (const [pl, rec] of this.planes) if (!set.has(pl)) { this.release(rec.ids); this.planes.delete(pl); }
    for (const pl of set) {
      const rec = this.planes.get(pl); if (rec && rec.changed >= pl.lastChangedTime) continue;
      const pose = frame.getPose(pl.planeSpace, ref); if (!pose) continue;
      if (rec) this.release(rec.ids);
      this.planes.set(pl, { changed: pl.lastChangedTime, ids: this.fill(pl.polygon, new THREE.Matrix4().fromArray(pose.transform.matrix)) });
    }
  }
  private release(ids: number[]) { for (const i of ids) { this.alive[i] = 0; this.free.push(i); this.o.scale.setScalar(0); this.o.updateMatrix(); this.mesh.setMatrixAt(i, this.o.matrix); } this.dirty = true; }
  private fill(poly: DOMPointReadOnly[], M: THREE.Matrix4) {
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity; for (const p of poly) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); z0 = Math.min(z0, p.z); z1 = Math.max(z1, p.z); }
    const n = new THREE.Vector3().setFromMatrixColumn(M, 1).normalize(), q = new THREE.Quaternion().setFromUnitVectors(this.z, n), ids: number[] = [];
    // grid in plane space, offset by half a step so neighbouring planes don't line up into seams
    for (let x = Math.ceil(x0 / SPACING) * SPACING + SPACING / 2; x < x1; x += SPACING) for (let z = Math.ceil(z0 / SPACING) * SPACING + SPACING / 2; z < z1; z += SPACING) {
      if (!inside(poly, x, z)) continue;
      const i = this.free.length ? this.free.pop()! : this.top < CAP ? this.top++ : -1; if (i < 0) return ids;
      this.v.set(x, 0, z).applyMatrix4(M).addScaledVector(n, LIFT);
      this.pos.set([this.v.x, this.v.y, this.v.z], i * 3); this.nrm.set([n.x, n.y, n.z], i * 3); this.cellOf[i] = this.cell(this.v.x, this.v.y, this.v.z); this.alive[i] = 1;
      this.o.position.copy(this.v); this.o.quaternion.copy(q); this.o.scale.setScalar(1); this.o.updateMatrix(); this.mesh.setMatrixAt(i, this.o.matrix); ids.push(i);
    }
    this.mesh.count = this.top; this.mesh.instanceMatrix.needsUpdate = true; this.dirty = true; return ids;
  }
  private cell(x: number, y: number, z: number) {
    const k = ((Math.floor(x / CELL) + 512) << 20) | ((Math.floor(y / CELL) + 512) << 10) | (Math.floor(z / CELL) + 512);
    let c = this.cells.get(k); if (c === undefined) { if (this.nc >= MAX_CELLS) return -1; c = this.nc++; this.cells.set(k, c); } return c;
  }

  /** A keyframe was saved: mark every dot it truly saw. m = camera-to-world, P = projection, depth = XRCPUDepthInformation. */
  observe(m: Float32Array, P: Float32Array, depth: any | null, kfId: number) {
    const Mi = new THREE.Matrix4().fromArray(m).invert(), Pm = new THREE.Matrix4().fromArray(P), cx = m[12], cy = m[13], cz = m[14], v = this.v;
    for (let i = 0; i < this.top; i++) {
      if (!this.alive[i]) continue; const c = this.cellOf[i]; if (c < 0 || this.lastKf[c] === kfId) continue;
      const px = this.pos[i * 3], py = this.pos[i * 3 + 1], pz = this.pos[i * 3 + 2];
      v.set(px, py, pz).applyMatrix4(Mi); const zc = -v.z; if (zc < .15 || zc > 4.5) continue;
      v.applyMatrix4(Pm); if (v.x < -1 || v.x > 1 || v.y < -1 || v.y > 1) continue;
      const dx = cx - px, dy = cy - py, dz = cz - pz, dl = Math.hypot(dx, dy, dz);
      if ((dx * this.nrm[i * 3] + dy * this.nrm[i * 3 + 1] + dz * this.nrm[i * 3 + 2]) / dl < .25) continue; // seen too edge-on
      if (depth) { try { const d = depth.getDepthInMeters((v.x + 1) / 2, (1 - v.y) / 2); if (d > 0 && d < zc * .9 - .1) continue; } catch { /* outside the depth map */ } } // something in front
      this.lastKf[c] = kfId; if (this.views[c] < 65535) this.views[c]++; if (dl < this.near[c]) this.near[c] = dl;
      this.dirs[c] |= 1 << (Math.floor((Math.atan2(dx, dz) + Math.PI) / (Math.PI * 2) * 12) % 12);
    }
    this.dirty = true;
  }
  grade(i: number) {
    const c = this.cellOf[i]; if (c < 0) return 2; let bins = 0; for (let d = this.dirs[c]; d; d &= d - 1) bins++;
    return this.views[c] >= 3 && bins >= 2 && this.near[c] <= 3 ? 0 : this.views[c] >= 1 ? 1 : 2;
  }
  /** Recolour (a few times a second): grades, counts for the HUD. */
  flush() {
    if (!this.dirty) return; this.dirty = false; this.counts = [0, 0, 0]; const col = this.mesh.instanceColor!.array as Float32Array;
    for (let i = 0; i < this.top; i++) { if (!this.alive[i]) continue; const g = this.grade(i); this.counts[g]++; const c = GRADE_RGB[g]; col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b; }
    this.mesh.instanceColor!.needsUpdate = true;
  }
  nearestRed(x: number, z: number) {
    let best = -1, bd = Infinity;
    for (let i = 0; i < this.top; i += 2) { if (!this.alive[i] || this.grade(i) !== 2) continue; const d = (this.pos[i * 3] - x) ** 2 + (this.pos[i * 3 + 2] - z) ** 2; if (d > .6 && d < bd) { bd = d; best = i; } }
    return best < 0 ? null : [this.pos[best * 3], this.pos[best * 3 + 2]] as const;
  }
  /** Dots and their colours, for the after-scan preview. */
  export() {
    this.dirty = true; this.flush(); const n = this.counts[0] + this.counts[1] + this.counts[2], pts = new Float32Array(n * 3), col = new Float32Array(n * 3), src = this.mesh.instanceColor!.array as Float32Array; let k = 0;
    for (let i = 0; i < this.top; i++) if (this.alive[i]) { pts.set(this.pos.subarray(i * 3, i * 3 + 3), k * 3); col.set(src.subarray(i * 3, i * 3 + 3), k * 3); k++; }
    return { pts, col, planes: this.planes.size };
  }
}

/** Point in polygon (plane space x, z), even-odd rule. */
function inside(poly: DOMPointReadOnly[], x: number, z: number) {
  let r = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const a = poly[i], b = poly[j]; if ((a.z > z) !== (b.z > z) && x < (b.x - a.x) * (z - a.z) / (b.z - a.z) + a.x) r = !r; } return r;
}
