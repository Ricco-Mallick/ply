// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Live coverage from depth: every few frames the depth map (160x90 on the S24 FE) is sampled and unprojected into 8 cm
   cells. A cell is only drawn once depth has hit it twice (single noisy hits never show), samples on depth edges are
   skipped (they produce "flying" points between objects), and only the first 4 m count.
   Each saved keyframe credits the cells its depth saw: red = in 0-1 keyframes, orange = 2-3, green = 4+ keyframes, or 3
   from 2+ directions, within 3.5 m. */
import * as THREE from 'three';

const CELL = .08, CAP = 250_000, NX = 20, NY = 36, EDGE = .012, FAR = 4;
export const GRADE_RGB = [new THREE.Color('#3DDC97'), new THREE.Color('#FFB84D'), new THREE.Color('#F2555A')];

export class Coverage {
  points: THREE.Points;
  private index = new Map<number, number>(); private n = 0;
  private hits = new Uint16Array(CAP); private kf = new Uint16Array(CAP); private dirs = new Uint16Array(CAP); private near = new Float32Array(CAP).fill(99);
  private lastKf = new Int32Array(CAP).fill(-1); private center = new Float32Array(CAP * 3); private drawIdx = new Int32Array(CAP).fill(-1); private grade = new Uint8Array(CAP).fill(255);
  // draw buffers hold only confirmed cells, appended as they confirm
  private dpos = new Float32Array(CAP * 3); private dcol = new Float32Array(CAP * 3); private dn = 0; private geo = new THREE.BufferGeometry();
  counts = [0, 0, 0]; private lo = Infinity; private hi = -1; private v = new THREE.Vector3(); private M = new THREE.Matrix4();

  constructor(material: THREE.Material) {
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.dpos, 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute('color', new THREE.BufferAttribute(this.dcol, 3).setUsage(THREE.DynamicDrawUsage)); this.geo.setDrawRange(0, 0);
    this.points = new THREE.Points(this.geo, material); this.points.frustumCulled = false;
  }

  /** One depth frame. m = camera-to-world, P = projection; kfId >= 0 when this frame was saved as a keyframe. */
  integrate(d: any, P: Float32Array, m: Float32Array, kfId: number) {
    this.M.fromArray(m); const cx = m[12], cz = m[14], v = this.v;
    const depth = (u: number, w: number) => { try { return d.getDepthInMeters(u, w) as number; } catch { return 0; } };
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      const u = (i + .5) / NX, w = (j + .5) / NY, z = depth(u, w);
      if (!(z > .25 && z < FAR)) continue;
      // depth edge (the boundary between a chair and the wall behind it): neighbours disagree -> skip
      const a = depth(u + EDGE, w), b = depth(u, w + EDGE);
      if (Math.abs(a - z) > z * .08 || Math.abs(b - z) > z * .08) continue;
      v.set((u * 2 - 1 + P[8]) * z / P[0], (1 - w * 2 + P[9]) * z / P[5], -z).applyMatrix4(this.M);
      this.add(v.x, v.y, v.z, cx, cz, z, kfId);
    }
  }
  private add(x: number, y: number, z: number, camX: number, camZ: number, dist: number, kfId: number) {
    const ix = Math.floor(x / CELL) + 512, iy = Math.floor(y / CELL) + 512, iz = Math.floor(z / CELL) + 512;
    if (ix < 0 || iy < 0 || iz < 0 || ix > 1023 || iy > 1023 || iz > 1023) return;
    const key = (ix << 20) | (iy << 10) | iz; let i = this.index.get(key);
    if (i === undefined) { if (this.n >= CAP) return; i = this.n++; this.index.set(key, i); }
    // running mean of where depth actually hit inside the cell (dots sit on the surface, not on a voxel grid)
    const h = this.hits[i]; if (h < 65535) this.hits[i]++; const k = 1 / Math.min(this.hits[i], 20);
    this.center[i * 3] += (x - this.center[i * 3]) * (h ? k : 1); this.center[i * 3 + 1] += (y - this.center[i * 3 + 1]) * (h ? k : 1); this.center[i * 3 + 2] += (z - this.center[i * 3 + 2]) * (h ? k : 1);
    if (dist < this.near[i]) this.near[i] = dist;
    if (kfId >= 0 && this.lastKf[i] !== kfId) { this.lastKf[i] = kfId; if (this.kf[i] < 65535) this.kf[i]++; this.dirs[i] |= 1 << (Math.floor((Math.atan2(camX - x, camZ - z) + Math.PI) / (Math.PI * 2) * 12) % 12); }
    if (this.hits[i] < 2) return; // not confirmed yet
    if (this.drawIdx[i] < 0) this.drawIdx[i] = this.dn++;
    const di = this.drawIdx[i]; this.dpos[di * 3] = this.center[i * 3]; this.dpos[di * 3 + 1] = this.center[i * 3 + 1]; this.dpos[di * 3 + 2] = this.center[i * 3 + 2];
    let bins = 0; for (let b = this.dirs[i]; b; b &= b - 1) bins++;
    const g = (this.kf[i] >= 4 || (this.kf[i] >= 3 && bins >= 2)) && this.near[i] <= 3.5 ? 0 : this.kf[i] >= 2 ? 1 : 2;
    if (g !== this.grade[i]) { if (this.grade[i] !== 255) this.counts[this.grade[i]]--; this.counts[g]++; this.grade[i] = g; const c = GRADE_RGB[g]; this.dcol[di * 3] = c.r; this.dcol[di * 3 + 1] = c.g; this.dcol[di * 3 + 2] = c.b; }
    if (di < this.lo) this.lo = di; if (di > this.hi) this.hi = di;
  }
  /** Push changed dots to the GPU. */
  flush() {
    if (this.hi < 0) return; const p = this.geo.attributes.position as THREE.BufferAttribute, c = this.geo.attributes.color as THREE.BufferAttribute;
    p.clearUpdateRanges(); c.clearUpdateRanges(); p.addUpdateRange(this.lo * 3, (this.hi - this.lo + 1) * 3); c.addUpdateRange(this.lo * 3, (this.hi - this.lo + 1) * 3);
    p.needsUpdate = c.needsUpdate = true; this.geo.setDrawRange(0, this.dn); this.lo = Infinity; this.hi = -1;
  }
  nearestRed(x: number, z: number) {
    let best = -1, bd = Infinity;
    for (let i = 0; i < this.n; i++) { if (this.grade[i] !== 2) continue; const dx = this.center[i * 3] - x, dz = this.center[i * 3 + 2] - z, d = dx * dx + dz * dz; if (d > .6 && d < bd) { bd = d; best = i; } }
    return best < 0 ? null : [this.center[best * 3], this.center[best * 3 + 2]] as const;
  }
  export() { return { pts: this.dpos.slice(0, this.dn * 3), col: this.dcol.slice(0, this.dn * 3) }; }
}
