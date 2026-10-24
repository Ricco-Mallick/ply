// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Scene manifest: what the pipeline hands the viewer (pipeline/README.md has the producing side).
   Convention: metres, +Y up, floor at floorY (0 for pipeline output), splat already aligned unless splatQuaternion says otherwise. */

export interface Occupancy { image: string; origin: [number, number]; cell: number; width: number; height: number; }
export interface SceneManifest {
  name: string;
  splat: string;
  /** Spark's prebuilt LOD tree (.rad + .radc chunks): streamed, so the twin appears before it has all downloaded */
  lod?: string;
  splatQuaternion?: [number, number, number, number];
  splatPosition?: [number, number, number];
  floorY?: number;
  eyeHeight?: number;
  start?: { position: [number, number, number]; yaw?: number };
  bounds?: { min: [number, number, number]; max: [number, number, number] };
  occupancy?: Occupancy;
  cameraPath?: [number, number, number][];
  /** cinematic tour keyframes (pipeline tour.py) */
  tour?: { pos: [number, number, number]; target: [number, number, number]; fov: number; dur: number }[];
}

/** Grid of blocked cells in the floor plane (x, z). 1 = blocked. */
export class OccupancyGrid {
  constructor(public data: Uint8Array, public meta: Occupancy) {}
  /** Blocked if outside the grid or the cell is occupied. */
  blocked(x: number, z: number) {
    const { origin, cell, width, height } = this.meta;
    const i = Math.floor((x - origin[0]) / cell), j = Math.floor((z - origin[1]) / cell);
    if (i < 0 || j < 0 || i >= width || j >= height) return true;
    return this.data[j * width + i] === 1;
  }
  /** A circle of radius r at (x, z) is clear. */
  clear(x: number, z: number, r: number) {
    if (this.blocked(x, z)) return false;
    for (let k = 0; k < 8; k++) { const a = k * Math.PI / 4; if (this.blocked(x + Math.cos(a) * r, z + Math.sin(a) * r)) return false; }
    return true;
  }
  /** Fallback when a scene ships no grid: build one from the splat centres with the pipeline's rule
      (anything between knee and head height blocks; cells need floor under them to be walkable). */
  static fromPoints(pts: Float32Array, floorY: number, cell = .1) {
    let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < pts.length; i += 3) { const y = pts[i + 1] - floorY; if (y > -.2 && y < .2) { minX = Math.min(minX, pts[i]); maxX = Math.max(maxX, pts[i]); minZ = Math.min(minZ, pts[i + 2]); maxZ = Math.max(maxZ, pts[i + 2]); } }
    if (!isFinite(minX)) return null;
    minX -= .3; minZ -= .3; const width = Math.ceil((maxX + .3 - minX) / cell), height = Math.ceil((maxZ + .3 - minZ) / cell);
    const block = new Uint16Array(width * height), floor = new Uint16Array(width * height);
    for (let i = 0; i < pts.length; i += 3) {
      const gx = Math.floor((pts[i] - minX) / cell), gz = Math.floor((pts[i + 2] - minZ) / cell); if (gx < 0 || gz < 0 || gx >= width || gz >= height) continue;
      const y = pts[i + 1] - floorY, k = gz * width + gx; if (y > -.2 && y < .2) floor[k]++; else if (y > .3 && y < 1.75) block[k]++;
    }
    const data = new Uint8Array(width * height);
    for (let k = 0; k < data.length; k++) data[k] = block[k] >= 4 || floor[k] === 0 ? 1 : 0;
    // close pinholes in the floor: a blocked cell with no wall points and 6+ free neighbours is floor we did not sample
    const out = data.slice();
    for (let z = 1; z < height - 1; z++) for (let x = 1; x < width - 1; x++) { const k = z * width + x; if (!data[k] || block[k] >= 4) continue; let n = 0; for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) n += data[k + dz * width + dx] ? 0 : 1; if (n >= 6) out[k] = 0; }
    return new OccupancyGrid(out, { image: '', origin: [minX, minZ], cell, width, height });
  }
  /** Floor height when the manifest does not give one: the densest thin layer in the lowest part of the scene. */
  static estimateFloor(pts: Float32Array) {
    const ys: number[] = []; for (let i = 1; i < pts.length; i += 3 * 7) ys.push(pts[i]); ys.sort((a, b) => a - b);
    const lo = ys[Math.floor(ys.length * .02)], hi = ys[Math.floor(ys.length * .45)], bins = new Array(60).fill(0);
    for (const y of ys) if (y >= lo && y <= hi) bins[Math.min(59, Math.floor((y - lo) / (hi - lo) * 60))]++;
    const b = bins.indexOf(Math.max(...bins)); return lo + (b + .5) / 60 * (hi - lo);
  }
  static async load(base: URL, meta: Occupancy) {
    const img = new Image(); img.crossOrigin = 'anonymous'; img.src = new URL(meta.image, base).href; await img.decode();
    const c = document.createElement('canvas'); c.width = meta.width; c.height = meta.height; const x = c.getContext('2d', { willReadFrequently: true })!;
    x.drawImage(img, 0, 0, meta.width, meta.height); const px = x.getImageData(0, 0, meta.width, meta.height).data;
    const data = new Uint8Array(meta.width * meta.height); for (let i = 0; i < data.length; i++) data[i] = px[i * 4] < 128 ? 1 : 0; // dark = blocked
    return { grid: new OccupancyGrid(data, meta), image: img };
  }
}

/** ?scene=<folder URL or scene.json URL>; defaults to the bundled demo. */
export async function loadManifest(): Promise<{ manifest: SceneManifest; base: URL }> {
  const q = new URLSearchParams(location.search).get('scene');
  const href = q ? (q.endsWith('.json') ? q : q.replace(/\/?$/, '/scene.json')) : `${import.meta.env.BASE_URL}scenes/demo/scene.json`;
  const url = new URL(href, location.href);
  const res = await fetch(url); if (!res.ok) throw new Error(`Could not load scene (${res.status})`);
  const raw = await res.json();
  return { manifest: await normalise(raw, url), base: url };
}

/** Pipeline output (pipeline/README.md, scene.json v1) -> the viewer's manifest. Hand-written manifests pass through. */
async function normalise(raw: any, base: URL): Promise<SceneManifest> {
  if (typeof raw.splat === 'string') return raw as SceneManifest;
  const phone = matchMedia('(max-width: 720px), (pointer: coarse)').matches;
  const m: SceneManifest = {
    name: raw.name ?? decodeURIComponent(base.pathname.split('/').filter(Boolean).slice(-2, -1)[0] ?? 'Captured space'),
    splat: phone && raw.splat?.lite ? raw.splat.lite : raw.splat?.file ?? 'scene.spz',
    // Streaming LOD shows a coarse twin first and refines it, which reads as "garbage that slowly improves" while
    // chunks come in. The single SPZ is edge-cached, so it arrives whole and sharp: use it by default and keep the
    // LOD behind ?lod=1 until the chunk path is as fast as a plain download.
    lod: (raw.splat?.lod && new URLSearchParams(location.search).has('lod')) ? raw.splat.lod : undefined,
    floorY: raw.floorY ?? 0, eyeHeight: raw.spawn?.position?.[1] ?? 1.6, bounds: raw.bounds, cameraPath: raw.cameraPath, tour: raw.tour,
  };
  if (raw.spawn?.position) {
    const [x, , z] = raw.spawn.position, l = raw.spawn.lookAt;
    m.start = { position: [x, m.floorY!, z], yaw: l ? Math.atan2(-(l[0] - x), -(l[2] - z)) : 0 }; // yaw 0 looks down -z
  }
  if (raw.occupancy?.meta) {
    const o = await (await fetch(new URL(raw.occupancy.meta, base))).json();
    m.occupancy = { image: raw.occupancy.image ?? o.image, origin: o.origin, cell: o.cellSize, width: o.width, height: o.height };
  }
  return m;
}
