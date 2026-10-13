/* Where takes live on the phone: Origin Private File System (survives reloads, not limited by RAM), chunks streamed in
   as MediaRecorder produces them. Browsers without OPFS writable streams keep the chunks in memory instead.
   A small index in localStorage lists the captures on this device. */
import type { Quality } from './analyzer';
import type { SensorSample } from './sensors';

export interface Clip { id: string; name: string; source: 'camera' | 'upload' | 'ar'; mime: string; size: number; duration: number; width: number; height: number; createdAt: number;
  quality: Quality[]; sensors: SensorSample[]; path: { x: number; z: number; t: number }[]; lock?: Record<string, unknown>; loopClosed?: boolean;
  /** AR scans: a tar of keyframes + poses instead of a video; the coverage cloud is stored next to it as `${id}.cloud` */
  ar?: { keyframes: number; coverage: { green: number; orange: number; red: number; voxels: number }; camera: string | null; walked: number }; }
export interface Capture { id: string; name: string; createdAt: number; clips: Clip[]; }

const INDEX = 'ply-captures';
const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
export const newId = uid;

/** The light list shown on the start screen. Full captures (with telemetry) live as JSON next to the videos. */
export interface CaptureSummary { id: string; name: string; createdAt: number; clips: number; duration: number; }
export function listCaptures(): CaptureSummary[] { try { return JSON.parse(localStorage.getItem(INDEX) || '[]'); } catch { return []; } }
const memJson = new Map<string, Capture>();
export async function saveCapture(c: Capture) {
  memJson.set(c.id, c);
  const d = await dir().catch(() => null);
  if (d) { try { const w = await (await (d as any).getFileHandle(`${c.id}.json`, { create: true })).createWritable(); await w.write(JSON.stringify(c)); await w.close(); } catch { /* memory copy remains */ } }
  const sum: CaptureSummary = { id: c.id, name: c.name, createdAt: c.createdAt, clips: c.clips.length, duration: c.clips.reduce((a, k) => a + k.duration, 0) };
  try { localStorage.setItem(INDEX, JSON.stringify([sum, ...listCaptures().filter(x => x.id !== c.id)])); } catch { /* private mode */ }
}
export async function loadCapture(id: string): Promise<Capture | null> {
  if (memJson.has(id)) return memJson.get(id)!;
  const d = await dir().catch(() => null); if (!d) return null;
  try { return JSON.parse(await (await (await (d as any).getFileHandle(`${id}.json`)).getFile()).text()); } catch { return null; }
}
export async function deleteCapture(c: Capture) {
  for (const k of c.clips) { await removeFile(k.id); if (k.ar) await removeFile(`${k.id}.cloud`); } await removeFile(`${c.id}.json`); memJson.delete(c.id);
  try { localStorage.setItem(INDEX, JSON.stringify(listCaptures().filter(x => x.id !== c.id))); } catch { /* ignore */ }
}

async function dir() { return (navigator.storage as any)?.getDirectory ? await navigator.storage.getDirectory() : null; }

/** A sink MediaRecorder chunks are written to as they arrive. */
export async function openSink(id: string) {
  const d = await dir().catch(() => null); const mem: Blob[] = [];
  let w: any = null;
  if (d) { try { const fh = await (d as any).getFileHandle(id, { create: true }); w = await fh.createWritable(); } catch { w = null; } }
  let queue = Promise.resolve();
  return {
    kind: w ? 'disk' as const : 'memory' as const,
    write(b: Blob) { if (w) queue = queue.then(() => w.write(b)); else mem.push(b); },
    async close(mime: string) { if (w) { await queue; await w.close(); } else memFiles.set(id, new Blob(mem, { type: mime })); },
  };
}
const memFiles = new Map<string, Blob>();

export async function readFile(id: string, mime = 'video/mp4'): Promise<Blob | null> {
  if (memFiles.has(id)) return memFiles.get(id)!;
  const d = await dir().catch(() => null); if (!d) return null;
  try { const f = await (await (d as any).getFileHandle(id)).getFile(); return f.type ? f : new Blob([f], { type: mime }); } catch { return null; }
}
export async function storeUpload(id: string, file: File) {
  const d = await dir().catch(() => null);
  if (d) { try { const w = await (await (d as any).getFileHandle(id, { create: true })).createWritable(); await w.write(file); await w.close(); return; } catch { /* fall through */ } }
  memFiles.set(id, file);
}
async function removeFile(id: string) { memFiles.delete(id); const d = await dir().catch(() => null); try { await (d as any)?.removeEntry(id); } catch { /* gone */ } }

export async function storageLeft() { try { const e = await navigator.storage.estimate(); return (e.quota ?? 0) - (e.usage ?? 0); } catch { return null; } }
export async function persist() { try { return await navigator.storage.persist?.() ?? false; } catch { return false; } }

/** The AR coverage cloud: float32 xyz then uint8 rgb per voxel, behind a 4-byte count. */
export async function saveCloud(id: string, pts: Float32Array, col: Float32Array) {
  const n = pts.length / 3, rgb = new Uint8Array(n * 3); for (let i = 0; i < n * 3; i++) rgb[i] = Math.round(col[i] * 255);
  await storeUpload(`${id}.cloud`, new File([new Uint32Array([n]), new Float32Array(pts), rgb], `${id}.cloud`));
}
export async function loadCloud(id: string) {
  const b = await readFile(`${id}.cloud`, 'application/octet-stream'); if (!b) return null; const buf = await b.arrayBuffer(), n = new Uint32Array(buf, 0, 1)[0];
  const pts = new Float32Array(buf, 4, n * 3), rgb = new Uint8Array(buf, 4 + n * 12, n * 3), col = new Float32Array(n * 3); for (let i = 0; i < n * 3; i++) col[i] = rgb[i] / 255;
  return { pts, col };
}
