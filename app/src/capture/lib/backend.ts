/* Getting a capture to the processing pipeline (cloud/src/index.ts is the other side).
   With VITE_API set, clips go straight to the bucket as resumable multipart uploads (the Ply API signs each part) and a job is queued for
   the GPU worker. The team key gates writes; it is entered once and kept on the device (or arrives as ?key=).
   Without VITE_API the app says so plainly and packages everything for the phone instead. */
import type { Capture } from './store';
import { readFile } from './store';

export interface Settings {
  preset: 'quick' | 'standard' | 'high';
  advanced: { iterations?: number; shDegree?: number; maxSplats?: number; mapper: 'global' | 'incremental'; features: 'aliked' | 'sift'; fps?: number; cameraHeight: number; focal35?: number; };
}
export const PRESETS: Record<Settings['preset'], { label: string; note: string; iterations: number; shDegree: number; maxSplats: number }> = {
  quick: { label: 'Quick', note: 'About 10 min on a T4. Good for checking a capture.', iterations: 7000, shDegree: 2, maxSplats: 600_000 },
  standard: { label: 'Standard', note: 'About 35 min. What judges should see.', iterations: 30000, shDegree: 3, maxSplats: 1_500_000 },
  high: { label: 'High', note: 'About 70 min on an L40S. For the final venue.', iterations: 40000, shDegree: 3, maxSplats: 3_000_000 },
};

export const API = (import.meta.env.VITE_API as string | undefined)?.replace(/\/$/, '');
export const backendReady = !!API;
const KEY = 'ply-team-key';
/** A ?key= link is read once at startup, saved, and removed from the address so a stale link can't override the field later. */
export function adoptKeyFromUrl() {
  const u = new URL(location.href), q = u.searchParams.get('key'); if (!q) return;
  try { localStorage.setItem(KEY, q.trim()); } catch { /* private mode */ }
  u.searchParams.delete('key'); history.replaceState(history.state, '', u.pathname + u.search + u.hash);
}
export function teamKey() { try { return localStorage.getItem(KEY) ?? ''; } catch { return ''; } }
export function setTeamKey(k: string) { try { localStorage.setItem(KEY, k.trim()); } catch { /* private mode */ } }

/** CLI flags for `python -m plypipe run`, from the chosen settings. */
export function pipelineArgs(s: Settings) {
  const p = PRESETS[s.preset], a = s.advanced, f: string[] = ['--preset', s.preset];
  if (a.iterations && a.iterations !== p.iterations) f.push('--iterations', String(a.iterations));
  if (a.shDegree !== undefined && a.shDegree !== p.shDegree) f.push('--sh-degree', String(a.shDegree));
  if (a.maxSplats && a.maxSplats !== p.maxSplats) f.push('--cap-max', String(a.maxSplats));
  if (a.mapper !== 'global') f.push('--mapper', a.mapper);
  if (a.features !== 'aliked') f.push('--features', a.features);
  if (a.fps) f.push('--fps', String(a.fps));
  f.push('--height-cam', String(a.cameraHeight));
  if (a.focal35) f.push('--focal-35mm', String(a.focal35));
  return f;
}

/** Everything the pipeline (and a human) needs besides the videos. */
export function captureJson(c: Capture, s: Settings) {
  return JSON.stringify({ version: 1, name: c.name, createdAt: new Date(c.createdAt).toISOString(), settings: s, pipelineArgs: pipelineArgs(s),
    clips: c.clips.map(k => ({ file: k.name, source: k.source, mime: k.mime, duration: k.duration, width: k.width, height: k.height, lock: k.lock, loopClosed: k.loopClosed,
      quality: k.quality.map(q => [Math.round(q.t), q.grade, +q.sharp.toFixed(1), Math.round(q.luma), +q.flow.toFixed(1), +q.turn.toFixed(1)]),
      qualityFields: ['tMs', 'grade', 'sharp', 'luma', 'flowPctPerS', 'turnDegPerS'],
      sensors: k.sensors.map(x => [Math.round(x.t), x.heading == null ? null : +x.heading.toFixed(1), +x.pitch.toFixed(1), +x.roll.toFixed(1), +x.turn.toFixed(1), +x.accel.toFixed(2)]),
      sensorFields: ['tMs', 'headingDeg', 'pitchDeg', 'rollDeg', 'turnDegPerS', 'accelMS2'], path: k.path })) }, null, 1);
}

const PART = 16 * 1024 * 1024; // S3 wants ≥5 MB parts; 16 MB keeps a dropped part cheap to redo on a phone connection
interface UpState { key: string; uploadId: string; parts: { partNumber: number; etag: string }[]; done?: boolean; }
const upKey = (clipId: string) => `ply-up-${clipId}`;
const loadUp = (id: string): UpState | null => { try { return JSON.parse(localStorage.getItem(upKey(id)) || 'null'); } catch { return null; } };
const saveUp = (id: string, s: UpState) => { try { localStorage.setItem(upKey(id), JSON.stringify(s)); } catch { /* ignore */ } };

async function call(path: string, init: RequestInit, signal: AbortSignal, tries = 5): Promise<any> {
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(`${API}${path}`, { ...init, signal, headers: { 'x-ply-key': teamKey(), ...(init.headers ?? {}) } });
      if (r.status === 401) throw Object.assign(new Error('The team key was not accepted'), { fatal: true });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error ?? `server said ${r.status}`);
      return r.status === 204 ? null : r.json();
    } catch (e) {
      if (signal.aborted || (e as any).fatal || i >= tries) throw e;
      await new Promise(res => setTimeout(res, [800, 2000, 5000, 10000, 20000][i]));
    }
  }
}

/** One part straight to storage; the bucket's ETag header is what completing the upload needs. */
async function putPart(url: string, body: Blob, signal: AbortSignal) {
  for (let i = 0; ; i++) {
    try {
      const r = await fetch(url, { method: 'PUT', body, signal });
      if (!r.ok) throw new Error(`storage said ${r.status}`);
      const etag = r.headers.get('etag'); if (!etag) throw Object.assign(new Error('storage hid the part ETag (bucket CORS must expose ETag)'), { fatal: true });
      return etag;
    } catch (e) { if (signal.aborted || (e as any).fatal || i >= 5) throw e; await new Promise(res => setTimeout(res, [800, 2000, 5000, 10000, 20000][i])); }
  }
}

/** Upload every clip (resuming finished parts), then capture.json, then queue the job. */
export async function upload(c: Capture, s: Settings, onProgress: (done: number, total: number, label: string) => void, signal: AbortSignal) {
  if (!backendReady) throw new Error('No processing server is connected yet');
  if (!teamKey()) throw Object.assign(new Error('Enter the team key first'), { fatal: true });
  const total = c.clips.reduce((a, k) => a + k.size, 0); let before = 0; const keys: string[] = [];
  for (const k of c.clips) {
    const blob = await readFile(k.id, k.mime); if (!blob) throw new Error(`${k.name} is missing from this phone`);
    let st = loadUp(k.id);
    if (!st?.done) {
      if (!st) { const r = await call('/v1/uploads', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ captureId: c.id, name: k.name, type: k.mime }) }, signal); st = { ...r, parts: [] }; saveUp(k.id, st!); }
      const n = Math.max(1, Math.ceil(blob.size / PART));
      const todo = Array.from({ length: n }, (_, i) => i + 1).filter(p => !st!.parts.some(x => x.partNumber === p));
      let done = blob.size - todo.reduce((a, p) => a + Math.min(PART, blob.size - (p - 1) * PART), 0); onProgress(before + done, total, k.name);
      // signed links come in batches (they expire after a few hours, so a resumed upload asks for fresh ones)
      for (let i = 0; i < todo.length; i += 20) {
        const batch = todo.slice(i, i + 20);
        const { urls } = await call('/v1/uploads/sign', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ key: st!.key, uploadId: st!.uploadId, parts: batch }) }, signal);
        for (const p of batch) {
          const body = blob.slice((p - 1) * PART, Math.min(blob.size, p * PART));
          const etag = await putPart(urls[p], body, signal);
          st!.parts.push({ partNumber: p, etag }); saveUp(k.id, st!); done += body.size; onProgress(before + done, total, k.name);
        }
      }
      await call('/v1/uploads/complete', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ key: st!.key, uploadId: st!.uploadId, parts: st!.parts.sort((a, b) => a.partNumber - b.partNumber) }) }, signal);
      st!.done = true; saveUp(k.id, st!);
    }
    keys.push(st!.key); before += k.size;
  }
  const cj = `captures/${c.id.replace(/[^\w.\-]+/g, '_')}/capture.json`;
  await call(`/v1/files?key=${encodeURIComponent(cj)}`, { method: 'PUT', headers: { 'content-type': 'application/json' }, body: captureJson(c, s) }, signal);
  keys.push(cj);
  const j = await call('/v1/jobs', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ captureId: c.id, name: c.name, settings: s, args: pipelineArgs(s), files: keys }) }, signal);
  return j as { jobId: string };
}

export interface Job { id: string; name: string; status: 'queued' | 'running' | 'done' | 'failed'; stage: string | null; progress: number; message: string | null;
  result: { base: string; files?: Record<string, string>; report?: any } | null; created_at: number; started_at: number | null; finished_at: number | null; updated_at: number; queuePosition: number; }
export async function getJob(id: string, signal?: AbortSignal): Promise<Job> {
  const r = await fetch(`${API}/v1/jobs/${encodeURIComponent(id)}`, { signal, cache: 'no-store' }); if (!r.ok) throw new Error(r.status === 404 ? 'No such job' : `server said ${r.status}`); return r.json();
}
export interface Health { workersOnline: number; workers: { name: string; busy: boolean; seenSecondsAgo: number }[]; queued: number; }
export async function getHealth(signal?: AbortSignal): Promise<Health | null> {
  try { const r = await fetch(`${API}/v1/health`, { signal, cache: 'no-store' }); return r.ok ? r.json() : null; } catch { return null; }
}
/** The walkthrough link for a finished job. */
export const sceneUrl = (jobId: string) => `${API}/v1/scenes/${jobId}/`;

/** No server: hand the files to the phone (share sheet where available, downloads otherwise). */
export async function saveToDevice(c: Capture, s: Settings) {
  const files: File[] = [];
  for (const k of c.clips) { const b = await readFile(k.id, k.mime); if (b) files.push(new File([b], k.name, { type: k.mime })); }
  files.push(new File([captureJson(c, s)], 'capture.json', { type: 'application/json' }));
  if ((navigator as any).canShare?.({ files })) { try { await navigator.share({ files, title: c.name }); return 'shared'; } catch (e) { if ((e as Error).name === 'AbortError') return 'cancelled'; } }
  for (const f of files) { const a = document.createElement('a'); a.href = URL.createObjectURL(f); a.download = f.name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); await new Promise(r => setTimeout(r, 400)); }
  return 'downloaded';
}
