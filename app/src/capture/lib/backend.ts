// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
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
export const PRESETS: Record<Settings['preset'], { label: string; note: string; multiplier: string; iterations: number; shDegree: number; maxSplats: number }> = {
  quick: { label: 'Quick', note: 'A fast look at the capture. Good for a coverage check.', multiplier: 'light', iterations: 7000, shDegree: 2, maxSplats: 600_000 },
  standard: { label: 'Standard', note: 'The normal run. Detail follows your clips.', multiplier: 'balanced', iterations: 30000, shDegree: 3, maxSplats: 1_500_000 },
  high: { label: 'High', note: 'More training time, finer detail. For the final venue.', multiplier: 'best', iterations: 40000, shDegree: 3, maxSplats: 2_000_000 },
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

/** Invite only: no signup, no account. A signed code is traded for GPU time; the worker verifies it. */
export interface WhoAmI { role: 'team' | 'invite'; label: string; expires?: number }
export async function whoami(code = teamKey()): Promise<WhoAmI> {
  if (!API) throw new Error('No processing server is connected');
  if (!code.trim()) throw new Error('Enter an invite code');
  const r = await fetch(`${API}/v1/whoami`, { headers: { 'x-ply-key': code.trim() } });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error ?? 'that invite is not valid or has expired');
  return body as WhoAmI;
}
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

/** The name a clip has on the server (the API's safeName): what the pipeline and its reports call it. */
export const serverName = (n: string) => n.replace(/[^\w.\-]+/g, '_').slice(0, 120) || 'file';

/** Everything the pipeline (and a human) needs besides the videos. */
export function captureJson(c: Capture, s: Settings) {
  return JSON.stringify({ version: 1, name: c.name, createdAt: new Date(c.createdAt).toISOString(), settings: s, pipelineArgs: pipelineArgs(s),
    clips: c.clips.map(k => ({ file: serverName(k.name), source: k.source, mime: k.mime, duration: k.duration, width: k.width, height: k.height, lock: k.lock, loopClosed: k.loopClosed, ar: k.ar, exclude: k.exclude,
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

/** Coverage check: frames + camera solving + align only, quick settings. Publishes preview.bin / preview.json. */
export const previewArgs = (s: Settings) => ['--preset', 'quick', '--until', 'align', '--features', 'sift', '--fps', '1.5', '--height-cam', String(s.advanced.cameraHeight)];
export const DEFAULT_SETTINGS: Settings = { preset: 'standard', advanced: { mapper: 'global', features: 'aliked', cameraHeight: 1.45 } };

/** Upload every clip (resuming finished parts), then capture.json, then queue the job (or a coverage check). */
export async function upload(c: Capture, s: Settings, onProgress: (done: number, total: number, label: string) => void, signal: AbortSignal, preview = false) {
  if (!backendReady) throw new Error('No processing server is connected yet');
  if (!teamKey()) throw Object.assign(new Error('Enter your invite code first'), { fatal: true });
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
  const j = await call('/v1/jobs', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ captureId: c.id, name: preview ? `${c.name} · coverage check` : c.name, settings: { ...s, preview }, args: preview ? previewArgs(s) : pipelineArgs(s), files: keys }) }, signal);
  return j as { jobId: string };
}

export interface Job { id: string; name: string; status: 'queued' | 'running' | 'review' | 'done' | 'failed'; stage: string | null; progress: number; message: string | null;
  result: { base: string; files?: Record<string, string>; report?: any;
    quality?: { ok: boolean; issues: { level: 'fail' | 'warn'; code: string; message: string }[] };
    review?: { verdict: string; confidence: number; recommend: string; good: string[]; risk: string[]; note: string; reviewer: string };
    attempts?: { attempt: number; args: string[]; quality?: { ok: boolean } }[] } | null; created_at: number; started_at: number | null; finished_at: number | null; updated_at: number; queuePosition: number; }
export async function decideJob(id: string, decision: 'approve' | 'reject', note = ''): Promise<{ ok: boolean; status: string }> {
  const r = await fetch(`${API}/v1/jobs/${encodeURIComponent(id)}/decision`, { method: 'POST',
    headers: { 'content-type': 'application/json', 'x-ply-key': teamKey() }, body: JSON.stringify({ decision, note }) });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error ?? 'the decision did not go through');
  return body;
}
export async function getJob(id: string, signal?: AbortSignal): Promise<Job> {
  const r = await fetch(`${API}/v1/jobs/${encodeURIComponent(id)}`, { signal, cache: 'no-store' }); if (!r.ok) throw new Error(r.status === 404 ? 'No such job' : `server said ${r.status}`); return r.json();
}
export interface Health { workersOnline: number; workersStarting?: number; workers: { name: string; busy: boolean; starting?: string | null; seenSecondsAgo: number }[]; queued: number; }
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

/** The coverage check's result: sparse points (xyz, rgb, how many frames saw each) and every frame's clip, second and fate. */
export interface PreviewFrame { frame: string; clip: string | null; t: number | null; registered: boolean; c: number[] | null; }
export interface Preview { pts: Float32Array; rgb: Uint8Array; views: Uint8Array; frames: PreviewFrame[]; registered: number; total: number; aligned: boolean; }
export async function getPreview(jobId: string, signal?: AbortSignal): Promise<Preview> {
  const [b, j] = await Promise.all([fetch(`${sceneUrl(jobId)}preview.bin`, { signal }), fetch(`${sceneUrl(jobId)}preview.json`, { signal })]);
  if (!b.ok || !j.ok) throw new Error('The coverage preview is not ready');
  const buf = await b.arrayBuffer(), n = new Uint32Array(buf, 0, 1)[0], meta = await j.json();
  return { pts: new Float32Array(buf.slice(4, 4 + n * 12)), rgb: new Uint8Array(buf, 4 + n * 12, n * 3), views: new Uint8Array(buf, 4 + n * 15, n), ...meta };
}
