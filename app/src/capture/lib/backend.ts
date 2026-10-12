/* Getting a capture to the processing pipeline.
   With VITE_UPLOAD_ENDPOINT (a tus server: Supabase Storage or a Cloudflare Worker in front of R2) and VITE_JOBS_ENDPOINT
   set, clips upload resumably and a job is queued. Without them the app says so plainly and packages everything for the
   phone instead: the videos, capture.json (quality + sensor log) and the exact pipeline command. */
import * as tus from 'tus-js-client';
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

const UPLOAD = import.meta.env.VITE_UPLOAD_ENDPOINT as string | undefined;
const JOBS = import.meta.env.VITE_JOBS_ENDPOINT as string | undefined;
export const backendReady = !!(UPLOAD && JOBS);

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

export async function upload(c: Capture, s: Settings, onProgress: (done: number, total: number, label: string) => void, signal: AbortSignal) {
  if (!backendReady) throw new Error('No processing server is connected yet');
  const total = c.clips.reduce((a, k) => a + k.size, 0); let before = 0; const uploaded: string[] = [];
  for (const k of c.clips) {
    const blob = await readFile(k.id, k.mime); if (!blob) throw new Error(`${k.name} is missing from this phone`);
    const url = await new Promise<string>((res, rej) => {
      const up = new tus.Upload(blob, { endpoint: UPLOAD!, retryDelays: [0, 1500, 4000, 10000, 20000], chunkSize: 8 * 1024 * 1024, metadata: { filename: k.name, filetype: k.mime, capture: c.id },
        onProgress: (b) => onProgress(before + b, total, k.name), onSuccess: () => res(up.url!), onError: rej });
      signal.addEventListener('abort', () => { up.abort(); rej(new DOMException('Cancelled', 'AbortError')); });
      up.findPreviousUploads().then(prev => { if (prev.length) up.resumeFromPreviousUpload(prev[0]); up.start(); });
    });
    uploaded.push(url); before += k.size;
  }
  const res = await fetch(JOBS!, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ capture: JSON.parse(captureJson(c, s)), files: uploaded }), signal });
  if (!res.ok) throw new Error(`The server refused the job (${res.status})`);
  return (await res.json()) as { jobId: string; statusUrl?: string };
}

/** No server: hand the files to the phone (share sheet where available, downloads otherwise). */
export async function saveToDevice(c: Capture, s: Settings) {
  const files: File[] = [];
  for (const k of c.clips) { const b = await readFile(k.id, k.mime); if (b) files.push(new File([b], k.name, { type: k.mime })); }
  files.push(new File([captureJson(c, s)], 'capture.json', { type: 'application/json' }));
  if ((navigator as any).canShare?.({ files })) { try { await navigator.share({ files, title: c.name }); return 'shared'; } catch (e) { if ((e as Error).name === 'AbortError') return 'cancelled'; } }
  for (const f of files) { const a = document.createElement('a'); a.href = URL.createObjectURL(f); a.download = f.name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); await new Promise(r => setTimeout(r, 400)); }
  return 'downloaded';
}
