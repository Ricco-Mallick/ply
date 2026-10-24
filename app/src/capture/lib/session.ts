// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* One live recording: MediaRecorder into the store, the quality loop on the preview, sensors logged at the same clock,
   and a screen wake lock so the phone doesn't dim mid-walk. The UI subscribes and redraws from `live`. */
import { Analyzer, TIP, type Issue, type Quality } from './analyzer';
import { bitrateFor, extFor, pickRecorderType, type CamInfo, type LockReport } from './camera';
import { Sensors, type SensorSample } from './sensors';
import { newId, openSink, type Clip } from './store';

export interface Live { recording: boolean; elapsed: number; q: Quality | null; tip: string | null; tipIssue: Issue | null; grades: number[]; steps: number; loop: boolean; storage: 'disk' | 'memory' | null; bytes: number; }

const SAMPLE_MS = 125; // 8 quality samples a second

export class Session {
  analyzer = new Analyzer(); sensors = new Sensors();
  live: Live = { recording: false, elapsed: 0, q: null, tip: null, tipIssue: null, grades: [], steps: 0, loop: false, storage: null, bytes: 0 };
  private rec: MediaRecorder | null = null; private sink: Awaited<ReturnType<typeof openSink>> | null = null; private clipId = '';
  quality: Quality[] = []; log: SensorSample[] = []; private t0 = 0; private timer = 0; private wake: any = null;
  private subs = new Set<() => void>(); private tipHold = 0; private mime = '';
  constructor(public video: HTMLVideoElement, public stream: MediaStream, public info: CamInfo, public lock: LockReport | null) { /* stream/info change when the lens is switched */ this.sensors.start(); this.loop(); }

  on(f: () => void) { this.subs.add(f); return () => { this.subs.delete(f); }; }
  private emit() { this.subs.forEach(f => f()); }

  /** Runs whether or not we're recording, so tips work while framing the first shot. */
  private loop() {
    const tick = () => {
      this.timer = window.setTimeout(tick, SAMPLE_MS);
      const v = this.video; if (v.readyState < 2 || !v.videoWidth) return;
      const now = performance.now(), t = this.live.recording ? now - this.t0 : 0;
      const q = this.analyzer.measure(v, v.videoWidth, v.videoHeight, now, this.sensors.available ? this.sensors.turn : 0, this.sensors.available ? this.sensors.pitch : null);
      q.t = t; this.live.q = q;
      // tips change at most every 1.2 s so they can be read; good frames clear them
      if (now > this.tipHold) { this.live.tipIssue = q.issue; this.live.tip = q.issue ? TIP[q.issue] : null; if (q.issue) this.tipHold = now + 1200; }
      if (this.live.recording) {
        this.quality.push(q); this.log.push(this.sensors.sample(t)); this.live.grades.push(q.grade); if (this.live.grades.length > 240) this.live.grades.shift();
        this.live.elapsed = t; this.live.steps = this.sensors.steps; this.live.loop = this.sensors.loopClosed;
      }
      this.emit();
    };
    tick();
  }

  async start() {
    this.mime = pickRecorderType(); this.clipId = newId(); this.sink = await openSink(this.clipId); this.live.storage = this.sink.kind; this.live.bytes = 0;
    this.rec = new MediaRecorder(this.stream, { ...(this.mime ? { mimeType: this.mime } : {}), videoBitsPerSecond: bitrateFor(this.info.width, this.info.height) });
    this.rec.ondataavailable = e => { if (e.data.size) { this.sink!.write(e.data); this.live.bytes += e.data.size; } };
    this.quality = []; this.log = []; this.live.grades = []; this.analyzer.reset(); this.sensors.resetPath();
    this.t0 = performance.now(); this.rec.start(1000); this.live.recording = true;
    try { this.wake = await (navigator as any).wakeLock?.request('screen'); } catch { /* not granted: the screen may dim */ }
    navigator.vibrate?.(30); this.emit();
  }

  async stop(name: string): Promise<Clip> {
    const rec = this.rec!; const done = new Promise<void>(r => { rec.onstop = () => r(); });
    rec.stop(); await done; this.live.recording = false;
    const mime = rec.mimeType || this.mime || 'video/webm'; await this.sink!.close(mime);
    try { await this.wake?.release(); } catch { /* already released */ }
    navigator.vibrate?.([20, 60, 20]);
    const clip: Clip = { id: this.clipId, name: `${name}.${extFor(mime)}`, source: 'camera', mime, size: this.live.bytes, duration: (performance.now() - this.t0) / 1000,
      width: this.info.width, height: this.info.height, createdAt: Date.now(), quality: this.quality, sensors: this.log, path: this.sensors.path.slice(),
      lock: this.lock ? { ...this.lock } : undefined, loopClosed: this.sensors.loopClosed };
    this.emit(); return clip;
  }

  dispose() { clearTimeout(this.timer); this.sensors.stop(); if (this.rec?.state === 'recording') this.rec.stop(); this.stream.getTracks().forEach(t => t.stop()); this.wake?.release?.().catch(() => {}); }
}

/** Uploaded videos get the same grading by seeking through them (no sensors, so no path or turn rate). */
export async function analyseFile(file: File, onProgress: (p: number) => void): Promise<Omit<Clip, 'id' | 'name' | 'source'>> {
  const v = document.createElement('video'); v.muted = true; v.playsInline = true; v.preload = 'auto'; v.src = URL.createObjectURL(file);
  await new Promise<void>((res, rej) => { v.onloadedmetadata = () => res(); v.onerror = () => rej(new Error('This video could not be read by the browser')); });
  const a = new Analyzer(), q: Quality[] = [], step = .5, dur = v.duration || 0;
  for (let t = 0; t < dur; t += step) {
    v.currentTime = t; await new Promise<void>(r => { v.onseeked = () => r(); });
    const s = a.measure(v, v.videoWidth, v.videoHeight, t * 1000); s.t = t * 1000; s.flow = 0; q.push(s); onProgress(t / dur);
  }
  const w = v.videoWidth, h = v.videoHeight; URL.revokeObjectURL(v.src);
  return { mime: file.type || 'video/mp4', size: file.size, duration: dur, width: w, height: h, createdAt: Date.now(), quality: q, sensors: [], path: [] };
}
