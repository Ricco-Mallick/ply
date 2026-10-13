/* Coverage check for any capture (videos or AR scans): a quick camera-solving run on the GPU worker, then the sparse
   point cloud in 3D (coloured by how many frames saw each point) and, per clip, the stretches where solving lost track,
   each with "Replace this part". Clips stay uploaded, so processing afterwards starts straight away. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Glass } from '../../shared/Glass';
import { DEFAULT_SETTINGS, backendReady, getHealth, getJob, getPreview, setTeamKey, teamKey, upload, type Health, type Job, type Preview } from '../lib/backend';
import type { Capture } from '../lib/store';
import { CloudView } from './CloudView';
import { stageLabel, stagePct } from './Status';
import { fmtTime } from './viz';

const G = [61 / 255, 220 / 255, 151 / 255], O = [1, 184 / 255, 77 / 255], R = [242 / 255, 85 / 255, 90 / 255];
export interface Span { clipId: string; from: number; to: number }

/** Runs of 2+ frames that camera solving could not place, padded a little: the parts worth reshooting. */
export function lostSpans(c: Capture, p: Preview): Span[] {
  const out: Span[] = [];
  for (const k of c.clips) {
    const fr = p.frames.filter(f => f.clip === k.name && f.t != null).sort((a, b) => a.t! - b.t!); let run: typeof fr = [];
    const flush = () => { if (run.length >= 2) out.push({ clipId: k.id, from: Math.max(0, run[0].t! - .5), to: Math.min(k.duration || 1e9, run[run.length - 1].t! + .5) }); run = []; };
    for (const f of fr) { if (!f.registered) run.push(f); else flush(); } flush();
  }
  return out;
}

export function CoverageCheck({ capture, onPreview, onReplace }: { capture: Capture; onPreview: (jobId: string) => void; onReplace: (s: Span) => void }) {
  const [phase, setPhase] = useState<'idle' | 'upload' | 'job' | 'ready' | 'error'>(capture.preview ? 'job' : 'idle');
  const [up, setUp] = useState(0); const [job, setJob] = useState<Job | null>(null); const [health, setHealth] = useState<Health | null>(null);
  const [prev, setPrev] = useState<Preview | null>(null); const [err, setErr] = useState<string | null>(null); const [key, setKey] = useState(teamKey());
  const ac = useRef<AbortController | null>(null);
  const stale = !!capture.preview && capture.preview.clips.join() !== capture.clips.map(k => `${k.id}:${(k.exclude ?? []).length}`).join();

  const run = async () => {
    if (!key.trim()) { setErr('Enter the team key first'); return; } setTeamKey(key); setErr(null); setPrev(null); setPhase('upload');
    ac.current?.abort(); ac.current = new AbortController();
    try { const j = await upload(capture, DEFAULT_SETTINGS, (d, t) => setUp(t ? d / t : 0), ac.current.signal, true); onPreview(j.jobId); setPhase('job'); }
    catch (e) { if (!ac.current.signal.aborted) { setErr((e as Error).message); setPhase('error'); } }
  };
  // follow the job; load the preview when it is done
  useEffect(() => {
    if (phase !== 'job' || !capture.preview) return; const a = new AbortController(); let t = 0;
    const poll = async () => {
      try {
        const j = await getJob(capture.preview!.jobId, a.signal); setJob(j);
        if (j.status === 'queued') setHealth(await getHealth(a.signal));
        if (j.status === 'done') { setPrev(await getPreview(capture.preview!.jobId, a.signal)); setPhase('ready'); return; }
        if (j.status === 'failed') { setErr(j.message ?? 'The coverage check failed'); setPhase('error'); return; }
      } catch (e) { if (a.signal.aborted) return; }
      t = window.setTimeout(poll, 4000);
    };
    poll(); return () => { a.abort(); clearTimeout(t); };
  }, [phase, capture.preview?.jobId]);
  useEffect(() => () => ac.current?.abort(), []);

  const view = useMemo(() => {
    if (!prev) return null; const n = prev.views.length, col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const v = prev.views[i], c = v >= 6 ? G : v >= 3 ? O : R; col.set(c, i * 3); }
    const path = prev.frames.filter(f => f.registered && f.c).map(f => f.c!) as number[][];
    let g = 0, o = 0; for (const v of prev.views) { if (v >= 6) g++; else if (v >= 3) o++; }
    return { col, path, g: g / Math.max(1, n), o: o / Math.max(1, n) };
  }, [prev]);
  const spans = prev ? lostSpans(capture, prev) : [];

  if (!backendReady) return null;
  return (
    <Glass className="covcheck" opts={{ radius: 20, shader: false }}>
      <div className="cc-h"><h2 className="wide">Coverage in 3D</h2>{phase === 'ready' && <span className="mono dim">{prev!.registered}/{prev!.total} frames placed · {prev!.views.length.toLocaleString()} points</span>}</div>
      {phase === 'idle' && <>
        <p className="cc-p">Run a quick camera-solving pass (about 3–6 minutes once a GPU is free) and see your capture as a point cloud: what's covered, what's thin, and which parts of which clip lost track. Your clips stay uploaded, so processing afterwards starts at once.</p>
        {!teamKey() && <input className="cc-key mono" placeholder="Team key" value={key} onChange={e => setKey(e.target.value)} />}
        <button className="btn" onClick={run}>Check coverage</button>
      </>}
      {phase === 'upload' && <><p className="cc-p mono">Uploading · {Math.round(up * 100)}%</p><div className="bar"><i style={{ transform: `scaleX(${up})` }} /></div></>}
      {phase === 'job' && <>
        <p className="cc-p mono">{!job ? 'Checking…' : job.status === 'queued' ? (health?.workersOnline ? 'In the queue · GPU busy' : 'Waiting for a GPU worker to start') : `${stageLabel(job.stage)} · ${stagePct(job)}%`}</p>
        <div className="bar"><i style={{ transform: `scaleX(${job?.status === 'running' ? Math.min(1, (job.progress ?? 0) / .29) : 0})` }} /></div>
      </>}
      {phase === 'error' && <><p className="msg bad" role="alert">{err}</p><button className="btn ghost" onClick={run}>Try again</button></>}
      {phase === 'ready' && prev && view && <>
        <CloudView pts={prev.pts} col={view.col} path={view.path} height={340} dot={.035} />
        <div className="bars cc-bars"><i style={{ flex: view.g, background: '#3DDC97' }} /><i style={{ flex: view.o, background: '#FFB84D' }} /><i style={{ flex: 1 - view.g - view.o, background: '#FF5A4E' }} /></div>
        <p className="cc-p mono dim">Points seen by 6+ frames green · 3–5 orange · fewer red. Orange line: where the camera was.</p>
        {capture.clips.map(k => { const fr = prev.frames.filter(f => f.clip === k.name && f.t != null), d = k.duration || Math.max(1, ...fr.map(f => f.t!));
          return <div className="cc-clip" key={k.id}><span className="mono">{k.ar ? 'AR ' : ''}{k.name.replace(/\.\w+$/, '')}</span>
            <div className="cc-strip">{fr.map(f => <i key={f.frame} className={f.registered ? 'ok' : 'lost'} style={{ left: `${f.t! / d * 100}%` }} />)}{(k.exclude ?? []).map(([a, b], i) => <b key={i} style={{ left: `${a / d * 100}%`, width: `${(b - a) / d * 100}%` }} />)}</div></div>; })}
        {spans.length ? <ul className="cc-spans">{spans.map((s, i) => { const k = capture.clips.find(x => x.id === s.clipId)!; return <li key={i}><span><b>{fmtTime(s.from)}–{fmtTime(s.to)}</b> of {k.name.replace(/\.\w+$/, '')} lost track</span><button className="btn ghost" onClick={() => onReplace(s)}>Replace this part</button></li>; })}</ul>
          : <p className="cc-p">Every part of every clip was placed. Good to process.</p>}
        {stale && <p className="cc-p mono">Clips changed since this check.</p>}
        <button className="btn ghost" onClick={run}>{stale ? 'Check again' : 'Run again'}</button>
      </>}
    </Glass>
  );
}
