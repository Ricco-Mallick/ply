/* Live view of a processing job (worker side: pipeline/plypipe/live.py). Polls live-index.json and shows what each
   stage has made so far, interactively: the kept frames, the camera-solving point cloud with the camera path, and the
   twin itself at each training snapshot, plus the loss curve. When the job is done it shows the finished twin. */
import { useEffect, useMemo, useState } from 'react';
import { Glass } from '../../shared/Glass';
import { sceneUrl, type Job } from '../lib/backend';
import { CloudView } from './CloudView';
import { SplatOrbit } from './SplatOrbit';

interface LiveIndex {
  updated: number; stages: Record<string, { status: string; t: number }>;
  frames?: { file: string; count: number; shown: number; cols: number; w: number; h: number; t: number };
  sparse?: { file: string; meta: string; after: string; aligned: boolean; points: number; registered: number; total: number; t: number };
  snapshots: { step: number; file: string; mb: number; t: number }[]; loss: [number, number][];
}
type Tab = 'frames' | 'cameras' | 'twin';

export function LiveTheatre({ jobId, job }: { jobId: string; job: Job }) {
  const [idx, setIdx] = useState<LiveIndex | null>(null); const [tab, setTab] = useState<Tab | null>(null); const [pinned, setPinned] = useState(false);
  const base = sceneUrl(jobId);
  useEffect(() => {
    let t = 0; const ac = new AbortController();
    const poll = async () => {
      try { const r = await fetch(`${base}live-index.json`, { signal: ac.signal, cache: 'no-store' }); if (r.ok) setIdx(await r.json()); } catch { /* not started yet */ }
      if (job.status === 'running' || job.status === 'queued') t = window.setTimeout(poll, document.hidden ? 15000 : 5000);
    };
    poll(); return () => { ac.abort(); clearTimeout(t); };
  }, [jobId, job.status]);
  // follow the newest artifact unless the viewer picked a tab themselves
  const done = job.status === 'done' && !!job.result && !(job.result as any).preview;
  const latest: Tab | null = done || idx?.snapshots.length ? 'twin' : idx?.sparse ? 'cameras' : idx?.frames ? 'frames' : null;
  useEffect(() => { if (!pinned && latest) setTab(latest); }, [latest, pinned]);
  if (!idx && !done) return null;
  const pick = (t: Tab) => { setTab(t); setPinned(true); };
  const tabs: [Tab, string, boolean][] = [['frames', `Frames${idx?.frames ? ` · ${idx.frames.count}` : ''}`, !!idx?.frames], ['cameras', `Cameras${idx?.sparse ? ` · ${idx.sparse.registered}/${idx.sparse.total}` : ''}`, !!idx?.sparse], ['twin', done ? 'Finished twin' : `Twin${idx?.snapshots.length ? ` · ${idx.snapshots.length}` : ''}`, done || !!idx?.snapshots.length]];
  return (
    <Glass className="theatre" opts={{ radius: 22, shader: false }}>
      <div className="th-tabs" role="tablist">{tabs.map(([k, l, ok]) => <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? 'on' : ''} disabled={!ok} onClick={() => pick(k)}>{l}</button>)}
        {job.status === 'running' && <span className="th-live mono"><i />Live</span>}</div>
      {tab === 'frames' && idx?.frames && <Frames base={base} f={idx.frames} />}
      {tab === 'cameras' && idx?.sparse && <Sparse base={base} s={idx.sparse} />}
      {tab === 'twin' && <Twin base={base} idx={idx} done={done} />}
    </Glass>
  );
}

/** The contact sheet, revealed tile by tile so new captures read as frames being picked. */
function Frames({ base, f }: { base: string; f: NonNullable<LiveIndex['frames']> }) {
  const url = `${base}${f.file}?v=${Math.round(f.t)}`;
  return (
    <div className="th-frames">
      <div className="th-grid" style={{ gridTemplateColumns: `repeat(${Math.min(16, f.cols)}, 1fr)` }}>
        {Array.from({ length: f.shown }, (_, i) => <span key={i} style={{ backgroundImage: `url(${url})`, backgroundSize: `${f.cols * 100}% ${Math.ceil(f.shown / f.cols) * 100}%`,
          backgroundPosition: `${(i % f.cols) / Math.max(1, f.cols - 1) * 100}% ${Math.floor(i / f.cols) / Math.max(1, Math.ceil(f.shown / f.cols) - 1) * 100}%`, aspectRatio: `${f.w} / ${f.h}`, animationDelay: `${i * 18}ms` }} />)}
      </div>
      <p className="mono th-cap">{f.count} sharp frames kept from the video{f.count > f.shown ? ` · ${f.shown} shown` : ''}</p>
    </div>
  );
}

/** Camera solving: sparse points coloured by how many frames saw them, the cameras as an orange path. */
function Sparse({ base, s }: { base: string; s: NonNullable<LiveIndex['sparse']> }) {
  const [d, setD] = useState<{ pts: Float32Array; col: Float32Array; path: number[][] } | null>(null);
  useEffect(() => {
    const v = Math.round(s.t); let gone = false;
    Promise.all([fetch(`${base}${s.file}?v=${v}`).then(r => r.arrayBuffer()), fetch(`${base}${s.meta}?v=${v}`).then(r => r.json())]).then(([buf, meta]) => {
      if (gone) return; const n = new Uint32Array(buf, 0, 1)[0], pts = new Float32Array(buf.slice(4, 4 + n * 12)), rgb = new Uint8Array(buf, 4 + n * 12, n * 3), col = new Float32Array(n * 3);
      for (let i = 0; i < n * 3; i++) col[i] = rgb[i] / 255; // the points' real colours: the room appears
      setD({ pts, col, path: meta.frames.filter((f: any) => f.c).map((f: any) => f.c) });
    }).catch(() => {});
    return () => { gone = true; };
  }, [base, s.t]);
  return <div>{d ? <CloudView pts={d.pts} col={d.col} path={d.path} height={420} dot={.03} /> : <div className="th-wait mono">Loading the point cloud…</div>}
    <p className="mono th-cap">{s.points.toLocaleString()} points · {s.registered} of {s.total} frames placed{s.aligned ? ' · levelled, real scale' : ' · before levelling'}</p></div>;
}

/** The twin: each training snapshot, or the finished model; plus the loss curve. */
function Twin({ base, idx, done }: { base: string; idx: LiveIndex | null; done: boolean }) {
  const snaps = idx?.snapshots ?? [];
  const [i, setI] = useState<number | null>(null);
  const cur = i == null ? (done ? -1 : snaps.length - 1) : i;
  const url = cur < 0 ? `${base}scene_lite.spz` : `${base}${snaps[cur].file}`;
  const loss = idx?.loss ?? [];
  // frame the room: the aligned sparse cloud's middle 90 % (from the cameras tab's data)
  const [frame, setFrame] = useState<{ target: number[]; radius: number }>({ target: [0, .9, 0], radius: 7 });
  useEffect(() => {
    if (!idx?.sparse?.aligned) return; let gone = false;
    fetch(`${base}${idx.sparse.file}?v=${Math.round(idx.sparse.t)}`).then(r => r.arrayBuffer()).then(buf => {
      if (gone) return; const n = new Uint32Array(buf, 0, 1)[0], p = new Float32Array(buf.slice(4, 4 + n * 12));
      const q = (ax: number, f: number) => { const a: number[] = []; for (let i = ax; i < p.length; i += 3 * Math.max(1, Math.floor(n / 4000))) a.push(p[i]); a.sort((x, y) => x - y); return a[Math.floor(f * (a.length - 1))] ?? 0; };
      const lo = [q(0, .05), q(2, .05)], hi = [q(0, .95), q(2, .95)];
      setFrame({ target: [(lo[0] + hi[0]) / 2, .8, (lo[1] + hi[1]) / 2], radius: Math.max(4, Math.hypot(hi[0] - lo[0], hi[1] - lo[1]) * .95) });
    }).catch(() => {});
    return () => { gone = true; };
  }, [base, idx?.sparse?.t]);
  const path = useMemo(() => { if (loss.length < 2) return ''; const ys = loss.map(l => l[1]), lo = Math.min(...ys), hi = Math.max(...ys);
    return loss.map(([x, y], k) => `${k ? 'L' : 'M'}${(x * 300).toFixed(1)},${(56 - (y - lo) / Math.max(1e-6, hi - lo) * 50).toFixed(1)}`).join(''); }, [loss.length]);
  return (
    <div>
      <SplatOrbit url={url} height={440} key={url} target={frame.target} radius={frame.radius} />
      <div className="th-snaps">
        {snaps.map((s, k) => <button key={s.step} className={cur === k ? 'on' : ''} onClick={() => setI(k)}>step {s.step.toLocaleString()}</button>)}
        {done && <button className={cur < 0 ? 'on' : ''} onClick={() => setI(-1)}>Finished</button>}
        {loss.length > 1 && <svg className="th-loss" viewBox="0 0 300 60" aria-label="Training loss over time (lower is better)"><path d={path} /></svg>}
      </div>
      <p className="mono th-cap">{cur < 0 ? 'The finished twin (phone version). Open the walkthrough to walk inside it.' : `The twin at step ${snaps[cur].step.toLocaleString()}: a preview with fewer splats; it sharpens as training goes on.`}</p>
    </div>
  );
}
