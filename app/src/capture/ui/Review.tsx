import { useEffect, useMemo, useRef, useState } from 'react';
import { Glass } from '../../shared/Glass';
import { TIP, problemSpans, summarise } from '../lib/analyzer';
import { loadCloud, readFile, type Capture, type Clip } from '../lib/store';
import { CloudView } from './CloudView';
import { HeadingRing, Legend, PathMap, Timeline, fmtBytes, fmtTime } from './viz';

const VERDICT = {
  good: { t: 'Good capture', d: 'Sharp, steady and well lit for most of the take. Ready to process.' },
  ok: { t: 'Usable, could be better', d: 'Some stretches are soft or fast. Processing will work; recapturing the red parts will look better.' },
  recapture: { t: 'Recapture recommended', d: 'Too much of this take is blurred, dark or rushed. Record the red stretches again before processing.' },
};

export function Review({ capture, onMore, onRetake, onContinue, onHome }: { capture: Capture; onMore: () => void; onRetake: (clipId: string) => void; onContinue: () => void; onHome: () => void }) {
  const [sel, setSel] = useState(capture.clips.length - 1); const clip = capture.clips[Math.min(sel, capture.clips.length - 1)];
  if (clip.ar) return <ArReview capture={capture} clip={clip} sel={sel} setSel={setSel} onMore={onMore} onRetake={onRetake} onContinue={onContinue} onHome={onHome} />;
  return <VideoReview capture={capture} clip={clip} sel={sel} setSel={setSel} onMore={onMore} onRetake={onRetake} onContinue={onContinue} onHome={onHome} />;
}

interface RP { capture: Capture; clip: Clip; sel: number; setSel: (i: number) => void; onMore: () => void; onRetake: (clipId: string) => void; onContinue: () => void; onHome: () => void }
const Tabs = ({ capture, sel, setSel }: { capture: Capture; sel: number; setSel: (i: number) => void }) => capture.clips.length > 1 ? <div className="clip-tabs" role="tablist">{capture.clips.map((k, i) => <button key={k.id} role="tab" aria-selected={i === sel} className={i === sel ? 'on' : ''} onClick={() => setSel(i)}><span className="mono">{String(i + 1).padStart(2, '0')}</span> {k.ar ? 'AR' : ''} {fmtTime(k.duration)}</button>)}</div> : null;

/** AR scans: the coverage cloud instead of a video, coverage instead of per-second quality. */
function ArReview({ capture, clip, sel, setSel, onMore, onRetake, onContinue, onHome }: RP) {
  const [cloud, setCloud] = useState<{ pts: Float32Array; col: Float32Array } | null | undefined>(undefined);
  useEffect(() => { setCloud(undefined); loadCloud(clip.id).then(setCloud); }, [clip.id]);
  const c = clip.ar!.coverage, tot = Math.max(1, c.voxels), g = c.green / tot, o = c.orange / tot, r = c.red / tot;
  const verdict = g > .55 && r < .2 ? VERDICT.good : g > .3 ? VERDICT.ok : VERDICT.recapture;
  return (
    <main className="review">
      <header className="cap-top"><button className="back mono" onClick={onHome}>← Captures</button><span className="mono dim">{capture.clips.length} part{capture.clips.length === 1 ? '' : 's'} · {fmtTime(capture.clips.reduce((a, k) => a + k.duration, 0))}</span></header>
      <div className="wide eyebrow">Review · AR scan</div>
      <h1>{verdict.t}</h1>
      <p className="lead">{verdict === VERDICT.good ? 'Most surfaces were seen from several angles. Ready to process.' : 'Some areas were only glimpsed (red). Another short scan over them will make the twin cleaner; scans are combined.'}</p>
      <Tabs capture={capture} sel={sel} setSel={setSel} />
      <div className="review-grid">
        <Glass className="player" opts={{ radius: 20, shader: false }}>
          {cloud ? <CloudView pts={cloud.pts} col={cloud.col} height={360} /> : <div className="player-na mono">{cloud === null ? 'Coverage not on this device' : 'Loading coverage…'}</div>}
          <div className="player-meta"><Legend /><span className="mono dim">{clip.ar!.keyframes} frames · {clip.ar!.camera} · {fmtBytes(clip.size)}</span></div>
        </Glass>
        <div className="side">
          <Glass className="stat-card" opts={{ radius: 20, shader: false }}>
            <div className="bars"><i style={{ flex: g, background: '#3DDC97' }} /><i style={{ flex: o, background: '#FFB84D' }} /><i style={{ flex: r, background: '#FF5A4E' }} /></div>
            <dl>
              <div><dt className="mono">Good</dt><dd>{Math.round(g * 100)}%</dd></div>
              <div><dt className="mono">Glimpsed</dt><dd>{Math.round(r * 100)}%</dd></div>
              <div><dt className="mono">Walked</dt><dd>{clip.ar!.walked} m</dd></div>
            </dl>
          </Glass>
        </div>
      </div>
      <div className="actions">
        <button className="btn ghost" onClick={() => onRetake(clip.id)}>Delete this scan</button>
        <button className="btn ghost" onClick={onMore}>Scan another part</button>
        <button className="btn big" onClick={onContinue}>{verdict === VERDICT.recapture ? 'Process anyway' : 'Continue'}</button>
      </div>
    </main>
  );
}

function VideoReview({ capture, clip, sel, setSel, onMore, onRetake, onContinue, onHome }: RP) {
  const [url, setUrl] = useState<string | null>(null); const [cur, setCur] = useState(0); const vid = useRef<HTMLVideoElement>(null);
  useEffect(() => { let u: string | null = null; readFile(clip.id, clip.mime).then(b => { if (b) { u = URL.createObjectURL(b); setUrl(u); } else setUrl(null); }); return () => { if (u) URL.revokeObjectURL(u); }; }, [clip.id]);
  const sum = useMemo(() => summarise(clip.quality), [clip]); const spans = useMemo(() => problemSpans(clip.quality), [clip]);
  const seek = (t: number) => { if (vid.current) { vid.current.currentTime = t; setCur(t); } };
  const all = useMemo(() => summarise(capture.clips.filter(k => !k.ar).flatMap(k => k.quality)), [capture]);

  return (
    <main className="review">
      <header className="cap-top"><button className="back mono" onClick={onHome}>← Captures</button><span className="mono dim">{capture.clips.length} clip{capture.clips.length === 1 ? '' : 's'} · {fmtTime(capture.clips.reduce((a, k) => a + k.duration, 0))}</span></header>
      <div className="wide eyebrow">Review</div>
      <h1>{VERDICT[sum.verdict].t}</h1>
      <p className="lead">{VERDICT[sum.verdict].d}</p>

      <Tabs capture={capture} sel={sel} setSel={setSel} />

      <div className="review-grid">
        <Glass className="player" opts={{ radius: 20, shader: false }}>
          {url ? <video ref={vid} src={url} controls playsInline onTimeUpdate={e => setCur((e.target as HTMLVideoElement).currentTime)} /> : <div className="player-na mono">Video not on this device</div>}
          <Timeline quality={clip.quality} duration={clip.duration} current={cur} onSeek={seek} />
          <div className="player-meta"><Legend /><span className="mono dim">{clip.width}×{clip.height} · {fmtBytes(clip.size)}{clip.lock?.exposure ? ` · ${clip.lock.exposure}` : ''}</span></div>
        </Glass>

        <div className="side">
          <Glass className="stat-card" opts={{ radius: 20, shader: false }}>
            <div className="bars" aria-label={`${Math.round(sum.good * 100)} percent good, ${Math.round(sum.ok * 100)} percent okay, ${Math.round(sum.bad * 100)} percent needs recapture`}>
              <i style={{ flex: sum.good, background: '#3DDC97' }} /><i style={{ flex: Math.max(0, sum.ok), background: '#FFB84D' }} /><i style={{ flex: sum.bad, background: '#FF5A4E' }} />
            </div>
            <dl>
              <div><dt className="mono">Good</dt><dd>{Math.round(sum.good * 100)}%</dd></div>
              <div><dt className="mono">Recapture</dt><dd>{Math.round(sum.bad * 100)}%</dd></div>
              <div><dt className="mono">Loop</dt><dd>{clip.source === 'upload' ? '—' : clip.loopClosed ? 'Closed ✓' : 'Open'}</dd></div>
            </dl>
          </Glass>
          {clip.path.length > 1 && (
            <Glass className="maps" opts={{ radius: 20, shader: false }}>
              <div><PathMap path={clip.path} quality={clip.quality} size={140} /><span className="mono">Path</span></div>
              <div><HeadingRing sensors={clip.sensors} quality={clip.quality} heading={null} size={140} /><span className="mono">Directions</span></div>
            </Glass>
          )}
          {spans.length > 0 && (
            <Glass className="issues" opts={{ radius: 20, shader: false }}>
              <h2 className="wide">Recapture these</h2>
              <ul>{spans.slice(0, 6).map((s, i) => <li key={i}><button onClick={() => seek(s.from)}><span className="mono">{fmtTime(s.from)}–{fmtTime(s.to)}</span>{TIP[s.issue]}</button></li>)}</ul>
            </Glass>
          )}
        </div>
      </div>

      <div className="actions">
        <button className="btn ghost" onClick={() => onRetake(clip.id)}>Delete this clip</button>
        <button className="btn ghost" onClick={onMore}>Record another part</button>
        <button className="btn big" onClick={onContinue} disabled={!capture.clips.length}>{all.verdict === 'recapture' ? 'Process anyway' : 'Continue'}</button>
      </div>
    </main>
  );
}
