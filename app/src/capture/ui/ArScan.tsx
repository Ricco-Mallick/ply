/* AR scan screen: an intro, then the phone's camera with the live coverage dots (drawn by WebXR) and this HUD as the
   DOM overlay, then a spin-around preview of what was covered before keeping the scan. */
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { ArCapture, GREEN, ORANGE, RED, type ArResult } from '../lib/ar';
import { newId, saveCloud, type Clip } from '../lib/store';
import { CloudView } from './CloudView';
import { fmtTime, fmtBytes } from './viz';

export function ArScan({ scanNo, mode = 'scan', onClip, onClose }: { scanNo: number; mode?: 'scan' | 'video'; onClip: (c: Clip) => void; onClose: () => void }) {
  const hud = useRef<HTMLDivElement>(null);
  const [cap, setCap] = useState<ArCapture | null>(null); const [phase, setPhase] = useState<'intro' | 'ar' | 'saving' | 'preview'>('intro');
  const [err, setErr] = useState<string | null>(null); const [res, setRes] = useState<{ r: ArResult; clip: Clip } | null>(null);
  const tick = useRef(0); useSyncExternalStore(cb => cap ? cap.on(() => { tick.current++; cb(); }) : () => {}, () => tick.current);

  const start = async () => {
    setErr(null); const id = newId(), c = new ArCapture(id, mode);
    try { await c.start(hud.current!); } catch (e) { setErr(`AR could not start: ${(e as Error).message}`); return; }
    setCap(c); setPhase('ar');
    c.result().then(async r => {
      setPhase('saving');
      if (r.keyframes.length < 8) { setErr('The scan ended before enough frames were saved. Start again and move slowly.'); setPhase('intro'); setCap(null); return; }
      await saveCloud(id, r.points, r.colors);
      const [w, h] = (r.camera ?? '0x0').split('x').map(Number);
      const clip: Clip = { id, name: `${mode === 'video' ? 'video' : 'scan'}-${String(scanNo).padStart(2, '0')}.tar`, source: 'ar', mime: 'application/x-tar', size: r.bytes, duration: r.seconds, width: w, height: h, createdAt: Date.now(),
        quality: [], sensors: [], path: r.path, ar: { keyframes: r.keyframes.length, coverage: r.coverage, camera: r.camera, walked: +r.path.reduce((s, p, i) => i ? s + Math.hypot(p.x - r.path[i - 1].x, p.z - r.path[i - 1].z) : 0, 0).toFixed(1) } };
      setRes({ r, clip }); setPhase('preview'); setCap(null);
    });
  };
  useEffect(() => () => { cap?.stop(); }, [cap]);

  const s = cap?.stats, total = s ? s.green + s.orange + s.red : 0, pct = (n: number) => (total ? n / total * 100 : 0);
  const tip = !s ? '' : s.tracking === 'starting' ? 'Move the phone slowly side to side so it can find the floor' : s.tracking === 'lost' ? 'Tracking lost: go back to where you were, slowly'
    : !cap?.recording ? 'Tracking. Tap record, then walk slowly around the room' : s.tooFast ? 'Slow down: frames blur when you move fast'
    : s.guide && s.guide.dist < 4 ? 'Head for the red patch and look at it from two sides' : pct(s.green) > 70 ? 'Looking good. Fill any red, then stop' : 'Keep walking slowly, look at every wall and the floor';

  return (
    <main className={`arscan ${phase}`}>
      {/* the DOM overlay: shown over the camera while the AR session runs */}
      <div ref={hud} className="ar-hud" hidden={phase !== 'ar'}>
        {s && <>
          <div className="ah-top">
            <span className={`ah-pill mono ${s.tracking}`}><i />{s.tracking === 'ok' ? (cap?.recording ? `REC ${fmtTime(s.seconds)}` : 'Tracking') : s.tracking === 'lost' ? 'Lost' : 'Finding floor'}</span>
            <span className="ah-pill mono">{s.keyframes} frames · {fmtBytes(s.bytes)}</span>
            <button className="ah-x" onClick={() => cap?.stop()} aria-label="End scan">×</button>
          </div>
          {s.thumb && cap?.recording && <figure className="ah-thumb"><img src={s.thumb} alt="Last saved frame" /><figcaption className="mono">#{s.keyframes}</figcaption></figure>}
          {s.guide && cap?.recording && s.guide.dist < 4 && <div className="ah-guide" style={{ transform: `rotate(${-s.guide.angle}rad)` }} aria-hidden="true">↑</div>}
          <div className="ah-bottom">
            <p className={`ah-tip ${s.tooFast ? 'warn' : ''}`}>{tip}</p>
            <div className="ah-cov" aria-label={`Coverage: ${Math.round(pct(s.green))}% good, ${Math.round(pct(s.orange))}% okay, ${Math.round(pct(s.red))}% seen in one frame only`}>
              <i style={{ width: `${pct(s.green)}%`, background: GREEN }} /><i style={{ width: `${pct(s.orange)}%`, background: ORANGE }} /><i style={{ width: `${pct(s.red)}%`, background: RED }} />
            </div>
            <div className="ah-legend mono"><span><b style={{ background: GREEN }} />good {Math.round(pct(s.green))}%</span><span><b style={{ background: ORANGE }} />2 frames {Math.round(pct(s.orange))}%</span><span><b style={{ background: RED }} />1 frame {Math.round(pct(s.red))}%</span></div>
            {cap?.recording
              ? <button className="ah-rec on" onClick={() => cap.stop()} aria-label="Stop scanning"><span /></button>
              : <button className="ah-rec" disabled={s.tracking !== 'ok'} onClick={() => { cap!.recording = true; }} aria-label="Start scanning"><span /></button>}
          </div>
        </>}
      </div>

      {phase === 'intro' && <section className="ar-intro">
        <header className="cap-top"><button className="back mono" onClick={onClose}>← Back</button><span className="mono dim">AR scan {scanNo}</span></header>
        <div className="wide eyebrow">{mode === 'video' ? 'Video with live coverage' : 'AR scan'}</div>
        <h1>{mode === 'video' ? <>Film the room,<br /><em>watch it fill in.</em></> : <>Paint the room<br /><em>with your phone.</em></>}</h1>
        <p className="lead">Everything the phone sees gets painted with dots: <b style={{ color: GREEN }}>green</b> is captured well, <b style={{ color: ORANGE }}>orange</b> needs another angle, <b style={{ color: RED }}>red</b> was seen in only one frame. Walk slowly until it's mostly green. The phone saves sharp frames with their exact positions, so the twin comes out at real scale with a level floor.</p>
        <ul className="ar-tips"><li>Start where you can see the floor</li><li>Walk the edge of the room, then a slow loop through the middle</li><li>Look at every wall from two sides</li></ul>
        {err && <p className="msg bad" role="alert">{err}</p>}
        {mode === 'video' && <p className="mono dim">Records 4 frames a second with their exact positions, more than the pipeline keeps from a normal video.</p>}
        <button className="btn big" onClick={start}>{mode === 'video' ? 'Start recording' : 'Start AR'}</button>
      </section>}

      {phase === 'saving' && <section className="ar-intro"><div className="wide eyebrow">Saving</div><h1>Packing your frames</h1><div className="bar"><i style={{ transform: 'scaleX(.6)' }} /></div></section>}

      {phase === 'preview' && res && <section className="ar-preview">
        <header className="cap-top"><span className="mono dim">Scan {scanNo} · {fmtTime(res.r.seconds)}</span></header>
        <div className="wide eyebrow">What you covered</div>
        <h1>{Math.round(res.r.coverage.green / Math.max(1, res.r.coverage.voxels) * 100)}% <em>captured well</em></h1>
        <CloudView pts={res.r.points} col={res.r.colors} keyframes={res.r.keyframes} height={Math.min(420, innerHeight * .48)} />
        <dl className="ar-stats mono">
          <div><dt>Frames</dt><dd>{res.r.keyframes.length}</dd></div><div><dt>Walked</dt><dd>{res.clip.ar!.walked} m</dd></div>
          <div><dt>Size</dt><dd>{fmtBytes(res.r.bytes)}</dd></div><div><dt>Camera</dt><dd>{res.r.camera}</dd></div>
        </dl>
        <p className="lead">{res.r.coverage.red / Math.max(1, res.r.coverage.voxels) > .2 ? 'There is a fair amount of red. Another short scan over those spots will help; scans of the same room are combined.' : 'Good coverage. Keep it, or add another scan for spots you missed.'}</p>
        <div className="actions">
          <button className="btn big" onClick={() => onClip(res.clip)}>Keep this scan</button>
        </div>
      </section>}
    </main>
  );
}
