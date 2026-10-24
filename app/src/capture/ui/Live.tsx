// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Glass } from '../../shared/Glass';
import { controls, listLenses, lockCamera, meterAt, openCamera, saveLens, setEv, setZoom, unlock, type CamControls, type CamInfo, type Lens, type LockReport } from '../lib/camera';
import { Session } from '../lib/session';
import type { Clip } from '../lib/store';
import type { Granted } from './Permissions';
import { GRADE_COL, HeadingRing, PathMap, fmtBytes, fmtTime } from './viz';

type Sheet = null | 'lens' | 'exposure';

export function Live({ granted, clipNo, onClip, onClose }: { granted: Granted; clipNo: number; onClip: (c: Clip) => void; onClose: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [cam, setCam] = useState<{ track: MediaStreamTrack; info: CamInfo }>({ track: granted.track, info: granted.info });
  const [ctl, setCtl] = useState<CamControls>(() => controls(granted.track));
  const [lenses, setLenses] = useState<Lens[]>([]); const [zoom, setZ] = useState(1); const [ev, setE] = useState(0);
  const [lock, setLock] = useState<LockReport | null>(null); const [busy, setBusy] = useState<string | null>(null);
  const [sheet, setSheet] = useState<Sheet>(null); const [ring, setRing] = useState<{ x: number; y: number; k: number } | null>(null);
  const [stopping, setStopping] = useState(false); const [loopToast, setLoopToast] = useState(false); const [note, setNote] = useState<string | null>(null);
  const tick = useRef(0);
  useSyncExternalStore(cb => session ? session.on(() => { tick.current++; cb(); }) : () => {}, () => tick.current);

  useEffect(() => {
    const v = video.current!; v.srcObject = granted.stream; v.play().catch(() => {});
    const s = new Session(v, granted.stream, granted.info, null); setSession(s);
    listLenses().then(setLenses).catch(() => {});
    return () => s.dispose();
  }, []);

  const L = session?.live, rec = !!L?.recording, q = L?.q, sens = session?.sensors;
  useEffect(() => { if (L?.loop && rec) { setLoopToast(true); navigator.vibrate?.([40, 40, 40]); const t = setTimeout(() => setLoopToast(false), 3500); return () => clearTimeout(t); } }, [L?.loop]);
  useEffect(() => { if (note) { const t = setTimeout(() => setNote(null), 3600); return () => clearTimeout(t); } }, [note]);

  const toggle = async () => {
    if (!session || stopping || busy) return;
    if (!rec) { setSheet(null); await session.start(); return; }
    setStopping(true); const clip = await session.stop(`clip-${String(clipNo).padStart(2, '0')}`); setStopping(false); onClip(clip);
  };

  const switchLens = async (l: Lens) => {
    if (!session || rec || l.deviceId === cam.info.deviceId) return;
    setBusy('Switching lens…'); session.stream.getTracks().forEach(t => t.stop());
    try {
      const g = await openCamera(l.deviceId); const v = video.current!; v.srcObject = g.stream; await v.play().catch(() => {});
      session.stream = g.stream; session.info = g.info; session.lock = null; session.analyzer.reset();
      setCam({ track: g.track, info: g.info }); setCtl(controls(g.track)); setZ(1); setE(0); setLock(null); saveLens(l.deviceId);
    } catch (e) { setNote(`That lens could not open (${(e as Error).message})`); }
    setBusy(null);
  };
  const doLock = async () => {
    if (!session) return; setBusy('Metering…');
    const r = await lockCamera(cam.track, () => session.live.q?.lumaMed ?? null, ring ? { x: ring.x, y: ring.y } : undefined);
    if (r.note) { setLock(null); setNote(r.note); } else { session.lock = r; setLock(r); }
    setBusy(null);
  };
  const doUnlock = async () => { if (!session) return; await unlock(cam.track); session.lock = null; setLock(null); };
  const tapMeter = async (e: React.PointerEvent) => {
    if (sheet) { setSheet(null); return; }
    const r = e.currentTarget.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    setRing({ x, y, k: Date.now() }); if (!lock) await meterAt(cam.track, x, y);
  };

  const locked = !!lock && !lock.unsupported.includes('exposure');
  const lensName = lenses.find(l => l.deviceId === cam.info.deviceId)?.name ?? 'Lens';
  const pitch = sens?.available ? sens.pitch : null, roll = sens?.available ? sens.roll : 0;
  const pitchOk = pitch != null && pitch > 5 && pitch < 25, grade = q?.grade ?? 0;
  const res = cam.info.width ? `${Math.max(cam.info.width, cam.info.height) >= 3000 ? '4K' : `${Math.min(cam.info.width, cam.info.height)}p`} · ${cam.info.fps} fps` : '';

  return (
    <main className={`live ${rec ? 'rec' : ''}`}>
      <video ref={video} className="preview" playsInline muted autoPlay aria-label="Camera preview" />
      <div className="tapzone" onPointerDown={tapMeter} aria-hidden="true" />
      {ring && <span className={`focus-ring ${locked ? 'locked' : ''}`} key={ring.k} style={{ left: `${ring.x * 100}%`, top: `${ring.y * 100}%` }} aria-hidden="true" />}
      <div className={`edge g${grade}`} aria-hidden="true" />

      <header className="live-top">
        <Glass as="button" className="icon pill" opts={{ shader: false, press: true }} onClick={onClose} aria-label="Close camera" disabled={rec}>✕</Glass>
        <Glass className="pill rec-chip" opts={{ shader: false }}>
          <span className={`rec-dot ${rec ? 'on' : ''}`} aria-hidden="true" />
          <span className="mono">{rec ? fmtTime((L?.elapsed ?? 0) / 1000) : `Clip ${clipNo}`}</span>
          {rec ? <span className="mono dim">{fmtBytes(L?.bytes ?? 0)}{L?.storage === 'memory' ? ' · RAM' : ''}</span> : <span className="mono dim">{res}</span>}
        </Glass>
      </header>

      <div className="tip-wrap" aria-live="polite">
        {busy ? <Glass className="tip pill" opts={{ shader: false }}>{busy}</Glass>
          : note ? <Glass className="tip pill g1" opts={{ shader: false }}>{note}</Glass>
          : L?.tip ? <Glass className={`tip pill g${grade}`} opts={{ shader: false }} key={L.tip}>{L.tip}</Glass>
          : rec ? <Glass className="tip pill g0" opts={{ shader: false }}>Looking good, keep going</Glass>
          : <Glass className="tip pill" opts={{ shader: false }}>Tap the screen to focus · press record when ready</Glass>}
        {loopToast && <Glass className="tip pill g0 loop" opts={{ shader: false }}>Loop closed ✓ You're back at the start</Glass>}
      </div>

      {pitch != null && (
        <div className="level" aria-hidden="true">
          <div className="horizon" style={{ transform: `rotate(${-roll}deg)` }}><i /><b /><i /></div>
          <div className="pitch"><span className="band" /><span className={`mark ${pitchOk ? 'ok' : ''}`} style={{ top: `${50 + Math.max(-45, Math.min(45, pitch - 15)) * 1.1}%` }} /></div>
        </div>
      )}

      {sheet && (
        <Glass className="sheet" opts={{ shader: false, radius: 22 }} role="dialog" aria-label={sheet === 'lens' ? 'Choose lens' : 'Exposure'}>
          {sheet === 'lens' ? (
            <>
              <div className="sheet-h"><span className="wide">Lens</span><button className="mono" onClick={() => setSheet(null)}>Done</button></div>
              <p className="sheet-p">Use the <b>Main</b> lens: it is the sharpest and has the least distortion. Ultra-wide bends straight walls, which the 3D solver dislikes.</p>
              <div className="lens-list">{lenses.map(l => <button key={l.deviceId} className={l.deviceId === cam.info.deviceId ? 'on' : ''} disabled={rec || !!busy} onClick={() => switchLens(l)}><b>{l.name}{l.name === 'Main' ? ' · recommended' : ''}</b><span className="mono">{l.label || 'camera'}</span></button>)}</div>
              {lenses.length <= 1 && <p className="sheet-p dim">This browser shows one rear camera. Phones that combine their lenses behind one camera switch with the zoom below.</p>}
              {ctl.zoom && ctl.zoom.min < ctl.zoom.max && (
                <div className="zooms">{[ctl.zoom.min < 1 ? +ctl.zoom.min.toFixed(1) : null, 1, ctl.zoom.max >= 2 ? 2 : null, ctl.zoom.max >= 3 ? 3 : null].filter((z): z is number => z !== null).map(z => <button key={z} className={Math.abs(zoom - z) < .05 ? 'on' : ''} disabled={rec} onClick={async () => { if (await setZoom(cam.track, z)) setZ(z); }}>{z}×</button>)}</div>
              )}
            </>
          ) : (
            <>
              <div className="sheet-h"><span className="wide">Exposure</span><button className="mono" onClick={() => setSheet(null)}>Done</button></div>
              <div className="seg2" role="radiogroup" aria-label="Exposure mode">
                <button role="radio" aria-checked={!locked} className={!locked ? 'on' : ''} disabled={rec || !!busy} onClick={doUnlock}>Auto</button>
                <button role="radio" aria-checked={locked} className={locked ? 'on' : ''} disabled={rec || !!busy || !ctl.canLock} onClick={doLock}>{locked ? 'Locked' : 'Lock'}</button>
              </div>
              <p className="sheet-p">{locked ? <>Held at <b>{lock!.exposure}</b>{lock!.focus ? ` · focus ${lock!.focus}` : ''}{lock!.whiteBalance ? ` · WB ${lock!.whiteBalance}` : ''}. Tap Auto to release.</>
                : ctl.canLock ? <><b>Auto is fine for most rooms</b>: Ply evens out brightness changes between frames. To lock, first tap a normally lit spot (a wall, not a lamp or window), then tap Lock.</>
                : <>This phone's browser does not allow locking exposure, so Auto it is. Ply evens out brightness changes during processing.</>}</p>
              {ctl.ev && (
                <label className="ev"><span className="mono">Brightness {ev > 0 ? '+' : ''}{ev.toFixed(1)} EV</span>
                  <input type="range" min={ctl.ev.min} max={ctl.ev.max} step={ctl.ev.step} value={ev} disabled={rec} onChange={async e => { const v = +e.target.value; setE(v); await setEv(cam.track, v); }} /></label>
              )}
              {q && <p className="mono dim meter">Scene brightness {Math.round(q.lumaMed / 2.55)}% · blown highlights {Math.round(q.clipped * 100)}%</p>}
            </>
          )}
        </Glass>
      )}

      <footer className="live-bottom">
        <div className="strip" aria-hidden="true">{(L?.grades ?? []).map((g, i) => <i key={i} style={{ background: GRADE_COL[g] }} />)}</div>
        <div className="quick">
          <Glass as="button" className="pill qbtn" opts={{ shader: false, press: true }} onClick={() => setSheet(sheet === 'lens' ? null : 'lens')} disabled={rec} aria-expanded={sheet === 'lens'}>{lensName}{zoom !== 1 ? ` ${zoom}×` : ''}</Glass>
          <Glass as="button" className={`pill qbtn ${locked ? 'hot' : ''}`} opts={{ shader: false, press: true }} onClick={() => setSheet(sheet === 'exposure' ? null : 'exposure')} disabled={rec} aria-expanded={sheet === 'exposure'}>{locked ? 'AE locked' : 'Auto'}{ev ? ` ${ev > 0 ? '+' : ''}${ev.toFixed(1)}` : ''}</Glass>
        </div>
        <div className="live-row">
          <Glass className="mini" opts={{ shader: false, radius: 18 }}>
            {sens?.available ? <PathMap path={sens.path} quality={session!.quality} size={92} live label={false} /> : <div className="mini-na mono">No motion sensors</div>}
            <span className="mono mini-l">{L?.steps ?? 0} steps{L?.loop ? ' · loop ✓' : ''}</span>
          </Glass>
          <button className={`shutter ${rec ? 'on' : ''}`} onClick={toggle} disabled={!session || stopping || !!busy} aria-label={rec ? 'Stop recording' : 'Start recording'}><span /></button>
          <Glass className="mini" opts={{ shader: false, radius: 18 }}>
            {sens?.available && sens.heading != null ? <HeadingRing sensors={session!.log} quality={session!.quality} heading={sens.heading} size={92} /> : <div className="mini-na mono">{q ? `Sharp ${Math.min(99, Math.round((q.sharpRel ?? 1) * 50))}` : '—'}</div>}
            <span className="mono mini-l">{q ? `Filmed · light ${Math.round(q.lumaMed / 2.55)}%` : ''}</span>
          </Glass>
        </div>
      </footer>
    </main>
  );
}
