import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Glass } from '../../shared/Glass';
import { lockCamera, type LockReport } from '../lib/camera';
import { Session } from '../lib/session';
import type { Clip } from '../lib/store';
import type { Granted } from './Permissions';
import { GRADE_COL, HeadingRing, PathMap, fmtBytes, fmtTime } from './viz';

export function Live({ granted, clipNo, onClip, onClose }: { granted: Granted; clipNo: number; onClip: (c: Clip) => void; onClose: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [lock, setLock] = useState<LockReport | null>(null); const [locking, setLocking] = useState(true);
  const [stopping, setStopping] = useState(false); const [loopToast, setLoopToast] = useState(false);
  const tick = useRef(0);
  useSyncExternalStore(cb => session ? session.on(() => { tick.current++; cb(); }) : () => {}, () => tick.current);

  useEffect(() => {
    const v = video.current!; v.srcObject = granted.stream; v.play().catch(() => {});
    const s = new Session(v, granted.stream, granted.info, null); setSession(s);
    // meter in auto for a moment, then hold exposure / focus / white balance for the whole take
    const t = setTimeout(async () => { const r = await lockCamera(granted.track); s.lock = r; setLock(r); setLocking(false); }, 1600);
    return () => { clearTimeout(t); s.dispose(); };
  }, []);

  const L = session?.live, rec = !!L?.recording, q = L?.q, sens = session?.sensors;
  useEffect(() => { if (L?.loop && rec) { setLoopToast(true); navigator.vibrate?.([40, 40, 40]); const t = setTimeout(() => setLoopToast(false), 3500); return () => clearTimeout(t); } }, [L?.loop]);

  const toggle = async () => {
    if (!session || stopping) return;
    if (!rec) { await session.start(); return; }
    setStopping(true); const clip = await session.stop(`clip-${String(clipNo).padStart(2, '0')}`); setStopping(false); onClip(clip);
  };
  const relock = async () => { if (!session) return; setLocking(true); try { await granted.track.applyConstraints({ advanced: [{ exposureMode: 'continuous', focusMode: 'continuous', whiteBalanceMode: 'continuous' } as any] }); } catch { /* fine */ } setTimeout(async () => { const r = await lockCamera(granted.track); session.lock = r; setLock(r); setLocking(false); }, 1400); };

  const lockText = locking ? 'Metering light…' : lock && !lock.unsupported.includes('exposure') ? `Locked · ${[lock.exposure, lock.focus && `focus ${lock.focus}`, lock.whiteBalance && `WB ${lock.whiteBalance}`].filter(Boolean).join(' · ')}` : 'Auto exposure · this phone cannot lock it';
  const pitch = sens?.available ? sens.pitch : null, roll = sens?.available ? sens.roll : 0;
  const pitchOk = pitch != null && pitch > 5 && pitch < 25;
  const grade = q?.grade ?? 0;

  return (
    <main className={`live ${rec ? 'rec' : ''}`}>
      <video ref={video} className="preview" playsInline muted autoPlay aria-label="Camera preview" />
      <div className={`edge g${grade}`} aria-hidden="true" />

      <header className="live-top">
        <Glass as="button" className="icon pill" opts={{ shader: false, press: true }} onClick={onClose} aria-label="Close camera" disabled={rec}>✕</Glass>
        <Glass className="pill rec-chip" opts={{ shader: false }}>
          <span className={`rec-dot ${rec ? 'on' : ''}`} aria-hidden="true" />
          <span className="mono">{rec ? fmtTime((L?.elapsed ?? 0) / 1000) : `Clip ${clipNo}`}</span>
          {rec && <span className="mono dim">{fmtBytes(L?.bytes ?? 0)}{L?.storage === 'memory' ? ' · RAM' : ''}</span>}
        </Glass>
        <Glass as="button" className="pill lock-chip mono" opts={{ shader: false, press: true }} onClick={relock} disabled={rec || locking} aria-label={`${lockText}. Tap to meter again.`}>{lockText}</Glass>
      </header>

      <div className="tip-wrap" aria-live="polite">
        {L?.tip ? <Glass className={`tip pill g${grade}`} opts={{ shader: false }} key={L.tip}>{L.tip}</Glass>
          : rec ? <Glass className="tip pill g0" opts={{ shader: false }}>Looking good, keep going</Glass>
          : <Glass className="tip pill" opts={{ shader: false }}>{granted.info.width ? `${Math.max(granted.info.width, granted.info.height) >= 3000 ? '4K' : `${Math.min(granted.info.width, granted.info.height)}p`} · ${granted.info.fps} fps · ready` : 'Starting camera…'}</Glass>}
        {loopToast && <Glass className="tip pill g0 loop" opts={{ shader: false }}>Loop closed ✓ You're back at the start</Glass>}
      </div>

      {pitch != null && (
        <div className="level" aria-hidden="true">
          <div className="horizon" style={{ transform: `rotate(${-roll}deg)` }}><i /><b /><i /></div>
          <div className="pitch"><span className="band" /><span className={`mark ${pitchOk ? 'ok' : ''}`} style={{ top: `${50 + Math.max(-45, Math.min(45, pitch - 15)) * 1.1}%` }} /></div>
        </div>
      )}

      <footer className="live-bottom">
        <div className="strip" aria-hidden="true">{(L?.grades ?? []).map((g, i) => <i key={i} style={{ background: GRADE_COL[g] }} />)}</div>
        <div className="live-row">
          <Glass className="mini" opts={{ shader: false, radius: 18 }}>
            {sens?.available ? <PathMap path={sens.path} quality={session!.quality} size={92} live label={false} /> : <div className="mini-na mono">No motion sensors</div>}
            <span className="mono mini-l">{L?.steps ?? 0} steps{L?.loop ? ' · loop ✓' : ''}</span>
          </Glass>
          <button className={`shutter ${rec ? 'on' : ''}`} onClick={toggle} disabled={!session || stopping || locking} aria-label={rec ? 'Stop recording' : 'Start recording'}><span /></button>
          <Glass className="mini" opts={{ shader: false, radius: 18 }}>
            {sens?.available && sens.heading != null ? <HeadingRing sensors={session!.log} quality={session!.quality} heading={sens.heading} size={92} /> : <div className="mini-na mono">{q ? `Sharp ${Math.min(99, Math.round((q.sharpRel ?? 1) * 50))}` : '—'}</div>}
            <span className="mono mini-l">{q ? `Filmed · light ${Math.round(q.luma / 2.55)}%` : ''}</span>
          </Glass>
        </div>
      </footer>
    </main>
  );
}
