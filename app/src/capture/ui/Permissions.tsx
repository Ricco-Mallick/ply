import { useState } from 'react';
import { Glass } from '../../shared/Glass';
import { openCamera, preferredLens, type CamInfo } from '../lib/camera';
import { requestMotionPermission } from '../lib/sensors';
import { persist, storageLeft } from '../lib/store';
import { fmtBytes } from './viz';

type St = 'idle' | 'asking' | 'ok' | 'denied' | 'unavailable' | 'auto';
export interface Granted { stream: MediaStream; track: MediaStreamTrack; info: CamInfo; motion: boolean; }

const FIX: Record<string, string> = {
  NotAllowedError: 'Camera access was blocked. Tap the icon left of the address bar → Permissions → Camera → Allow, then try again.',
  NotFoundError: 'No camera was found on this device.',
  NotReadableError: 'Another app is using the camera. Close it and try again.',
  InsecureContext: 'The camera only works on a secure (https) page. Open the https link.',
  NoMediaDevices: 'This browser cannot open the camera. Use Chrome on Android or Safari on iPhone.',
};

/** Everything is asked for from one tap, because iOS only shows the motion prompt inside a user gesture. */
export function Permissions({ onReady, onBack }: { onReady: (g: Granted) => void; onBack: () => void }) {
  const [cam, setCam] = useState<St>('idle'); const [mot, setMot] = useState<St>('idle'); const [sto, setSto] = useState<St>('idle');
  const [err, setErr] = useState<string | null>(null); const [left, setLeft] = useState<number | null>(null);

  const ask = async () => {
    setErr(null); setMot('asking'); setCam('asking'); setSto('asking');
    const m = await requestMotionPermission();
    let motionOk = m !== 'denied';
    if (m === 'not-needed') motionOk = await new Promise<boolean>(r => { const t = setTimeout(() => { removeEventListener('devicemotion', h); r(false); }, 900); const h = () => { clearTimeout(t); removeEventListener('devicemotion', h); r(true); }; addEventListener('devicemotion', h); });
    setMot(motionOk ? 'ok' : m === 'denied' ? 'denied' : 'unavailable');
    const p = await persist(); setSto(p ? 'ok' : 'auto'); setLeft(await storageLeft());
    try {
      let g = await openCamera(); setCam('ok');
      // default to the main sensor (some phones hand out the ultra-wide for "environment")
      const better = await preferredLens(g.info.deviceId);
      if (better) { g.stream.getTracks().forEach(t => t.stop()); g = await openCamera(better); }
      setTimeout(() => onReady({ ...g, motion: motionOk }), 450);
    } catch (e) { const n = (e as Error).name; setCam('denied'); setErr(FIX[n] ?? `The camera could not start (${(e as Error).message}).`); }
  };

  const row = (title: string, why: string, st: St, need: string) => (
    <li className={`perm ${st}`}>
      <span className="perm-dot" aria-hidden="true" />
      <div><div className="perm-t">{title} <span className="mono perm-need">{need}</span></div><div className="perm-d">{why}</div></div>
      <span className="mono perm-s">{{ idle: '', asking: '…', ok: 'Allowed', denied: 'Blocked', unavailable: 'Not on this device', auto: 'Best effort' }[st]}</span>
    </li>
  );
  return (
    <main className="perms">
      <button className="back mono" onClick={onBack}>← Back</button>
      <div className="wide eyebrow">Before you record</div>
      <h1>Ply needs a few things</h1>
      <Glass className="perm-card" opts={{ radius: 22 }}>
        <ul>
          {row('Camera', 'The rear camera records the walkthrough. Nothing leaves the phone until you choose to process.', cam, 'Required')}
          {row('Motion & orientation', 'Gyro and compass tell Ply how fast you turn, how you hold the phone, and where you have walked.', mot, 'Recommended')}
          {row('Storage', `Keeps long recordings safe on the phone if the page reloads.${left ? ` ${fmtBytes(left)} free.` : ''}`, sto, 'Automatic')}
          {row('Screen stays on', 'The screen will not dim while you record.', cam === 'ok' ? 'auto' : 'idle', 'Automatic')}
        </ul>
      </Glass>
      {err && <p className="perm-err" role="alert">{err}</p>}
      <button className="btn big" onClick={ask} disabled={cam === 'asking'}>{cam === 'denied' ? 'Try again' : 'Allow and continue'}</button>
    </main>
  );
}
