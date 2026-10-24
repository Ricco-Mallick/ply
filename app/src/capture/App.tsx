// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The capture web app: start → permissions → (first time) tutorial → live → review → process.
   Uploaded videos skip the camera and go through the same analysis and review. */
import { useEffect, useState } from 'react';
import { analyseFile } from './lib/session';
import { deleteCapture, loadCapture, newId, saveCapture, storeUpload, type Capture, type Clip } from './lib/store';
import { Live } from './ui/Live';
import type { Span } from './ui/CoverageCheck';
import { ArScan } from './ui/ArScan';
import { Permissions, type Granted } from './ui/Permissions';
import { Review } from './ui/Review';
import { Start } from './ui/Start';
import { Submit } from './ui/Submit';
import { Status, rememberJob } from './ui/Status';
import { Tutorial } from './ui/Tutorial';

type Screen = 'start' | 'perms' | 'tutorial' | 'live' | 'ar' | 'arvideo' | 'review' | 'submit' | 'analysing' | 'status';
const SEEN = 'ply-tutorial-seen';
const seen = () => { try { return localStorage.getItem(SEEN) === '1'; } catch { return false; } };

export function App() {
  const [jobId, setJobId] = useState<string | null>(() => new URLSearchParams(location.search).get('job'));
  const [screen, setScreen] = useState<Screen>(() => (new URLSearchParams(location.search).get('job') ? 'status' : 'start'));
  const [granted, setGranted] = useState<Granted | null>(null);
  const [cap, setCap] = useState<Capture | null>(null);
  const [prog, setProg] = useState<{ name: string; p: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // the browser back button walks back through the flow instead of leaving the page
  useEffect(() => { history.replaceState({ screen }, ''); const f = (e: PopStateEvent) => setScreen((e.state?.screen as Screen) ?? 'start'); addEventListener('popstate', f); return () => removeEventListener('popstate', f); }, []);
  const go = (s: Screen) => { setScreen(s); history.pushState({ screen: s }, ''); scrollTo(0, 0); };
  useEffect(() => { if (screen !== 'live' && screen !== 'tutorial' && granted) { granted.stream.getTracks().forEach(t => t.stop()); setGranted(null); } }, [screen]);

  const ensureCapture = () => cap ?? { id: newId(), name: `Capture ${new Date().toLocaleDateString(undefined, { day: 'numeric', month: 'short' })}`, createdAt: Date.now(), clips: [] };
  const update = (c: Capture) => { setCap(c); saveCapture(c); };

  const record = () => { setCap(ensureCapture()); go('perms'); };
  const scan = () => { setCap(ensureCapture()); go('ar'); };
  const onGranted = (g: Granted) => { setGranted(g); go(seen() ? 'live' : 'tutorial'); };
  const onClip = (clip: Clip) => { const c = ensureCapture(); update({ ...c, clips: [...c.clips, clip] }); go('review'); };

  const onUpload = async (files: File[]) => {
    const c = ensureCapture(); setErr(null); go('analysing'); const clips: Clip[] = [];
    for (const [i, f] of files.entries()) {
      try {
        setProg({ name: f.name, p: 0 }); const id = newId(); await storeUpload(id, f);
        const meta = await analyseFile(f, p => setProg({ name: f.name, p }));
        clips.push({ ...meta, id, name: `clip-${String(c.clips.length + i + 1).padStart(2, '0')}-${f.name.replace(/[^\w.-]+/g, '_')}`, source: 'upload' });
      } catch (e) { setErr(`${f.name}: ${(e as Error).message}`); }
    }
    setProg(null); if (!clips.length) { go('start'); return; }
    update({ ...c, clips: [...c.clips, ...clips] }); go('review');
  };
  // coverage check: remember the job and what it covered; replacing a lost stretch marks it skipped and records a new clip
  const sig = (c: Capture) => c.clips.map(k => `${k.id}:${(k.exclude ?? []).length}`);
  const onPreview = (jobId: string) => { if (cap) update({ ...cap, preview: { jobId, clips: sig(cap), at: Date.now() } }); };
  const onReplace = (s: Span) => {
    if (!cap) return; const clips = cap.clips.map(k => k.id === s.clipId ? { ...k, exclude: [...(k.exclude ?? []), [+s.from.toFixed(2), +s.to.toFixed(2)] as [number, number]] } : k);
    update({ ...cap, clips }); const k = clips.find(x => x.id === s.clipId); go(k?.ar ? (k.name.startsWith('video') ? 'arvideo' : 'ar') : 'perms');
  };
  const open = async (id: string) => { const c = await loadCapture(id); if (c) { setCap(c); go('review'); } else setErr('That capture is no longer on this device.'); };
  const retake = async (clipId: string) => {
    if (!cap) return; const clips = cap.clips.filter(k => k.id !== clipId);
    if (!clips.length) { await deleteCapture(cap); setCap(null); go('start'); return; }
    await deleteCapture({ ...cap, clips: cap.clips.filter(k => k.id === clipId) }).catch(() => {}); update({ ...cap, clips });
  };

  return (
    <>
      {err && <div className="toast" role="alert" onClick={() => setErr(null)}>{err}</div>}
      {screen === 'start' && <Start onRecord={record} onScan={scan} onUpload={onUpload} onOpen={open} onJob={id => { setJobId(id); history.pushState({ screen: 'status' }, '', `?job=${id}`); setScreen('status'); }} />}
      {screen === 'perms' && <Permissions onReady={onGranted} onBack={() => history.back()} />}
      {screen === 'tutorial' && <Tutorial onDone={() => { try { localStorage.setItem(SEEN, '1'); } catch { /* ignore */ } setScreen('live'); history.replaceState({ screen: 'live' }, ''); }} />}
      {screen === 'live' && (granted ? <Live granted={granted} clipNo={(cap?.clips.length ?? 0) + 1} onClip={onClip} onClose={() => go(cap?.clips.length ? 'review' : 'start')} /> : <Relaunch onGo={() => go('perms')} />)}
      {screen === 'ar' && <ArScan scanNo={(cap?.clips.length ?? 0) + 1} onClip={onClip} onClose={() => go(cap?.clips.length ? 'review' : 'start')} />}
      {screen === 'arvideo' && <ArScan key="video" mode="video" scanNo={(cap?.clips.length ?? 0) + 1} onClip={onClip} onClose={() => go(cap?.clips.length ? 'review' : 'start')} />}
      {screen === 'review' && cap && <Review capture={cap} onMore={() => go(cap.clips.at(-1)?.ar ? (cap.clips.at(-1)!.name.startsWith('video') ? 'arvideo' : 'ar') : 'perms')} onRetake={retake} onContinue={() => go('submit')} onHome={() => { setCap(null); go('start'); }} onPreview={onPreview} onReplace={onReplace} />}
      {screen === 'submit' && cap && <Submit capture={cap} onRename={n => update({ ...cap, name: n })} onBack={() => history.back()} onQueued={id => { rememberJob({ id, name: cap.name, createdAt: Date.now() }); setJobId(id); history.replaceState({ screen: 'status' }, '', `?job=${id}`); setScreen('status'); }} />}
      {screen === 'status' && jobId && <Status jobId={jobId} onHome={() => { history.replaceState({ screen: 'start' }, '', location.pathname); setScreen('start'); }} />}
      {screen === 'analysing' && (
        <main className="analysing" aria-live="polite">
          <div className="wide eyebrow">Checking your videos</div>
          <h1>{prog?.name ?? 'Reading'}</h1>
          <div className="bar"><i style={{ transform: `scaleX(${prog?.p ?? 0})` }} /></div>
          <p className="mono dim">Sharpness, light and steadiness, every half second · stays on this device</p>
        </main>
      )}
    </>
  );
}

/** Coming back to the live screen via history: the camera was released, so ask for it again. */
function Relaunch({ onGo }: { onGo: () => void }) { useEffect(onGo, []); return null; }
