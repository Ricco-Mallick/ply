// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
import { useRef, useState } from 'react';
import { Glass } from '../../shared/Glass';
import { DEFAULT_SETTINGS, PRESETS, backendReady, pipelineArgs, saveToDevice, setTeamKey, teamKey, upload, type Settings } from '../lib/backend';
import type { Capture } from '../lib/store';
import { fmtBytes, fmtTime } from './viz';

export function Submit({ capture, onRename, onBack, onQueued }: { capture: Capture; onRename: (n: string) => void; onBack: () => void; onQueued: (jobId: string) => void }) {
  const [key, setKey] = useState(teamKey());
  const [s, setS] = useState<Settings>(DEFAULT_SETTINGS);
  const [adv, setAdv] = useState(false); const [busy, setBusy] = useState<string | null>(null); const [pct, setPct] = useState(0);
  const [msg, setMsg] = useState<{ ok: boolean; t: string } | null>(null); const [copied, setCopied] = useState(false); const abort = useRef<AbortController | null>(null);
  const p = PRESETS[s.preset], a = s.advanced, size = capture.clips.reduce((x, k) => x + k.size, 0);
  const setA = (k: keyof Settings['advanced'], v: any) => setS(o => ({ ...o, advanced: { ...o.advanced, [k]: v } }));
  const input = `captures/${slug(capture.name)}/`; // the clips + capture.json, together in one folder
  const cmd = `python -m plypipe run ${input} --out work/${slug(capture.name)} ${pipelineArgs(s).join(' ')}`;

  const go = async () => {
    setMsg(null);
    if (!backendReady) { setBusy('Preparing files'); const r = await saveToDevice(capture, s).catch(e => (setMsg({ ok: false, t: e.message }), null)); setBusy(null); if (r && r !== 'cancelled') setMsg({ ok: true, t: r === 'shared' ? 'Shared. Put the files in one folder and run the command below.' : 'Saved to Downloads. Put the files in one folder and run the command below.' }); return; }
    abort.current = new AbortController(); setBusy('Uploading');
    setTeamKey(key);
    try { const j = await upload(capture, s, (d, t, l) => { setPct(d / t); setBusy(`Uploading ${l}`); }, abort.current.signal); setBusy(null); onQueued(j.jobId); return; }
    catch (e) { const m = (e as Error).message; setMsg({ ok: false, t: (e as Error).name === 'AbortError' ? 'Upload paused. Tap again to resume where it stopped.' : /team key/i.test(m) ? `${m}. Check the Team key field above and tap again.` : `${m}. Tap again to resume.` }); }
    setBusy(null);
  };

  return (
    <main className="submit">
      <header className="cap-top"><button className="back mono" onClick={onBack}>← Review</button><span className="mono dim">{capture.clips.length} clip{capture.clips.length === 1 ? '' : 's'} · {fmtTime(capture.clips.reduce((x, k) => x + k.duration, 0))} · {fmtBytes(size)}</span></header>
      <div className="wide eyebrow">Process</div>
      <h1>Make the twin</h1>

      <label className="field"><span className="mono">Name this space</span><input value={capture.name} onChange={e => onRename(e.target.value)} placeholder="LHC · Floor 1" maxLength={60} /></label>

      <div className="presets" role="radiogroup" aria-label="Quality preset">
        {(Object.keys(PRESETS) as Settings['preset'][]).map(k => (
          <Glass as="button" key={k} role="radio" aria-checked={s.preset === k} className={`preset ${s.preset === k ? 'on' : ''}`} opts={{ shader: false, press: true, radius: 18 }} onClick={() => setS(o => ({ ...o, preset: k, advanced: { ...o.advanced, iterations: undefined, shDegree: undefined, maxSplats: undefined } }))}>
            <span className="preset-t">{PRESETS[k].label}</span><span className="preset-d">{PRESETS[k].note}</span>
            <span className="mono preset-m">{PRESETS[k].multiplier} · SH{PRESETS[k].shDegree} · sized from your capture</span>
          </Glass>
        ))}
      </div>

      <button className="adv-toggle mono" aria-expanded={adv} onClick={() => setAdv(v => !v)}>{adv ? '−' : '+'} Advanced settings</button>
      {adv && (
        <Glass className="adv" opts={{ shader: false, radius: 18 }}>
          <Num label="Training steps" v={a.iterations ?? p.iterations} min={1000} max={60000} step={1000} on={v => setA('iterations', v)} hint="Auto by default. More steps, finer detail, longer wait." />
          <Num label="Max splats" v={a.maxSplats ?? p.maxSplats} min={200000} max={6000000} step={100000} on={v => setA('maxSplats', v)} hint="Auto by default, from the room's surface area. Phones load ~1M comfortably." />
          <Sel label="Colour detail (SH degree)" v={String(a.shDegree ?? p.shDegree)} opts={[['0', '0 · flat colour'], ['1', '1'], ['2', '2'], ['3', '3 · view-dependent shine']]} on={v => setA('shDegree', +v)} />
          <Sel label="Camera solver" v={a.mapper} opts={[['global', 'Global (fast, default)'], ['incremental', 'Incremental (if global folds the path)']]} on={v => setA('mapper', v)} />
          <Sel label="Feature matching" v={a.features} opts={[['aliked', 'ALIKED + LightGlue (GPU)'], ['sift', 'SIFT (classic, CPU-friendly)']]} on={v => setA('features', v)} />
          <Num label="Frames per second kept" v={a.fps ?? 2.5} min={1} max={6} step={.5} on={v => setA('fps', v)} hint="Auto by default, from how fast you walked. Raise it for a sprint or a tight space." />
          <Num label="Phone height (m)" v={a.cameraHeight} min={.8} max={2} step={.05} on={v => setA('cameraHeight', v)} hint="Sets real-world scale. Chest height is about 1.45 m." />
        </Glass>
      )}

      {backendReady && <label className="field"><span className="mono">Team key</span><input value={key} onChange={e => setKey(e.target.value)} placeholder="From the Ply team" autoComplete="off" spellCheck={false} style={{ fontFamily: 'var(--font-mono)', fontSize: 16 }} /></label>}
      {!backendReady && <p className="note"><b>No processing server is connected yet.</b> Ply will save the videos and a capture file (quality and sensor log) to this phone; run them through the pipeline with the command below.</p>}
      <div className="cmd"><code className="mono">{cmd}</code><button className="mono" onClick={() => navigator.clipboard?.writeText(cmd).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1600); })}>{copied ? 'Copied' : 'Copy'}</button></div>

      {busy && <div className="upbar" role="progressbar" aria-valuenow={Math.round(pct * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={busy}><i style={{ transform: `scaleX(${pct})` }} /><span className="mono">{busy}{backendReady ? ` · ${Math.round(pct * 100)}%` : ''}</span></div>}
      {msg && <p className={`msg ${msg.ok ? 'ok' : 'bad'}`} role="status">{msg.t}</p>}
      <div className="actions">
        {busy && backendReady && <button className="btn ghost" onClick={() => abort.current?.abort()}>Pause</button>}
        <button className="btn big" onClick={go} disabled={!!busy || !capture.name.trim() || (backendReady && !key.trim())}>{backendReady ? 'Upload and process' : 'Save files to this phone'}</button>
      </div>
    </main>
  );
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '') || 'capture';
function Num({ label, v, min, max, step, on, hint }: { label: string; v: number; min: number; max: number; step: number; on: (v: number) => void; hint?: string }) {
  return <label className="adv-row"><span><span className="adv-l">{label}</span>{hint && <span className="adv-h">{hint}</span>}</span><input type="number" value={v} min={min} max={max} step={step} onChange={e => on(Math.min(max, Math.max(min, +e.target.value || min)))} /></label>;
}
function Sel({ label, v, opts, on }: { label: string; v: string; opts: [string, string][]; on: (v: string) => void }) {
  return <label className="adv-row"><span className="adv-l">{label}</span><select value={v} onChange={e => on(e.target.value)}>{opts.map(([k, t]) => <option key={k} value={k}>{t}</option>)}</select></label>;
}
