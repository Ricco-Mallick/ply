// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
import { useEffect, useRef, useState } from 'react';
import QRCode from 'qrcode';
import { Glass } from '../../shared/Glass';
import { listCaptures, type CaptureSummary } from '../lib/store';
import { fmtTime } from './viz';
import { listJobs, stageLabel, stagePct, stepText } from './Status';
import { arSupported } from '../lib/ar';
import { getJob, type Job } from '../lib/backend';

const BASE = import.meta.env.BASE_URL;
const PHONE = matchMedia('(pointer: coarse)').matches;


export function Start({ onRecord, onScan, onUpload, onOpen, onJob }: { onRecord: () => void; onScan: () => void; onUpload: (files: File[]) => void; onOpen: (id: string) => void; onJob: (id: string) => void }) {
  const [ar, setAr] = useState(false); useEffect(() => { arSupported().then(setAr); }, []);
  const [jobs] = useState(() => listJobs());
  const live = useLiveJobs(jobs.map(j => j.id));
  const [saved] = useState<CaptureSummary[]>(() => listCaptures());
  const file = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const pick = (fl: FileList | null) => { const v = [...(fl ?? [])].filter(f => f.type.startsWith('video/') || /\.(mp4|mov|webm|mkv)$/i.test(f.name)); if (v.length) onUpload(v); };
  return (
    <main className="start" onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={e => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files); }}>
      <header className="cap-top"><Glass as="a" href={BASE} className="brand pill" opts={{ shader: false, press: true }}><span className="mark" aria-hidden="true" />PLY</Glass><a className="cta-ghost" href={`${BASE}guide.html`}>Capture guide</a></header>
      <section className="start-hero">
        <div className="wide eyebrow">New capture</div>
        <h1>Walk it once.<br /><em>We'll map it.</em></h1>
        <p className="lead">Record the space with this phone's camera and Ply coaches you live, or upload videos you already have. Both end up as the same walkable twin.</p>
      </section>
      <div className="choices">
        {ar && <Glass as="button" className="choice primary" opts={{ press: true, radius: 22 }} onClick={onScan}>
          <span className="choice-ico" aria-hidden="true"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3" /><circle cx="8" cy="10" r="1" fill="currentColor" /><circle cx="12" cy="8" r="1" fill="currentColor" /><circle cx="16" cy="11" r="1" fill="currentColor" /><circle cx="10" cy="14" r="1" fill="currentColor" /><circle cx="14" cy="15" r="1" fill="currentColor" /></svg></span>
          <span className="choice-t">AR scan</span>
          <span className="choice-d">Watch the room fill in with live coverage dots. Real scale, level floor. Best results.</span>
          <span className="choice-note mono">Recommended on this phone</span>
        </Glass>}
        <Glass as="button" className={`choice ${ar ? '' : 'primary'}`} opts={{ press: true, radius: 22 }} onClick={onRecord}>
          <span className="choice-ico" aria-hidden="true"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="2.5" y="6" width="14" height="12" rx="2.5" /><path d="m16.5 10.5 5-3v9l-5-3" /><circle cx="9.5" cy="12" r="2.4" fill="currentColor" stroke="none" /></svg></span>
          <span className="choice-t">Record with camera</span>
          <span className="choice-d">{ar ? 'Video with live tips, alignment, sensors and a quality map.' : 'Live guidance, sensors, quality map. Best results.'}</span>
          {!PHONE && <span className="choice-note mono">Works best on a phone</span>}
        </Glass>
        <Glass as="button" className={`choice ${drag ? 'drag' : ''}`} opts={{ press: true, radius: 22 }} onClick={() => file.current?.click()}>
          <span className="choice-ico" aria-hidden="true"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M12 15V4m0 0-4 4m4-4 4 4" /><path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" /></svg></span>
          <span className="choice-t">Upload videos</span>
          <span className="choice-d">One or more clips of the same space. Drop them here or pick from your files.</span>
        </Glass>
        <input ref={file} type="file" accept="video/*" multiple hidden onChange={e => pick(e.target.files)} />
        <Glass as="div" className="choice soon" aria-disabled="true" opts={{ radius: 22 }}>
          <span className="choice-ico" aria-hidden="true"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="12" cy="12" r="2.6" fill="currentColor" stroke="none" /><path d="M6.4 6.4a7.9 7.9 0 0 0 0 11.2M17.6 17.6a7.9 7.9 0 0 0 0-11.2" /><path d="M3.4 3.4a12.2 12.2 0 0 0 0 17.2M20.6 20.6a12.2 12.2 0 0 0 0-17.2" opacity=".5" /></svg></span>
          <span className="soon-body">
            <span className="choice-t">LiDAR + video<i className="pill-ios mono">iPhone only</i></span>
            <span className="choice-d">The Pro sensor's depth, recorded alongside the video: true metric geometry in dim or textureless rooms, with photo colour on top.</span>
          </span>
          <span className="choice-note mono soon-note"><b aria-hidden="true" /><span className="soon-lines"><span>iPhone 12 Pro or later</span><span className="soon-sub">Coming soon · in development</span></span></span>
        </Glass>
      </div>
      {!PHONE && <PhoneQR />}
      {jobs.length > 0 && (
        <section className="saved">
          <h2 className="wide">Sent for processing</h2>
          <ul>{jobs.map(j => <li key={j.id}><button onClick={() => onJob(j.id)}><span className="saved-n">{j.name}</span><span className="mono">Job {j.id.slice(0, 8)} · {new Date(j.createdAt).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}</span>{live[j.id] && <JobChip job={live[j.id]} />}</button></li>)}</ul>
        </section>
      )}
      {saved.length > 0 && (
        <section className="saved">
          <h2 className="wide">On this device</h2>
          <ul>{saved.map(c => <li key={c.id}><button onClick={() => onOpen(c.id)}><span className="saved-n">{c.name}</span><span className="mono">{c.clips} clip{c.clips === 1 ? '' : 's'} · {fmtTime(c.duration)} · {new Date(c.createdAt).toLocaleDateString()}</span></button></li>)}</ul>
        </section>
      )}
    </main>
  );
}

/** On a laptop: a code to open this page on the phone that will do the walking. */
function PhoneQR() {
  const [src, setSrc] = useState(''); const url = (location.hostname === 'localhost' || location.hostname === '127.0.0.1') && __LAN_URL__ ? `${__LAN_URL__}${location.pathname.replace(/^\//, '')}` : location.href;
  useEffect(() => { QRCode.toDataURL(url, { margin: 1, width: 240, color: { dark: '#0B1220', light: '#F3F4F6' } }).then(setSrc).catch(() => setSrc('')); }, [url]);
  if (!src) return null;
  return (
    <Glass className="qr" opts={{ radius: 18 }}>
      <img src={src} width={120} height={120} alt={`QR code to open ${url} on your phone`} />
      <div><div className="wide">Open on your phone</div><p>Scan with the phone camera. Recording needs the rear camera and motion sensors, so the phone does the walking.</p><code className="mono">{url}</code></div>
    </Glass>
  );
}

/** Live status for the remembered jobs: unfinished ones are re-checked every 5 s while the page is visible. */
function useLiveJobs(ids: string[]) {
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const key = ids.join(',');
  useEffect(() => {
    if (!ids.length) return;
    const ac = new AbortController(); let t = 0; const done = new Set<string>();
    const poll = async () => {
      const got = await Promise.all(ids.filter(id => !done.has(id)).map(id => getJob(id, ac.signal).catch(() => null)));
      if (ac.signal.aborted) return;
      setJobs(prev => { const n = { ...prev }; for (const j of got) if (j) { n[j.id] = j; if (j.status === 'done' || j.status === 'failed') done.add(j.id); } return n; });
      if (done.size < ids.length) t = window.setTimeout(poll, document.hidden ? 20000 : 5000);
    };
    poll(); return () => { ac.abort(); clearTimeout(t); };
  }, [key]);
  return jobs;
}

function JobChip({ job }: { job: Job }) {
  const text = job.status === 'done' ? 'Ready to walk' : job.status === 'failed' ? 'Failed' : job.status === 'queued' ? `Queued · #${job.queuePosition}`
    : `${stageLabel(job.stage)} · ${stagePct(job)}%${stepText(job.message) ? ` · ${stepText(job.message)}` : ''}`;
  return <span className={`job-chip mono ${job.status}`}><i aria-hidden="true" />{text}</span>;
}
