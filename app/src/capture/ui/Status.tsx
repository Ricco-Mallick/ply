// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
import { useEffect, useState } from 'react';
import { Glass } from '../../shared/Glass';
import { getHealth, getJob, sceneUrl, type Health, type Job } from '../lib/backend';
import { LiveTheatre } from './LiveTheatre';

const BASE = import.meta.env.BASE_URL;
const STAGES: [string, string][] = [['frames', 'Picking sharp frames'], ['sfm', 'Solving camera positions'], ['align', 'Levelling the floor, real scale'], ['occupancy', 'Mapping walkable floor'], ['label', 'Finding objects (AI)'], ['train', 'Training gaussian splats'], ['export', 'Compressing for the web']];
// same shares of wall time as the worker's WEIGHTS (pipeline/plypipe/worker.py): turns overall progress into each stage's own
const WEIGHTS: Record<string, number> = { frames: .05, sfm: .20, align: .02, occupancy: .02, label: .04, train: .62, export: .05 };
export function stagePct(j: Pick<Job, 'stage' | 'progress'>) {
  if (!j.stage || !(j.stage in WEIGHTS)) return 0;
  let start = 0; for (const [k] of STAGES) { if (k === j.stage) break; start += WEIGHTS[k]; }
  return Math.max(0, Math.min(100, Math.round((j.progress - start) / WEIGHTS[j.stage] * 100)));
}
/** The worker's message without its "stage: " prefix ("sfm: matching frames · 40% · ~6 min left" -> "Matching frames · …"). */
export const stepText = (m: string | null) => { const t = (m ?? '').replace(/^[a-z]+:\s*/, ''); return t ? t[0].toUpperCase() + t.slice(1) : ''; };
export const stageLabel = (stage: string | null) => STAGES.find(s => s[0] === stage)?.[1] ?? 'Starting';
const JOBS = 'ply-jobs';
export interface JobRef { id: string; name: string; createdAt: number; }
export function listJobs(): JobRef[] { try { return JSON.parse(localStorage.getItem(JOBS) || '[]'); } catch { return []; } }
export function rememberJob(j: JobRef) { try { localStorage.setItem(JOBS, JSON.stringify([j, ...listJobs().filter(x => x.id !== j.id)].slice(0, 30))); } catch { /* ignore */ } }

const ago = (t: number) => { const s = Math.max(0, (Date.now() - t) / 1000); return s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`; };

/** Live status of one processing job. Polls every few seconds; the page can be closed and reopened by link. */
export function Status({ jobId, onHome }: { jobId: string; onHome: () => void }) {
  const [job, setJob] = useState<Job | null>(null); const [health, setHealth] = useState<Health | null>(null); const [err, setErr] = useState<string | null>(null); const [copied, setCopied] = useState(false);
  useEffect(() => {
    let t = 0; const ac = new AbortController();
    const poll = async () => {
      try { const j = await getJob(jobId, ac.signal); setJob(j); setErr(null); if (j.status === 'queued') setHealth(await getHealth(ac.signal)); if (j.status === 'done' || j.status === 'failed') return; }
      catch (e) { if (ac.signal.aborted) return; setErr((e as Error).message); }
      t = window.setTimeout(poll, document.hidden ? 15000 : 4000);
    };
    poll(); return () => { ac.abort(); clearTimeout(t); };
  }, [jobId]);
  const [, tick] = useState(0);  // re-render every second so "updated N s ago" and the elapsed time count live
  useEffect(() => { const i = window.setInterval(() => tick(n => n + 1), 1000); return () => clearInterval(i); }, []);
  useEffect(() => { if (job) document.title = `${job.status === 'done' ? '✓' : job.status === 'failed' ? '✕' : `${Math.round(job.progress * 100)}%`} ${job.name} · Ply`; }, [job?.status, job?.progress]);

  const si = job?.stage ? STAGES.findIndex(s => s[0] === job.stage) : -1;
  const link = `${location.origin}${BASE}capture.html?job=${jobId}`;
  const walk = `${BASE}walk.html?scene=${encodeURIComponent(sceneUrl(jobId))}`;
  const head = !job ? 'Checking…' : job.status === 'queued' ? 'Waiting for a GPU' : job.status === 'running' ? 'Building your twin' : job.status === 'done' ? 'Your twin is ready' : 'Processing failed';

  return (
    <main className={`status ${job?.status === 'running' || job?.status === 'done' ? 'wide-status' : ''}`}>
      <header className="cap-top"><button className="back mono" onClick={onHome}>← Captures</button><span className="mono dim">Job {jobId.slice(0, 8)}</span></header>
      <div className="wide eyebrow">{job?.name ?? 'Processing'}</div>
      <h1>{head}</h1>
      {job?.status === 'queued' && <p className="lead">Number {job.queuePosition} in the queue. The GPU worker picks jobs up in order; you can close this page and come back with the link below.</p>}
      {job?.status === 'queued' && health && <p className={`gpu-chip mono ${health.workersOnline ? 'on' : health.workersStarting ? 'warm' : ''}`}><span aria-hidden="true" />{health.workersOnline ? `GPU worker online${health.workers.some(w => w.busy) ? ' · busy with another job' : ' · starting yours shortly'}` : health.workersStarting ? `GPU worker starting up · ${(health.workers.find(w => w.starting)?.starting ?? '').replace(/^installing/, 'setting up')}` : 'No GPU worker running right now · the job waits safely until one starts'}</p>}
      {job?.status === 'running' && <p className="lead">Started {ago(job.started_at ?? job.created_at)} ago.</p>}
      {job?.status === 'running' && <p className={`gpu-chip mono ${Date.now() - job.updated_at < 90_000 ? 'on is-live' : 'warm'}`}><span aria-hidden="true" />{stepText(job.message) || 'Working'} · {Date.now() - job.updated_at < 90_000 ? `live, updated ${ago(job.updated_at)} ago` : `no word from the GPU for ${ago(job.updated_at)}`}</p>}
      {job?.status === 'failed' && <p className="lead bad-t">{job.message ?? 'The pipeline stopped with an error.'} Try the Incremental camera solver or SIFT matching in Advanced settings, or recapture the red stretches.</p>}
      {job?.result?.quality && (
        <div className={`verdict ${job.result.quality.ok ? 'ok' : 'bad'}`}>
          <span className="mono">{job.result.quality.ok ? 'Quality gate passed' : 'Quality gate failed'}{job.result.attempts && job.result.attempts.length > 1 ? ` · ${job.result.attempts.length} attempts` : ''}</span>
          {!!job.result.quality.issues.length && <ul>{job.result.quality.issues.map(i => <li key={i.code} className={i.level}>{i.message}</li>)}</ul>}
        </div>
      )}
      {job?.status === 'done' && <p className="lead">Finished in {job.started_at && job.finished_at ? ago(Date.now() - (job.finished_at - job.started_at)) : '—'}. Walk it in the browser, share the link, or open it on the big screen.</p>}

      {job && (job.status === 'running' || job.status === 'done') && <LiveTheatre jobId={jobId} job={job} />}

      <Glass className="stages" opts={{ radius: 20, shader: false }}>
        <ol>{STAGES.map(([k, label], i) => {
          const state = job?.status === 'done' ? 'done' : i < si ? 'done' : i === si ? (job?.status === 'failed' ? 'fail' : 'now') : 'todo';
          return <li key={k} className={state}><span className="st-dot" aria-hidden="true" /><span className="st-l">{label}</span>{state === 'now' && job && <span className="mono st-p">{stagePct(job)}%</span>}{state === 'done' && <span className="mono st-p">✓</span>}</li>;
        })}</ol>
        {job?.status === 'running' && <div className="bar"><i style={{ transform: `scaleX(${job.progress})` }} /></div>}
      </Glass>
      {err && <p className="msg bad" role="alert">{err}. Retrying…</p>}

      <div className="actions">
        <button className="btn ghost" onClick={() => navigator.clipboard?.writeText(link).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); })}>{copied ? 'Link copied' : 'Copy status link'}</button>
        {job?.status === 'done' && <a className="btn big" href={walk}>Open walkthrough</a>}
      </div>
    </main>
  );
}
