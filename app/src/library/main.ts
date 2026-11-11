// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Library: every capture the team has run, from any device.
   The job list lives in the Worker's database, not in this browser, so a phone capture shows up on the
   laptop as soon as the team code is entered. Nothing is stored here except the code itself. */
import '../shared/tokens.css';
import './library.css';

const KEY = 'ply-team-key';
const API = (import.meta.env.VITE_API as string | undefined)?.replace(/\/$/, '') || 'https://ply-api.ricco-ply.workers.dev';

interface Job { id: string; name: string; status: string; stage: string | null; progress: number | null; message: string | null; createdAt: number; updatedAt: number; files: string[] | null; }

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;
const app = $('#root');
const code = () => { try { return localStorage.getItem(KEY) ?? ''; } catch { return ''; } };
const setCode = (v: string) => { try { v ? localStorage.setItem(KEY, v.trim()) : localStorage.removeItem(KEY); } catch { /* private mode */ } };
const ago = (t: number) => { const s = Math.max(0, (Date.now() - t) / 1000); return s < 90 ? `${Math.round(s)} s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 172800 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`; };
const esc = (s: string) => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c] as string));
const sceneUrl = (id: string) => `${API}/v1/scenes/${id}/scene.json`;

const BADGE: Record<string, string> = { done: 'ready', review: 'needs review', running: 'running', queued: 'queued', failed: 'failed' };

app.innerHTML = `
<header class="top">
  <a class="brand" href="./"><span class="mark" aria-hidden="true"></span>PLY</a>
  <span class="sec mono">Library</span>
  <span class="grow"></span>
  <a class="ghost" href="./walk.html">Walkthrough</a>
  <a class="ghost" href="./capture.html">New capture</a>
</header>

<main>
  <section class="intro">
    <div class="wide eyebrow">Your team</div>
    <h1>Every capture, <em>one place</em></h1>
    <p class="lead">Finished twins, jobs still running and runs that failed, whichever device they came from. Enter the team code once; this browser remembers it.</p>
    <form class="code" id="codeform">
      <input id="code" name="code" placeholder="team code, e.g. team.1801318764.1e9b…" autocomplete="off" spellcheck="false" />
      <button class="btn" type="submit">Open library</button>
      <button class="ghost" type="button" id="forget">Forget</button>
    </form>
    <p class="note mono" id="note">no code yet</p>
  </section>

  <section class="list" id="list" aria-live="polite"></section>

  <section class="demo">
    <div class="wide eyebrow">Built-in demos</div>
    <div class="cards" id="demos"></div>
  </section>
</main>
<footer class="foot mono">Ply · Namma Space · Techfest 2026-27</footer>
`;

const DEMOS: { name: string; note: string; walk: string; model?: string }[] = [
  { name: 'Classroom', note: '10.5 × 6.3 m · 194 places · one pipeline version behind the current one', walk: './walk.html?scene=' + encodeURIComponent(`${API}/v1/scenes/hq-classroom-1/scene.json`), model: './model.html?scene=' + encodeURIComponent(`${API}/v1/scenes/hq-classroom-1/scene.json`) + '&module=/ply/models/classroom2.js' },
  { name: 'Chapel', note: 'vaulted interior · current pipeline, high preset', walk: './walk.html?scene=' + encodeURIComponent(`${API}/v1/scenes/chapel-high-2/scene.json`) },
  { name: 'Office (LiDAR)', note: '6.15 M points · measured shell · under development', walk: './lidar.html' },
  // Deep Blending's Dr Johnson house (263 photos) through the current pipeline, one twin per preset; numbers in docs/BENCHMARK.md
  ...([
    ['low', 'quick preset · 277k splats · 5 MB · 29.0 dB on held-out photos'],
    ['medium', 'standard preset · 554k splats · 12 MB · 30.8 dB on held-out photos'],
    ['high', 'high preset · 961k splats · 22 MB · 31.0 dB on held-out photos'],
  ] as const).map(([tier, note]) => ({
    name: `Dr Johnson house · ${tier}`, note,
    walk: './walk.html?scene=' + encodeURIComponent(`${API}/v1/scenes/drjohnson-${tier}/scene.json`),
  })),
];

$('#demos').innerHTML = DEMOS.map(d => `
  <article class="card">
    <h3>${d.name}</h3>
    <p>${d.note}</p>
    <div class="row"><a class="btn sm" href="${d.walk}">Walk it</a>${d.model ? `<a class="ghost sm" href="${d.model}">Model</a>` : ''}</div>
  </article>`).join('');

function card(j: Job) {
  const pct = Math.round((j.progress ?? 0) * 100);
  const ready = j.status === 'done' && j.files?.includes('scene.json');
  const action = ready ? `<a class="btn sm" href="./walk.html?scene=${encodeURIComponent(sceneUrl(j.id))}">Walk it</a>`
    : j.status === 'review' ? `<a class="btn sm" href="./capture.html?job=${encodeURIComponent(j.id)}">Review</a>`
    : j.status === 'running' || j.status === 'queued' ? `<a class="ghost sm" href="./capture.html?job=${encodeURIComponent(j.id)}">Watch</a>`
    : '';
  const legacy = /demo|rehearsal/i.test(j.name) && !ready ? '' : /demo|rehearsal/i.test(j.name) ? 'legacy' : '';
  return `<article class="card ${ready ? '' : 'dim'}">
    <div class="chead"><h3>${esc(j.name || 'Capture')}</h3><span class="badge ${esc(j.status)}">${BADGE[j.status] ?? esc(j.status)}</span></div>
    ${legacy ? '<p class="meta legacy-note">Bad capture · very old development pipeline</p>' : ''}
    <p class="meta mono">${j.stage ? esc(j.stage) + ' · ' : ''}${j.createdAt ? ago(j.createdAt) : ''}${j.message && j.status !== 'done' ? ' · ' + esc(String(j.message).slice(0, 70)) : ''}</p>
    ${j.status === 'running' || j.status === 'queued' ? `<div class="bar"><i style="transform:scaleX(${(pct / 100).toFixed(3)})"></i></div>` : ''}
    <div class="row">${action}<span class="id mono">${esc(j.id.slice(0, 8))}</span></div>
  </article>`;
}

async function load(teamCode: string) {
  const note = $('#note'), list = $('#list');
  if (!teamCode) { list.innerHTML = ''; note.textContent = 'no code yet'; return; }
  note.textContent = 'reading the team library…';
  try {
    const r = await fetch(`${API}/v1/jobs`, { headers: { 'x-ply-key': teamCode } });
    if (r.status === 401) { note.innerHTML = '<span class="bad">that code was not accepted (expired, or not a team code)</span>'; list.innerHTML = ''; return; }
    if (!r.ok) throw new Error(`server said ${r.status}`);
    const { jobs } = await r.json() as { jobs: Job[] };
    const live = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;
    const ready = jobs.filter(j => j.status === 'done').length;
    note.textContent = `${jobs.length} jobs · ${ready} ready · ${live} in flight · updated ${new Date().toLocaleTimeString()}`;
    list.innerHTML = jobs.length
      ? `<div class="wide eyebrow">Captures</div><div class="cards">${jobs.map(card).join('')}</div>`
      : '<p class="lead">No jobs yet. Record something with the capture app and it appears here.</p>';
  } catch (e) {
    note.innerHTML = `<span class="bad">${esc((e as Error).message)}</span>`;
  }
}

const form = $('#codeform') as HTMLFormElement;
const input = $('#code') as HTMLInputElement;
// a shared link can carry the code: library.html?code=team.…
const fromUrl = new URLSearchParams(location.search).get('code');
if (fromUrl) { setCode(fromUrl); history.replaceState(null, '', location.pathname); }
input.value = code();
$('#forget').addEventListener('click', () => { setCode(''); input.value = ''; load(''); });
form.addEventListener('submit', e => { e.preventDefault(); const v = input.value.trim(); setCode(v); load(v); });
load(code());
setInterval(() => load(code()), 15000);   // a job the phone just queued shows up without a reload
