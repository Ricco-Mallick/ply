// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Ply API: one Cloudflare Worker for uploads, the job queue (D1) and serving results.
 * Storage is any S3-compatible bucket (Backblaze B2 by default: free 10 GB, no card). Bytes never pass through the
 * Worker on the way in: it signs short-lived URLs and the phone / GPU worker PUT parts straight to the bucket.
 *
 *  Capture app (header X-Ply-Key: APP_KEY)
 *    POST /v1/uploads            {captureId, name, type}        -> {key, uploadId}       start a multipart upload
 *    POST /v1/uploads/sign       {key, uploadId, parts:[N…]}     -> {urls: {N: url}}     PUT each part there; keep the ETag header
 *    POST /v1/uploads/complete   {key, uploadId, parts}          -> {key}
 *    PUT  /v1/files              ?key  <bytes>  (≤ 50 MB)        -> {key}                capture.json and other small files
 *    POST /v1/jobs               {captureId, name, settings, args, files} -> {jobId}
 *  Anyone
 *    GET  /v1/jobs/:id                                           -> status, stage, progress, result
 *    GET  /v1/health                                             -> GPU workers online, queue length
 *    GET  /v1/scenes/:jobId/places                               -> the walkthrough's pins (never edge-cached)
 *    GET  /v1/scenes/:jobId/:file                                -> results (Range supported) for the viewer
 *  Team key (X-Ply-Key: APP_KEY) or the GPU worker, owner roles only
 *    PUT  /v1/scenes/:jobId/places   {places, view?}             -> {ok, updatedAt, count}  add/move pins, crop; cloud/PLACES-API.md
 *  GPU worker (Authorization: Bearer WORKER_TOKEN)
 *    POST /v1/worker/claim       {worker, prefix?}               -> next job or 204       stale running jobs (no heartbeat 20 min) are re-queued;
 *                                                                  prefix limits it to captures whose id starts with it (rehearsals)
 *    POST /v1/worker/jobs/:id    {status?, stage?, progress?, message?, result?}          heartbeat + progress
 *    GET  /v1/worker/file        ?key                            -> {url} signed download link (inputs)
 *    (and the upload routes above, which accept any key under scenes/<jobId>/ for results)
 */

import { completeMultipart, createMultipart, presign, signedFetch, type S3Conf } from './s3.ts';
import { cdnRedirect, isBulk } from './cdn.ts';
import { kickModal } from './modal.ts';
import { MAX_BODY_BYTES, document as placesDocument, emptyDoc, normalize as normalizePlaces, validatePlaces, validateView } from './places.ts';

interface Env { DB: any; APP_KEY: string; WORKER_TOKEN: string; ALLOWED_ORIGINS: string;
  S3_ENDPOINT: string; S3_REGION: string; S3_BUCKET: string; S3_KEY_ID: string; S3_SECRET: string;
  /** Modal GPU worker (pipeline/modal/ply_modal.py, see src/modal.ts): trigger URL (wrangler.toml, else the older
      secret) + shared token; unset = Kaggle only */
  MODAL_TRIGGER_URL?: string; MODAL_TRIGGER?: string; MODAL_TRIGGER_TOKEN?: string;
  /** results on Hugging Face (HF_REPO in wrangler.toml, HF_TOKEN a secret); unset = bucket only */
  HF_REPO?: string; HF_TOKEN?: string;
  /** commit and time of this deploy, set by .github/workflows/api.yml (--var); unset for a hand deploy */
  VERSION?: string;
  CF_VERSION?: { id: string; tag: string; timestamp: string }; DEPLOYED_AT?: string; }
type Ctx = { waitUntil(p: Promise<unknown>): void };

const s3 = (e: Env): S3Conf => ({ endpoint: e.S3_ENDPOINT, region: e.S3_REGION, bucket: e.S3_BUCKET, keyId: e.S3_KEY_ID, secret: e.S3_SECRET });
const SIGN_TTL = 6 * 3600;

const STALE_MS = 20 * 60 * 1000;
const MAX_PAUSES = 8;
const MAX_ATTEMPTS = 2;
const json = (data: unknown, status = 200, h: Record<string, string> = {}) => new Response(JSON.stringify(data), { status, headers: { 'content-type': 'application/json', ...h } });
/** Anything a person can change between two views: no browser, proxy or edge copy may answer the next read. */
const NO_STORE = { 'cache-control': 'no-store' };
const safeName = (s: string) => s.replace(/[^\w.\-]+/g, '_').slice(0, 120) || 'file';
const id = () => crypto.randomUUID();
/** The Workers runtime has caches.default; the DOM type in lib does not, and cloud/ has no workers-types dependency. */
const edge = () => (caches as unknown as { default: Cache }).default;

function cors(req: Request, env: Env) {
  const o = req.headers.get('origin') ?? '', ok = env.ALLOWED_ORIGINS.split(',').includes(o) || /^https?:\/\/(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?$/.test(o);
  return {
    'access-control-allow-origin': ok ? o : '*', vary: 'origin',
    'access-control-allow-methods': 'GET,HEAD,POST,PUT,OPTIONS', 'access-control-allow-headers': 'content-type,x-ply-key,authorization,range',
    'access-control-expose-headers': 'content-length,content-range,accept-ranges,etag', 'access-control-max-age': '86400',
  };
}

type Role = 'app' | 'worker' | null;

import { inviteBudget, resultsVisible } from './limits.ts';
import { summarize, summaryText } from './summary.ts';
import { PAIR_PER_HOUR, PAIR_TTL_MS, machineKeyAllowed, machineName, machinePathAllowed, machineReadAllowed, normCode, pairCode, parseMachineAuth, randomHex, sha256hex, trimSpecs } from './machines.ts';


/* Invite only. There is no signup: an invite is a signed code, label.expiry.mac, minted with the
   app key (scripts/invite.py), so the Worker can verify one with Web Crypto and no database. Expiry
   bounds the damage of a leaked code; rotating APP_KEY revokes every outstanding invite at once. */
const enc = new TextEncoder();
async function hmacHex(secret: string, msg: string): Promise<string> {
  const k = await crypto.subtle.importKey('raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const sig = new Uint8Array(await crypto.subtle.sign('HMAC', k, enc.encode(msg)));
  return Array.from(sig, b => b.toString(16).padStart(2, '0')).join('');
}
async function inviteOf(code: string, env: Env): Promise<{ label: string; expires: number } | null> {
  const m = /^([a-z0-9][a-z0-9-]{1,23})\.(\d{10})\.([0-9a-f]{32})$/.exec(code);
  if (!m || !env.APP_KEY) return null;
  const [, label, exp, mac] = m;
  const expires = Number(exp);
  if (expires * 1000 < Date.now()) return null;
  const want = await hmacHex(env.APP_KEY, `${label}.${exp}`);
  return want.slice(0, 32) === mac ? { label, expires } : null;
}
async function identity(req: Request, env: Env): Promise<{ role: Role; owner: string }> {
  const key = req.headers.get('x-ply-key') ?? '';
  if (env.WORKER_TOKEN && req.headers.get('authorization') === `Bearer ${env.WORKER_TOKEN}`) return { role: 'worker', owner: 'worker' };
  if (env.APP_KEY && key === env.APP_KEY) return { role: 'app', owner: 'team' };
  const inv = key ? await inviteOf(key, env) : null;
  return { role: inv ? 'app' : null, owner: inv?.label ?? '' };
}

async function role(req: Request, env: Env): Promise<Role> {
  if (env.WORKER_TOKEN && req.headers.get('authorization') === `Bearer ${env.WORKER_TOKEN}`) return 'worker';
  const key = req.headers.get('x-ply-key') ?? '';
  if (!key) return null;
  if (env.APP_KEY && key === env.APP_KEY) return 'app';
  return (await inviteOf(key, env)) ? 'app' : null;
}
/** The app may only write its own captures; the worker may only write results. */
function keyAllowed(key: string, r: Role) {
  if (key.includes('..') || key.startsWith('/')) return false;
  return r === 'app' ? key.startsWith('captures/') : r === 'worker' ? key.startsWith('scenes/') : false;
}

type Machine = { name: string; busyJob: string | null; files: string[] };
/** A paired PC (src/machines.ts): its own token, and the job it is working on, if any. */
async function machineOf(req: Request, env: Env, now: number): Promise<Machine | null> {
  const auth = parseMachineAuth(req.headers.get('authorization'));
  if (!auth) return null;
  const row = await env.DB.prepare("SELECT name FROM machines WHERE name = ? AND status = 'active' AND token_hash = ?")
    .bind(auth.name, await sha256hex(auth.secret)).first() as any;
  if (!row) return null;
  await env.DB.prepare('UPDATE machines SET last_seen = ? WHERE name = ?').bind(now, row.name).run();
  // its job: running now, or ended in the last half hour (the log and late files go up after the final status)
  const job = await env.DB.prepare(`SELECT id, files FROM jobs WHERE assigned_worker = ? AND worker = ? AND
      (status = 'running' OR (status IN ('done','failed','review') AND updated_at > ?)) ORDER BY started_at DESC LIMIT 1`).bind(row.name, row.name, now - 30 * 60_000).first() as any;
  let files: string[] = [];
  try { files = job ? JSON.parse(job.files) : []; } catch { /* malformed row: no inputs */ }
  return { name: row.name, busyJob: job?.id ?? null, files };
}
const isTeam = async (req: Request, env: Env) => (await identity(req, env)).owner === 'team';

export default {
  async fetch(req: Request, env: Env, ctx: Ctx): Promise<Response> {
    const c = cors(req, env);
    if (req.method === 'OPTIONS') return new Response(null, { status: 204, headers: c });
    try { const r = await route(req, env, ctx); const h = new Headers(r.headers); for (const [k, v] of Object.entries(c)) h.set(k, v); return new Response(r.body, { status: r.status, headers: h }); }
    catch (e) { return json({ error: (e as Error).message || 'server error' }, 500, c); }
  },
  /** Every 10 minutes: work waiting (a missed trigger, or a job whose worker died) and no worker seen for 90 s -> wake Modal. */
  async scheduled(_e: unknown, env: Env, ctx: Ctx) {
    const now = Date.now();
    // queued for 2+ minutes, or 'running' with no heartbeat for 20 minutes (its worker died) and still allowed a retry
    const waiting = await env.DB.prepare(`SELECT COUNT(*) n FROM jobs WHERE capture_id NOT LIKE 'rehearsal-%' AND assigned_worker IS NULL AND
      ((status='queued' AND created_at < ?) OR (status='running' AND updated_at < ? AND attempts < ?))`).bind(now - 120_000, now - STALE_MS, MAX_ATTEMPTS).first();
    const live = await env.DB.prepare('SELECT COUNT(*) n FROM workers WHERE last_seen > ? AND name NOT IN (SELECT name FROM machines)').bind(now - 90_000).first();
    if (waiting.n > 0 && live.n === 0) ctx.waitUntil(kickModal(env, 'cron'));
  },
};

async function route(req: Request, env: Env, ctx: Ctx): Promise<Response> {
  const url = new URL(req.url), p = url.pathname, q = url.searchParams, now = Date.now();
  let r = await role(req, env);
  // a paired PC: a worker for its assigned jobs only, and nobody anywhere else in the API
  const mach = r ? null : await machineOf(req, env, now);
  if (mach) r = machinePathAllowed(p) ? 'worker' : null;
  const allowed = (key: string) => mach ? machineKeyAllowed(key, mach.busyJob) : keyAllowed(key, r);
  // who am I: the app checks an invite before it lets someone record and spend GPU time
  if (p === '/v1/whoami') {
    const key = req.headers.get('x-ply-key') ?? '';
    if (!key) return json({ error: 'no invite' }, 401);
    if (env.APP_KEY && key === env.APP_KEY) return json({ role: 'team', label: 'team' });
    const inv = await inviteOf(key, env);
    return inv ? json({ role: 'invite', label: inv.label, expires: inv.expires }) : json({ error: 'that invite is not valid or has expired' }, 401);
  }

  // which code is live, in one request: a stale deploy shows here instead of as a slow viewer
  // ---- paired PCs (src/machines.ts) ----
  if (p === '/v1/machines/pair' && req.method === 'POST') {  // no key: the PC has none yet. It gets a code to show
    await env.DB.prepare("DELETE FROM machines WHERE status = 'pending' AND created_at < ?").bind(now - PAIR_TTL_MS).run();
    const recent = await env.DB.prepare('SELECT COUNT(*) n FROM machines WHERE created_at > ?').bind(now - 3600_000).first() as any;
    if ((recent?.n ?? 0) >= PAIR_PER_HOUR) return json({ error: 'too many pairing requests this hour, try later' }, 429);
    const b = await req.json().catch(() => ({})) as any;
    const name = machineName(b.hint), code = pairCode(), poll = randomHex(32);
    await env.DB.prepare('INSERT INTO machines (name,status,pair_code,poll_hash,specs,created_at) VALUES (?,?,?,?,?,?)')
      .bind(name, 'pending', code, await sha256hex(poll), trimSpecs(b.specs), now).run();
    return json({ name, code, poll, expiresInSec: PAIR_TTL_MS / 1000 }, 201, NO_STORE);
  }
  if (p === '/v1/machines/poll' && req.method === 'POST') {  // the PC waits here until the team approves its code
    const b = await req.json().catch(() => ({})) as any;
    const row = await env.DB.prepare('SELECT name,status,poll_hash FROM machines WHERE name = ?').bind(String(b.name ?? '')).first() as any;
    if (!row || !row.poll_hash || row.poll_hash !== await sha256hex(String(b.poll ?? ''))) return json({ error: 'unknown pairing' }, 404, NO_STORE);
    if (row.status === 'pending') return json({ status: 'pending' }, 200, NO_STORE);
    if (row.status !== 'approved') return json({ status: row.status }, 200, NO_STORE);
    const secret = randomHex(32);  // handed over once; only its hash is kept, and the poll secret dies with it
    await env.DB.prepare("UPDATE machines SET status='active', token_hash=?, poll_hash=NULL, pair_code=NULL WHERE name=? AND status='approved'")
      .bind(await sha256hex(secret), row.name).run();
    return json({ status: 'active', name: row.name, token: `pm_${row.name}.${secret}` }, 200, NO_STORE);
  }
  if (p === '/v1/machines' || p.startsWith('/v1/machines/')) {
    if (r !== 'app' || !(await isTeam(req, env))) return json({ error: 'team key required' }, 401);
    if (p === '/v1/machines' && req.method === 'GET') {
      const rows = await env.DB.prepare(`SELECT m.name,m.status,m.pair_code,m.specs,m.label,m.created_at,m.approved_at,m.last_seen,
          (SELECT COUNT(*) FROM jobs j WHERE j.assigned_worker = m.name AND j.status IN ('queued','running')) AS queued
          FROM machines m WHERE m.status != 'revoked' ORDER BY COALESCE(m.last_seen, m.created_at) DESC`).all();
      return json({ machines: (rows.results as any[]).map(x => ({ ...x, specs: x.specs ? JSON.parse(x.specs) : null,
        online: !!x.last_seen && now - x.last_seen < 90_000 })) }, 200, NO_STORE);
    }
    if (p === '/v1/machines/approve' && req.method === 'POST') {
      const b = await req.json().catch(() => ({})) as any;
      const code = normCode(b.code);
      const row = code ? await env.DB.prepare("SELECT name FROM machines WHERE pair_code = ? AND status = 'pending'").bind(code).first() as any : null;
      if (!row) return json({ error: 'no PC is waiting with that code (expired, or already approved)' }, 404);
      await env.DB.prepare("UPDATE machines SET status='approved', approved_at=?, label=? WHERE name=?")
        .bind(now, b.label ? String(b.label).slice(0, 80) : null, row.name).run();
      return json({ ok: true, name: row.name });
    }
    const mm = p.match(/^\/v1\/machines\/([a-z0-9-]{3,48})\/(revoke|label)$/);
    if (mm && req.method === 'POST') {
      const b = await req.json().catch(() => ({})) as any;
      if (mm[2] === 'revoke') {
        await env.DB.prepare("UPDATE machines SET status='revoked', token_hash=NULL, poll_hash=NULL, pair_code=NULL WHERE name=?").bind(mm[1]).run();
        // its open jobs stop here, loudly: nothing else may claim them (and they must not drift onto paid Modal)
        await env.DB.prepare(`UPDATE jobs SET status='failed', message=?, finished_at=?, updated_at=?
            WHERE assigned_worker=? AND status IN ('queued','running')`)
          .bind(`stopped: its PC ${mm[1]} was revoked. Assign it again (plyctl assign) to run it elsewhere`, now, now, mm[1]).run();
      } else {
        await env.DB.prepare('UPDATE machines SET label=? WHERE name=?').bind(String(b.label ?? '').slice(0, 80) || null, mm[1]).run();
      }
      return json({ ok: true });
    }
    return json({ error: 'not found' }, 404);
  }
  // assign a job to a paired PC (or back to the shared queue with worker: null). A failed or finished job is queued again
  const assign = p.match(/^\/v1\/jobs\/([\w-]{6,64})\/assign$/);
  if (assign && req.method === 'POST') {
    if (r !== 'app' || !(await isTeam(req, env))) return json({ error: 'team key required' }, 401);
    const b = await req.json().catch(() => ({})) as any;
    const worker = b.worker ? String(b.worker) : null;
    if (worker) {
      const m2 = await env.DB.prepare("SELECT name FROM machines WHERE name=? AND status IN ('approved','active')").bind(worker).first();
      if (!m2) return json({ error: `no paired machine called ${worker}` }, 404);
    }
    const job = await env.DB.prepare('SELECT * FROM jobs WHERE id=?').bind(assign[1]).first() as any;
    if (!job) return json({ error: 'no such job' }, 404);
    const args = Array.isArray(b.args) ? JSON.stringify(b.args.map(String)) : job.args;  // what to run on it: --until, --from, --preset ...
    // queued or failed: moved as it is. Done, in review or running: a copy with the same clips and a new id, so the
    // PC never writes over scenes/<id>/ of a twin that is published or still being made
    if (!b.copy && (job.status === 'queued' || job.status === 'failed')) {
      await env.DB.prepare("UPDATE jobs SET assigned_worker=?, args=?, status='queued', attempts=0, progress=0, stage=NULL, message=?, updated_at=? WHERE id=?")
        .bind(worker, args, worker ? `assigned to ${worker}` : 'back in the shared queue', now, job.id).run();
      if (!worker && !String(job.capture_id ?? '').startsWith('rehearsal-')) ctx.waitUntil(kickModal(env, `job ${job.id}`));
      return json({ ok: true, jobId: job.id, copied: false, worker, args: JSON.parse(args) });
    }
    const copyId = id();
    await env.DB.prepare('INSERT INTO jobs (id,name,capture_id,settings,args,files,status,progress,created_at,updated_at,owner,assigned_worker,message) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)')
      .bind(copyId, `${String(job.name).slice(0, 64)} (${worker ? 'on ' + worker.slice(0, 12) : 'again'})`, job.capture_id, job.settings, args, job.files,
        'queued', 0, now, now, job.owner ?? 'team', worker, `copy of ${job.id}${worker ? `, assigned to ${worker}` : ''}`).run();
    if (!worker && !String(job.capture_id ?? '').startsWith('rehearsal-')) ctx.waitUntil(kickModal(env, `job ${copyId}`));
    return json({ ok: true, jobId: copyId, copied: true, from: job.id, worker, args: JSON.parse(args) });
  }

  if (p === '/' || p === '/v1') return json({ ok: true, service: 'ply-api', version: env.CF_VERSION?.tag ?? env.VERSION ?? 'manual', deployedAt: env.CF_VERSION?.timestamp ?? env.DEPLOYED_AT ?? null, id: env.CF_VERSION?.id ?? null }, 200, { 'cache-control': 'no-store' });
  if (p === '/v1/health' && req.method === 'GET') {
    const w = await env.DB.prepare('SELECT name,last_seen,busy_job,note FROM workers WHERE last_seen > ? ORDER BY last_seen DESC').bind(now - 90_000).all();
    const q = await env.DB.prepare("SELECT COUNT(*) n FROM jobs WHERE status='queued'").first();
    const pcs = new Set(((await env.DB.prepare('SELECT name FROM machines').all()).results as any[]).map(x => x.name));
    const pcsOnline = w.results.filter((x: any) => pcs.has(x.name)).length;
    w.results = w.results.filter((x: any) => !pcs.has(x.name));
    return json({ ok: true, pcsOnline, workersOnline: w.results.filter((x: any) => !x.note).length, workersStarting: w.results.filter((x: any) => x.note).length, workers: w.results.map((x: any) => ({ name: x.name, busy: !!x.busy_job, starting: x.note || null, seenSecondsAgo: Math.round((now - x.last_seen) / 1000) })), queued: q.n }, 200, { 'cache-control': 'no-store' });
  }

  // ---- places: the walkthrough's pins (round 2, interactive spatial POI tagging) ----
  // This must come before the generic /v1/scenes/:id/:file route below: "places" matches it, and that route would
  // answer with a cached, publish-gated 404 instead of the empty document.
  const placesPath = p.match(/^\/v1\/scenes\/([\w-]+)\/places$/);
  if (placesPath) {
    const scene = placesPath[1], key = `scenes/${scene}/places.json`;
    if (req.method === 'GET' || req.method === 'HEAD') {
      // Read straight from storage, never through the shared edge cache: a pin added a second ago must show now.
      const text = await objectText(env, key);
      if (text === null) return json(emptyDoc(), 200, NO_STORE);            // no pins yet is a valid scene
      const doc = normalizePlaces(text);
      if (!doc) return json({ error: 'places.json is not readable' }, 502, NO_STORE);
      return json(doc, 200, NO_STORE);
    }
    if (req.method === 'PUT') {
      // Owner roles only: the team key that runs the account, or the GPU worker that publishes. An invite is a
      // guest identity (it may spend GPU time on a job) and cannot rewrite a scene's pins.
      if (!r) return json({ error: 'missing or wrong key' }, 401, NO_STORE);
      if (r === 'app' && (await identity(req, env)).owner !== 'team') return json({ error: 'only the team key can edit places' }, 403, NO_STORE);
      const declared = Number(req.headers.get('content-length') ?? 0);
      if (declared > MAX_BODY_BYTES) return json({ error: `body is larger than ${MAX_BODY_BYTES / 1024} KB` }, 413, NO_STORE);
      const raw = await req.text();
      if (raw.length > MAX_BODY_BYTES) return json({ error: `body is larger than ${MAX_BODY_BYTES / 1024} KB` }, 413, NO_STORE);
      let body: unknown;
      try { body = JSON.parse(raw); } catch { return json({ ok: false, error: 'body is not valid JSON', index: null, field: 'body' }, 400, NO_STORE); }
      const checked = validatePlaces(body);
      if (!checked.ok) return json(checked, 400, NO_STORE);
      // a body without `view` (an older viewer, a pin edit) keeps the crop already stored; null clears it
      const view = validateView(body);
      if (!view.ok) return json(view, 400, NO_STORE);
      let keep = view.view;
      if (keep === undefined) { const prev = await objectText(env, key); keep = prev ? normalizePlaces(prev)?.view : undefined; }
      const doc = placesDocument(checked.places, undefined, keep);
      const res = await signedFetch(s3(env), 'PUT', key, {}, JSON.stringify(doc), { 'content-type': 'application/json' });
      if (!res.ok) return json({ error: `storage refused the file (${res.status})` }, 502, NO_STORE);
      return json({ ok: true, updatedAt: doc.updatedAt, count: doc.places.length }, 200, NO_STORE);
    }
    return json({ error: 'method not allowed' }, 405, { ...NO_STORE, allow: 'GET,HEAD,PUT,OPTIONS' });
  }

  // ---- public reads ----
  let m = p.match(/^\/v1\/jobs\/([\w-]+)$/);
  if (m && req.method === 'GET') {
    const row = await env.DB.prepare('SELECT id,name,status,stage,progress,message,result,created_at,started_at,finished_at,updated_at FROM jobs WHERE id=?').bind(m[1]).first();
    if (!row) return json({ error: 'no such job' }, 404);
    return json({ ...row, result: row.result ? JSON.parse(row.result) : null, queuePosition: row.status === 'queued' ? (await env.DB.prepare("SELECT COUNT(*) n FROM jobs WHERE status='queued' AND created_at<?").bind(row.created_at).first()).n + 1 : 0 }, 200, { 'cache-control': 'no-store' });
  }
  // ---- a twin described for machines (agents, chatbots): built from the published files, cached briefly ----
  const sum = p.match(/^\/v1\/scenes\/([\w-]+)\/summary(\.txt|\.json)?$/);
  if (sum && (req.method === 'GET' || req.method === 'HEAD')) {
    const id = sum[1], row = await env.DB.prepare('SELECT status FROM jobs WHERE id = ?').bind(id).first() as any;
    if (!resultsVisible(row?.status, 'scene.json', r !== 'worker')) return json({ error: 'this twin is not published yet' }, 403);
    const read = async (f: string) => { const t = await objectText(env, `scenes/${id}/${f}`); try { return t ? JSON.parse(t) : null; } catch { return null; } };
    const [scene, occupancy, shell, places] = await Promise.all([read('scene.json'), read('occupancy.json'), read('shell.json'), read('places.json')]);
    if (!scene) return json({ error: 'no such twin' }, 404);
    const j = summarize({ id, site: 'https://ricco.codes/ply', api: new URL(req.url).origin, scene, occupancy, shell, places });
    const head = { 'cache-control': 'public, max-age=120' };
    if (sum[2] === '.txt' || (!sum[2] && /text\/plain/.test(req.headers.get('accept') ?? ''))) {
      return new Response(summaryText(j), { headers: { ...head, 'content-type': 'text/plain; charset=utf-8' } });
    }
    return json(j, 200, head);
  }

  m = p.match(/^\/v1\/scenes\/([\w-]+)\/([\w.\-\/]+)$/);  // subfolders allowed: lod/class-lod-3.radc
  // the live view's index changes every few seconds while a job runs; everything else is immutable or cache-busted
  if (m && (req.method === 'GET' || req.method === 'HEAD')) {
    // results are public only once a human has looked at them; a job still running is the operator's business
    const row = await env.DB.prepare('SELECT status FROM jobs WHERE id = ?').bind(m[1]).first() as any;
    if (!resultsVisible(row?.status, m[2], r !== 'worker')) return json({ error: 'this twin is not published yet' }, 403);
    // live-index.json, scene.json and places.json are small and change while a job is fixed or a pin is moved
    const nocache = m[2] === 'live-index.json' || m[2] === 'scene.json' || m[2] === 'places.json';
    return serve(env, `scenes/${m[1]}/${m[2]}`, req, nocache ? 'no-store' : 'public, max-age=3600', ctx);
  }

  // ---- uploads (app: captures/, worker: scenes/) ----
  if (p === '/v1/uploads' && req.method === 'POST') {
    if (!r) return json({ error: 'missing or wrong key' }, 401);
    const b = await req.json() as any;
    const key = r === 'app' ? `captures/${safeName(b.captureId)}/${safeName(b.name)}` : String(b.key ?? '');
    if (!allowed(key)) return json({ error: 'key not allowed' }, 403);
    const uploadId = await createMultipart(s3(env), key, b.type || 'application/octet-stream');
    return json({ key, uploadId });
  }
  if (p === '/v1/uploads/sign' && req.method === 'POST') {
    const b = await req.json() as any;
    if (!allowed(b.key)) return json({ error: 'not allowed' }, r ? 403 : 401);
    const parts: number[] = (b.parts ?? []).map(Number).filter((n: number) => n >= 1 && n <= 10000).slice(0, 200);
    const urls: Record<number, string> = {};
    for (const n of parts) urls[n] = await presign(s3(env), 'PUT', b.key, { partNumber: String(n), uploadId: String(b.uploadId) }, SIGN_TTL);
    return json({ urls });
  }
  if (p === '/v1/uploads/complete' && req.method === 'POST') {
    const b = await req.json() as any;
    if (!allowed(b.key)) return json({ error: 'not allowed' }, r ? 403 : 401);
    await completeMultipart(s3(env), b.key, String(b.uploadId), (b.parts ?? []).map((x: any) => ({ partNumber: Number(x.partNumber), etag: String(x.etag) })));
    return json({ key: b.key });
  }
  if (p === '/v1/files' && req.method === 'PUT') {
    const key = q.get('key') ?? '';
    if (!allowed(key)) return json({ error: 'not allowed' }, r ? 403 : 401);
    const body = await req.arrayBuffer();
    if (body.byteLength > 25e6) return json({ error: 'too large: use a multipart upload' }, 413);
    const res = await signedFetch(s3(env), 'PUT', key, {}, body, { 'content-type': req.headers.get('content-type') ?? 'application/octet-stream' });
    if (!res.ok) return json({ error: `storage refused the file (${res.status})` }, 502);
    return json({ key });
  }

  // ---- jobs ----
  // the library: every job this team has run, newest first (the app shows it after a team code)
  if (p === '/v1/jobs' && req.method === 'GET') {
    if (r !== 'app') return json({ error: 'missing or wrong key' }, 401);
    const rows = await env.DB.prepare('SELECT id,name,status,stage,progress,message,created_at,updated_at,result FROM jobs ORDER BY created_at DESC LIMIT 60').all();
    return json({
      jobs: (rows.results as any[]).map(x => {
        let files: string[] | null = null;
        try { const res = x.result ? JSON.parse(x.result) : null; files = res && res.files ? Object.keys(res.files) : null; } catch { /* older rows */ }
        return { id: x.id, name: x.name, status: x.status, stage: x.stage, progress: x.progress, message: x.message, createdAt: x.created_at, updatedAt: x.updated_at, files };
      })
    }, 200, { 'cache-control': 'no-store' });
  }

  if (p === '/v1/jobs' && req.method === 'POST') {
    if (r !== 'app') return json({ error: 'missing or wrong key' }, 401);
    const me = await identity(req, env);
    const b = await req.json() as any;
    const idem = String(req.headers.get('x-idem') ?? b.idem ?? '').slice(0, 120);
    if (idem) {  // the same submission twice is one job, not two runs paid for twice
      const seen = await env.DB.prepare('SELECT id FROM jobs WHERE idem = ?').bind(idem).first() as any;
      if (seen) return json({ jobId: seen.id, duplicate: true }, 200);
    }
    if (me.owner && me.owner !== 'team') {
      const active = await env.DB.prepare("SELECT COUNT(*) n FROM jobs WHERE owner = ? AND status IN ('queued','running')").bind(me.owner).first() as any;
      const today = await env.DB.prepare('SELECT COUNT(*) n FROM jobs WHERE owner = ? AND created_at > ?').bind(me.owner, now - 86_400_000).first() as any;
      const over = inviteBudget(active?.n ?? 0, today?.n ?? 0);
      if (over) return json({ error: over }, 429);
    }
    // uploaded clips under captures/<id>/, or shared Google Drive files 'gdrive:<fileId>:<name>' (the worker fetches those itself)
    const files: string[] = (b.files ?? []).filter((k: string) => typeof k === 'string' && (k.startsWith(`captures/${safeName(b.captureId)}/`) || /^gdrive:[\w-]{20,}:[\w.\-]{1,120}$/.test(k)));
    if (!files.length) return json({ error: 'no uploaded files for this capture' }, 400);
    const jobId = id();
    const assignTo = b.worker && me.owner === 'team' ? String(b.worker).slice(0, 60) : null;  // a paired PC (team key only)
    await env.DB.prepare('INSERT INTO jobs (id,name,capture_id,settings,args,files,status,progress,created_at,updated_at,owner,idem,assigned_worker) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)')
      .bind(jobId, String(b.name ?? 'Capture').slice(0, 80), safeName(b.captureId), JSON.stringify(b.settings ?? {}), JSON.stringify((b.args ?? []).map(String)), JSON.stringify(files), 'queued', 0, now, now, me.owner || null, idem || null, assignTo).run();
    if (!assignTo && !safeName(b.captureId).startsWith('rehearsal-')) ctx.waitUntil(kickModal(env, `job ${jobId}`)); // rehearsals are for Kaggle CPU
    return json({ jobId }, 201);
  }

  // ---- the human gate: a finished run waits here until someone decides ----
  const decision = p.match(/^\/v1\/jobs\/([\w-]{6,64})\/decision$/);
  if (decision && req.method === 'POST') {
    if (r !== 'app') return json({ error: 'missing or wrong invite' }, 401);
    const b = await req.json().catch(() => ({})) as any;
    const approve = String(b.decision ?? '') === 'approve';
    const note = String(b.note ?? '').slice(0, 500);
    const row = await env.DB.prepare('SELECT id,status,args,result FROM jobs WHERE id = ?').bind(decision[1]).first() as any;
    if (!row) return json({ error: 'no such job' }, 404);
    if (row.status !== 'review') return json({ error: `job is ${row.status}, not in review` }, 409);
    if (approve) {
      await env.DB.prepare("UPDATE jobs SET status='done', message=?, updated_at=? WHERE id=?")
        .bind(note ? `approved: ${note}` : 'approved', now, row.id).run();
      return json({ ok: true, status: 'done' });
    }
    // rejected: back in the queue, and the next run starts from the frames again
    await env.DB.prepare("UPDATE jobs SET status='queued', progress=0, stage=NULL, message=?, updated_at=?, args=? WHERE id=?")
      .bind(note ? `rejected: ${note}` : 'rejected, queued again', now,
        JSON.stringify([...JSON.parse(row.args || '[]').filter((a: string) => a !== '--force' && a !== 'frames'), '--force', 'frames']), row.id).run();
    return json({ ok: true, status: 'queued' });
  }

  // ---- GPU worker ----
  if (p.startsWith('/v1/worker/')) {
    if (r !== 'worker') return json({ error: 'worker token required' }, 401);
    if (mach && p === '/v1/worker/kick') return json({ error: 'not for paired machines' }, 403);
    if (p === '/v1/worker/kick' && req.method === 'POST') return json({ ok: true, modal: await kickModal(env, 'manual') });
    if (p === '/v1/worker/claim' && req.method === 'POST') {
      const b = await req.json().catch(() => ({})) as any;
      const wname = String(b.worker ?? 'worker').slice(0, 60);
      // a job whose worker died twice (crash, out of memory, stalled) is failed for good instead of being picked up
      // again and again: each retry costs GPU time
      await env.DB.prepare(`UPDATE jobs SET status='failed', message='Stopped after 2 attempts: the GPU worker crashed or stalled each time. Check the capture, then queue it again.', finished_at=?, updated_at=?
        WHERE status='running' AND updated_at < ? AND attempts >= ?`).bind(now, now, now - STALE_MS, MAX_ATTEMPTS).run();
      // a paired PC takes only what was assigned to it by name; the shared-token workers (Modal, Kaggle) only the rest
      const mine = mach ? 'assigned_worker = ?' : 'assigned_worker IS NULL';
      const claimer = mach ? mach.name : wname;
      const row = await env.DB.prepare(`UPDATE jobs SET status='running', worker=?, stage=NULL, progress=0, message='claimed', started_at=?, updated_at=?, attempts=attempts+1
        WHERE id = (SELECT id FROM jobs WHERE (status='queued' OR (status='running' AND updated_at < ? AND attempts < ?)) AND capture_id LIKE ? AND ${mine} ORDER BY created_at LIMIT 1) RETURNING *`)
        .bind(claimer, now, now, now - STALE_MS, MAX_ATTEMPTS, `${String(b.prefix ?? '').replace(/[%_]/g, '')}%`, ...(mach ? [mach.name] : [])).first();
      if (mach && b.specs) await env.DB.prepare('UPDATE machines SET specs=? WHERE name=?').bind(trimSpecs(b.specs), mach.name).run();
      if (!claimer.startsWith('preflight')) // the pre-flight check's token probe is not a worker
        await env.DB.prepare('INSERT INTO workers (name,last_seen,busy_job,note) VALUES (?,?,?,NULL) ON CONFLICT(name) DO UPDATE SET last_seen=excluded.last_seen, busy_job=excluded.busy_job, note=NULL').bind(claimer, now, row?.id ?? null).run();
      if (!row) return new Response(null, { status: 204 });
      return json({ ...row, settings: JSON.parse(row.settings), args: JSON.parse(row.args), files: JSON.parse(row.files) });
    }
    if (p === '/v1/worker/hello' && req.method === 'POST') {  // a starting worker says what it is doing (install progress)
      const b = await req.json().catch(() => ({})) as any;
      await env.DB.prepare('INSERT INTO workers (name,last_seen,busy_job,note) VALUES (?,?,NULL,?) ON CONFLICT(name) DO UPDATE SET last_seen=excluded.last_seen, note=excluded.note').bind(mach ? mach.name : String(b.worker ?? 'worker').slice(0, 60), now, String(b.note ?? '').slice(0, 200)).run();
      return json({ ok: true });
    }
    m = p.match(/^\/v1\/worker\/jobs\/([\w-]+)$/);
    if (m && req.method === 'POST') {
      if (mach && m[1] !== mach.busyJob) return json({ error: 'not your job' }, 403);
      const b = await req.json() as any, sets: string[] = ['updated_at=?'], vals: unknown[] = [now];
      if (b.status) { sets.push('status=?'); vals.push(b.status); if (b.status === 'done' || b.status === 'failed') { sets.push('finished_at=?'); vals.push(now); } }
      // a worker re-queueing its own job is a planned hand-over (the CPU camera solve hands back to a GPU), not a crash:
      // the next claim must not use up the job's crash retries
      if (b.status === 'queued') sets.push('attempts=0');
      // paused (a restore point saved, the job handed back): counted, and failed past MAX_PAUSES
      if (b.status === 'queued' && b.paused) {
        const row = await env.DB.prepare('SELECT pauses FROM jobs WHERE id=?').bind(m[1]).first() as any;
        if ((row?.pauses ?? 0) + 1 > MAX_PAUSES) {
          await env.DB.prepare("UPDATE jobs SET status='failed', message=?, finished_at=?, updated_at=? WHERE id=?")
            .bind(`stopped after ${MAX_PAUSES} pauses: its workers keep being stopped. Its restore point is kept; queue it again to continue.`, now, now, m[1]).run();
          return json({ ok: true, failed: 'too many pauses' });
        }
        sets.push('pauses=pauses+1');
      }
      if (b.stage !== undefined) { sets.push('stage=?'); vals.push(b.stage); }
      if (b.progress !== undefined) { sets.push('progress=?'); vals.push(Math.max(0, Math.min(1, Number(b.progress) || 0))); }
      if (b.message !== undefined) { sets.push('message=?'); vals.push(String(b.message).slice(0, 4000)); }
      if (b.result !== undefined) { sets.push('result=?'); vals.push(JSON.stringify(b.result)); }
      await env.DB.prepare(`UPDATE jobs SET ${sets.join(',')} WHERE id=?`).bind(...vals, m[1]).run();
      await env.DB.prepare('UPDATE workers SET last_seen=? WHERE busy_job=?').bind(now, m[1]).run();
      return json({ ok: true });
    }
    if (mach && p === '/v1/worker/file' && !machineReadAllowed(q.get('key') ?? '', mach.busyJob, mach.files)) return json({ error: 'not an input of your job' }, 403);
    if (p === '/v1/worker/file' && req.method === 'GET') return json({ url: await presign(s3(env), 'GET', q.get('key') ?? '', {}, SIGN_TTL) });
  }
  return json({ error: 'not found' }, 404);
}

/** Results for the viewer: fetched from the bucket with a signed URL and streamed through, Range passed along. */
async function serve(env: Env, key: string, req: Request, cache: string, ctx?: Ctx) {
  // splats, meshes and LOD chunks: the browser fetches them from Hugging Face's CDN (src/cdn.ts), not through here
  if (env.HF_REPO && env.HF_TOKEN && key.startsWith('scenes/') && isBulk(key)) {
    const r = await cdnRedirect(env.HF_REPO, env.HF_TOKEN, key, req.url, edge(), ctx ? p => ctx.waitUntil(p) : undefined);
    if (r) return r;
  }
  // Whole-file GETs are cached at Cloudflare's edge. The bucket's free tier counts every download (1 GB/day),
  // and the same twin is fetched again on every view; a cache hit costs neither bucket bandwidth nor quota.
  const range = req.headers.get('range');
  const cacheable = !!ctx && !range && req.method === 'GET' && cache !== 'no-store';
  const cacheKey = cacheable ? new Request(new URL(req.url).toString(), { method: 'GET' }) : null;
  if (cacheKey) {
    const hit = await edge().match(cacheKey);
    if (hit) { const h = new Headers(hit.headers); h.set('x-ply-cache', 'hit'); return new Response(hit.body, { status: hit.status, headers: h }); }
  }
  let up: Response | null = null;
  // results live on Hugging Face when configured (100 GB, no daily download cap); inputs and old jobs stay on the bucket
  if (env.HF_REPO && env.HF_TOKEN && key.startsWith('scenes/')) {
    const h = new Headers({ authorization: `Bearer ${env.HF_TOKEN}` });
    if (range) h.set('range', range);
    up = await fetch(`https://huggingface.co/datasets/${env.HF_REPO}/resolve/main/${key}`, { method: req.method, headers: h });
    if (up.status === 404 || up.status === 403 || up.status === 401) up = null;
  }
  if (!up) {
    const url = await presign(s3(env), req.method === 'HEAD' ? 'HEAD' : 'GET', key, {}, 600);
    up = await fetch(url, { method: req.method, headers: range ? { range } : {} });
  }
  if (up.status === 404 || up.status === 403) return json({ error: 'not found' }, 404);
  const h = new Headers({ 'cache-control': cache, 'accept-ranges': 'bytes', 'x-ply-cache': 'miss' });
  for (const k of ['content-type', 'content-length', 'content-range', 'etag', 'last-modified']) { const v = up.headers.get(k); if (v) h.set(k, v); }
  const resp = new Response(up.body, { status: up.status, headers: h });
  if (cacheKey && up.ok) ctx!.waitUntil(edge().put(cacheKey, resp.clone()));
  return resp;
}

/** Text of one object, read around the edge cache entirely: the bucket first, then Hugging Face.
    The bucket is where an interactive edit lands (PUT /v1/scenes/:id/places); HF is where the GPU worker publishes
    when PLY_HF_REPO is set, so both are consulted. A scene id is a job id, so a published scene never gets a second
    places.json from a re-run: whichever copy exists for that id is the current one. */
async function objectText(env: Env, key: string): Promise<string | null> {
  const r = await fetch(await presign(s3(env), 'GET', key, {}, 600), { method: 'GET' });
  if (r.ok) return await r.text();
  if (env.HF_REPO && env.HF_TOKEN && key.startsWith('scenes/')) {
    const h = await fetch(`https://huggingface.co/datasets/${env.HF_REPO}/resolve/main/${key}`, { headers: { authorization: `Bearer ${env.HF_TOKEN}` } });
    if (h.ok) return await h.text();
  }
  return null;
}
