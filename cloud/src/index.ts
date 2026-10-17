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
 *    GET  /v1/scenes/:jobId/:file                                -> results (Range supported) for the viewer
 *  GPU worker (Authorization: Bearer WORKER_TOKEN)
 *    POST /v1/worker/claim       {worker, prefix?}               -> next job or 204       stale running jobs (no heartbeat 20 min) are re-queued;
 *                                                                  prefix limits it to captures whose id starts with it (rehearsals)
 *    POST /v1/worker/jobs/:id    {status?, stage?, progress?, message?, result?}          heartbeat + progress
 *    GET  /v1/worker/file        ?key                            -> {url} signed download link (inputs)
 *    (and the upload routes above, which accept any key under scenes/<jobId>/ for results)
 */

import { completeMultipart, createMultipart, presign, signedFetch, type S3Conf } from './s3';

interface Env { DB: any; APP_KEY: string; WORKER_TOKEN: string; ALLOWED_ORIGINS: string;
  S3_ENDPOINT: string; S3_REGION: string; S3_BUCKET: string; S3_KEY_ID: string; S3_SECRET: string;
  /** Modal GPU worker (pipeline/modal/ply_modal.py): trigger URL + shared token; unset = Kaggle only */
  MODAL_TRIGGER?: string; MODAL_TRIGGER_TOKEN?: string; }
type Ctx = { waitUntil(p: Promise<unknown>): void };

/** Wake the Modal GPU worker (it starts unless one is already up, and stops itself when the queue is empty). */
async function kickModal(env: Env, why: string) {
  if (!env.MODAL_TRIGGER || !env.MODAL_TRIGGER_TOKEN) return null;
  try { const r = await fetch(env.MODAL_TRIGGER, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ token: env.MODAL_TRIGGER_TOKEN, why }) }); return await r.json(); }
  catch (e) { console.log('modal trigger failed', (e as Error).message); return null; }
}
const s3 = (e: Env): S3Conf => ({ endpoint: e.S3_ENDPOINT, region: e.S3_REGION, bucket: e.S3_BUCKET, keyId: e.S3_KEY_ID, secret: e.S3_SECRET });
const SIGN_TTL = 6 * 3600;

const STALE_MS = 20 * 60 * 1000;
const MAX_ATTEMPTS = 2;
const json = (data: unknown, status = 200, h: Record<string, string> = {}) => new Response(JSON.stringify(data), { status, headers: { 'content-type': 'application/json', ...h } });
const safeName = (s: string) => s.replace(/[^\w.\-]+/g, '_').slice(0, 120) || 'file';
const id = () => crypto.randomUUID();

function cors(req: Request, env: Env) {
  const o = req.headers.get('origin') ?? '', ok = env.ALLOWED_ORIGINS.split(',').includes(o) || /^https?:\/\/(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?$/.test(o);
  return {
    'access-control-allow-origin': ok ? o : '*', vary: 'origin',
    'access-control-allow-methods': 'GET,HEAD,POST,PUT,OPTIONS', 'access-control-allow-headers': 'content-type,x-ply-key,authorization,range',
    'access-control-expose-headers': 'content-length,content-range,accept-ranges,etag', 'access-control-max-age': '86400',
  };
}

type Role = 'app' | 'worker' | null;
function role(req: Request, env: Env): Role {
  if (env.WORKER_TOKEN && req.headers.get('authorization') === `Bearer ${env.WORKER_TOKEN}`) return 'worker';
  if (env.APP_KEY && req.headers.get('x-ply-key') === env.APP_KEY) return 'app';
  return null;
}
/** The app may only write its own captures; the worker may only write results. */
function keyAllowed(key: string, r: Role) {
  if (key.includes('..') || key.startsWith('/')) return false;
  return r === 'app' ? key.startsWith('captures/') : r === 'worker' ? key.startsWith('scenes/') : false;
}

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
    const waiting = await env.DB.prepare(`SELECT COUNT(*) n FROM jobs WHERE capture_id NOT LIKE 'rehearsal-%' AND
      ((status='queued' AND created_at < ?) OR (status='running' AND updated_at < ? AND attempts < ?))`).bind(now - 120_000, now - STALE_MS, MAX_ATTEMPTS).first();
    const live = await env.DB.prepare('SELECT COUNT(*) n FROM workers WHERE last_seen > ?').bind(now - 90_000).first();
    if (waiting.n > 0 && live.n === 0) ctx.waitUntil(kickModal(env, 'cron'));
  },
};

async function route(req: Request, env: Env, ctx: Ctx): Promise<Response> {
  const url = new URL(req.url), p = url.pathname, q = url.searchParams, r = role(req, env), now = Date.now();

  if (p === '/' || p === '/v1') return json({ ok: true, service: 'ply-api' });
  if (p === '/v1/health' && req.method === 'GET') {
    const w = await env.DB.prepare('SELECT name,last_seen,busy_job,note FROM workers WHERE last_seen > ? ORDER BY last_seen DESC').bind(now - 90_000).all();
    const q = await env.DB.prepare("SELECT COUNT(*) n FROM jobs WHERE status='queued'").first();
    return json({ ok: true, workersOnline: w.results.filter((x: any) => !x.note).length, workersStarting: w.results.filter((x: any) => x.note).length, workers: w.results.map((x: any) => ({ name: x.name, busy: !!x.busy_job, starting: x.note || null, seenSecondsAgo: Math.round((now - x.last_seen) / 1000) })), queued: q.n }, 200, { 'cache-control': 'no-store' });
  }

  // ---- public reads ----
  let m = p.match(/^\/v1\/jobs\/([\w-]+)$/);
  if (m && req.method === 'GET') {
    const row = await env.DB.prepare('SELECT id,name,status,stage,progress,message,result,created_at,started_at,finished_at,updated_at FROM jobs WHERE id=?').bind(m[1]).first();
    if (!row) return json({ error: 'no such job' }, 404);
    return json({ ...row, result: row.result ? JSON.parse(row.result) : null, queuePosition: row.status === 'queued' ? (await env.DB.prepare("SELECT COUNT(*) n FROM jobs WHERE status='queued' AND created_at<?").bind(row.created_at).first()).n + 1 : 0 }, 200, { 'cache-control': 'no-store' });
  }
  m = p.match(/^\/v1\/scenes\/([\w-]+)\/([\w.\-]+)$/);
  // the live view's index changes every few seconds while a job runs; everything else is immutable or cache-busted
  if (m && (req.method === 'GET' || req.method === 'HEAD')) return serve(env, `scenes/${m[1]}/${m[2]}`, req, m[2] === 'live-index.json' ? 'no-store' : 'public, max-age=3600', ctx);

  // ---- uploads (app: captures/, worker: scenes/) ----
  if (p === '/v1/uploads' && req.method === 'POST') {
    if (!r) return json({ error: 'missing or wrong key' }, 401);
    const b = await req.json() as any;
    const key = r === 'app' ? `captures/${safeName(b.captureId)}/${safeName(b.name)}` : String(b.key ?? '');
    if (!keyAllowed(key, r)) return json({ error: 'key not allowed' }, 403);
    const uploadId = await createMultipart(s3(env), key, b.type || 'application/octet-stream');
    return json({ key, uploadId });
  }
  if (p === '/v1/uploads/sign' && req.method === 'POST') {
    const b = await req.json() as any;
    if (!keyAllowed(b.key, r)) return json({ error: 'not allowed' }, r ? 403 : 401);
    const parts: number[] = (b.parts ?? []).map(Number).filter((n: number) => n >= 1 && n <= 10000).slice(0, 200);
    const urls: Record<number, string> = {};
    for (const n of parts) urls[n] = await presign(s3(env), 'PUT', b.key, { partNumber: String(n), uploadId: String(b.uploadId) }, SIGN_TTL);
    return json({ urls });
  }
  if (p === '/v1/uploads/complete' && req.method === 'POST') {
    const b = await req.json() as any;
    if (!keyAllowed(b.key, r)) return json({ error: 'not allowed' }, r ? 403 : 401);
    await completeMultipart(s3(env), b.key, String(b.uploadId), (b.parts ?? []).map((x: any) => ({ partNumber: Number(x.partNumber), etag: String(x.etag) })));
    return json({ key: b.key });
  }
  if (p === '/v1/files' && req.method === 'PUT') {
    const key = q.get('key') ?? '';
    if (!keyAllowed(key, r)) return json({ error: 'not allowed' }, r ? 403 : 401);
    const body = await req.arrayBuffer();
    if (body.byteLength > 25e6) return json({ error: 'too large: use a multipart upload' }, 413);
    const res = await signedFetch(s3(env), 'PUT', key, {}, body, { 'content-type': req.headers.get('content-type') ?? 'application/octet-stream' });
    if (!res.ok) return json({ error: `storage refused the file (${res.status})` }, 502);
    return json({ key });
  }

  // ---- jobs ----
  if (p === '/v1/jobs' && req.method === 'POST') {
    if (r !== 'app') return json({ error: 'missing or wrong key' }, 401);
    const b = await req.json() as any;
    // uploaded clips under captures/<id>/, or shared Google Drive files 'gdrive:<fileId>:<name>' (the worker fetches those itself)
    const files: string[] = (b.files ?? []).filter((k: string) => typeof k === 'string' && (k.startsWith(`captures/${safeName(b.captureId)}/`) || /^gdrive:[\w-]{20,}:[\w.\-]{1,120}$/.test(k)));
    if (!files.length) return json({ error: 'no uploaded files for this capture' }, 400);
    const jobId = id();
    await env.DB.prepare('INSERT INTO jobs (id,name,capture_id,settings,args,files,status,progress,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)')
      .bind(jobId, String(b.name ?? 'Capture').slice(0, 80), safeName(b.captureId), JSON.stringify(b.settings ?? {}), JSON.stringify((b.args ?? []).map(String)), JSON.stringify(files), 'queued', 0, now, now).run();
    if (!safeName(b.captureId).startsWith('rehearsal-')) ctx.waitUntil(kickModal(env, `job ${jobId}`)); // rehearsals are for Kaggle CPU
    return json({ jobId }, 201);
  }

  // ---- GPU worker ----
  if (p.startsWith('/v1/worker/')) {
    if (r !== 'worker') return json({ error: 'worker token required' }, 401);
    if (p === '/v1/worker/kick' && req.method === 'POST') return json({ ok: true, modal: await kickModal(env, 'manual') });
    if (p === '/v1/worker/claim' && req.method === 'POST') {
      const b = await req.json().catch(() => ({})) as any;
      const wname = String(b.worker ?? 'worker').slice(0, 60);
      // a job whose worker died twice (crash, out of memory, stalled) is failed for good instead of being picked up
      // again and again: each retry costs GPU time
      await env.DB.prepare(`UPDATE jobs SET status='failed', message='Stopped after 2 attempts: the GPU worker crashed or stalled each time. Check the capture, then queue it again.', finished_at=?, updated_at=?
        WHERE status='running' AND updated_at < ? AND attempts >= ?`).bind(now, now, now - STALE_MS, MAX_ATTEMPTS).run();
      const row = await env.DB.prepare(`UPDATE jobs SET status='running', worker=?, stage=NULL, progress=0, message='claimed', started_at=?, updated_at=?, attempts=attempts+1
        WHERE id = (SELECT id FROM jobs WHERE (status='queued' OR (status='running' AND updated_at < ? AND attempts < ?)) AND capture_id LIKE ? ORDER BY created_at LIMIT 1) RETURNING *`)
        .bind(String(b.worker ?? 'worker').slice(0, 60), now, now, now - STALE_MS, MAX_ATTEMPTS, `${String(b.prefix ?? '').replace(/[%_]/g, '')}%`).first();
      if (!wname.startsWith('preflight')) // the pre-flight check's token probe is not a worker
        await env.DB.prepare('INSERT INTO workers (name,last_seen,busy_job,note) VALUES (?,?,?,NULL) ON CONFLICT(name) DO UPDATE SET last_seen=excluded.last_seen, busy_job=excluded.busy_job, note=NULL').bind(wname, now, row?.id ?? null).run();
      if (!row) return new Response(null, { status: 204 });
      return json({ ...row, settings: JSON.parse(row.settings), args: JSON.parse(row.args), files: JSON.parse(row.files) });
    }
    if (p === '/v1/worker/hello' && req.method === 'POST') {  // a starting worker says what it is doing (install progress)
      const b = await req.json().catch(() => ({})) as any;
      await env.DB.prepare('INSERT INTO workers (name,last_seen,busy_job,note) VALUES (?,?,NULL,?) ON CONFLICT(name) DO UPDATE SET last_seen=excluded.last_seen, note=excluded.note').bind(String(b.worker ?? 'worker').slice(0, 60), now, String(b.note ?? '').slice(0, 200)).run();
      return json({ ok: true });
    }
    m = p.match(/^\/v1\/worker\/jobs\/([\w-]+)$/);
    if (m && req.method === 'POST') {
      const b = await req.json() as any, sets: string[] = ['updated_at=?'], vals: unknown[] = [now];
      if (b.status) { sets.push('status=?'); vals.push(b.status); if (b.status === 'done' || b.status === 'failed') { sets.push('finished_at=?'); vals.push(now); } }
      if (b.stage !== undefined) { sets.push('stage=?'); vals.push(b.stage); }
      if (b.progress !== undefined) { sets.push('progress=?'); vals.push(Math.max(0, Math.min(1, Number(b.progress) || 0))); }
      if (b.message !== undefined) { sets.push('message=?'); vals.push(String(b.message).slice(0, 4000)); }
      if (b.result !== undefined) { sets.push('result=?'); vals.push(JSON.stringify(b.result)); }
      await env.DB.prepare(`UPDATE jobs SET ${sets.join(',')} WHERE id=?`).bind(...vals, m[1]).run();
      await env.DB.prepare('UPDATE workers SET last_seen=? WHERE busy_job=?').bind(now, m[1]).run();
      return json({ ok: true });
    }
    if (p === '/v1/worker/file' && req.method === 'GET') return json({ url: await presign(s3(env), 'GET', q.get('key') ?? '', {}, SIGN_TTL) });
  }
  return json({ error: 'not found' }, 404);
}

/** Results for the viewer: fetched from the bucket with a signed URL and streamed through, Range passed along. */
async function serve(env: Env, key: string, req: Request, cache: string, ctx?: ExecutionContext) {
  // Whole-file GETs are cached at Cloudflare's edge. The bucket's free tier counts every download (1 GB/day),
  // and the same twin is fetched again on every view; a cache hit costs neither bucket bandwidth nor quota.
  const range = req.headers.get('range');
  const cacheable = !!ctx && !range && req.method === 'GET' && cache !== 'no-store';
  const cacheKey = cacheable ? new Request(new URL(req.url).toString(), { method: 'GET' }) : null;
  if (cacheKey) {
    const hit = await caches.default.match(cacheKey);
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
  if (cacheKey && up.ok) ctx!.waitUntil(caches.default.put(cacheKey, resp.clone()));
  return resp;
}
