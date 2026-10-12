/* Ply API: one Cloudflare Worker for uploads (R2 multipart, resumable), the job queue (D1) and serving results.
 *
 *  Capture app (header X-Ply-Key: APP_KEY)
 *    POST /v1/uploads            {captureId, name, type}        -> {key, uploadId}       start a multipart upload
 *    PUT  /v1/uploads/part       ?key&uploadId&part=N  <bytes>   -> {partNumber, etag}
 *    POST /v1/uploads/complete   {key, uploadId, parts}          -> {key, size}
 *    PUT  /v1/files              ?key  <bytes>  (≤ 50 MB)        -> {key}                capture.json and other small files
 *    POST /v1/jobs               {captureId, name, settings, args, files} -> {jobId}
 *  Anyone
 *    GET  /v1/jobs/:id                                           -> status, stage, progress, result
 *    GET  /v1/scenes/:jobId/:file                                -> results (Range supported) for the viewer
 *  GPU worker (Authorization: Bearer WORKER_TOKEN)
 *    POST /v1/worker/claim       {worker}                        -> next job or 204       stale running jobs (no heartbeat 20 min) are re-queued
 *    POST /v1/worker/jobs/:id    {status?, stage?, progress?, message?, result?}          heartbeat + progress
 *    GET  /v1/worker/file        ?key                            -> the object (inputs)
 *    (and the upload routes above, which accept any key under scenes/<jobId>/ for results)
 */

interface Env { BUCKET: any; DB: any; APP_KEY: string; WORKER_TOKEN: string; ALLOWED_ORIGINS: string; }

const STALE_MS = 20 * 60 * 1000;
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
  async fetch(req: Request, env: Env): Promise<Response> {
    const c = cors(req, env);
    if (req.method === 'OPTIONS') return new Response(null, { status: 204, headers: c });
    try { const r = await route(req, env); const h = new Headers(r.headers); for (const [k, v] of Object.entries(c)) h.set(k, v); return new Response(r.body, { status: r.status, headers: h }); }
    catch (e) { return json({ error: (e as Error).message || 'server error' }, 500, c); }
  },
};

async function route(req: Request, env: Env): Promise<Response> {
  const url = new URL(req.url), p = url.pathname, q = url.searchParams, r = role(req, env), now = Date.now();

  if (p === '/' || p === '/v1') return json({ ok: true, service: 'ply-api' });

  // ---- public reads ----
  let m = p.match(/^\/v1\/jobs\/([\w-]+)$/);
  if (m && req.method === 'GET') {
    const row = await env.DB.prepare('SELECT id,name,status,stage,progress,message,result,created_at,started_at,finished_at,updated_at FROM jobs WHERE id=?').bind(m[1]).first();
    if (!row) return json({ error: 'no such job' }, 404);
    return json({ ...row, result: row.result ? JSON.parse(row.result) : null, queuePosition: row.status === 'queued' ? (await env.DB.prepare("SELECT COUNT(*) n FROM jobs WHERE status='queued' AND created_at<?").bind(row.created_at).first()).n + 1 : 0 }, 200, { 'cache-control': 'no-store' });
  }
  m = p.match(/^\/v1\/scenes\/([\w-]+)\/([\w.\-]+)$/);
  if (m && (req.method === 'GET' || req.method === 'HEAD')) return serve(env, `scenes/${m[1]}/${m[2]}`, req, 'public, max-age=3600');

  // ---- uploads (app: captures/, worker: scenes/) ----
  if (p === '/v1/uploads' && req.method === 'POST') {
    if (!r) return json({ error: 'missing or wrong key' }, 401);
    const b = await req.json() as any;
    const key = r === 'app' ? `captures/${safeName(b.captureId)}/${safeName(b.name)}` : String(b.key ?? '');
    if (!keyAllowed(key, r)) return json({ error: 'key not allowed' }, 403);
    const up = await env.BUCKET.createMultipartUpload(key, { httpMetadata: { contentType: b.type || 'application/octet-stream' } });
    return json({ key, uploadId: up.uploadId });
  }
  if (p === '/v1/uploads/part' && req.method === 'PUT') {
    const key = q.get('key') ?? '', uploadId = q.get('uploadId') ?? '', part = Number(q.get('part'));
    if (!keyAllowed(key, r)) return json({ error: 'not allowed' }, r ? 403 : 401);
    if (!(part >= 1 && part <= 10000) || !req.body) return json({ error: 'bad part' }, 400);
    const done = await env.BUCKET.resumeMultipartUpload(key, uploadId).uploadPart(part, req.body);
    return json({ partNumber: done.partNumber, etag: done.etag });
  }
  if (p === '/v1/uploads/complete' && req.method === 'POST') {
    const b = await req.json() as any;
    if (!keyAllowed(b.key, r)) return json({ error: 'not allowed' }, r ? 403 : 401);
    const obj = await env.BUCKET.resumeMultipartUpload(b.key, b.uploadId).complete(b.parts);
    return json({ key: b.key, size: obj.size });
  }
  if (p === '/v1/files' && req.method === 'PUT') {
    const key = q.get('key') ?? '';
    if (!keyAllowed(key, r)) return json({ error: 'not allowed' }, r ? 403 : 401);
    if (Number(req.headers.get('content-length') ?? 0) > 50e6) return json({ error: 'too large: use a multipart upload' }, 413);
    await env.BUCKET.put(key, req.body, { httpMetadata: { contentType: req.headers.get('content-type') ?? 'application/octet-stream' } });
    return json({ key });
  }

  // ---- jobs ----
  if (p === '/v1/jobs' && req.method === 'POST') {
    if (r !== 'app') return json({ error: 'missing or wrong key' }, 401);
    const b = await req.json() as any;
    const files: string[] = (b.files ?? []).filter((k: string) => typeof k === 'string' && k.startsWith(`captures/${safeName(b.captureId)}/`));
    if (!files.length) return json({ error: 'no uploaded files for this capture' }, 400);
    const jobId = id();
    await env.DB.prepare('INSERT INTO jobs (id,name,capture_id,settings,args,files,status,progress,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)')
      .bind(jobId, String(b.name ?? 'Capture').slice(0, 80), safeName(b.captureId), JSON.stringify(b.settings ?? {}), JSON.stringify((b.args ?? []).map(String)), JSON.stringify(files), 'queued', 0, now, now).run();
    return json({ jobId }, 201);
  }

  // ---- GPU worker ----
  if (p.startsWith('/v1/worker/')) {
    if (r !== 'worker') return json({ error: 'worker token required' }, 401);
    if (p === '/v1/worker/claim' && req.method === 'POST') {
      const b = await req.json().catch(() => ({})) as any;
      const row = await env.DB.prepare(`UPDATE jobs SET status='running', worker=?, stage=NULL, progress=0, message='claimed', started_at=?, updated_at=?
        WHERE id = (SELECT id FROM jobs WHERE status='queued' OR (status='running' AND updated_at < ?) ORDER BY created_at LIMIT 1) RETURNING *`)
        .bind(String(b.worker ?? 'worker').slice(0, 60), now, now, now - STALE_MS).first();
      if (!row) return new Response(null, { status: 204 });
      return json({ ...row, settings: JSON.parse(row.settings), args: JSON.parse(row.args), files: JSON.parse(row.files) });
    }
    m = p.match(/^\/v1\/worker\/jobs\/([\w-]+)$/);
    if (m && req.method === 'POST') {
      const b = await req.json() as any, sets: string[] = ['updated_at=?'], vals: unknown[] = [now];
      if (b.status) { sets.push('status=?'); vals.push(b.status); if (b.status === 'done' || b.status === 'failed') { sets.push('finished_at=?'); vals.push(now); } }
      if (b.stage !== undefined) { sets.push('stage=?'); vals.push(b.stage); }
      if (b.progress !== undefined) { sets.push('progress=?'); vals.push(Math.max(0, Math.min(1, Number(b.progress) || 0))); }
      if (b.message !== undefined) { sets.push('message=?'); vals.push(String(b.message).slice(0, 500)); }
      if (b.result !== undefined) { sets.push('result=?'); vals.push(JSON.stringify(b.result)); }
      await env.DB.prepare(`UPDATE jobs SET ${sets.join(',')} WHERE id=?`).bind(...vals, m[1]).run();
      return json({ ok: true });
    }
    if (p === '/v1/worker/file' && req.method === 'GET') return serve(env, q.get('key') ?? '', req, 'no-store');
  }
  return json({ error: 'not found' }, 404);
}

/** Stream an R2 object with Range support (the viewer and ffmpeg both seek). */
async function serve(env: Env, key: string, req: Request, cache: string) {
  const range = req.headers.get('range'); let opts: any = {};
  const mr = range?.match(/bytes=(\d*)-(\d*)/);
  if (mr) { const s = mr[1] ? Number(mr[1]) : undefined, e = mr[2] ? Number(mr[2]) : undefined; opts.range = s !== undefined ? { offset: s, ...(e !== undefined ? { length: e - s + 1 } : {}) } : { suffix: e }; }
  const obj = req.method === 'HEAD' ? await env.BUCKET.head(key) : await env.BUCKET.get(key, opts);
  if (!obj) return json({ error: 'not found' }, 404);
  const h = new Headers({ 'content-type': obj.httpMetadata?.contentType ?? 'application/octet-stream', 'cache-control': cache, 'accept-ranges': 'bytes', etag: obj.httpEtag });
  if (req.method === 'HEAD') { h.set('content-length', String(obj.size)); return new Response(null, { headers: h }); }
  if (opts.range && obj.range) { const off = obj.range.offset ?? (obj.size - obj.range.suffix), len = obj.range.length ?? (obj.size - off); h.set('content-range', `bytes ${off}-${off + len - 1}/${obj.size}`); h.set('content-length', String(len)); return new Response(obj.body, { status: 206, headers: h }); }
  h.set('content-length', String(obj.size)); return new Response(obj.body, { headers: h });
}
