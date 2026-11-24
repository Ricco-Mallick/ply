/* The places API: public read, owner write, hard validation, and a round trip through storage.
   Run: node --experimental-strip-types test/places.test.mts
   The worker is exercised through its real fetch handler; only the two storage endpoints (the S3 bucket and Hugging
   Face) and the edge cache are stand-ins, so the routing, auth, validation and headers are the deployed ones. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import api from '../src/index.ts';

const BASE = 'https://api.test';
const bucket = new Map<string, string>();
const hf = new Map<string, string>();
let puts: { key: string; body: string; contentType: string | null }[] = [];
let cacheCalls = 0;

/** Stand-in for the edge cache: the places routes must never ask it. */
(globalThis as any).caches = {
  default: {
    match: async () => { cacheCalls++; return undefined; },
    put: async () => { cacheCalls++; },
  },
};

/** Stand-in for storage: presigned GET / signed PUT to the bucket, and the resolve URL on Hugging Face. */
globalThis.fetch = (async (input: any, init: any = {}) => {
  const url = new URL(typeof input === 'string' ? input : input.url);
  const method = String(init.method ?? 'GET').toUpperCase();
  if (url.hostname === 's3.test') {
    const key = decodeURIComponent(url.pathname.replace(/^\/bucket\//, ''));
    if (method === 'GET') {
      const body = bucket.get(key);
      return body === undefined ? new Response('', { status: 404 }) : new Response(body, { status: 200, headers: { 'content-type': 'application/json' } });
    }
    if (method === 'PUT') {
      puts.push({ key, body: String(init.body), contentType: init.headers?.['content-type'] ?? null });
      bucket.set(key, String(init.body));
      return new Response('', { status: 200 });
    }
  }
  if (url.hostname === 'huggingface.co') {
    const key = decodeURIComponent(url.pathname.split('/resolve/main/')[1] ?? '');
    const body = hf.get(key);
    return body === undefined ? new Response('', { status: 404 }) : new Response(body, { status: 200 });
  }
  return new Response('', { status: 404 });
}) as any;

const env: any = {
  DB: {}, APP_KEY: 'team-secret', WORKER_TOKEN: 'worker-token', ALLOWED_ORIGINS: 'https://app.test',
  S3_ENDPOINT: 'https://s3.test', S3_REGION: 'test-region', S3_BUCKET: 'bucket', S3_KEY_ID: 'key-id', S3_SECRET: 'secret',
};
const hfEnv: any = { ...env, HF_REPO: 'Owner/repo', HF_TOKEN: 'hf-token' };
const ctx = { waitUntil: (_p: Promise<unknown>) => {} };

const place = (over: Record<string, unknown> = {}) => ({
  id: 'main-door', name: 'Main door', cat: 'Exits', p: [-0.98, 1.2, 2.19], pinned: true,
  ai: { cls: 'door', conf: 0.91, half: [0.45, 1.0, 0.08] }, ...over,
});

const call = (method: string, scene: string, e: any = env, body?: unknown, headers: Record<string, string> = {}) =>
  (api.fetch as any)(new Request(`${BASE}/v1/scenes/${scene}/places`, {
    method,
    ...(body === undefined ? {} : { body: typeof body === 'string' ? body : JSON.stringify(body) }),
    headers: { 'content-type': 'application/json', ...headers },
  }), e, ctx);

const read = async (scene: string, e: any = env) => {
  const res = await call('GET', scene, e);
  return { res, json: await res.json() as any };
};

const write = (scene: string, body: unknown, headers: Record<string, string> = { 'x-ply-key': env.APP_KEY }, e: any = env) =>
  call('PUT', scene, e, body, headers);

let n = 0;
const scene = (label: string) => `${label}-${++n}`;

/** An invite minted the way scripts/invite.py does: label.expiry.mac, mac = HMAC(APP_KEY, label.expiry) cut to 32 hex. */
async function invite(label = 'friend'): Promise<string> {
  const exp = String(Math.floor(Date.now() / 1000) + 3600);
  const enc = new TextEncoder();
  const k = await crypto.subtle.importKey('raw', enc.encode(env.APP_KEY), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const sig = new Uint8Array(await crypto.subtle.sign('HMAC', k, enc.encode(`${label}.${exp}`)));
  return `${label}.${exp}.${Array.from(sig, b => b.toString(16).padStart(2, '0')).join('').slice(0, 32)}`;
}

test('a scene with no places.json reads as an empty document, never edge-cached', async () => {
  const { res, json } = await read(scene('empty'));
  assert.equal(res.status, 200);
  assert.deepEqual(json, { schema: 1, updatedAt: null, places: [] });
  assert.equal(res.headers.get('cache-control'), 'no-store');
  assert.equal(cacheCalls, 0, 'the places read must bypass the shared edge cache');
});

test('a published file (pipeline shape: version/source) reads as schema 1', async () => {
  const id = scene('published');
  bucket.set(`scenes/${id}/places.json`, JSON.stringify({ version: 1, source: 'sam3', places: [place()] }));
  const { res, json } = await read(id);
  assert.equal(res.status, 200);
  assert.equal(json.schema, 1);
  assert.equal(json.updatedAt, null, 'the pipeline file has no timestamp');
  assert.equal(json.places.length, 1);
  assert.equal(json.places[0].id, 'main-door');
  assert.equal(res.headers.get('cache-control'), 'no-store');
  assert.equal(cacheCalls, 0);
});

test('a file written by this API keeps its timestamp, and reads fall back to Hugging Face', async () => {
  const id = scene('hf');
  const updatedAt = '2026-11-20T10:11:12.000Z';
  hf.set(`scenes/${id}/places.json`, JSON.stringify({ schema: 1, updatedAt, places: [place()] }));
  const { res, json } = await read(id, hfEnv);   // bucket has no copy: the worker published to HF
  assert.equal(res.status, 200);
  assert.equal(json.updatedAt, updatedAt);
  assert.deepEqual(json.places.map((p: any) => p.id), ['main-door']);
  assert.equal(res.headers.get('cache-control'), 'no-store');
});

test('an unreadable places.json is a 502, not an empty scene', async () => {
  const id = scene('broken');
  bucket.set(`scenes/${id}/places.json`, '<html>not json</html>');
  const { res, json } = await read(id);
  assert.equal(res.status, 502);
  assert.match(json.error, /not readable/);
  assert.equal(res.headers.get('cache-control'), 'no-store');
});

test('writing without a key is 401, with a wrong key is 401, with an invite is 403', async () => {
  const id = scene('auth');
  let res = await write(id, { places: [place()] }, { 'x-ply-key': '' });
  assert.equal(res.status, 401);
  res = await write(id, { places: [place()] }, { 'x-ply-key': 'not-the-key' });
  assert.equal(res.status, 401);
  res = await write(id, { places: [place()] }, { 'x-ply-key': await invite() });
  assert.equal(res.status, 403);
  assert.match((await res.json() as any).error, /team key/);
  assert.equal(bucket.has(`scenes/${id}/places.json`), false, 'a refused write must not reach storage');
});

test('the team invite code can write, because the app tags with it', async () => {
  // The app has no secret in the browser: it holds the team invite code the capture flow already uses. Only an
  // invite whose label is not "team" (a judges code, a friend code) is refused, so this path must keep working.
  const id = scene('team-invite');
  const res = await write(id, { places: [place()] }, { 'x-ply-key': await invite('team') });
  assert.equal(res.status, 200, 'the team invite should write places');
  assert.equal((await res.json() as any).count, 1);
  const judges = await write(scene('judges-invite'), { places: [place()] }, { 'x-ply-key': await invite('judges') });
  assert.equal(judges.status, 403, 'a judges invite may look, not tag');
});

test('the team key and the worker token can write', async () => {
  for (const [label, headers] of [
    ['team', { 'x-ply-key': env.APP_KEY }],
    ['worker', { authorization: `Bearer ${env.WORKER_TOKEN}` }],
  ] as const) {
    const id = scene(label);
    const res = await write(id, { places: [place()] }, headers);
    assert.equal(res.status, 200, `${label} should write`);
    assert.equal((await res.json() as any).count, 1);
  }
});

test('more than 500 places is 400, naming the list', async () => {
  const places = Array.from({ length: 501 }, (_, i) => place({ id: `p-${i}`, name: `Place ${i}` }));
  const res = await write(scene('limit'), { places });
  assert.equal(res.status, 400);
  const body = await res.json() as any;
  assert.equal(body.index, null);
  assert.equal(body.field, 'places');
  assert.match(body.error, /500/);
});

test('a bad id is 400, naming place index and field', async () => {
  const res = await write(scene('id'), { places: [place({ id: 'Main Door!' })] });
  assert.equal(res.status, 400);
  const body = await res.json() as any;
  assert.equal(body.index, 0);
  assert.equal(body.field, 'id');
  assert.match(body.error, /place 0: id must match/);
});

test('a category outside the four is 400', async () => {
  const res = await write(scene('cat'), { places: [place({ cat: 'Doors' })] });
  assert.equal(res.status, 400);
  const body = await res.json() as any;
  assert.equal(body.index, 0);
  assert.equal(body.field, 'cat');
  assert.match(body.error, /Furniture, Storage, Exits, Things/);
});

test('a non-finite or out-of-world coordinate is 400', async () => {
  for (const [i, p] of [[0, [0, null, 0]], [0, [0, 'far', 0]], [1, [6000, 1, 0]], [2, [0, 1, -6000]]] as const) {
    const places = [place({ id: 'a-1' }), place({ id: 'b-2' }), place({ id: 'c-3' })];
    (places[i] as any).p = p;
    const res = await write(scene('xyz'), { places });
    assert.equal(res.status, 400, `p=${JSON.stringify(p)}`);
    const body = await res.json() as any;
    assert.equal(body.index, i);
    assert.equal(body.field, 'p');
  }
});

test('a duplicate id is 400 and names the first place that used it', async () => {
  const res = await write(scene('dupe'), { places: [place({ id: 'desk' }), place({ id: 'desk' })] });
  assert.equal(res.status, 400);
  const body = await res.json() as any;
  assert.equal(body.index, 1);
  assert.equal(body.field, 'id');
  assert.match(body.error, /already used by place 0/);
});

test('the rest of the field rules hold: name, pinned, ai', async () => {
  const cases: [Record<string, unknown>, string][] = [
    [{ name: '' }, 'name'],
    [{ name: 'x'.repeat(81) }, 'name'],
    [{ pinned: 'yes' }, 'pinned'],
    [{ ai: { cls: '', conf: 0.5, half: [1, 1, 1] } }, 'ai.cls'],
    [{ ai: { cls: 'door', conf: null, half: [1, 1, 1] } }, 'ai.conf'],
    [{ ai: { cls: 'door', conf: 0.5, half: [1, 'wide', 1] } }, 'ai.half'],
  ];
  for (const [over, field] of cases) {
    const res = await write(scene('fields'), { places: [place(over)] });
    assert.equal(res.status, 400, `${field} should be refused`);
    const body = await res.json() as any;
    assert.equal(body.field, field);
    assert.equal(body.index, 0);
  }
});

test('a bad body is 400 and POST is 405', async () => {
  let res = await write(scene('body'), 'not json');
  assert.equal(res.status, 400);
  assert.equal((await res.json() as any).field, 'body');
  res = await write(scene('body'), { pins: [] });
  assert.equal(res.status, 400);
  assert.equal((await res.json() as any).field, 'places');
  res = await call('POST', scene('body'));
  assert.equal(res.status, 405);
  assert.match(String(res.headers.get('allow')), /PUT/);
});

test('a successful write round-trips: schema, fresh ISO updatedAt, count and places', async () => {
  const id = scene('roundtrip');
  puts = [];
  const sent = [
    place({ id: 'desk', name: 'Study desk', cat: 'Furniture', p: [0.62, 0.86, 1.62] }),
    place({ id: 'window', name: 'Window', cat: 'Things', p: [0.43, 0.57, 2.39], pinned: false,
      ai: { cls: 'window', conf: 0.96, half: [0.55, 0.75, 0.1], frames: 4 } }),   // an extra ai key survives
  ];
  const res = await write(id, { places: sent }, { 'x-ply-key': env.APP_KEY });
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('cache-control'), 'no-store');
  const body = await res.json() as any;
  assert.equal(body.ok, true);
  assert.equal(body.count, 2);
  assert.deepEqual(Object.keys(body).sort(), ['count', 'ok', 'updatedAt']);
  assert.match(body.updatedAt, /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
  assert.ok(Math.abs(Date.now() - Date.parse(body.updatedAt)) < 60_000, 'updatedAt is fresh');

  assert.equal(puts.length, 1, 'one PUT to storage');
  assert.equal(puts[0].key, `scenes/${id}/places.json`);
  assert.equal(puts[0].contentType, 'application/json');
  const stored = JSON.parse(puts[0].body);
  assert.equal(stored.schema, 1);
  assert.equal(stored.updatedAt, body.updatedAt);
  assert.deepEqual(stored.places, sent);
  assert.equal(stored.places[1].ai.frames, 4);

  const got = await read(id);
  assert.equal(got.res.status, 200);
  assert.deepEqual(got.json, { schema: 1, updatedAt: body.updatedAt, places: sent });
  assert.equal(cacheCalls, 0, 'the read after the write never consults the edge cache');
});

test('info: a note, an http(s) link and a small JPEG photo round-trip; anything else is 400', async () => {
  const id = scene('info');
  const photo = 'data:image/jpeg;base64,' + 'A'.repeat(400);
  const ok = await write(id, { places: [place({ info: { note: '  Fire extinguisher behind the door  ', link: 'https://example.org/map', photo } })] });
  assert.equal(ok.status, 200);
  const got = await read(id);
  assert.deepEqual(got.json.places[0].info, { note: 'Fire extinguisher behind the door', link: 'https://example.org/map', photo });
  for (const [info, field] of [
    [{ link: 'javascript:alert(1)' }, 'info.link'],
    [{ note: 'x'.repeat(601) }, 'info.note'],
    [{ photo: 'data:image/svg+xml;base64,PHN2Zz4=' }, 'info.photo'],
    [{ photo: 'data:image/jpeg;base64,' + 'A'.repeat(70_000) }, 'info.photo'],
    [{ html: '<b>x</b>' }, 'info'],
    ['text', 'info'],
  ] as const) {
    const res = await write(scene('info-bad'), { places: [place({ info })] });
    assert.equal(res.status, 400, JSON.stringify(info).slice(0, 60));
    assert.equal((await res.json() as any).field, field);
  }
});

test('view.crop is stored with the places, kept by a write without view, cleared by null', async () => {
  const id = scene('crop');
  const crop = { min: [-3, -0.2, -4], max: [3, 3, 4] };
  assert.equal((await write(id, { places: [place()], view: { crop } })).status, 200);
  assert.deepEqual((await read(id)).json.view, { crop });
  assert.equal((await write(id, { places: [place({ name: 'Front door' })] })).status, 200);   // a pin edit from an older viewer
  const kept = (await read(id)).json;
  assert.equal(kept.places[0].name, 'Front door');
  assert.deepEqual(kept.view, { crop });
  assert.equal((await write(id, { places: [place()], view: null })).status, 200);
  assert.equal('view' in (await read(id)).json, false);
  for (const view of [{ crop: { min: [0, 0, 0], max: [0, 1, 1] } }, { crop: { min: [0, 0], max: [1, 1, 1] } }, { spin: 1 }, 'x']) {
    const res = await write(scene('crop-bad'), { places: [], view });
    assert.equal(res.status, 400, JSON.stringify(view));
    assert.match((await res.json() as any).field, /^view/);
  }
});
