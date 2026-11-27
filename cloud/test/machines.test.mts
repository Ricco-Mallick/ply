// Paired PCs end to end against the real route code, with node:sqlite standing in for D1: pairing, approval, the
// token handed over once, claims limited to assigned jobs, and everything a machine token must not reach.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import api from '../src/index.ts';
import { machineKeyAllowed, machinePathAllowed, normCode, parseMachineAuth } from '../src/machines.ts';

function d1() {
  const db = new DatabaseSync(':memory:');
  db.exec(readFileSync(new URL('../schema.sql', import.meta.url), 'utf8'));
  for (const f of readdirSync(new URL('../migrations/', import.meta.url)).sort())
    db.exec(readFileSync(new URL(`../migrations/${f}`, import.meta.url), 'utf8'));
  const stmt = (sql: string, args: unknown[] = []) => ({
    bind: (...a: unknown[]) => stmt(sql, a),
    first: async () => db.prepare(sql).get(...(args as any[])) ?? null,
    all: async () => ({ results: db.prepare(sql).all(...(args as any[])) }),
    run: async () => db.prepare(sql).run(...(args as any[])),
  });
  return { prepare: (sql: string) => stmt(sql), raw: db };
}

const APP_KEY = 'team-key-for-tests', WORKER_TOKEN = 'shared-worker-token';
const DB = d1();
const env: any = { DB, APP_KEY, WORKER_TOKEN, ALLOWED_ORIGINS: '', S3_ENDPOINT: 'https://s3.example.com', S3_REGION: 'r', S3_BUCKET: 'b', S3_KEY_ID: 'k', S3_SECRET: 's' };
const ctx = { waitUntil() {} };
async function call(method: string, path: string, body?: unknown, headers: Record<string, string> = {}) {
  const res = await api.fetch(new Request(`https://api.test${path}`, { method, headers: { 'content-type': 'application/json', ...headers }, body: body === undefined ? undefined : JSON.stringify(body) }), env, ctx);
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null };
}
const team = { 'x-ply-key': APP_KEY };
const shared = { authorization: `Bearer ${WORKER_TOKEN}` };
function job(id: string, capture = 'cap-1') {
  const now = Date.now();
  DB.raw.prepare("INSERT INTO jobs (id,name,capture_id,settings,args,files,status,progress,created_at,updated_at) VALUES (?,?,?,?,?,?,'queued',0,?,?)")
    .run(id, id, capture, '{}', '[]', JSON.stringify([`captures/${capture}/a.mp4`]), now, now);
}

test('helpers', () => {
  assert.equal(normCode('ab cd-ef23'), 'ABCD-EF23');
  assert.equal(normCode('abc'), '');
  assert.equal(parseMachineAuth(`Bearer pm_pc-lab-ab12.${'a'.repeat(64)}`)?.name, 'pc-lab-ab12');
  assert.equal(parseMachineAuth(`Bearer ${WORKER_TOKEN}`), null);
  assert.ok(machineKeyAllowed('scenes/j1/scene.spz', 'j1'));
  assert.ok(!machineKeyAllowed('scenes/j2/scene.spz', 'j1'));
  assert.ok(!machineKeyAllowed('scenes/j1/../j2/x', 'j1'));
  assert.ok(!machinePathAllowed('/v1/jobs') && !machinePathAllowed('/v1/worker/kick') && machinePathAllowed('/v1/worker/claim'));
});

test('a PC pairs, is approved, collects its token once, and works only on what it is given', async () => {
  const pair = await call('POST', '/v1/machines/pair', { hint: 'DESKTOP-LAB3', specs: { machine: { gpu: { name: 'GTX 1650', totalGB: 4, sm: 7.5 }, ramGB: 16 }, trainable: true, sam3: false, why: 'x' } });
  assert.equal(pair.status, 201);
  assert.match(pair.body.code, /^[A-Z2-9]{4}-[A-Z2-9]{4}$/);
  assert.match(pair.body.name, /^pc-desktop-lab3-[0-9a-f]{4}$/);
  const name = pair.body.name;

  assert.equal((await call('POST', '/v1/machines/poll', { name, poll: pair.body.poll })).body.status, 'pending');
  assert.equal((await call('POST', '/v1/machines/poll', { name, poll: 'f'.repeat(64) })).status, 404);  // only that PC
  assert.equal((await call('POST', '/v1/machines/approve', { code: pair.body.code }, shared)).status, 401);  // not the worker token
  assert.equal((await call('POST', '/v1/machines/approve', { code: pair.body.code.toLowerCase(), label: 'college lab 3' }, team)).status, 200);

  const got = await call('POST', '/v1/machines/poll', { name, poll: pair.body.poll });
  assert.equal(got.body.status, 'active');
  const mine = { authorization: `Bearer ${got.body.token}` };
  assert.equal((await call('POST', '/v1/machines/poll', { name, poll: pair.body.poll })).status, 404);  // handed over once

  const list = await call('GET', '/v1/machines', undefined, team);
  assert.equal(list.body.machines[0].name, name);
  assert.equal(list.body.machines[0].specs.gpu.name, 'GTX 1650');

  job('job-shared');
  job('job-lab1');
  // nothing assigned yet: the PC gets nothing, though a shared job is waiting
  assert.equal((await call('POST', '/v1/worker/claim', { worker: 'ignored' }, mine)).status, 204);
  assert.equal((await call('POST', '/v1/jobs/job-lab1/assign', { worker: name, args: ['--preset', 'quick', '--until', 'occupancy'] }, team)).status, 200);
  // the shared-token workers (Modal, Kaggle) never take an assigned job
  const sh = await call('POST', '/v1/worker/claim', { worker: 'modal-l40s' }, shared);
  assert.equal(sh.body.id, 'job-shared');
  assert.equal((await call('POST', '/v1/worker/claim', { worker: 'modal-l40s' }, shared)).status, 204);

  const c = await call('POST', '/v1/worker/claim', { worker: 'pretend-to-be-modal', specs: { trainable: true } }, mine);
  assert.equal(c.body.id, 'job-lab1');
  assert.equal(c.body.worker, name);  // claims under its own name, whatever it says
  assert.deepEqual(c.body.args, ['--preset', 'quick', '--until', 'occupancy']);

  // its job: report, read its inputs, write its results
  assert.equal((await call('POST', '/v1/worker/jobs/job-lab1', { stage: 'frames', progress: 0.1 }, mine)).status, 200);
  assert.equal((await call('GET', '/v1/worker/file?key=captures/cap-1/a.mp4', undefined, mine)).status, 200);
  assert.equal((await call('POST', '/v1/uploads/sign', { key: 'scenes/job-lab1/scene.spz', uploadId: 'u', parts: [1] }, mine)).status, 200);
  // and nothing else
  assert.equal((await call('POST', '/v1/worker/jobs/job-shared', { status: 'failed' }, mine)).status, 403);
  assert.equal((await call('GET', '/v1/worker/file?key=captures/other/b.mp4', undefined, mine)).status, 403);
  assert.equal((await call('POST', '/v1/uploads/sign', { key: 'scenes/job-shared/scene.spz', uploadId: 'u', parts: [1] }, mine)).status, 403);
  assert.equal((await call('POST', '/v1/worker/kick', {}, mine)).status, 401);
  assert.equal((await call('GET', '/v1/jobs', undefined, mine)).status, 401);
  assert.equal((await call('GET', '/v1/machines', undefined, mine)).status, 401);

  // revoked: the token stops working at once, and its running job fails loudly instead of waiting forever
  assert.equal((await call('POST', `/v1/machines/${name}/revoke`, {}, team)).status, 200);
  assert.equal((await call('POST', '/v1/worker/jobs/job-lab1', { progress: 0.2 }, mine)).status, 401);
  const gone = DB.raw.prepare("SELECT status, message FROM jobs WHERE id='job-lab1'").get() as any;
  assert.equal(gone.status, 'failed');
  assert.match(gone.message, /revoked/);
  assert.equal((await call('POST', '/v1/worker/claim', { worker: 'modal-l40s' }, shared)).status, 204);  // not handed to Modal
});

test('assigning needs the team key and a real machine; a finished twin is copied, never overwritten', async () => {
  job('job-x01');
  assert.equal((await call('POST', '/v1/jobs/job-x01/assign', { worker: 'pc-nope-0000' }, team)).status, 404);
  assert.equal((await call('POST', '/v1/jobs/job-x01/assign', { worker: null }, shared)).status, 401);
  for (const status of ['done', 'review', 'running']) {
    DB.raw.prepare('UPDATE jobs SET status=? WHERE id=?').run(status, 'job-x01');
    const r = await call('POST', '/v1/jobs/job-x01/assign', { worker: null, args: ['--preset', 'quick'] }, team);
    assert.equal(r.status, 200);
    assert.equal(r.body.copied, true);
    assert.notEqual(r.body.jobId, 'job-x01');
    assert.equal((DB.raw.prepare('SELECT status FROM jobs WHERE id=?').get('job-x01') as any).status, status);  // untouched
    const c = DB.raw.prepare('SELECT status, files, args FROM jobs WHERE id=?').get(r.body.jobId) as any;
    assert.equal(c.status, 'queued');
    assert.equal(c.files, JSON.stringify(['captures/cap-1/a.mp4']));
    assert.equal(c.args, JSON.stringify(['--preset', 'quick']));
  }
});

test('health keeps paired PCs apart from the shared GPU workers', async () => {
  const now = Date.now();
  DB.raw.prepare("INSERT INTO machines (name,status,created_at) VALUES ('pc-h-0001','active',?)").run(now);
  DB.raw.prepare("INSERT OR REPLACE INTO workers (name,last_seen,busy_job,note) VALUES ('pc-h-0001',?,NULL,NULL)").run(now);
  const h = await call('GET', '/v1/health');
  assert.ok(h.body.pcsOnline >= 1);
  assert.ok(!h.body.workers.some((w: any) => w.name === 'pc-h-0001'));
});

test('pairing is rate limited', async () => {
  let last = 0;
  for (let i = 0; i < 25; i++) last = (await call('POST', '/v1/machines/pair', { hint: `pc${i}` })).status;
  assert.equal(last, 429);
});

test('a paused job goes back to the queue, its pauses are counted, and too many fail it', async () => {
  job('job-pause1');
  DB.raw.prepare("UPDATE jobs SET status='running', attempts=1 WHERE id='job-pause1'").run();
  for (let i = 1; i <= 8; i++) {
    const r = await call('POST', '/v1/worker/jobs/job-pause1', { status: 'queued', paused: true, message: 'paused' }, shared);
    assert.equal(r.status, 200);
    const row = DB.raw.prepare("SELECT status, pauses, attempts FROM jobs WHERE id='job-pause1'").get() as any;
    assert.deepEqual([row.status, row.pauses, row.attempts], ['queued', i, 0]);
  }
  await call('POST', '/v1/worker/jobs/job-pause1', { status: 'queued', paused: true }, shared);
  const row = DB.raw.prepare("SELECT status, message FROM jobs WHERE id='job-pause1'").get() as any;
  assert.equal(row.status, 'failed');
  assert.match(row.message, /pauses/);
});
