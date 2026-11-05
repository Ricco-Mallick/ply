/* Waking the Modal worker against stand-in triggers. Run: node --experimental-strip-types test/modal.test.mts */
import { kickModal } from '../src/modal.ts';

const log = console.log; let fail = 0; const check = (name: string, ok: boolean) => { log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
const calls: { url: string; body: any }[] = [];
const NEW = 'https://riccomarcelino--ply-trigger.us-east.modal.run', OLD = 'https://ricco-mallick--ply-trigger.modal.run';
globalThis.fetch = (async (url: string, init: any) => {
  const body = JSON.parse(init.body); calls.push({ url, body });
  if (url === OLD) return new Response('modal-http: invalid function call', { status: 404 });   // app not deployed there
  if (body.workerToken !== 'wt' && body.token !== 'tt') return Response.json({ ok: false, error: 'unauthorised' });
  return Response.json({ ok: true, started: true });
}) as any;
console.log = () => {};   // kickModal logs failures

check('nothing configured -> null', await kickModal({}, 'x') === null);
let r = await kickModal({ MODAL_TRIGGER_URL: NEW, MODAL_TRIGGER: OLD, WORKER_TOKEN: 'wt' }, 'job j1');
check('the URL in wrangler.toml comes first and the worker token is enough', r?.started === true && calls.length === 1 && calls[0].url === NEW && calls[0].body.why === 'job j1');
calls.length = 0;
r = await kickModal({ MODAL_TRIGGER_URL: OLD, MODAL_TRIGGER: NEW, MODAL_TRIGGER_TOKEN: 'tt' }, 'cron');
check('a dead URL falls through to the next one', r?.ok === true && calls.map(c => c.url).join() === `${OLD},${NEW}`);
r = await kickModal({ MODAL_TRIGGER: OLD, WORKER_TOKEN: 'wt' }, 'manual');
check('a failure says why instead of null', r?.ok === false && r.status === 404 && r.url === 'ricco-mallick--ply-trigger' && /invalid function call/.test(String(r.error)));
r = await kickModal({ MODAL_TRIGGER_URL: NEW, MODAL_TRIGGER_TOKEN: 'wrong', WORKER_TOKEN: 'nope' }, 'manual');
check('a wrong token is reported, without the token', r?.ok === false && r.error === 'unauthorised' && !JSON.stringify(r).includes('nope'));

if (fail) { console.error(`${fail} failed`); process.exit(1); }
