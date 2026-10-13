// Headless Chrome over CDP for screenshot checks: const b = await open({ w, h, mobile }); b.ev(js); b.shot(name); b.close()
// GPU flags (not swiftshader): Spark needs real WebGL2 to stream splats in reasonable time.
import { spawn } from 'node:child_process'; import fs from 'node:fs'; import os from 'node:os'; import path from 'node:path';
export const sl = ms => new Promise(r => setTimeout(r, ms));
export const SHOTS = process.env.SHOTS || path.join(os.tmpdir(), 'ply-shots');
export async function open({ port = 9360, w = 1280, h = 800, mobile = false } = {}) {
  fs.mkdirSync(SHOTS, { recursive: true });
  const prof = fs.mkdtempSync(path.join(os.tmpdir(), 'ply-chrome-'));
  const ch = spawn('google-chrome', ['--headless=new', '--use-gl=angle', '--use-angle=gl-egl', '--ignore-gpu-blocklist', `--remote-debugging-port=${port}`, `--user-data-dir=${prof}`, `--window-size=${w},${h}`, 'about:blank'], { detached: true });
  for (let i = 0; i < 30; i++) { await sl(400); try { await fetch(`http://127.0.0.1:${port}/json/version`); break; } catch { /* starting */ } }
  const t = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(x => x.type === 'page');
  const ws = new WebSocket(t.webSocketDebuggerUrl); await new Promise(r => ws.onopen = r);
  let id = 0; const pend = {}; const logs = [];
  ws.onmessage = m => { const d = JSON.parse(m.data); if (d.method === 'Runtime.exceptionThrown') logs.push('EXC ' + JSON.stringify(d.params.exceptionDetails.exception?.description ?? d.params.exceptionDetails.text).slice(0, 300)); if (pend[d.id]) { pend[d.id](d); delete pend[d.id]; } };
  const send = (method, params = {}) => new Promise(r => { const i = ++id; pend[i] = r; ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async e => { const r = await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true }); const x = r.result?.exceptionDetails ?? r.error; if (x) return 'ERR ' + String(x.exception?.description ?? JSON.stringify(x)).slice(0, 300); return r.result?.result?.value; };
  const shot = async n => { const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 82 }); fs.writeFileSync(path.join(SHOTS, `${n}.jpg`), Buffer.from(r.result.data, 'base64')); };
  await send('Runtime.enable'); await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile });
  const close = () => { try { process.kill(-ch.pid); } catch { ch.kill(); } setTimeout(() => fs.rmSync(prof, { recursive: true, force: true, maxRetries: 5 }), 800); };
  return { ev, shot, send, logs, close };
}
