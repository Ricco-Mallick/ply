// node scripts/lab-shots.mjs 1440x900 [m] steps.mjs [query]: open the design lab, run [name, js, waitMs] steps, screenshot each
import { open, sl, SHOTS } from './cdp.mjs'; import path from 'node:path';
const [w, h] = process.argv[2].split('x').map(Number), mobile = process.argv[3] === 'm';
const steps = (await import(path.resolve(process.argv[4]))).default;
const b = await open({ w, h, mobile, port: 9362 });
await b.send('Page.navigate', { url: `http://localhost:5199/ply/lab.html${process.argv[5] ?? ''}` });
for (let i = 0; i < 90; i++) { await sl(2000); if (await b.ev('!!window.__lab')) break; }
await sl(2000); await b.ev(`[...document.querySelectorAll('.btn')].find(x=>x.textContent.includes('Open the lab'))?.click()`); await sl(1200);
await b.ev(`window.opt=(r,v)=>{const row=[...document.querySelectorAll('.lab-row')].find(x=>x.querySelector('.lab-row-l').textContent===r);[...row.querySelectorAll('button')].find(b=>b.textContent===v).click()};1`);
for (const [name, js, wait] of steps) { const r = await b.ev(js); if (r !== undefined && r !== 1) console.log(name, String(r).slice(0, 200)); await sl(wait ?? 1800); await b.shot(name); }
console.log(b.logs.slice(0, 6).join('\n'), '\nshots in', SHOTS); b.close(); setTimeout(() => process.exit(0), 1000);
