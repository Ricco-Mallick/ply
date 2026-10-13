// node scripts/probe-views.mjs [scene-url]  -> 8 views from the middle of the walkable floor (for placing pins by eye)
import { open, sl } from './cdp.mjs';
const b = await open({ w: 960, h: 600 });
const q = process.argv[2] ? `?scene=${encodeURIComponent(process.argv[2])}` : '';
await b.send('Page.navigate', { url: `http://localhost:5199/ply/walk.html${q}` });
for (let i = 0; i < 90; i++) { await sl(2000); if (await b.ev('!!window.__ply?.state?.loaded')) break; }
await b.ev(`document.querySelector('.btn')?.click()`); await sl(800);
console.log(await b.ev(`(()=>{const g=__ply.grid,{width:w,height:h,cell,origin}=g.meta;let sx=0,sz=0,n=0;for(let j=0;j<h;j++)for(let i=0;i<w;i++)if(!g.data[j*w+i]){sx+=origin[0]+(i+.5)*cell;sz+=origin[1]+(j+.5)*cell;n++}const p=__ply.state.pos;p.x=sx/n;p.z=sz/n;if(!g.clear(p.x,p.z,.22))__ply.unstick();return JSON.stringify({pos:p.toArray().map(v=>+v.toFixed(2)),floor:__ply.floorY,grid:g.meta})})()`));
await sl(2500);
for (let k = 0; k < 8; k++) { await b.ev(`(__ply.state.yaw = ${k} * Math.PI / 4, __ply.pitch = -0.2, 1)`); await sl(1800); await b.shot('probe' + k); }
console.log(b.logs.slice(0, 5).join('\n')); b.close(); process.exit(0);
