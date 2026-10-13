// node scripts/probe-pick.mjs picks.json [scene-url]: picks = [[viewIndex, x, y, name], ...] on probe-views' 960x600 shots
// -> median splat hit around each pixel (world metres), from the same spot probe-views used
import { open, sl } from './cdp.mjs'; import fs from 'node:fs';
const picks = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const b = await open({ w: 960, h: 600, port: 9361 });
const q = process.argv[3] ? `?scene=${encodeURIComponent(process.argv[3])}` : '';
await b.send('Page.navigate', { url: `http://localhost:5199/ply/walk.html${q}` });
for (let i = 0; i < 90; i++) { await sl(2000); if (await b.ev('!!window.__ply?.state?.loaded')) break; }
await b.ev(`document.querySelector('.btn')?.click()`); await sl(800);
await b.ev(`(()=>{const g=__ply.grid,{width:w,height:h,cell,origin}=g.meta;let sx=0,sz=0,n=0;for(let j=0;j<h;j++)for(let i=0;i<w;i++)if(!g.data[j*w+i]){sx+=origin[0]+(i+.5)*cell;sz+=origin[1]+(j+.5)*cell;n++}const p=__ply.state.pos;p.x=sx/n;p.z=sz/n;if(!g.clear(p.x,p.z,.22))__ply.unstick();return 1})()`); await sl(1500);
for (const [k, x, y, name] of picks) {
  await b.ev(`(__ply.state.yaw = ${k} * Math.PI / 4, __ply.pitch = -0.2, 1)`); await sl(700);
  const r = await b.ev(`(async()=>{const THREE=await import('/ply/node_modules/.vite/deps/three.js'); __ply.camera.updateMatrixWorld(true); const rc=new THREE.Raycaster(); const ps=[];
    for(let dx=-20;dx<=20;dx+=8)for(let dy=-20;dy<=20;dy+=8){rc.setFromCamera(new THREE.Vector2((${x}+dx)/480-1,1-(${y}+dy)/300),__ply.camera);const h=rc.intersectObject(__ply.splat,false)[0];if(h&&h.distance<7)ps.push(h.point.toArray());}
    if(!ps.length)return 'none';const med=i=>{const a=ps.map(p=>p[i]).sort((a,b)=>a-b);return +a[a.length>>1].toFixed(2)};return JSON.stringify([med(0),med(1),med(2)])+' n='+ps.length})()`);
  console.log(name.padEnd(16), r);
}
b.close(); process.exit(0);
