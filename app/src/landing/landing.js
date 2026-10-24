// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Landing film: three.js scroll-driven story (ported from the design artifact). */
import '../shared/tokens.css';
import './landing.css';
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const RM = matchMedia('(prefers-reduced-motion: reduce)').matches;
const MOBILE = matchMedia('(max-width: 860px), (pointer: coarse)').matches;
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v)), lerp = (a, b, t) => a + (b - a) * t;
const ss = t => t * t * (3 - 2 * t), win = (p, a, b) => ss(clamp((p - a) / (b - a)));
const easeOutCubic = t => 1 - Math.pow(1 - t, 3);
const $ = id => document.getElementById(id);
let seed = 9; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

/* ================= renderer ================= */
let renderer;
try { renderer = new THREE.WebGLRenderer({ canvas: $('gl'), antialias: false, powerPreference: 'high-performance' }); }
catch (e) { $('err').style.display = 'grid'; throw e; }
let DPR = Math.min(devicePixelRatio || 1, MOBILE ? 1.25 : 1.5);
renderer.setPixelRatio(DPR); renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
const scene = new THREE.Scene(); scene.background = new THREE.Color('#070B14');
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, .05, 400); scene.add(camera);

/* ================= shared GLSL ================= */
const NOISE = `
float hash(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
float vnoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+1.),f.x),f.y);}
float fbm(vec2 p){float a=.5,s=0.;for(int i=0;i<4;i++){s+=a*vnoise(p);p*=2.03;a*=.5;}return s;}`;
const AMBER = new THREE.Color('#FFB84D'), CORE = new THREE.Color('#FFFAEB'), ORANGE = new THREE.Color('#FF6B00');

/* ================= splat/point material (morph scatter→shape, dot→gaussian, scan line, cursor push) ================= */
function pointMaterial(extra = {}) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uTime: { value: 0 }, uMorph: { value: 0 }, uSplat: { value: 0 }, uColorMix: { value: 0 }, uOpacity: { value: 0 }, uScan: { value: -10 }, uReveal: { value: 1 },
      uPx: { value: 600 }, uMouse: { value: new THREE.Vector2(9, 9) }, uPush: { value: 0 }, uAspect: { value: 1 }, uAmber: { value: AMBER }, uSize: { value: 1 }, uSweep: { value: -99 }, uSegMix: { value: 0 }, uFrontOn: { value: 0 }, uBang: { value: 1 }, uSing: { value: new THREE.Vector3(0, 1.4, 0) }, ...extra },
    vertexShader: `
      attribute vec3 aScatter; attribute vec3 color; attribute float aSeed; attribute vec3 aSeg;
      uniform float uTime,uMorph,uSplat,uPx,uPush,uAspect,uScan,uReveal,uSize,uSweep,uSegMix,uFrontOn,uBang; uniform vec2 uMouse; uniform vec3 uSing;
      varying vec3 vCol, vSeg; varying float vSeed, vScan, vVis, vSplat, vNear, vFront, vSegK, vHot, vBangA;
      void main(){
        float m = smoothstep(0., 1., clamp(uMorph*1.35 - aSeed*.35, 0., 1.));
        float bk = clamp(uBang*1.35 - aSeed*.35, 0., 1.), be = 1. - pow(1.-bk, 4.);   /* 0 = collapsed into the singularity */
        vec3 p = mix(mix(uSing, aScatter, be), position, m); vHot = (1.-be) * step(uBang, .999); vBangA = smoothstep(0., .22, be);
        p += (1.-m) * vec3(sin(uTime*.6+aSeed*40.), cos(uTime*.5+aSeed*30.), sin(uTime*.7+aSeed*20.)) * .08;
        float dx = uSweep - position.x; vFront = exp(-dx*dx/.09) * uFrontOn; vSegK = smoothstep(0., 1.2, dx) * uSegMix; vSeg = aSeg;
        float lift = smoothstep(-.1, .25, dx) * exp(-max(dx,0.)*1.6) * uFrontOn; p.y += lift * (.1 + aSeed*.35); p.xz += lift * (vec2(fract(aSeed*91.), fract(aSeed*57.))-.5) * .18;
        vec4 mv = modelViewMatrix * vec4(p,1.);
        vec4 cl = projectionMatrix * mv;
        vec2 nd = cl.xy/cl.w; vec2 d = (nd - uMouse) * vec2(uAspect,1.); float f = exp(-dot(d,d)/.018);
        cl.xy += normalize(d+1e-4) * f * uPush * .06 * cl.w;
        gl_Position = cl;
        vSplat = clamp(uSplat*1.25 - aSeed*.25, 0., 1.);
        float sz = mix(1.1, 11., vSplat) * uSize * (1. + vFront*1.6) * (1. + be*(1.-be)*7.*step(uBang, .999));
        gl_PointSize = clamp(sz * uPx / -mv.z * .05, 1., 30.);
        vNear = smoothstep(.9, 3.6, -mv.z);
        vCol = color; vSeed = aSeed;
        vScan = exp(-pow((position.y - uScan)/.06, 2.)) * m;
        vVis = step(length(position.xz)*.18 + aSeed*.25, uReveal*1.25);
      }`,
    fragmentShader: `
      uniform float uOpacity,uColorMix,uTime; uniform vec3 uAmber;
      varying vec3 vCol, vSeg; varying float vSeed, vScan, vVis, vSplat, vNear, vFront, vSegK, vHot, vBangA;
      void main(){
        vec2 c = gl_PointCoord*2.-1.; float r2 = dot(c,c); if(r2>1.) discard;
        float dotA = smoothstep(1.,.45,r2);
        float gA = exp(-r2*5.);
        float a = mix(dotA, gA*.75, vSplat);
        vec3 col = mix(uAmber*(.75+.35*sin(uTime*3.+vSeed*50.)), vCol*.82, uColorMix);
        col += uAmber * vScan * 3.;
        col = mix(col, vSeg * (.85 + .3*sin(uTime*2.+vSeed*30.)), vSegK * .8);
        col += mix(uAmber, vec3(1.), .55) * vFront * 1.1;
        col = mix(col, vec3(1., .93, .8) * 1.7, vHot * .85);
        gl_FragColor = vec4(col, min(1., a * (1. + vFront)) * uOpacity * vVis * vNear * vBangA);
      }` });
}
function buildPoints(list, mat) {
  const n = list.length, pos = new Float32Array(n * 3), sc = new Float32Array(n * 3), col = new Float32Array(n * 3), sg = new Float32Array(n * 3), sd = new Float32Array(n);
  const c = new THREE.Color();
  list.forEach((q, i) => { pos.set(q.p, i * 3); sc.set(q.s, i * 3); c.set(q.c); col.set([c.r, c.g, c.b], i * 3); c.set(q.g || q.c); sg.set([c.r, c.g, c.b], i * 3); sd[i] = rnd(); });
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('aScatter', new THREE.BufferAttribute(sc, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3)); g.setAttribute('aSeed', new THREE.BufferAttribute(sd, 1)); g.setAttribute('aSeg', new THREE.BufferAttribute(sg, 3));
  const pts = new THREE.Points(g, mat); pts.frustumCulled = false; return pts;
}
const shade = (hex, f) => { const n = parseInt(hex.slice(1), 16); return '#' + [n >> 16 & 255, n >> 8 & 255, n & 255].map(v => clamp(Math.round(v * f), 0, 255).toString(16).padStart(2, '0')).join(''); };
const sphere = r => { const u = rnd() * 2 - 1, t = rnd() * 6.283, s = Math.sqrt(1 - u * u), k = Math.cbrt(rnd()) * r; return [s * Math.cos(t) * k, u * k + 1.4, s * Math.sin(t) * k]; };

/* ================= ROOM (capture + reconstruct), world origin ================= */
const roomList = []; const DENS = MOBILE ? .45 : 1;
function plane(o, u, v, dens, colFn) { const area = Math.hypot(...u) * Math.hypot(...v), n = Math.round(area * dens * DENS);
  for (let i = 0; i < n; i++) { const a = rnd(), b = rnd(); const p = [o[0] + u[0] * a + v[0] * b, o[1] + u[1] * a + v[1] * b, o[2] + u[2] * a + v[2] * b]; const c = colFn(a, b, p); if (c) roomList.push({ p, s: sphere(6.5), c }); } }
function box(mn, mx, dens, col) { const [x0, y0, z0] = mn, [x1, y1, z1] = mx;
  plane([x0, y1, z0], [x1 - x0, 0, 0], [0, 0, z1 - z0], dens, () => shade(col, 1.08 + rnd() * .1));
  plane([x0, y0, z1], [x1 - x0, 0, 0], [0, y1 - y0, 0], dens, () => shade(col, .85 + rnd() * .15));
  plane([x0, y0, z0], [0, 0, z1 - z0], [0, y1 - y0, 0], dens, () => shade(col, .75 + rnd() * .15));
  plane([x1, y0, z0], [0, 0, z1 - z0], [0, y1 - y0, 0], dens, () => shade(col, .8 + rnd() * .15)); }
seed = 5;
plane([-3, 0, -2], [6, 0, 0], [0, 0, 4], 1300, (a, b) => { const plank = Math.floor(a * 30) % 2 ? .92 : 1.04; const rug = ((a - .5) / .32) ** 2 + ((b - .62) / .22) ** 2 < 1; return rug ? shade('#a2563f', .85 + rnd() * .3) : shade('#8a6748', plank * (.85 + rnd() * .25)); });
plane([-3, 0, -2], [6, 0, 0], [0, 2.8, 0], 1100, (a, b) => { const win = a > .35 && a < .65 && b > .38 && b < .76; if (win) return shade(b > .57 ? '#9fc2de' : '#e3ecf1', .95 + rnd() * .1); const frame = a > .34 && a < .66 && b > .37 && b < .77; return frame ? '#f4f1ea' : shade('#d2c9b8', .9 + rnd() * .15); });
plane([-3, 0, -2], [0, 0, 4], [0, 2.8, 0], 900, () => shade('#c3b8a5', .82 + rnd() * .2));
plane([3, 0, -2], [0, 0, 4], [0, 2.8, 0], 900, (a, b) => (a > .55 && a < .82 && b < .74) ? shade('#6b4f3a', .9 + rnd() * .2) : shade('#b6ab98', .82 + rnd() * .2));
plane([-3, 2.8, -2], [6, 0, 0], [0, 0, 4], 180, () => shade('#e4dfd5', .9 + rnd() * .1));
box([-2.4, 0, -1.9], [-.5, .45, -1.1], 2600, '#4b6680'); box([-2.4, .45, -2], [-.5, .95, -1.75], 2600, '#5b7896'); box([-2.5, 0, -1.95], [-2.35, .62, -1.1], 2600, '#3f566c'); box([-.55, 0, -1.95], [-.4, .62, -1.1], 2600, '#3f566c');
box([-.6, .4, -.35], [.7, .46, .35], 2600, '#6b5a48'); box([-.55, 0, -.3], [-.5, .4, -.25], 3000, '#4a3d30'); box([.62, 0, .25], [.67, .4, .3], 3000, '#4a3d30');
box([2.2, 0, -1.8], [2.55, .45, -1.45], 2600, '#7a5a43');
for (let i = 0; i < 2200 * DENS; i++) { const t = rnd() * 6.283, r = Math.sqrt(rnd()) * .45, h = .45 + rnd() * 1.1; roomList.push({ p: [2.37 + Math.cos(t) * r * (1.2 - (h - .45) * .5), h, -1.62 + Math.sin(t) * r], s: sphere(6.5), c: shade(rnd() < .5 ? '#4f7a4a' : '#638f55', .8 + rnd() * .4) }); }
const roomMat = pointMaterial(); roomMat.uniforms.uSize.value = .8; const room = buildPoints(roomList, roomMat); scene.add(room);

/* camera frustums along the capture path + plexus link */
const POSES = 40, poseGeo = new THREE.BufferGeometry(), pv = [], pidx = [], linkV = [];
for (let i = 0; i < POSES; i++) { const t = i / POSES * Math.PI * 1.85 + .3, x = Math.cos(t) * 2.1, z = Math.sin(t) * 1.35, y = 1.35 + Math.sin(i * .7) * .12;
  const o = new THREE.Vector3(x, y, z), look = new THREE.Vector3(-x * .2, 1.2, -z * .2), fw = look.clone().sub(o).normalize(), rt = new THREE.Vector3().crossVectors(fw, new THREE.Vector3(0, 1, 0)).normalize(), up = new THREE.Vector3().crossVectors(rt, fw);
  const d = .22, w = .13, h = .085, cs = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([a, b]) => o.clone().addScaledVector(fw, d).addScaledVector(rt, a * w).addScaledVector(up, b * h));
  const segs = [[o, cs[0]], [o, cs[1]], [o, cs[2]], [o, cs[3]], [cs[0], cs[1]], [cs[1], cs[2]], [cs[2], cs[3]], [cs[3], cs[0]]];
  for (const [a, b] of segs) { pv.push(a.x, a.y, a.z, b.x, b.y, b.z); pidx.push(i, i); } linkV.push(x, y, z); }
poseGeo.setAttribute('position', new THREE.Float32BufferAttribute(pv, 3)); poseGeo.setAttribute('aI', new THREE.Float32BufferAttribute(pidx, 1));
const lineMat = (col = AMBER, extra = '') => new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  uniforms: { uCount: { value: 0 }, uOpacity: { value: 0 }, uCol: { value: col }, uTime: { value: 0 } },
  vertexShader: `attribute float aI; uniform float uCount; varying float vA; void main(){ vA = clamp(uCount - aI, 0., 1.); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `uniform float uOpacity; uniform vec3 uCol; varying float vA; void main(){ if(vA<=0.) discard; gl_FragColor = vec4(uCol*1.6*vA, vA*uOpacity); }` });
const poseMat = lineMat(new THREE.Color('#FFE7C2')); poseMat.depthTest = false; const poses = new THREE.LineSegments(poseGeo, poseMat); poses.frustumCulled = false; scene.add(poses);
const linkGeo = new THREE.BufferGeometry(); linkGeo.setAttribute('position', new THREE.Float32BufferAttribute(linkV, 3)); linkGeo.setAttribute('aI', new THREE.Float32BufferAttribute([...Array(POSES).keys()], 1));
const linkMat = lineMat(new THREE.Color('#ffffff')); const link = new THREE.Line(linkGeo, linkMat); link.frustumCulled = false; scene.add(link);

/* infinity floor grid with shockwave ignite (infraGrid / dotGrid / infinityFloor) */
function floorGrid(size, shock) { return new THREE.Mesh(new THREE.PlaneGeometry(size, size).rotateX(-Math.PI / 2), new THREE.ShaderMaterial({ transparent: true, depthWrite: false,
  uniforms: { uR: { value: 0 }, uOpacity: { value: 0 }, uTime: { value: 0 }, uFade: { value: new THREE.Vector2(size * .2, size * .48) }, uAmber: { value: AMBER } },
  vertexShader: `varying vec2 vW; void main(){ vW = position.xz; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `${NOISE}
    uniform float uR,uOpacity,uTime; uniform vec2 uFade; uniform vec3 uAmber; varying vec2 vW;
    float line(float x, float w){ float d = abs(fract(x)-.5); return smoothstep(w, 0., .5-d); }
    void main(){
      float g1 = max(line(vW.x,.012), line(vW.y,.012));
      float g5 = max(line(vW.x/5.,.006), line(vW.y/5.,.006));
      vec2 cell = fract(vW/2.)-.5; vec2 ac = abs(cell); float cross = (step(ac.x,.006)*step(ac.y,.07) + step(ac.y,.006)*step(ac.x,.07)) * step(length(cell),.1);
      float d = length(vW); float ring = exp(-pow((uR-d)/1.1,2.)); float ign = smoothstep(0.,4.,uR-d) * (.35+.65*fbm(vW*.15+uTime*.05));
      vec3 base = vec3(.32,.4,.58) * (g1*.10 + g5*.22 + cross*.5);
      vec3 hot = uAmber * (g1*.5 + g5*1.1 + cross*1.6) * (ign*.55 + ring*2.4) + uAmber*ring*.12;
      float fade = 1. - smoothstep(uFade.x, uFade.y, d);
      gl_FragColor = vec4(base + hot, (g1*.35 + g5*.6 + cross + ring*.25) * fade * uOpacity);
    }` })); }
/* boot scene objects: singularity (core + anamorphic streak) and a floor shockwave */
const glowTex = (() => { const c = document.createElement('canvas'); c.width = c.height = 128; const x = c.getContext('2d'); const g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(.12, 'rgba(255,240,215,.9)'); g.addColorStop(.35, 'rgba(255,170,80,.28)'); g.addColorStop(1, 'rgba(255,120,30,0)'); x.fillStyle = g; x.fillRect(0, 0, 128, 128); return new THREE.CanvasTexture(c); })();
const singCore = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false }));
const singStreak = new THREE.Sprite(singCore.material.clone()); [singCore, singStreak].forEach(o => { o.position.set(0, 1.4, 0); o.visible = false; o.renderOrder = 9; scene.add(o); });
const shockMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uK: { value: 0 }, uCol: { value: new THREE.Color('#FFD9A0') } },
  vertexShader: `varying vec2 vW; void main(){ vW = position.xz; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `uniform float uK; uniform vec3 uCol; varying vec2 vW; void main(){ float d = length(vW), R = uK*26.; float a = exp(-pow((d-R)/(.25+uK*1.6),2.)) * (1.-uK) + exp(-pow((d-R*.62)/(.15+uK),2.)) * (1.-uK)*.4; gl_FragColor = vec4(uCol*a*2.2, a); }` });
const shock = new THREE.Mesh(new THREE.PlaneGeometry(64, 64).rotateX(-Math.PI / 2), shockMat); shock.position.y = .01; shock.visible = false; scene.add(shock);
const roomFloor = floorGrid(40); roomFloor.position.y = -.002; scene.add(roomFloor);

/* dust motes with RGB split */
function motes(n, box, color) { const g = new THREE.BufferGeometry(), p = new Float32Array(n * 3), s = new Float32Array(n);
  for (let i = 0; i < n; i++) { p.set([box[0] + rnd() * box[3], box[1] + rnd() * box[4], box[2] + rnd() * box[5]], i * 3); s[i] = rnd(); }
  g.setAttribute('position', new THREE.BufferAttribute(p, 3)); g.setAttribute('aSeed', new THREE.BufferAttribute(s, 1));
  const m = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uTime: { value: 0 }, uOpacity: { value: 0 }, uPx: { value: 600 }, uH: { value: box[4] }, uCol: { value: new THREE.Color(color) } },
    vertexShader: `attribute float aSeed; uniform float uTime,uPx,uH; varying float vTw;
      void main(){ vec3 p = position; p.y = position.y + mod(uTime*(.05+aSeed*.08) + aSeed*uH, uH) - position.y + ${box[1].toFixed(2)};
        p.x += sin(uTime*.4+aSeed*30.)*.25; p.z += cos(uTime*.3+aSeed*20.)*.25;
        vec4 mv = modelViewMatrix*vec4(p,1.); gl_Position = projectionMatrix*mv; gl_PointSize = (2.+aSeed*5.)*uPx/-mv.z*.02; vTw = .7+.3*sin(uTime*6.+aSeed*40.); }`,
    fragmentShader: `uniform float uOpacity; uniform vec3 uCol; varying float vTw;
      void main(){ vec2 c = gl_PointCoord*2.-1.; float g = exp(-dot(c,c)*6.); float r = exp(-dot(c-vec2(.3,0.),c-vec2(.3,0.))*6.); float b = exp(-dot(c+vec2(.3,0.),c+vec2(.3,0.))*6.);
        gl_FragColor = vec4(uCol*vec3(r,g,b)*2.5*vTw, g*uOpacity); }` });
  const pts = new THREE.Points(g, m); pts.frustumCulled = false; return pts; }
const roomDust = motes(MOBILE ? 140 : 300, [-4, 0, -3, 8, 3.2, 6], '#FBE08D'); scene.add(roomDust);

/* ================= TWIN (floor plan → extruded splat twin), group at y=-30 ================= */
const TW = new THREE.Group(); TW.position.set(0, -30, 0); scene.add(TW);
const PLAN = { wallH: 2.8,
  walls: [[0,0,20,0],[20,0,20,12],[20,12,0,12],[0,12,0,6.7],[0,5.3,0,0],[0,5,2.5,5],[3.7,5,9,5],[10.2,5,16,5],[17.2,5,20,5],[0,7,4,7],[5.2,7,9,7],[10,7,15,7],[16.2,7,20,7],[6,0,6,5],[13,0,13,5],[8,7,8,12],[11,7,11,12]],
  rooms: [['Reception',0,0,6,5,'#8c8274'],['Meeting Room 4',6,0,13,5,'#5d6672'],['Lab 2',13,0,20,5,'#9a9a94'],['Corridor',0,5,20,7,'#a89c88'],['Lounge',0,7,8,12,'#7b6046'],['Washroom',8,7,11,12,'#b8bcbf'],['Lecture Hall 21',11,7,20,12,'#6e5440']],
  boxes: [[8,1.5,11.5,3.2,.75,'#6b5a48','TABLE · 0.97'],[1,1.2,4,2.2,1.05,'#8b7f70','DESK · 0.93'],[14,1,19,2,.9,'#9aa3a8','LAB BENCH · 0.91'],[1,9.6,4,10.7,.8,'#46617a','3-SEATER SOFA · 0.94'],[12.2,8.4,19.4,8.95,.5,'#7a3f3a','SEATS ×42'],[12.2,9.5,19.4,10.05,.5,'#7a3f3a',''],[12.2,10.6,19.4,11.15,.5,'#7a3f3a','']],
  route: [[0.4,6],[9.6,6],[9.6,4.2],[12.4,4.2],[12.4,2.5]],
  pois: [['ENTRANCE',.4,6,'start'],['RECEPTION',3,3.4],['MEETING ROOM 4',12.4,2.5,'dest'],['LAB 2',16.5,3.4],['LOUNGE',5.6,8.4],['WASHROOM',9.5,9.6],['LECTURE HALL 21',15.8,11.6]] };
const W = (x, y, h = 0) => [x - 10, h, y - 6];
const twinList = []; seed = 13;
const pushT = (x, y, h, c, g) => twinList.push({ p: W(x, y, h), s: W(x, y, 0), c, g });
/* segmentation classes (SAM-style mask colours) revealed by the Label sweep */
const SEG_ROOM = ['#2FD3B9', '#FF7A1A', '#8C7BFF', '#3A4C70', '#B7DA4A', '#4FB2FF', '#FF4F7B'], SEG_WALL = '#93A8CC', SEG_OBJ = '#FFD24A';
PLAN.rooms.forEach(([, x0, y0, x1, y1, col], ri) => { const n = Math.round((x1 - x0) * (y1 - y0) * 60 * DENS); for (let i = 0; i < n; i++) pushT(lerp(x0, x1, rnd()), lerp(y0, y1, rnd()), 0, shade(col, .82 + rnd() * .36), SEG_ROOM[ri]); });
for (const [x0, y0, x1, y1] of PLAN.walls) { const n = Math.round(Math.hypot(x1 - x0, y1 - y0) * 180 * DENS); for (let i = 0; i < n; i++) { const t = rnd(); pushT(lerp(x0, x1, t) + (rnd() - .5) * .08, lerp(y0, y1, t) + (rnd() - .5) * .08, rnd() * PLAN.wallH, shade('#cfc8bb', .78 + rnd() * .3), SEG_WALL); } }
for (const [x0, y0, x1, y1, h, col] of PLAN.boxes) { const n = Math.round(((x1 - x0) * (y1 - y0) * 60 + (x1 - x0 + y1 - y0) * 30) * DENS);
  for (let i = 0; i < n; i++) { const top = rnd() < .6; let x = lerp(x0, x1, rnd()), y = lerp(y0, y1, rnd()); const z = top ? h : rnd() * h; if (!top) { if (rnd() < .5) y = rnd() < .5 ? y0 : y1; else x = rnd() < .5 ? x0 : x1; } pushT(x, y, z, shade(col, .8 + rnd() * .4), SEG_OBJ); } }
const twinMat = pointMaterial(); twinMat.uniforms.uSize.value = .42; const twin = buildPoints(twinList, twinMat); TW.add(twin);
const twinFloor = floorGrid(90); twinFloor.position.y = -.01; TW.add(twinFloor);

/* blueprint wall strokes drawn on the floor, then extruded wireframe walls */
const strokeMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uDraw: { value: 0 }, uOpacity: { value: 0 }, uCol: { value: AMBER } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `uniform float uDraw,uOpacity; uniform vec3 uCol; varying vec2 vUv; void main(){ if(vUv.x > uDraw) discard; float head = exp(-pow((uDraw-vUv.x)*14.,2.)); float core = 1.-abs(vUv.y*2.-1.); gl_FragColor = vec4(uCol*(1.4+head*4.)*core, core*uOpacity); }` });
const wallMeshes = [], edgeMat = new THREE.LineBasicMaterial({ color: '#FFD9A0', transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
const fillMat = new THREE.MeshBasicMaterial({ color: '#0d1626', transparent: true, opacity: 0, depthWrite: false });
for (const [x0, y0, x1, y1] of PLAN.walls) { const L = Math.hypot(x1 - x0, y1 - y0), ang = Math.atan2(y1 - y0, x1 - x0), [cx, , cz] = W((x0 + x1) / 2, (y0 + y1) / 2);
  const st = new THREE.Mesh(new THREE.PlaneGeometry(L, .14).rotateX(-Math.PI / 2), strokeMat); st.position.set(cx, .02, cz); st.rotation.y = -ang; TW.add(st);
  const g = new THREE.BoxGeometry(L, PLAN.wallH, .12); g.translate(0, PLAN.wallH / 2, 0); const m = new THREE.Mesh(g, fillMat); m.position.set(cx, 0, cz); m.rotation.y = -ang; m.scale.y = .001;
  m.add(new THREE.LineSegments(new THREE.EdgesGeometry(g), edgeMat)); TW.add(m); wallMeshes.push(m); }

/* Label sweep: a vertical sheet of light crosses the twin; points recolour by class behind it */
const sweepMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, uniforms: { uTime: { value: 0 }, uOpacity: { value: 0 }, uCol: { value: new THREE.Color('#FFD9A0') } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `${NOISE} uniform float uTime,uOpacity; uniform vec3 uCol; varying vec2 vUv;
    void main(){ float ends = smoothstep(0.,.12,vUv.x)*smoothstep(1.,.88,vUv.x); float h = vUv.y;
      float body = (.08 + .5*exp(-h*7.) + .25*smoothstep(.75,1.,h)) * (.6 + .4*vnoise(vec2(vUv.x*30., h*4. - uTime*3.)));
      float ticks = step(.93, fract(h*26. - uTime*1.5)) * .35 + step(.985, fract(vUv.x*48.)) * .25 * (1.-h);
      float a = (body + ticks + exp(-h*60.)*1.6) * ends * uOpacity; gl_FragColor = vec4(uCol*a*.55, a*.6); }` });
const sweep = new THREE.Mesh(new THREE.PlaneGeometry(14, 3.4).rotateY(Math.PI / 2).translate(0, 1.7, 0), sweepMat); sweep.visible = false; TW.add(sweep);
const trailMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uOpacity: { value: 0 }, uCol: { value: ORANGE } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `uniform float uOpacity; uniform vec3 uCol; varying vec2 vUv; void main(){ float a = pow(vUv.x, 3.) * smoothstep(0.,.1,vUv.y)*smoothstep(1.,.9,vUv.y) * .4 * uOpacity; gl_FragColor = vec4(uCol*a*1.5, a); }` });
const trail = new THREE.Mesh(new THREE.PlaneGeometry(4, 14).rotateX(-Math.PI / 2).translate(-2, .03, 0), trailMat); trail.visible = false; TW.add(trail);

/* AI detection boxes */
const aiBoxes = PLAN.boxes.map(([x0, y0, x1, y1, h, , lab], i) => { const g = new THREE.BoxGeometry(x1 - x0 + .2, h + .25, y1 - y0 + .2); g.translate(0, (h + .25) / 2, 0);
  const l = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: '#FF8A2A', transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
  const [cx, , cz] = W((x0 + x1) / 2, (y0 + y1) / 2); l.position.set(cx, 0, cz); l.scale.y = .001; TW.add(l); return { l, lab, top: h + .3, cx }; });

/* route ribbon with flowing data + riding particles + destination ring */
const curve = new THREE.CatmullRomCurve3(PLAN.route.flatMap((p, i, a) => { const v = new THREE.Vector3(...W(p[0], p[1], .06)); return [v]; }), false, 'catmullrom', .05);
const routeMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uDraw: { value: 0 }, uTime: { value: 0 }, uCol: { value: ORANGE }, uCore: { value: CORE } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `uniform float uDraw,uTime; uniform vec3 uCol,uCore; varying vec2 vUv;
    void main(){ if(vUv.x > uDraw) discard; float side = abs(vUv.y*2.-1.); float core = pow(1.-side, 3.);
      float flow = .55 + .45*sin(vUv.x*90. - uTime*6.); float head = exp(-pow((uDraw-vUv.x)*40.,2.));
      vec3 c = mix(uCol, uCore, core*.6) * (1.6*flow + head*6.); gl_FragColor = vec4(c, (.35+core)*.9); }` });
const route = new THREE.Mesh(new THREE.TubeGeometry(curve, 300, .07, 8, false), routeMat); TW.add(route);
const RIDERS = 60, riderGeo = new THREE.BufferGeometry(), riderPos = new Float32Array(RIDERS * 3), riderSeed = Array.from({ length: RIDERS }, () => rnd());
riderGeo.setAttribute('position', new THREE.BufferAttribute(riderPos, 3));
const riderMat = new THREE.PointsMaterial({ color: '#FFE0AE', size: .16, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }); const riders = new THREE.Points(riderGeo, riderMat); riders.frustumCulled = false; TW.add(riders);
const ringMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uTime: { value: 0 }, uOpacity: { value: 0 }, uCol: { value: new THREE.Color('#FFEEAC') } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `uniform float uTime,uOpacity; uniform vec3 uCol; varying vec2 vUv; void main(){ float d = length(vUv-.5)*2.; float a = 0.;
    for(int i=0;i<2;i++){ float t = fract(uTime*.5 + float(i)*.5); float R = t; a += exp(-pow((d-R)/(.06+t*.05),2.)) * (1.-t); }
    a += exp(-d*d*60.)*1.5; gl_FragColor = vec4(uCol*a*2., a*uOpacity); }` });
const destRing = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 3.2).rotateX(-Math.PI / 2), ringMat); destRing.position.set(...W(12.4, 2.5, .05)); TW.add(destRing);
const startRing = destRing.clone(); startRing.material = ringMat.clone(); startRing.position.set(...W(.4, 6, .05)); TW.add(startRing);

/* AR chevrons along the route (instanced) */
const chevShape = new THREE.Shape(); chevShape.moveTo(.35, 0); chevShape.lineTo(-.2, -.28); chevShape.lineTo(-.06, 0); chevShape.lineTo(-.2, .28); chevShape.closePath();
const CHEV = 26, chevMat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#FF7A1A').multiplyScalar(2.2), transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
const chevs = new THREE.InstancedMesh(new THREE.ShapeGeometry(chevShape).rotateX(-Math.PI / 2), chevMat, CHEV); chevs.frustumCulled = false; TW.add(chevs);
const twinDust = motes(MOBILE ? 200 : 450, [-12, 0, -8, 24, 3.5, 16], '#FFE0AE'); TW.add(twinDust);

/* ================= post: bloom + final (cubes wipe, chromatic intro, zoom-through, grain, vignette) ================= */
const composer = new EffectComposer(renderer); composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth / 2, innerHeight / 2), .5, .5, .78); composer.addPass(bloom);
const finalPass = new ShaderPass({ uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uWipe: { value: 0 }, uWipeOn: { value: 0 }, uDir: { value: 0 }, uCA: { value: 0 }, uCAg: { value: 0 }, uChrom: { value: 0 }, uZoom: { value: 0 }, uAspect: { value: 1 }, uGrain: { value: RM ? 0 : .045 }, uVig: { value: .55 }, uAmber: { value: AMBER } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `${NOISE}
    uniform sampler2D tDiffuse; uniform float uTime,uWipe,uWipeOn,uDir,uCA,uCAg,uChrom,uZoom,uAspect,uGrain,uVig; uniform vec3 uAmber; varying vec2 vUv;
    vec3 ca(vec2 uv, float amt){ vec2 cc = uv-.5; vec2 o = cc*dot(cc,cc)*amt;
      return vec3(texture2D(tDiffuse, uv+o).r, texture2D(tDiffuse, uv+o*.5).g*.5 + texture2D(tDiffuse, uv).g*.5, texture2D(tDiffuse, uv-o).b); }
    void main(){
      vec2 uv = vUv; float band = 0.; float n = 0.;
      if(uWipeOn > .5){
        vec2 q = vUv - .5; q.x *= uAspect;
        float coord = uDir < .5 ? vUv.y + (vUv.x-.5)*.24 : length(q)/(.5*length(vec2(uAspect,1.)));
        n = fbm(vUv*vec2(3.,7.) + vec2(0., uTime*.25));
        float front = coord + (n-.5)*.35;
        float prog = uWipe*1.6 - .3;
        band = exp(-pow((front-prog)/.085, 2.));
        float slab = hash(vec2(floor(vUv.y*96.), floor(uTime*20.)));
        uv.x += (slab-.5)*.07*band; uv.y -= (uDir<.5 ? .05 : 0.) * band;
        if(uDir > .5){ uv = .5 + (uv-.5)*(1. - .06*band); }
      }
      float amt = uCA*band*1.2 + uChrom*.9*smoothstep(1.4,.2,abs(vUv.x-.5)+abs(vUv.y-.5)) + uCAg + .02;
      vec3 col;
      if(uZoom > .001){ vec2 toC = .5 - uv; col = vec3(0.); float tot = 0.; for(float t=0.; t<14.; t++){ float pc = (t + hash(vUv*90.+t))/14.; float w = pc - pc*pc; col += ca(uv + toC*pc*uZoom, amt)*w; tot += w; } col /= tot; }
      else col = ca(uv, amt);
      col += uAmber * band * (.25 + n*.6);
      col = mix(col, vec3(1.), .12*uChrom*smoothstep(.2,.9, vnoise(vUv*4.+uTime)));
      vec2 vq = (vUv-.5)*vec2(uAspect*.75,1.); col *= 1. - uVig*smoothstep(.35,1.05,length(vq)*1.45);
      float tt = floor(uTime*24.); float g = (hash(gl_FragCoord.xy + tt*100.)*.7 + hash(gl_FragCoord.xy*.5 + tt*200.)*.3)*2.-1.;
      float lum = dot(col, vec3(.299,.587,.114)); col += g*uGrain*(1.-abs(clamp(lum,0.,1.)-.5)*2.) * vec3(1.,.9,1.1);
      gl_FragColor = vec4(col, 1.);
    }` });
composer.addPass(finalPass); composer.addPass(new OutputPass());

/* ================= camera timeline (hermite keyframes, hidden jumps behind wipes) ================= */
const K = [
  [0.00, [0, 1.7, 9.4], [0, 1.25, 0]], [0.06, [3.6, 2.4, 7.4], [0, 1.2, 0]], [0.125, [5.2, 3.4, 5.6], [0, 1.0, 0]],
  [0.17, [2.6, 2.1, 5.8], [0, 1.25, -.4]], [0.235, [-2.4, 1.7, 4.7], [0, 1.25, -.6]], [0.262, [-3.6, 2.6, 5.4], [0, 1.1, 0], 'cut'],
  [0.278, [0, -1.0, 4.5], [0, -30, 0], 'cut'], [0.34, [0, -10, 21], [0, -30, 0]], [0.43, [15, -14.5, 20], [0, -30.5, 0]],
  [0.535, [15, -14.5, 20], [0, -30.5, 0]], [0.6, [11, -21, 12], [1.5, -30, -1]], [0.69, [-4, -18, 17], [1, -30, -1]],
  [0.80, [-15, -22.5, 9], [0, -30, -.5]], [0.845, [-12.4, -28.3, 0.2], [-6, -28.6, 0], 'cut2'], [0.885, [-4.2, -28.3, 0.05], [3, -28.7, -.6]],
  [0.915, [-1.2, -28.3, -.4], [2.6, -29, -4]], [0.95, [0, -18, 24], [0, -30, 0]], [1.0, [0, -6, 36], [0, -30, 0]] ];
function hermite(arr, idx, p) { let i = 0; while (i < K.length - 2 && p > K[i + 1][0]) i++; const a = K[i], b = K[i + 1], t = clamp((p - a[0]) / (b[0] - a[0]));
  if (a[3] === 'cut' && b[3] === 'cut') { const e = ss(t); return a[arr].map((v, k) => lerp(v, b[arr][k], e)); }
  const tan = j => { const pv = K[Math.max(0, j - 1)], nx = K[Math.min(K.length - 1, j + 1)]; if (K[j][3]) return K[j][arr].map(() => 0); return K[j][arr].map((v, k) => (nx[arr][k] - pv[arr][k]) / (nx[0] - pv[0])); };
  const ta = tan(i), tb = tan(i + 1), dt = b[0] - a[0], t2 = t * t, t3 = t2 * t;
  return a[arr].map((v, k) => (2 * t3 - 3 * t2 + 1) * v + (t3 - 2 * t2 + t) * dt * ta[k] + (-2 * t3 + 3 * t2) * b[arr][k] + (t3 - t2) * dt * tb[k]); }

/* ================= virtual scroll: stretch bands, snap, intro gate (Edolus ScrollManager) ================= */
const TOTAL = 30000, TOUCH = 4200;
const STRETCH = [[.06, .085, 1.4], [.20, .27, 1.3], [.44, .535, 1.6], [.70, .83, 1.25]];
const SNAPS = [.13, .49, .985], SNAP_R = .016;
const segs = (() => { let acc = 0; const out = []; let last = 0; for (const [a, b, f] of STRETCH) { out.push([last, a, acc, acc + (a - last)]); acc += a - last; out.push([a, b, acc, acc + (b - a) * f]); acc += (b - a) * f; last = b; } out.push([last, 1, acc, acc + 1 - last]); acc += 1 - last; return { out, len: acc }; })();
const rawToP = r => { const v = r * segs.len; for (const [a, b, ra, rb] of segs.out) if (v <= rb) return lerp(a, b, (v - ra) / (rb - ra)); return 1; };
const pToRaw = p => { for (const [a, b, ra, rb] of segs.out) if (p <= b) return lerp(ra, rb, (p - a) / (b - a)) / segs.len; return 1; };
let rawT = 0, rawC = 0, lastIn = 0, gate = 'locked', intro = 0, introT = 0, bootT0 = 0, bootStage = -1, bootDone = 0;
function input(dpx) { if (gate === 'locked') return; lastIn = performance.now();
  if (gate === 'boot') return;
  if (gate === 'intro') { introT = clamp(introT + dpx / 1200); if (introT >= 1 && dpx > 0) gate = 'open'; return; }
  if (rawT <= 0 && dpx < 0) { gate = 'intro'; introT = clamp(introT + dpx / 1200); return; }
  rawT = clamp(rawT + dpx / TOTAL); }
addEventListener('wheel', e => { e.preventDefault(); input(e.deltaY * (e.deltaMode === 1 ? 18 : 1)); }, { passive: false });
let ty = null, dragging = false; addEventListener('touchstart', e => { ty = e.touches[0].clientY; }, { passive: true });
addEventListener('touchmove', e => { if (ty === null) return; if (!dragging) e.preventDefault(); const y = e.touches[0].clientY; if (!dragging) input((ty - y) * TOTAL / TOUCH / 2.2); ty = y; }, { passive: false });
addEventListener('keydown', e => { const k = e.key; if (['ArrowDown', 'PageDown', ' '].includes(k)) { e.preventDefault(); input(TOTAL * .03); } if (['ArrowUp', 'PageUp'].includes(k)) { e.preventDefault(); input(-TOTAL * .03); } });

/* ================= audio: synth stems mixed per band + sfx, with reverb ================= */
let AC = null, master, st = {}, soundOn = false, nextNote = 0, stepN = 0;
const MIX = [[0, .14, .9, 0, 0], [.14, .28, .9, .6, .2], [.28, .44, .8, .8, .5], [.44, .57, .7, .5, .6], [.57, .83, .8, 1, .9], [.83, .93, .9, .6, .7], [.93, 1.01, .7, 0, .1]];
function startAudio() { if (AC) return; AC = new (window.AudioContext || window.webkitAudioContext)(); master = AC.createGain(); master.gain.value = 0;
  const comp = AC.createDynamicsCompressor(); const verb = AC.createConvolver(); const len = AC.sampleRate * 2.6, ir = AC.createBuffer(2, len, AC.sampleRate);
  for (let c = 0; c < 2; c++) { const d = ir.getChannelData(c); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 2.6); } verb.buffer = ir;
  const wet = AC.createGain(); wet.gain.value = .35; master.connect(comp); master.connect(verb).connect(wet).connect(comp); comp.connect(AC.destination);
  const mk = () => { const g = AC.createGain(); g.gain.value = 0; g.connect(master); return g; }; st = { pad: mk(), bass: mk(), arp: mk(), fx: mk() }; st.fx.gain.value = .6;
  const lp = AC.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 700; lp.Q.value = .5; lp.connect(st.pad); st.lp = lp;
  [110, 130.81, 164.81, 196, 220].forEach(f => [-7, 7].forEach(dt => { const o = AC.createOscillator(); o.type = 'sawtooth'; o.frequency.value = f; o.detune.value = dt; const g = AC.createGain(); g.gain.value = .035; o.connect(g).connect(lp); o.start(); }));
  nextNote = AC.currentTime + .1; setInterval(sched, 50); }
const ARP = [220, 261.63, 329.63, 392, 440, 392, 329.63, 261.63], BPM = 88;
function sched() { if (!AC) return; const e8 = 60 / BPM / 2; while (nextNote < AC.currentTime + .25) { const t = nextNote;
  if (stepN % 4 === 0) { const o = AC.createOscillator(), g = AC.createGain(); o.type = 'sine'; o.frequency.setValueAtTime(stepN % 16 < 8 ? 55 : 49, t); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(.6, t + .03); g.gain.exponentialRampToValueAtTime(.001, t + e8 * 3.6); o.connect(g).connect(st.bass); o.start(t); o.stop(t + e8 * 4); }
  const o = AC.createOscillator(), g = AC.createGain(); o.type = 'triangle'; o.frequency.setValueAtTime(ARP[stepN % 8] * (stepN % 32 >= 16 ? 1.335 : 1), t); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(.14, t + .008); g.gain.exponentialRampToValueAtTime(.001, t + e8 * 1.4); o.connect(g).connect(st.arp); o.start(t); o.stop(t + e8 * 1.5);
  nextNote += e8; stepN++; } }
function mix(p) { if (!AC || !soundOn) return; const r = MIX.find(m => p >= m[0] && p < m[1]) || MIX[MIX.length - 1], t = AC.currentTime;
  st.pad.gain.setTargetAtTime(r[2] * .55, t, .7); st.bass.gain.setTargetAtTime(r[3] * .4, t, .7); st.arp.gain.setTargetAtTime(r[4] * .28, t, .7); st.lp.frequency.setTargetAtTime(500 + p * 2200, t, .9); }
function sfx(kind) { if (!AC || !soundOn) return; const t = AC.currentTime, o = AC.createOscillator(), g = AC.createGain();
  if (kind === 'whoosh') { const b = AC.createBuffer(1, AC.sampleRate * 1.2, AC.sampleRate), d = b.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.sin(Math.PI * i / d.length);
    const s = AC.createBufferSource(), f = AC.createBiquadFilter(); f.type = 'bandpass'; f.frequency.setValueAtTime(300, t); f.frequency.exponentialRampToValueAtTime(3000, t + 1.1); f.Q.value = 1.2; s.buffer = b; const gg = AC.createGain(); gg.gain.value = .5; s.connect(f).connect(gg).connect(st.fx); s.start(t); return; }
  if (kind === 'charge') { o.type = 'sawtooth'; const lp = AC.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.setValueAtTime(200, t); lp.frequency.exponentialRampToValueAtTime(2400, t + 1.7); o.frequency.setValueAtTime(55, t); o.frequency.exponentialRampToValueAtTime(440, t + 1.75);
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(.09, t + 1.6); g.gain.linearRampToValueAtTime(0, t + 1.78); o.connect(lp).connect(g).connect(st.fx); o.start(t); o.stop(t + 1.8); return; }
  if (kind === 'boom') { o.type = 'sine'; o.frequency.setValueAtTime(120, t); o.frequency.exponentialRampToValueAtTime(28, t + 1.4); g.gain.setValueAtTime(.5, t); g.gain.exponentialRampToValueAtTime(.001, t + 2.2); o.connect(g).connect(st.fx); o.start(t); o.stop(t + 2.3); return; }
  if (kind === 'chapter') { o.type = 'sine'; o.frequency.setValueAtTime(660, t); o.frequency.exponentialRampToValueAtTime(990, t + .3); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(.14, t + .02); g.gain.exponentialRampToValueAtTime(.001, t + .8); }
  else { o.type = 'square'; o.frequency.setValueAtTime(2000 + Math.random() * 1200, t); g.gain.setValueAtTime(.02, t); g.gain.exponentialRampToValueAtTime(.001, t + .03); }
  o.connect(g).connect(st.fx); o.start(t); o.stop(t + .9); }
function setSound(on) { soundOn = on; $('audio').classList.toggle('playing', on); $('audio').setAttribute('aria-pressed', String(on)); $('audio-l').textContent = on ? 'Audio on' : 'Audio off';
  if (on) { startAudio(); AC.resume(); master.gain.setTargetAtTime(.75, AC.currentTime, .6); } else if (AC) master.gain.setTargetAtTime(0, AC.currentTime, .25); }
$('audio').addEventListener('click', () => setSound(!soundOn));
document.addEventListener('visibilitychange', () => { if (AC) document.hidden ? AC.suspend() : soundOn && AC.resume(); });

/* ================= text scramble (Edolus TextScramble) ================= */
const GLYPHS = '!<>-_\\/[]{}—=+*^?#________';
function scramble(el, text, speed = 1) { if (RM) { el.textContent = text; return; } const q = [...text].map(ch => ({ ch, s: Math.floor(Math.random() * 24 / speed), e: 0, g: '' })); q.forEach(o => o.e = o.s + Math.floor(Math.random() * 30 / speed)); let f = 0;
  cancelAnimationFrame(el._raf); const tick = () => { let out = '', done = 0; for (const o of q) { if (f >= o.e) { out += o.ch; done++; } else if (f >= o.s) { if (!o.g || Math.random() < .28) o.g = GLYPHS[Math.random() * GLYPHS.length | 0]; out += `<span style="opacity:.55">${o.g}</span>`; } else out += ' '; }
    el.innerHTML = out; f++; if (done < q.length) el._raf = requestAnimationFrame(tick); }; tick(); }

/* ================= DOM: chapters, HUD typewriter, anchors, search, ending ================= */
const chaps = [...document.querySelectorAll('.chap')].map(el => { const [a, b] = el.dataset.w.split(',').map(Number); return { el, a, b, h: el.querySelector('h2'), on: false }; });
const HUDS = [[.18, .255, ['/// RECONSTRUCTION', 'FRAMES   : 1,280', 'POSES    : 1,254  98%', 'SPLATS   : 2,431,882', 'EXPOSURE : CORRECTED', 'TIME     : 24 MIN']],
  [.36, .435, ['/// TWIN · FLOOR 01', 'AREA     : 240 M²', 'WALLS    : 17', 'DOORS    : 8', 'WALKABLE : 61%', 'ACCURACY : ±3 CM']],
  [.548, .608, ['/// SEGMENTING', 'GAUSSIANS : 2,431,882', 'MODEL     : SAM 3.1 → 3D', 'CLASSES   : 14', 'OCR       : PP-OCRv6']],
  [.615, .695, ['/// DETECTIONS', 'SOFA     : 0.94', 'TABLE    : 0.97', 'BENCH    : 0.91', 'SIGN     : "LH 21"', 'POIS      : 23']]];
let hudCur = -1, hudStart = 0;
function hudUpdate(p, now) { const i = HUDS.findIndex(([a, b]) => p >= a && p <= b); const el = $('hud');
  if (i !== hudCur) { hudCur = i; hudStart = now; } if (i < 0) { el.style.opacity = 0; return; } const [a, b, lines] = HUDS[i]; el.style.opacity = String(clamp(Math.min((p - a) / .01, (b - p) / .01)));
  const chars = Math.floor((now - hudStart) / 1000 * 60); let left = chars, out = '';
  for (const ln of lines) { if (left <= 0) break; const s = ln.slice(0, left); left -= ln.length + 18; const parts = s.split(':'); out += parts.length > 1 ? `<span class="k">${parts[0]}:</span>${parts.slice(1).join(':')}\n` : s + '\n'; }
  $('hudtx').innerHTML = out.trimEnd() + '<span class="cur">_</span>'; }
const anchorWrap = $('anchors'); const anchors = PLAN.pois.map(([n, x, y, k]) => { const el = document.createElement('div'); el.className = 'anchor' + (k === 'dest' ? ' dest' : ''); el.innerHTML = `<span class="ring"></span><span class="leader"></span><span class="tx"></span>`; anchorWrap.appendChild(el); return { el, n, v: new THREE.Vector3(...W(x, y, 1.3)), on: false }; });
aiBoxes.forEach(b => { if (!b.lab) return; const el = document.createElement('div'); el.className = 'anchor ai'; el.innerHTML = `<span class="ring" style="border-color:#FF8A2A"></span><span class="leader"></span><span class="tx"></span>`; anchorWrap.appendChild(el);
  anchors.push({ el, n: b.lab, v: b.l.position.clone().setY(b.top), on: false, ai: true }); });
const tmp = new THREE.Vector3();

/* ================= loader → bands → hero → CTA ================= */
const lpct = $('lpct'), lword = [...$('lword').children]; let loadT = performance.now();
const heroText = ['Walk a building once.', 'Navigate it forever.'];
$('title').innerHTML = heroText.map((l, li) => `<span class="l l${li}">` + l.split(' ').map(w => `<span class="w">${[...w].map(c => `<span class="c">${c}</span>`).join('')}</span>`).join(' ') + '</span>').join('');
function runLoader() { const start = performance.now(), dur = RM ? 300 : 3000;
  lword.forEach((s, i) => s.animate([{ backgroundPosition: '100% 0' }, { backgroundPosition: '0% 0' }], { duration: RM ? 1 : 1100, delay: RM ? 0 : i * 220, easing: 'cubic-bezier(.65,0,.35,1)', fill: 'forwards' }));
  const step = () => { const k = clamp((performance.now() - start) / dur); lpct.textContent = Math.round(easeOutCubic(k) * 100) + '%'; if (k < 1) requestAnimationFrame(step); else reveal(); }; step(); }
function reveal() { renderer.compile(scene, camera); $('loader').classList.add('opened'); setTimeout(() => { $('loader').style.display = 'none'; }, 1700);
  setTimeout(() => { $('logo').classList.add('on'); $('audio').classList.add('on');
    const cs = [...document.querySelectorAll('#title .c')]; const order = cs.map((_, i) => i).sort(() => Math.random() - .5);
    order.forEach((ci, k) => cs[ci].animate([{ opacity: 0, filter: 'blur(12px)' }, { opacity: 1, filter: 'blur(0px)' }], { duration: RM ? 1 : 1200, delay: RM ? 0 : 200 + k * 28, easing: 'cubic-bezier(.16,1,.3,1)', fill: 'forwards' }));
    setTimeout(() => { $('hero').classList.add('ready'); const c = $('cta'); c.classList.add('line'); setTimeout(() => { c.classList.add('open'); scramble($('cta-l'), 'Initiate capture', 1.2); }, 520); }, RM ? 0 : 1100); }, RM ? 0 : 900); }
let chromT = -1;
function begin(withSound) { if (gate !== 'locked') return; if (withSound) setSound(true);
  $('hero').animate([{ opacity: 1, filter: 'blur(0px)' }, { opacity: 0, filter: 'blur(10px)' }], { duration: 900, easing: 'cubic-bezier(.65,0,.35,1)', fill: 'forwards' }).finished.then(() => $('hero').style.visibility = 'hidden');
  $('foot').classList.add('on'); $('pline').classList.add('on');
  /* boot scene: the cloud collapses into a singularity, the floor renders out from it, then a big bang re-expands the cloud.
     After that the visitor scrolls the cloud into the room, as before. */
  gate = 'boot'; bootT0 = performance.now(); bootStage = -1; $('icap').classList.add('on'); sfx('charge'); }
$('cta').addEventListener('click', () => begin(true)); $('silent').addEventListener('click', () => begin(false));
$('replay').addEventListener('click', () => { rawT = 0; gate = 'intro'; introT = 1; chromT = performance.now(); sfx('whoosh'); });
// skip: fast-forward through the rest of the film (it still plays, quickly) and land on the final card
let skipping = null;
$('skip').addEventListener('click', () => {
  if (gate === 'boot') { bootT0 = 0; bootDone = 1; $('icap').classList.remove('on'); }
  gate = 'open'; introT = 1; skipping = { from: rawT, t0: performance.now() }; sfx('whoosh');
});

/* ================= cursor ================= */
const cur = $('cursor'), orb = $('orb'), hintEl = $('hint'); let mx = innerWidth / 2, my = innerHeight / 2, cx = mx, cy = my, mN = new THREE.Vector2(9, 9), lastX = null, rotT = 0, rotV = 0, rot = 0;
let hintTxt = ''; function setHint(t) { if (t === hintTxt) return; hintTxt = t; if (!t) { hintEl.textContent = ''; return; } scramble(hintEl, t, 1.4); }
addEventListener('pointermove', e => { const dx = lastX === null ? 0 : e.clientX - lastX; lastX = e.clientX; mx = e.clientX; my = e.clientY; mN.set(mx / innerWidth * 2 - 1, -(my / innerHeight * 2 - 1)); if (dragging) rotV += dx * .0045; });
const exploring = p => p > .445 && p < .535;
$('gl').addEventListener('pointerdown', e => { lastX = e.clientX; if (exploring(pNow)) { dragging = true; $('gl').setPointerCapture(e.pointerId); sfx('blip'); } });
addEventListener('pointerup', () => dragging = false); addEventListener('pointercancel', () => dragging = false);

/* ================= resize + adaptive DPR ================= */
function resize() { const w = innerWidth, h = innerHeight; renderer.setPixelRatio(DPR); renderer.setSize(w, h); composer.setPixelRatio(DPR); composer.setSize(w, h); bloom.resolution.set(w / 2, h / 2);
  camera.aspect = w / h; camera.updateProjectionMatrix(); const px = h * DPR; [roomMat, twinMat].forEach(m => { m.uniforms.uPx.value = px; m.uniforms.uAspect.value = w / h; }); roomDust.material.uniforms.uPx.value = px; twinDust.material.uniforms.uPx.value = px;
  finalPass.uniforms.uAspect.value = w / h; }
addEventListener('resize', resize); resize();
let fpsAcc = 0, fpsN = 0, fpsT = 0;

/* ================= liquid glass (port of the portfolio's glass-rect.js + glass-touch.js) =================
   Refraction: raymarch-free rounded-box SDF lens, IOR 1.5, fresnel^4, R/B chromatic split, golden-angle frost,
   adaptive dimming where the scene behind is bright, materialises by bending harder while it appears.
   The plate is one quarter-size copy of the film canvas, taken right after the frame renders (the drawing
   buffer is still valid then) and shared by every panel. Touch/light: springs, not eases; at rest it does nothing.
   Phones and coarse pointers get the CSS glass only (no GPU or battery for the shader), as on the portfolio. */
const GLASS_PHONE = matchMedia('(max-width: 720px), (pointer: coarse)').matches, GLASS_RM = matchMedia('(prefers-reduced-motion: reduce)').matches;
const GLASS_FS = `precision highp float;
uniform vec2 uRes; uniform sampler2D uPlate; uniform vec4 uRect; uniform float uRadius,uDisp,uFrost,uReveal,uThick,uAlpha; uniform vec3 uTint;
const float IOR = 1.5;
float sdRoundBox(vec2 p, vec2 b, float r){ vec2 q = abs(p) - b + r; return min(max(q.x,q.y),0.0) + length(max(q,0.0)) - r; }
vec4 plate(vec2 uv){ return texture2D(uPlate, clamp(uv,0.0,1.0)); }
float rand(vec2 c){ return fract(sin(dot(c,vec2(12.9898,78.233)))*43758.5453); }
void main(){
  vec2 uv = gl_FragCoord.xy / uRes; vec2 p = (uv - 0.5) * 2.0; float aspect = uRes.x / max(uRes.y,1.0); p.x *= aspect;
  vec2 hb = vec2(aspect, 1.0) * 0.985; float r = min(uRadius, 0.98); float d = sdRoundBox(p, hb, r); if (d > 0.0) discard;
  float e = 0.004; vec2 g = vec2(sdRoundBox(p+vec2(e,0.),hb,r) - sdRoundBox(p-vec2(e,0.),hb,r), sdRoundBox(p+vec2(0.,e),hb,r) - sdRoundBox(p-vec2(0.,e),hb,r)) / (2.0*e);
  float edge = smoothstep(0.0, -0.16, d); vec3 n = normalize(vec3(g * edge, 1.0 - edge*0.55)); vec3 vd = vec3(0.,0.,1.);
  float fres = pow(1.0 - max(dot(n,vd),0.0), 4.0);
  vec3 rv = refract(-vd, n, 1.0/IOR); float mat = 1.0 - clamp(uReveal, 0.0, 1.0);
  vec2 off = rv.xy * (uThick * 0.055 * (1.0 + 2.2 * mat)); vec2 base = uRect.xy + uv * uRect.zw; vec2 sc = uRect.zw;
  vec3 col; float a = 0.0; vec2 uR = base + (off*(1.0+uDisp))*sc, uG = base + off*sc, uB = base + (off*(1.0-uDisp))*sc;
  float rnd = rand(uv)*6.2831853; const int N = 16; const float GA = 2.39996323; float rr = 0.0, st = 1.0/float(N); float bm = uFrost * 0.02 * (1.0 + 1.4 * mat);
  vec3 acc = vec3(0.0);
  for (int i=0;i<N;i++){ float th = float(i)*GA + rnd; rr += st; vec2 bo = vec2(cos(th),sin(th))*rr*bm*sc; acc.r += plate(uR+bo).r; vec4 gg = plate(uG+bo); acc.g += gg.g; a += gg.a; acc.b += plate(uB+bo).b; }
  col = acc/float(N);
  float lum = dot(col, vec3(0.299,0.587,0.114)); col *= 1.0 - 0.5 * smoothstep(0.22, 0.75, lum); col *= uTint;
  col += vec3(0.55,0.68,0.80) * fres * 0.34; col = mix(col, col*1.06 + 0.035, edge);
  float alpha = mix(uAlpha, 0.9, fres) * uReveal; alpha = max(alpha, uAlpha * 0.55 * uReveal);
  gl_FragColor = vec4(col, alpha);
}`;
const glassPlate = document.createElement('canvas'), glassPlateCtx = glassPlate.getContext('2d', { alpha: false }); let glassPlateAt = -1e9, glassPlateId = 0;
function Spring(x, k, ratio) { this.x = this.t = x; this.v = 0; this.k = k; this.c = 2 * ratio * Math.sqrt(k); }
Spring.prototype.step = function (dt) { const n = Math.max(1, Math.ceil(dt * 120)), h = dt / n; for (let i = 0; i < n; i++) { this.v += (this.k * (this.t - this.x) - this.c * this.v) * h; this.x += this.v * h; } return this.x; };
Spring.prototype.at = function (x) { this.x = this.t = x; this.v = 0; };
const glasses = [];
function glassAttach(el, o = {}) {
  const spec = document.createElement('div'), rim = document.createElement('div'); spec.className = 'glass-spec'; rim.className = 'glass-rim'; el.prepend(rim); el.prepend(spec);
  const G = { el, spec, rim, o, vis: o.vis || (() => 1), sx: new Spring(.5, 190, 1), sy: new Spring(.4, 190, 1), hov: new Spring(0, 130, 1), prs: new Spring(0, 320, .5), lx: new Spring(-.7, 110, 1), ly: new Spring(-.7, 110, 1), out: {}, gl: null, down: false };
  el.addEventListener('pointerdown', () => G.down = true); addEventListener('pointerup', () => G.down = false);
  if (!GLASS_PHONE && o.shader !== false) {
    const cv = document.createElement('canvas'); cv.className = 'glass-gl'; const gl = cv.getContext('webgl', { alpha: true, premultipliedAlpha: false, antialias: false });
    if (gl) { const sh = (t, src) => { const x = gl.createShader(t); gl.shaderSource(x, src); gl.compileShader(x); return x; };
      const pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, 'attribute vec2 aPos; void main(){ gl_Position = vec4(aPos,0.0,1.0); }')); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, GLASS_FS)); gl.linkProgram(pr);
      const par = gl.getExtension('KHR_parallel_shader_compile'); const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex); [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T].forEach(k => gl.texParameteri(gl.TEXTURE_2D, k, gl.CLAMP_TO_EDGE)); [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER].forEach(k => gl.texParameteri(gl.TEXTURE_2D, k, gl.LINEAR));
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); el.prepend(cv); el.classList.add('has-gl');
      G.gl = { gl, cv, pr, par, buf, tex, U: {}, linked: false, up: -1 }; }
  }
  glasses.push(G); return G;
}
function glassReady(R) { if (R.linked) return R.linked === true; const gl = R.gl; if (R.par && !gl.getProgramParameter(R.pr, R.par.COMPLETION_STATUS_KHR)) return false;
  if (!gl.getProgramParameter(R.pr, gl.LINK_STATUS)) { R.linked = 'bad'; R.cv.remove(); return false; }
  ['uRes', 'uPlate', 'uRect', 'uRadius', 'uDisp', 'uFrost', 'uTint', 'uReveal', 'uThick', 'uAlpha'].forEach(k => R.U[k] = gl.getUniformLocation(R.pr, k)); R.linked = true; return true; }
function glassFrame(dt) {
  let plateDone = false; const now = performance.now();
  for (const G of glasses) {
    const v = clamp(G.vis()); if (v < .005) continue; const r = G.el.getBoundingClientRect(); if (r.width < 2) continue;
    /* light: the pointer anywhere near the panel lights it (the panels themselves ignore the pointer) */
    const rx = (cx - r.left) / r.width, ry = (cy - r.top) / r.height, dxp = Math.max(r.left - cx, 0, cx - r.right), dyp = Math.max(r.top - cy, 0, cy - r.bottom);
    const near = Math.exp(-Math.hypot(dxp, dyp) / 160) * (MOBILE ? 0 : 1);
    G.sx.t = clamp(rx, -.2, 1.2); G.sy.t = clamp(ry, -.2, 1.2); G.hov.t = near; G.prs.t = G.down ? 1 : 0;
    const S = [G.sx, G.sy, G.hov, G.prs]; if (GLASS_RM) S.forEach(s => s.at(s.t)); else S.forEach(s => s.step(dt));
    const asp = r.width / r.height, ddx = (G.sx.x - .5) * asp, ddy = G.sy.x - .5, len = Math.hypot(ddx, ddy) || 1, lit = Math.min(1, G.hov.x * 1.4);
    G.lx.t = ddx / len * lit - .7 * (1 - lit); G.ly.t = ddy / len * lit - .7 * (1 - lit); if (GLASS_RM) { G.lx.at(G.lx.t); G.ly.at(G.ly.t); } else { G.lx.step(dt); G.ly.step(dt); }
    const glow = clamp(G.hov.x * .62 + Math.max(0, G.prs.x) * .95, 0, 1.3), ang = Math.atan2(G.lx.x, -G.ly.x) * 180 / Math.PI;
    const put = (k, val, eps, fn) => { if (!(Math.abs(val - (G.out[k] ?? 1e9)) <= eps)) { G.out[k] = val; fn(val); } };
    put('gx', G.sx.x, .003, x => G.spec.style.setProperty('--gx', x.toFixed(3))); put('gy', G.sy.x, .003, x => G.spec.style.setProperty('--gy', x.toFixed(3)));
    put('gh', glow, .006, x => G.spec.style.setProperty('--gh', x.toFixed(3))); put('la', ang, .8, x => G.rim.style.setProperty('--la', x.toFixed(1))); put('gr', clamp(.5 + glow * .5), .01, x => G.rim.style.setProperty('--gr', x.toFixed(2)));
    if (G.o.press) { const scl = 1 + .010 * G.prs.x + .003 * G.hov.x; put('sc', scl, 1e-4, x => G.el.style.scale = x.toFixed(4)); }
    /* refraction */
    const R = G.gl; if (!R || !glassReady(R)) continue; const gl = R.gl;
    if (!plateDone) { plateDone = true; const pw = Math.max(480, Math.round(innerWidth / 2)), ph = Math.max(270, Math.round(innerHeight / 2));
      if (glassPlate.width !== pw || glassPlate.height !== ph) { glassPlate.width = pw; glassPlate.height = ph; glassPlateAt = -1e9; }
      if (now - glassPlateAt > 45) { glassPlateAt = now; glassPlateId++; glassPlateCtx.drawImage(renderer.domElement, 0, 0, pw, ph); } }
    const w = Math.max(2, Math.round(r.width * .5)), h = Math.max(2, Math.round(r.height * .5)); if (R.cv.width !== w || R.cv.height !== h) { R.cv.width = w; R.cv.height = h; }
    gl.bindTexture(gl.TEXTURE_2D, R.tex); if (R.up !== glassPlateId) { R.up = glassPlateId; gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, glassPlate); }
    gl.viewport(0, 0, w, h); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.useProgram(R.pr); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, R.tex);
    const U = R.U, rad = (G.o.radius ?? 14) / (r.height / 2); gl.uniform1i(U.uPlate, 0); gl.uniform2f(U.uRes, w, h); gl.uniform4f(U.uRect, r.left / innerWidth, 1 - r.bottom / innerHeight, r.width / innerWidth, r.height / innerHeight);
    gl.uniform1f(U.uRadius, rad); gl.uniform1f(U.uDisp, (G.o.chromatic ?? 22) / 1000); gl.uniform1f(U.uFrost, (G.o.frost ?? 8) / 100); gl.uniform3f(U.uTint, .8, .86, .94); gl.uniform1f(U.uReveal, v);
    gl.uniform1f(U.uThick, G.o.thick ?? 1.5); gl.uniform1f(U.uAlpha, G.o.alpha ?? .2);
    gl.bindBuffer(gl.ARRAY_BUFFER, R.buf); const ap = gl.getAttribLocation(R.pr, 'aPos'); gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap, 2, gl.FLOAT, false, 0, 0); gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
const op = id => () => +($(id).style.opacity || 0);
glassAttach($('hud'), { vis: op('hud'), radius: 14, thick: 1.8 });
glassAttach($('search'), { vis: op('search'), radius: 16, thick: 1.6 });
glassAttach($('audio'), { shader: false, press: true, vis: () => $('audio').classList.contains('on') ? 1 : 0 });
document.querySelectorAll('.btn.ghost').forEach(b => glassAttach(b, { shader: false, press: true, vis: () => +($('final').style.opacity || 0) }));

/* ================= frame ================= */
let pNow = 0, pPrev = 0, last = performance.now(), lastChap = -1;
const CHN = ['Capture', 'Reconstruct', 'Twin', 'Explore', 'Label', 'Navigate', 'Arrive', 'Map your own'];
const QUERY = 'meeting room 4';
const dummy = new THREE.Object3D();
function frame(now) { requestAnimationFrame(frame); const dt = Math.min(.05, (now - last) / 1000); last = now; const time = now / 1000;
  if (document.hidden) return;
  /* scroll */
  if (skipping) { const k = Math.min(1, (now - skipping.t0) / 2600), e = k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2; rawT = lerp(skipping.from, 1, e); lastIn = now; if (k >= 1) skipping = null; }
  rawC = RM ? rawT : lerp(rawC, rawT, 1 - Math.pow(1 - .1, dt * 60));
  if (!RM && gate === 'open' && now - lastIn > 1000 && !dragging) { const p0 = rawToP(rawT); for (const s of SNAPS) if (Math.abs(s - p0) < SNAP_R && Math.abs(s - p0) > .0008) { rawT = pToRaw(s); lastIn = now; } }
  /* boot timeline (seconds): 0-.8 collapse, .8-1.75 floor renders, 1.75 bang, 3.1 hand over to scroll */
  let bang = 1, singI = 0, shockK = 0, bangPulse = 0, shake = 0, floorBoot = bootDone;
  if (bootT0) { const bt = BTFIX !== null ? +BTFIX : (now - bootT0) / 1000 * (RM ? 8 : 1), cr = clamp(bt / .8), fk = clamp((bt - .8) / .95), bk = clamp((bt - 1.75) / 1.35);
    const stage = bt < .8 ? 0 : bt < 1.75 ? 1 : 2; if (stage !== bootStage) { bootStage = stage; scramble($('icap-t'), ['Initialising spatial engine', 'Rendering ground plane', 'Expanding point field'][stage], 2.8);
      if (stage === 2) { chromT = now; sfx('boom'); sfx('whoosh'); } }
    bang = bt < 1.75 ? 1 - cr * cr * cr : 1 - Math.pow(1 - bk, 5);
    singI = bt < 1.75 ? ss(clamp((bt - .35) / .5)) * (1 + .25 * Math.sin(bt * 30) * fk) : Math.max(0, 1 - (bt - 1.75) / .5) * 2.2;
    floorBoot = bt < .8 ? 0 : 1 - Math.pow(1 - fk, 3); shockK = clamp((bt - 1.75) / 1.6); bangPulse = bt > 1.75 ? Math.max(0, Math.sin(Math.PI * clamp((bt - 1.75) / .7))) : 0; shake = bt > 1.75 ? Math.exp(-(bt - 1.75) * 5) : fk * .08;
    if (bt >= 3.1) { bootT0 = 0; bootDone = 1; gate = 'intro'; lastIn = now; $('icap').classList.remove('on'); } }
  const preBoot = gate === 'locked' ? 1 : 0;
  intro = RM ? introT : lerp(intro, introT, 1 - Math.pow(.9, dt * 60));
  const p = rawToP(rawC); pNow = p; mix(p);
  /* camera */
  const ip = ss(intro); const cpos = hermite(1, 0, p), ctgt = hermite(2, 0, p);
  const introOff = (1 - ip) * (gate === 'locked' ? 1 : 1); const shk = RM ? 0 : shake * .09; camera.position.set(cpos[0] + introOff * 2 + (Math.random() - .5) * shk, cpos[1] + introOff * 2.6 + (Math.random() - .5) * shk, cpos[2] + introOff * 9);
  const look = new THREE.Vector3(...ctgt); const ml = (win(p, .27, .30) * (1 - win(p, .43, .45)) + win(p, .71, .74)) * (MOBILE ? 0 : 1);
  look.x += (cx / innerWidth - .5) * 1.4 * ml + (cx / innerWidth - .5) * .5 * (1 - win(p, .25, .28)); look.y += -(cy / innerHeight - .5) * .9 * ml - (cy / innerHeight - .5) * .3 * (1 - win(p, .25, .28));
  camera.lookAt(look);
  /* room: capture + reconstruct */
  const roomOn = p < .3; room.visible = poses.visible = link.visible = roomFloor.visible = roomDust.visible = roomOn;
  if (roomOn) { const U = roomMat.uniforms; U.uTime.value = time; U.uMorph.value = .25 + .75 * Math.max(ip * .6, win(p, 0, .07)); U.uOpacity.value = (.45 + .35 * ip) * lerp(.7, 1, win(p, .145, .2)) * (1 - win(p, .255, .275));
    U.uColorMix.value = win(p, .145, .2); U.uSplat.value = win(p, .17, .245); U.uScan.value = p < .14 ? lerp(-.2, 3.1, (time * .25) % 1) : -10; U.uMouse.value.copy(mN); U.uPush.value = RM ? 0 : (1 - U.uSplat.value * .7);
    poseMat.uniforms.uCount.value = lerp(3, POSES + 1, win(p, 0, .12)) * Math.max(ip, .15); poseMat.uniforms.uOpacity.value = .9 * (1 - win(p, .17, .22)); linkMat.uniforms.uCount.value = poseMat.uniforms.uCount.value; linkMat.uniforms.uOpacity.value = .35 * (1 - win(p, .17, .22));
    roomFloor.material.uniforms.uOpacity.value = .9 * (1 - win(p, .24, .27)); roomFloor.material.uniforms.uR.value = (preBoot ? 3 : floorBoot * 13) + ip * 4 + win(p, 0, .14) * 10; U.uBang.value = bang;
    singCore.visible = singStreak.visible = singI > .001; singCore.material.opacity = singStreak.material.opacity = Math.min(1, singI); singCore.scale.setScalar(.5 + singI * .9); singStreak.scale.set(1.5 + singI * 9, .05 + singI * .06, 1);
    shock.visible = shockK > 0 && shockK < 1; shockMat.uniforms.uK.value = shockK; roomFloor.material.uniforms.uTime.value = time;
    roomDust.material.uniforms.uTime.value = time; roomDust.material.uniforms.uOpacity.value = win(p, .14, .18) * (1 - win(p, .25, .27)) * .9; }
  /* twin */
  const sweepX = lerp(-11.5, 11.8, ss(win(p, .548, .6))), frontOn = win(p, .543, .552) * (1 - win(p, .598, .608));
  if (p > .548 && pPrev <= .548) sfx('whoosh'); pPrev = p;
  const twinOn = p > .262; TW.visible = twinOn;
  if (twinOn) { rotV *= .95; rotT += rotV * dt * 60 * .016; rotT = clamp(rotT, -1.1, 1.1); if (!exploring(p) && !dragging) rotT *= .97; rot = lerp(rot, rotT, .1); TW.rotation.y = rot + (RM ? 0 : Math.sin(time * .12) * .025 * win(p, .4, .45));
    const fl = twinFloor.material.uniforms; fl.uOpacity.value = win(p, .27, .3) * (1 - win(p, .93, .99) * .7); fl.uR.value = 2 + win(p, .29, .37) * 40; fl.uTime.value = time;
    strokeMat.uniforms.uDraw.value = win(p, .285, .335); strokeMat.uniforms.uOpacity.value = win(p, .28, .29) * (1 - win(p, .4, .44) * .7);
    const ext = win(p, .33, .39); wallMeshes.forEach(m => m.scale.y = Math.max(.001, ext)); edgeMat.opacity = ext * .55 * (1 - win(p, .405, .44) * .75) + win(p, .835, .86) * .0; fillMat.opacity = ext * .5 * (1 - win(p, .4, .43));
    const T = twinMat.uniforms; T.uTime.value = time; T.uMorph.value = win(p, .345, .405); T.uOpacity.value = win(p, .33, .36); T.uColorMix.value = win(p, .37, .42); T.uSplat.value = win(p, .375, .43); T.uMouse.value.copy(mN); T.uPush.value = 0; T.uReveal.value = 3;
    /* label sweep */
    T.uSweep.value = sweepX; T.uFrontOn.value = frontOn; T.uSegMix.value = win(p, .545, .56) * (1 - .55 * win(p, .615, .66)) * (1 - win(p, .69, .715));
    sweep.visible = trail.visible = frontOn > 0; sweep.position.x = trail.position.x = sweepX; sweepMat.uniforms.uOpacity.value = trailMat.uniforms.uOpacity.value = frontOn; sweepMat.uniforms.uTime.value = time;
    aiBoxes.forEach((b, i) => { const k = ss(clamp((sweepX - b.cx) / 1.6)) * (1 - win(p, .7, .72)); b.l.scale.y = Math.max(.001, k); b.l.material.opacity = k * (.7 + .3 * Math.sin(time * 5 + i)); });
    const rd = win(p, .715, .765); routeMat.uniforms.uDraw.value = rd; routeMat.uniforms.uTime.value = time; route.visible = rd > 0;
    riderMat.opacity = rd * (1 - win(p, .92, .95)); for (let i = 0; i < RIDERS; i++) { const u = Math.min(rd, (riderSeed[i] + time * .09) % 1); const v = curve.getPointAt(u); riderPos.set([v.x + Math.sin(i) * .05, v.y + .05 + Math.cos(i * 3) * .04, v.z], i * 3); } riderGeo.attributes.position.needsUpdate = true;
    ringMat.uniforms.uTime.value = time; ringMat.uniforms.uOpacity.value = win(p, .755, .78) * (1 - win(p, .93, .96)); startRing.material.uniforms.uTime.value = time + .4; startRing.material.uniforms.uOpacity.value = rd > 0 ? win(p, .71, .73) * (1 - win(p, .84, .86)) : 0;
    const chA = win(p, .838, .86) * (1 - win(p, .918, .935)); chevMat.opacity = chA; chevs.visible = chA > 0;
    if (chA > 0) { for (let i = 0; i < CHEV; i++) { const u = clamp(((i + (RM ? 0 : (time * 1.4) % 1)) / CHEV)); const pt = curve.getPointAt(u), tg = curve.getTangentAt(u); dummy.position.set(pt.x, pt.y + .01, pt.z); dummy.rotation.set(0, -Math.atan2(tg.z, tg.x), 0); dummy.scale.setScalar(.9); dummy.updateMatrix(); chevs.setMatrixAt(i, dummy.matrix); } chevs.instanceMatrix.needsUpdate = true; }
    twinDust.material.uniforms.uTime.value = time; twinDust.material.uniforms.uOpacity.value = win(p, .4, .45) * .8; }
  /* post */
  const F = finalPass.uniforms; F.uTime.value = time;
  const w1 = clamp((p - .248) / (.292 - .248)), w3 = clamp((p - .818) / (.858 - .818));
  if (w1 > 0 && w1 < 1) { F.uWipeOn.value = 1; F.uWipe.value = w1; F.uDir.value = 0; F.uCA.value = 7.5; }
  else F.uWipeOn.value = 0;
  F.uZoom.value = RM ? 0 : Math.sin(Math.PI * w3) * .28 + bangPulse * .3; F.uCAg.value = RM ? 0 : frontOn * .05;
  const ce = chromT < 0 ? 0 : clamp((now - chromT) / 2600); F.uChrom.value = RM ? 0 : ss(1 - Math.abs(2 * ce - 1)) * (ce < 1 ? 1 : 0);
  bloom.strength = .5 - win(p, .15, .2) * .2 + win(p, .27, .3) * .2 - win(p, .4, .44) * .15 + win(p, .7, .76) * .3 + frontOn * .05 + bangPulse * .9;
  /* DOM */
  const ci = p < .14 ? 0 : p < .28 ? 1 : p < .44 ? 2 : p < .555 ? 3 : p < .7 ? 4 : p < .83 ? 5 : p < .925 ? 6 : 7;
  if (ci !== lastChap && gate !== 'locked') { if (lastChap !== -1) { sfx('chapter'); if (!RM) { $('logo').classList.add('glitch'); setTimeout(() => $('logo').classList.remove('glitch'), 160); setTimeout(() => $('logo').classList.add('glitch'), 240); setTimeout(() => $('logo').classList.remove('glitch'), 320); } } lastChap = ci; }
  $('readout').textContent = String(Math.round(p * 100)).padStart(3, '0') + ' · ' + CHN[ci]; $('pbar').style.width = p * 100 + '%';
  chaps.forEach(c => { const a = gate === 'locked' ? 0 : clamp(Math.min((p - c.a) / .012, (c.b - p) / .012)); c.el.style.opacity = a; c.el.style.filter = `blur(${(1 - a) * 8}px)`; c.el.style.transform = `translateY(${(1 - a) * 14}px)`;
    if (a > .5 && !c.on) { c.on = true; scramble(c.h, c.h.dataset.t); } if (a === 0) c.on = false; });
  hudUpdate(p, now);
  anchors.forEach((A, i) => { const a = A.ai ? ss(clamp((sweepX - A.v.x) / 1.6 - .4)) * (1 - win(p, .69, .7)) : win(p, .628 + i * .009, .64 + i * .009) * (1 - win(p, .815, .83)) + (A.el.classList.contains('dest') ? win(p, .76, .78) * (1 - win(p, .815, .83)) : 0);
    A.el.style.opacity = clamp(a); if (a > 0) { tmp.copy(A.v); TW.localToWorld(tmp); tmp.project(camera); const x = (tmp.x * .5 + .5) * innerWidth, y = (-tmp.y * .5 + .5) * innerHeight; A.el.style.transform = `translate(${x}px,${y}px)`;
      if (!A.on && a > .3) { A.on = true; scramble(A.el.querySelector('.tx'), A.n, 1.2); sfx('blip'); } A.el.querySelector('.leader').style.transform = `scaleX(${clamp(a * 1.5)})`; } else A.on = false; });
  const sA = clamp(Math.min((p - .705) / .01, (.83 - p) / .01)); $('search').style.opacity = sA; $('q').textContent = QUERY.slice(0, Math.round(win(p, .708, .73) * QUERY.length));
  const e = clamp((p - .925) / .075), bandK = clamp(e / .55); $('bt').style.transform = $('bb').style.transform = `scaleY(${bandK})`;
  $('dimblur').style.background = `rgba(0,0,0,${clamp(e / .22) * .5})`; const bl = ss(clamp(e / .4)) * 12; $('dimblur').style.backdropFilter = $('dimblur').style.webkitBackdropFilter = bl > .1 ? `blur(${bl}px)` : 'none';
  const o1 = ss(clamp(Math.min((e - .12) / .1, (.4 - e) / .08))), o2 = ss(clamp(Math.min((e - .42) / .09, (.64 - e) / .06))), o3 = ss(clamp((e - .66) / .11)); /* o3 must be 1 by the .985 snap (e = .8) */
  [['e1', o1], ['e2', o2]].forEach(([id, o]) => { $(id).style.opacity = o; $(id).style.filter = `blur(${(1 - o) * 10}px)`; });
  $('final').style.opacity = o3; $('final').style.transform = `translateY(${(1 - o3) * 18}px)`; $('final').style.pointerEvents = o3 > .5 ? 'auto' : 'none'; $('replay').style.opacity = o3; $('replay').style.pointerEvents = o3 > .5 ? 'auto' : 'none';
  const sk = gate !== 'locked' && !skipping && p < .9 ? 1 : 0; $('skip').style.opacity = sk; $('skip').style.pointerEvents = sk ? 'auto' : 'none';
  /* cursor */
  cx = lerp(cx, mx, 1 - Math.exp(-dt * 12)); cy = lerp(cy, my, 1 - Math.exp(-dt * 12)); cur.style.transform = `translate(${cx}px,${cy}px)`;
  const ex = gate === 'open' && exploring(p); orb.classList.toggle('on', ex);
  setHint(gate === 'locked' ? '' : ex ? (dragging ? 'Rotating' : 'Drag to rotate') : '');
  $('cue').classList.toggle('on', ((gate === 'intro' && introT < .05 && now - lastIn > 600) || (gate === 'open' && p < .1 && now - lastIn > 1300)));
  /* render */
  composer.render(dt);
  glassFrame(dt);
  /* adaptive DPR after warm-up */
  if (time > 4) { fpsAcc += dt; fpsN++; if (fpsAcc > 2) { const fps = fpsN / fpsAcc; fpsAcc = fpsN = 0; if (fps < 42 && DPR > .65) { DPR = Math.max(.65, DPR - .15); resize(); } else if (fps > 58 && DPR < Math.min(devicePixelRatio, 1.5) - .1) { DPR += .1; resize(); } } }
}
const DBG = new URLSearchParams(location.search).get('p'), BTFIX = new URLSearchParams(location.search).get('bt');
(document.fonts ? document.fonts.ready : Promise.resolve()).then(() => { if (BTFIX !== null) { $('loader').style.display = 'none'; $('hero').style.display = 'none'; gate = 'boot'; bootT0 = 1; $('icap').classList.add('on'); ['logo', 'audio', 'foot', 'pline'].forEach(i => $(i).classList.add('on')); } else if (DBG !== null) { $('loader').style.display = 'none'; $('hero').style.display = 'none'; gate = 'open'; introT = intro = 1; rawT = rawC = pToRaw(+DBG); ['logo', 'audio', 'foot', 'pline'].forEach(i => $(i).classList.add('on')); } else runLoader(); requestAnimationFrame(frame); });
