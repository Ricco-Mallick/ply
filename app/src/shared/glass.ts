/* Liquid glass: a port of the portfolio's glass-rect.js (refraction) + glass-touch.js (light and press springs).
   Refraction: rounded-box SDF lens, IOR 1.5, fresnel^4, R/B chromatic split, golden-angle frost, adaptive dimming where
   the scene behind is bright, and it materialises by bending harder while it appears. Every panel samples one shared
   half-size copy of the scene canvas, taken right after the scene renders (the drawing buffer is still valid then).
   Phones and coarse pointers get the CSS glass only, as on the portfolio. At rest it does no work. */

const PHONE = matchMedia('(max-width: 720px), (pointer: coarse)').matches;
const RM = matchMedia('(prefers-reduced-motion: reduce)').matches;
const clamp = (v: number, a = 0, b = 1) => Math.min(b, Math.max(a, v));

const FS = `precision highp float;
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
  vec2 uR = base + (off*(1.0+uDisp))*sc, uG = base + off*sc, uB = base + (off*(1.0-uDisp))*sc;
  float rnd = rand(uv)*6.2831853; const int N = 16; const float GA = 2.39996323; float rr = 0.0, st = 1.0/float(N); float bm = uFrost * 0.02 * (1.0 + 1.4 * mat);
  vec3 acc = vec3(0.0);
  for (int i=0;i<N;i++){ float th = float(i)*GA + rnd; rr += st; vec2 bo = vec2(cos(th),sin(th))*rr*bm*sc; acc.r += plate(uR+bo).r; acc.g += plate(uG+bo).g; acc.b += plate(uB+bo).b; }
  vec3 col = acc/float(N);
  float lum = dot(col, vec3(0.299,0.587,0.114)); col *= 1.0 - 0.5 * smoothstep(0.22, 0.75, lum); col *= uTint;
  col += vec3(0.55,0.68,0.80) * fres * 0.34; col = mix(col, col*1.06 + 0.035, edge);
  float alpha = mix(uAlpha, 0.9, fres) * uReveal; alpha = max(alpha, uAlpha * 0.55 * uReveal);
  gl_FragColor = vec4(col, alpha);
}`;

class Spring {
  x: number; t: number; v = 0; c: number;
  constructor(x: number, public k: number, ratio: number) { this.x = this.t = x; this.c = 2 * ratio * Math.sqrt(k); }
  step(dt: number) { const n = Math.max(1, Math.ceil(dt * 120)), h = dt / n; for (let i = 0; i < n; i++) { this.v += (this.k * (this.t - this.x) - this.c * this.v) * h; this.x += this.v * h; } return this.x; }
  at(x: number) { this.x = this.t = x; this.v = 0; }
}

export interface GlassOptions { shader?: boolean; press?: boolean; radius?: number; frost?: number; chromatic?: number; alpha?: number; thick?: number; vis?: () => number; }
interface GL { gl: WebGLRenderingContext; cv: HTMLCanvasElement; pr: WebGLProgram; par: any; buf: WebGLBuffer; tex: WebGLTexture; U: Record<string, WebGLUniformLocation | null>; linked: boolean | 'bad'; up: number; }
interface Glass { el: HTMLElement; spec: HTMLDivElement; rim: HTMLDivElement; o: GlassOptions; sx: Spring; sy: Spring; hov: Spring; prs: Spring; lx: Spring; ly: Spring; out: Record<string, number>; gl: GL | null; down: boolean; }

const glasses: Glass[] = [];
const plate = document.createElement('canvas'), plateCtx = plate.getContext('2d', { alpha: false })!;
let plateAt = -1e9, plateId = 0, source: HTMLCanvasElement | null = null, px = innerWidth / 2, py = innerHeight / 2;
addEventListener('pointermove', e => { px = e.clientX; py = e.clientY; }, { passive: true });

/** The canvas the glass refracts (the 3D scene). */
export function setGlassSource(c: HTMLCanvasElement) { source = c; }

export function attachGlass(el: HTMLElement, o: GlassOptions = {}) {
  if (el.querySelector(':scope > .glass-rim')) return () => {};
  const spec = document.createElement('div'), rim = document.createElement('div'); spec.className = 'glass-spec'; rim.className = 'glass-rim';
  spec.setAttribute('aria-hidden', 'true'); rim.setAttribute('aria-hidden', 'true'); el.prepend(rim); el.prepend(spec);
  const G: Glass = { el, spec, rim, o, sx: new Spring(.5, 190, 1), sy: new Spring(.4, 190, 1), hov: new Spring(0, 130, 1), prs: new Spring(0, 320, .5), lx: new Spring(-.7, 110, 1), ly: new Spring(-.7, 110, 1), out: {}, gl: null, down: false };
  const down = () => { G.down = true; }, up = () => { G.down = false; };
  el.addEventListener('pointerdown', down); addEventListener('pointerup', up);
  if (!PHONE && o.shader !== false) {
    const cv = document.createElement('canvas'); cv.className = 'glass-gl'; cv.setAttribute('aria-hidden', 'true');
    const gl = cv.getContext('webgl', { alpha: true, premultipliedAlpha: false, antialias: false });
    if (gl) {
      const sh = (t: number, src: string) => { const x = gl.createShader(t)!; gl.shaderSource(x, src); gl.compileShader(x); return x; };
      const pr = gl.createProgram()!; gl.attachShader(pr, sh(gl.VERTEX_SHADER, 'attribute vec2 aPos; void main(){ gl_Position = vec4(aPos,0.0,1.0); }')); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(pr);
      const buf = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      const tex = gl.createTexture()!; gl.bindTexture(gl.TEXTURE_2D, tex);
      [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T].forEach(k => gl.texParameteri(gl.TEXTURE_2D, k, gl.CLAMP_TO_EDGE)); [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER].forEach(k => gl.texParameteri(gl.TEXTURE_2D, k, gl.LINEAR));
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); el.prepend(cv); el.classList.add('has-gl');
      G.gl = { gl, cv, pr, par: gl.getExtension('KHR_parallel_shader_compile'), buf, tex, U: {}, linked: false, up: -1 };
    }
  }
  glasses.push(G);
  return () => { const i = glasses.indexOf(G); if (i >= 0) glasses.splice(i, 1); spec.remove(); rim.remove(); G.gl?.cv.remove(); el.classList.remove('has-gl'); el.removeEventListener('pointerdown', down); removeEventListener('pointerup', up); };
}

function ready(R: GL) {
  if (R.linked === true) return true; if (R.linked === 'bad') return false; const gl = R.gl;
  if (R.par && !gl.getProgramParameter(R.pr, R.par.COMPLETION_STATUS_KHR)) return false;
  if (!gl.getProgramParameter(R.pr, gl.LINK_STATUS)) { R.linked = 'bad'; R.cv.remove(); return false; }
  for (const k of ['uRes', 'uPlate', 'uRect', 'uRadius', 'uDisp', 'uFrost', 'uTint', 'uReveal', 'uThick', 'uAlpha']) R.U[k] = gl.getUniformLocation(R.pr, k);
  R.linked = true; return true;
}

/** Call once per frame right after the scene renders. */
export function glassFrame(dt: number) {
  let plateDone = false; const now = performance.now();
  for (const G of glasses) {
    const v = clamp(G.o.vis ? G.o.vis() : 1); if (v < .005) continue;
    const r = G.el.getBoundingClientRect(); if (r.width < 2) continue;
    const rx = (px - r.left) / r.width, ry = (py - r.top) / r.height, dxp = Math.max(r.left - px, 0, px - r.right), dyp = Math.max(r.top - py, 0, py - r.bottom);
    const near = PHONE ? 0 : Math.exp(-Math.hypot(dxp, dyp) / 160);
    G.sx.t = clamp(rx, -.2, 1.2); G.sy.t = clamp(ry, -.2, 1.2); G.hov.t = near; G.prs.t = G.down ? 1 : 0;
    for (const s of [G.sx, G.sy, G.hov, G.prs]) RM ? s.at(s.t) : s.step(dt);
    const asp = r.width / r.height, ddx = (G.sx.x - .5) * asp, ddy = G.sy.x - .5, len = Math.hypot(ddx, ddy) || 1, lit = Math.min(1, G.hov.x * 1.4);
    G.lx.t = ddx / len * lit - .7 * (1 - lit); G.ly.t = ddy / len * lit - .7 * (1 - lit); for (const s of [G.lx, G.ly]) RM ? s.at(s.t) : s.step(dt);
    const glow = clamp(G.hov.x * .62 + Math.max(0, G.prs.x) * .95, 0, 1.3), ang = Math.atan2(G.lx.x, -G.ly.x) * 180 / Math.PI;
    const put = (k: string, val: number, eps: number, fn: (x: number) => void) => { if (!(Math.abs(val - (G.out[k] ?? 1e9)) <= eps)) { G.out[k] = val; fn(val); } };
    put('gx', G.sx.x, .003, x => G.spec.style.setProperty('--gx', x.toFixed(3))); put('gy', G.sy.x, .003, x => G.spec.style.setProperty('--gy', x.toFixed(3)));
    put('gh', glow, .006, x => G.spec.style.setProperty('--gh', x.toFixed(3))); put('la', ang, .8, x => G.rim.style.setProperty('--la', x.toFixed(1)));
    put('gr', clamp(.5 + glow * .5), .01, x => G.rim.style.setProperty('--gr', x.toFixed(2)));
    if (G.o.press) put('sc', 1 + .010 * G.prs.x + .003 * G.hov.x, 1e-4, x => { G.el.style.scale = x.toFixed(4); });
    const R = G.gl; if (!R || !source || !ready(R)) continue; const gl = R.gl;
    if (!plateDone) {
      plateDone = true; const pw = Math.max(480, Math.round(innerWidth / 2)), ph = Math.max(270, Math.round(innerHeight / 2));
      if (plate.width !== pw || plate.height !== ph) { plate.width = pw; plate.height = ph; plateAt = -1e9; }
      if (now - plateAt > 45) { plateAt = now; plateId++; plateCtx.drawImage(source, 0, 0, pw, ph); }
    }
    const w = Math.max(2, Math.round(r.width * .5)), h = Math.max(2, Math.round(r.height * .5)); if (R.cv.width !== w || R.cv.height !== h) { R.cv.width = w; R.cv.height = h; }
    gl.bindTexture(gl.TEXTURE_2D, R.tex);
    if (R.up !== plateId) { R.up = plateId; gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, plate); }
    gl.viewport(0, 0, w, h); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.useProgram(R.pr); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, R.tex);
    const U = R.U; gl.uniform1i(U.uPlate, 0); gl.uniform2f(U.uRes, w, h); gl.uniform4f(U.uRect, r.left / innerWidth, 1 - r.bottom / innerHeight, r.width / innerWidth, r.height / innerHeight);
    gl.uniform1f(U.uRadius, (G.o.radius ?? 14) / (r.height / 2)); gl.uniform1f(U.uDisp, (G.o.chromatic ?? 22) / 1000); gl.uniform1f(U.uFrost, (G.o.frost ?? 8) / 100);
    gl.uniform3f(U.uTint, .8, .86, .94); gl.uniform1f(U.uReveal, v); gl.uniform1f(U.uThick, G.o.thick ?? 1.5); gl.uniform1f(U.uAlpha, G.o.alpha ?? .2);
    gl.bindBuffer(gl.ARRAY_BUFFER, R.buf); const ap = gl.getAttribLocation(R.pr, 'aPos'); gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap, 2, gl.FLOAT, false, 0, 0); gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
