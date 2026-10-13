/* Frame quality from the video itself, on a small grey copy (~160 px wide) several times a second:
   - sharpness: variance of the Laplacian (what the pipeline's frame picker uses), judged against this take's own median
     so a dim room isn't marked blurry everywhere;
   - exposure: mean luma and the share of clipped pixels;
   - motion: global image shift between samples by coarse block matching, in % of frame width per second.
   Combined with the gyro turn rate into a grade per sample: good / ok / bad. */

export type Grade = 0 | 1 | 2; // 0 good, 1 ok, 2 bad
export interface Quality { t: number; sharp: number; sharpRel: number; luma: number; lumaMed: number; clipped: number; flow: number; turn: number; grade: Grade; issue: Issue | null; }
export type Issue = 'blur' | 'fast' | 'turn' | 'dark' | 'bright' | 'tilt';

const W = 160;

export class Analyzer {
  private c = document.createElement('canvas'); // GPU-backed on purpose: willReadFrequently makes the canvas CPU-side, so every sample copied the full camera frame to
  // the CPU before shrinking it. This way the GPU shrinks it and only the 160 px copy is read back.
  private x = this.c.getContext('2d')!;
  private prev: Float32Array | null = null; private prevT = 0; private h = 90; private sharpHist: number[] = [];

  /** Grey pixels of the current frame at W px wide. */
  private grab(src: CanvasImageSource, sw: number, sh: number) {
    this.h = Math.max(2, Math.round(W * sh / sw)); if (this.c.width !== W || this.c.height !== this.h) { this.c.width = W; this.c.height = this.h; }
    this.x.drawImage(src, 0, 0, W, this.h); const d = this.x.getImageData(0, 0, W, this.h).data, g = new Float32Array(W * this.h);
    for (let i = 0; i < g.length; i++) g[i] = .299 * d[i * 4] + .587 * d[i * 4 + 1] + .114 * d[i * 4 + 2];
    return g;
  }

  measure(src: CanvasImageSource, sw: number, sh: number, t: number, turn = 0, pitch: number | null = null): Quality {
    const g = this.grab(src, sw, sh), w = W, h = this.h;
    let sum = 0, sum2 = 0, n = 0, lsum = 0, clip = 0;
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x, L = g[i - 1] + g[i + 1] + g[i - w] + g[i + w] - 4 * g[i]; sum += L; sum2 += L * L; n++;
    }
    for (let i = 0; i < g.length; i++) { lsum += g[i]; if (g[i] > 250 || g[i] < 4) clip++; }
    const sharp = sum2 / n - (sum / n) ** 2, luma = lsum / g.length, clipped = clip / g.length;
    // median brightness ignoring blown highlights: what exposure metering should aim at (a lamp in view shouldn't darken the room)
    const hist = new Uint32Array(64); let m = 0; for (let i = 0; i < g.length; i++) if (g[i] < 248) { hist[g[i] >> 2]++; m++; }
    let acc = 0, lumaMed = luma; for (let b = 0; b < 64; b++) { acc += hist[b]; if (acc >= m / 2) { lumaMed = b * 4 + 2; break; } }
    // global shift: try offsets on a coarse grid, keep the one with the lowest mean absolute difference
    let flow = 0;
    if (this.prev && t > this.prevT) {
      let best = Infinity, bx = 0, by = 0; const R = 12, step = 2;
      for (let oy = -R; oy <= R; oy += step) for (let ox = -R; ox <= R; ox += step) {
        let s = 0, k = 0; for (let y = R; y < h - R; y += 3) for (let x = R; x < w - R; x += 3) { s += Math.abs(g[y * w + x] - this.prev[(y + oy) * w + x + ox]); k++; }
        if (s / k < best) { best = s / k; bx = ox; by = oy; }
      }
      flow = Math.hypot(bx, by) / w * 100 / ((t - this.prevT) / 1000); // % of width per second
    }
    this.prev = g; this.prevT = t;
    this.sharpHist.push(sharp); if (this.sharpHist.length > 150) this.sharpHist.shift();
    const med = [...this.sharpHist].sort((a, b) => a - b)[Math.floor(this.sharpHist.length / 2)] || sharp;
    const sharpRel = sharp / Math.max(med, 1e-6);
    return { t, sharp, sharpRel, luma, lumaMed, clipped, flow, turn, ...grade(sharp, sharpRel, luma, clipped, flow, turn, pitch) };
  }
  reset() { this.prev = null; this.sharpHist = []; }
}

/** Worst problem wins; the order is what the live tip mentions first. */
function grade(sharp: number, rel: number, luma: number, clipped: number, flow: number, turn: number, pitch: number | null): { grade: Grade; issue: Issue | null } {
  const bad: Issue[] = [], ok: Issue[] = [];
  if (turn > 75 || flow > 55) bad.push(turn > 75 ? 'turn' : 'fast'); else if (turn > 40 || flow > 32) ok.push(turn > 40 ? 'turn' : 'fast');
  if (rel < .35 || sharp < 6) bad.push('blur'); else if (rel < .6) ok.push('blur');
  if (luma < 28) bad.push('dark'); else if (luma < 55) ok.push('dark');
  if (clipped > .35) bad.push('bright'); else if (clipped > .18) ok.push('bright');
  if (pitch != null && (pitch < -20 || pitch > 45)) ok.push('tilt');
  return bad.length ? { grade: 2, issue: bad[0] } : ok.length ? { grade: 1, issue: ok[0] } : { grade: 0, issue: null };
}

export const TIP: Record<Issue, string> = {
  turn: 'Turn slower: rotate with your whole body', fast: 'Slow down: about one step a second', blur: 'Hold steady, the frame is blurring',
  dark: 'Too dark: switch on more lights', bright: 'Too bright: angle away from the window', tilt: 'Hold the phone upright, tilted slightly down',
};

/** Merge runs of bad samples longer than `min` seconds into "recapture this" spans. */
export function problemSpans(q: Quality[], min = 1.5) {
  const out: { from: number; to: number; issue: Issue }[] = []; let start = -1, issue: Issue | null = null;
  for (let i = 0; i <= q.length; i++) {
    const bad = i < q.length && q[i].grade === 2;
    if (bad && start < 0) { start = i; issue = q[i].issue; }
    if (!bad && start >= 0) { const from = q[start].t / 1000, to = q[i - 1].t / 1000; if (to - from >= min) out.push({ from, to, issue: issue! }); start = -1; }
  }
  return out;
}
export function summarise(q: Quality[]) {
  const n = Math.max(1, q.length), good = q.filter(s => s.grade === 0).length / n, bad = q.filter(s => s.grade === 2).length / n;
  const verdict: 'good' | 'ok' | 'recapture' = bad > .25 ? 'recapture' : good > .6 && bad < .1 ? 'good' : 'ok';
  return { good, ok: 1 - good - bad, bad, verdict };
}
