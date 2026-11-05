// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Device tiering for the walkthrough: where the splat budget and render resolution start, how far the adaptive loop may
   go, and what it learned on this device last time. Starting near what the device can carry matters more on old
   hardware than the adaptive loop itself: every step down costs a few seconds of stutter on the way. */

export type Tier = 'low' | 'mid' | 'high';
export interface PerfProfile {
  tier: Tier; mobile: boolean;
  /** splat budget: initial, adaptive floor, adaptive ceiling */
  budget: number; floor: number; max: number;
  /** device-pixel-ratio cap: initial and adaptive floor */
  dpr: number; dprMin: number;
  /** float32 centres + half-float attributes (doubles GPU memory) */
  extSplats: boolean;
  /** prefer the scene's lite SPZ when it has one */
  lite: boolean;
}

const KEY = 'ply.perf.v1';
// GPUs that struggle with a million-splat sort and fill: old Mali/Adreno/PowerVR phones, Intel iGPUs before Iris Xe,
// and software rasterisers
const WEAK_GPU = /Mali-(T|G[357]\d\b|G5[12])|Adreno \(TM\) ([1-5]\d\d|6[0-3]\d)\b|PowerVR|Intel\(R\) (HD|UHD) Graphics|Mesa Intel\(R\) (HD|UHD)|SwiftShader|llvmpipe|softpipe/i;

export const PROFILES: Record<Tier, (mobile: boolean) => Omit<PerfProfile, 'tier' | 'mobile'>> = {
  low: () => ({ budget: 600_000, floor: 300_000, max: 1_000_000, dpr: 1.0, dprMin: 0.6, extSplats: false, lite: true }),
  mid: m => ({ budget: m ? 1_000_000 : 1_500_000, floor: 500_000, max: m ? 1_600_000 : 2_000_000, dpr: 1.25, dprMin: 0.75, extSplats: false, lite: m }),
  high: m => ({ budget: m ? 1_600_000 : 2_500_000, floor: 700_000, max: 2_500_000, dpr: 1.5, dprMin: 0.75, extSplats: !m, lite: m }),
};

function gpuInfo(): { renderer: string; maxTexture: number } {
  try {
    const gl = document.createElement('canvas').getContext('webgl2') as WebGL2RenderingContext | null;
    if (!gl) return { renderer: '', maxTexture: 0 };
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    const info = { renderer: String(gl.getParameter(dbg ? dbg.UNMASKED_RENDERER_WEBGL : gl.RENDERER) ?? ''), maxTexture: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number };
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return info;
  } catch { return { renderer: '', maxTexture: 0 }; }
}

export function detectTier(s: { mobile: boolean; memory?: number; cores?: number; renderer: string; maxTexture: number }): Tier {
  const { mobile, memory, cores, renderer, maxTexture } = s;
  // a 0 here means WebGL 2 itself failed to answer; the renderer will tell the user, not us
  if ((maxTexture && maxTexture < 8192) || WEAK_GPU.test(renderer) || (memory !== undefined && memory <= 2) || (cores !== undefined && cores <= 2)) return 'low';
  if ((memory !== undefined && memory <= 4) || (cores !== undefined && cores <= 4)) return 'mid';
  // Android phones have a wide spread; only call one high when it says it has the memory and cores for it
  // (Spark's own Android default is 1M splats)
  if (mobile && /Android/i.test(navigator.userAgent) && !((memory ?? 0) >= 8 && (cores ?? 0) >= 8)) return 'mid';
  return 'high';
}

let cached: PerfProfile | null = null;
export function deviceProfile(): PerfProfile {
  if (cached) return cached;
  const mobile = matchMedia('(pointer: coarse), (max-width: 720px)').matches;
  const forced = new URLSearchParams(location.search).get('quality');  // ?quality=low|mid|high to test a tier
  const nav = navigator as Navigator & { deviceMemory?: number };
  const tier: Tier = forced === 'low' || forced === 'mid' || forced === 'high' ? forced
    : detectTier({ mobile, memory: nav.deviceMemory, cores: nav.hardwareConcurrency || undefined, ...gpuInfo() });
  const p: PerfProfile = { tier, mobile, ...PROFILES[tier](mobile) };
  // what the adaptive loop settled on here last time: start there instead of re-learning it with a stutter
  const learned = forced ? null : loadLearned();
  if (learned) {
    p.budget = Math.max(p.floor, Math.min(p.budget, learned.budget));
    p.dpr = Math.max(p.dprMin, Math.min(p.dpr, learned.dpr));
  }
  return cached = p;
}

function loadLearned(): { budget: number; dpr: number } | null {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? 'null');
    return v && Number.isFinite(v.budget) && Number.isFinite(v.dpr) ? v : null;
  } catch { return null; }
}
/** Quality picked in the viewer: auto adapts to the frame rate, a tier is held fixed. */
export type QualityMode = 'auto' | Tier;
const MODE_KEY = 'ply.quality.v1';
export function loadMode(): QualityMode {
  try { const v = localStorage.getItem(MODE_KEY); return v === 'low' || v === 'mid' || v === 'high' ? v : 'auto'; } catch { return 'auto'; }
}
export function saveMode(m: QualityMode) { try { localStorage.setItem(MODE_KEY, m); } catch { /* private mode */ } }

export function saveLearned(budget: number, dpr: number) {
  try { localStorage.setItem(KEY, JSON.stringify({ budget: Math.round(budget), dpr })); } catch { /* private mode */ }
}
