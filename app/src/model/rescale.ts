// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The viewer's side of `plypipe rescale` (plan section 9, 2r). Two taps give a length in the twin's units; the true
   length typed in gives F = true / measured, and every metric thing in the model moves by F about the scene origin.
   rescale.py is the authority. These are its limits, its rounding, its `reference` block and its scene.json transform,
   so a correction measured here means the same thing as one the command applies to the twin's files. */
import type { SceneDocument } from './types';

export interface Point { x: number; y: number; z: number }

/** rescale.py's LIMITS: a factor outside this is a mistyped length, not a correction. */
export const LIMITS = [0.2, 5] as const;

/** The distance between the two points picked in the twin, in twin units (rescale.py's `measure`). */
export function measure(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

/** True length / the same length in the twin (rescale.py's `factor_from`), or null when either is not a length. */
export function factorFrom(knownM: number, measured: number): number | null {
  if (!(knownM > 0 && measured > 0 && Number.isFinite(knownM) && Number.isFinite(measured))) return null;
  return knownM / measured;
}

const r = (v: number) => Math.round(v * 1e6) / 1e6;
const vec = (v: number[], f: number, keep: number[] = []) => v.map((x, i) => (keep.includes(i) ? x : r(x * f)));

/** scene.json after a rescale by f (rescale.py's `scene_json`, returning a new object). `reference` is what the panel
    recorded: the true length typed in and the length the taps gave. */
export function rescaleScene(scene: SceneDocument, f: number, reference: { knownM: number; measuredM: number }): SceneDocument {
  const s = structuredClone(scene);
  const old = s.scaleSource === 'reference' ? s.reference : undefined;
  const firstSource = old ? old.previousSource : s.scaleSource;
  const firstScale = old ? old.previousScale : s.scale;
  if (typeof s.scale === 'number') s.scale = s.scale * f;
  if (Array.isArray(s.transform)) s.transform = s.transform.map((row, i) => (i < 3 ? row.map((v) => v * f) : row.slice()));
  if (s.bounds) s.bounds = Object.fromEntries(Object.entries(s.bounds).map(([k, v]) => [k, vec(v, f)]));
  if (Array.isArray(s.cameraPath)) s.cameraPath = s.cameraPath.map((p) => vec(p, f));
  // the eye height in spawn and tour is a person's, not the room's (the viewer reads spawn's y as the eye)
  if (s.spawn) s.spawn = Object.fromEntries(Object.entries(s.spawn).map(([k, v]) => [k, Array.isArray(v) ? vec(v, f, [1]) : v]));
  if (Array.isArray(s.tour)) s.tour = s.tour.map((kf) => Object.fromEntries(Object.entries(kf).map(([k, v]) => [k, (k === 'pos' || k === 'target') && Array.isArray(v) ? vec(v, f, [1]) : v])));
  if (s.topview) {
    for (const k of ['metersPerPixel', 'cutY']) if (typeof s.topview[k] === 'number') s.topview[k] = r(s.topview[k] as number * f);
    if (Array.isArray(s.topview.origin)) s.topview.origin = vec(s.topview.origin, f);
  }
  s.scaleSource = 'reference';
  s.reference = {
    factor: r(f * (old?.factor ?? 1)),
    appliedFactor: r(f),
    previousSource: firstSource,
    previousScale: firstScale,
    scale: s.scale,
    knownM: reference.knownM,
    measuredM: reference.measuredM,
  };
  return s;
}

/** The scale as the viewer prints it: 4 decimals at most, no trailing zeros. */
export const fmtScale = (v: number) => String(Number(v.toFixed(4)));
