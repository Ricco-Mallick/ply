// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Places: the walkthrough's pins, readable and writable over HTTP (round 2, interactive spatial POI tagging).

   A scene's places live at `scenes/<sceneId>/places.json`, next to scene.json. The GPU worker writes the file when it
   publishes (the label stage's AI detections, see pipeline/plypipe/stages/label.py); this module validates the body a
   person sends when they add or move a pin, so one bad place cannot corrupt the set the viewer reads.

   The stored document is `{schema: 1, updatedAt, places}`. Older published files use `{version: 1, source, places}`
   (the pipeline shape); `normalize()` reads either and always answers with schema 1, so the app has one contract. */

export const CATEGORIES = ['Furniture', 'Storage', 'Exits', 'Things'] as const;
export type Category = (typeof CATEGORIES)[number];

/** A place, as app/src/lab/places.ts defines it. Unknown keys are kept: the pipeline puts extras such as ai.frames in
    the same object, and the app round-trips what it read. */
export interface Place {
  id: string; name: string; cat: Category; p: [number, number, number];
  ai?: { cls: string; conf: number; half: [number, number, number]; [k: string]: unknown };
  pinned?: boolean; [k: string]: unknown;
}

/** The document the API reads and writes. `updatedAt` is null for files written before this endpoint existed. */
export interface PlacesDoc { schema: 1; updatedAt: string | null; places: Place[] }

export const MAX_PLACES = 500;
export const MAX_NAME = 80;
export const MAX_ID = 41;                 // ^[a-z0-9][a-z0-9-]{0,40}$
export const MAX_AI_CLS = 80;
export const ID_RE = /^[a-z0-9][a-z0-9-]{0,40}$/;
/** Points are metres, floor y = 0. Without a cheap per-scene bound, the XZ bound is a sanity check on the units. */
export const WORLD_LIMIT = 5000;
/** A full 500-place set is well under 100 KB; the cap keeps a hand-written body from being buffered without limit. */
export const MAX_BODY_BYTES = 512 * 1024;

export interface PlaceError { ok: false; error: string; index: number | null; field: string }
export type Validated = { ok: true; places: Place[] } | PlaceError;

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

function bad(index: number | null, field: string, why: string): PlaceError {
  return { ok: false, error: index === null ? why : `place ${index}: ${why}`, index, field };
}

/** `places` -> a clean list, or the first problem, named by place index and field. */
export function validatePlaces(body: unknown): Validated {
  if (!isObj(body)) return bad(null, 'places', 'body must be {"places": Place[]}');
  const list = body.places;
  if (!Array.isArray(list)) return bad(null, 'places', 'body.places must be an array');
  if (list.length > MAX_PLACES) return bad(null, 'places', `at most ${MAX_PLACES} places per scene (got ${list.length})`);

  const seen = new Map<string, number>();
  const out: Place[] = [];
  for (let i = 0; i < list.length; i++) {
    const raw = list[i];
    if (!isObj(raw)) return bad(i, 'place', 'must be an object');
    const { id, name, cat, p, ai, pinned, ...rest } = raw;

    if (typeof id !== 'string' || !ID_RE.test(id)) {
      return bad(i, 'id', `id must match ^[a-z0-9][a-z0-9-]{0,40}$ (got ${JSON.stringify(id)})`);
    }
    const first = seen.get(id);
    if (first !== undefined) return bad(i, 'id', `id ${JSON.stringify(id)} is already used by place ${first}`);
    seen.set(id, i);

    if (typeof name !== 'string') return bad(i, 'name', 'name must be a string');
    const trimmed = name.trim();
    if (!trimmed.length || trimmed.length > MAX_NAME) return bad(i, 'name', `name must be 1..${MAX_NAME} characters (got ${trimmed.length})`);

    if (typeof cat !== 'string' || !(CATEGORIES as readonly string[]).includes(cat)) {
      return bad(i, 'cat', `cat must be one of ${CATEGORIES.join(', ')} (got ${JSON.stringify(cat)})`);
    }

    if (!Array.isArray(p) || p.length !== 3) return bad(i, 'p', 'p must be [x, y, z] in metres');
    for (const v of p) if (!finite(v)) return bad(i, 'p', `p must be three finite numbers in metres (got ${JSON.stringify(p)})`);
    if (Math.abs(p[0]) >= WORLD_LIMIT || Math.abs(p[2]) >= WORLD_LIMIT) {
      return bad(i, 'p', `x and z must be within ±${WORLD_LIMIT} m (got ${JSON.stringify(p)})`);
    }

    let cleanAi: Place['ai'];
    if (ai !== undefined) {
      if (!isObj(ai)) return bad(i, 'ai', 'ai must be an object when present');
      const { cls, conf, half, ...aiRest } = ai;
      if (typeof cls !== 'string' || !cls.trim() || cls.length > MAX_AI_CLS) return bad(i, 'ai.cls', `ai.cls must be 1..${MAX_AI_CLS} characters`);
      if (!finite(conf)) return bad(i, 'ai.conf', 'ai.conf must be a finite number');
      if (!Array.isArray(half) || half.length !== 3 || !half.every(finite)) return bad(i, 'ai.half', 'ai.half must be three finite numbers');
      cleanAi = { ...aiRest, cls, conf, half: half as [number, number, number] };
    }

    if (pinned !== undefined && typeof pinned !== 'boolean') return bad(i, 'pinned', 'pinned must be true or false');

    out.push({ ...rest, id, name: trimmed, cat: cat as Category, p: p as [number, number, number], ...(cleanAi ? { ai: cleanAi } : {}), ...(pinned !== undefined ? { pinned } : {}) });
  }
  return { ok: true, places: out };
}

/** The empty answer for a scene with no places.json yet (never a 404: no pins is a valid state). */
export const emptyDoc = (): PlacesDoc => ({ schema: 1, updatedAt: null, places: [] });

/** Parse a stored places.json. The pipeline's older `{version, source, places}` reads as schema 1 with no timestamp.
    Returns null for text that is not a places document at all. */
export function normalize(text: string): PlacesDoc | null {
  let j: unknown;
  try { j = JSON.parse(text); } catch { return null; }
  if (!isObj(j) || !Array.isArray(j.places)) return null;
  return { schema: 1, updatedAt: typeof j.updatedAt === 'string' && j.updatedAt ? j.updatedAt : null, places: j.places as Place[] };
}

/** The write body: schema, a fresh timestamp and the validated list. */
export function document(places: Place[], updatedAt = new Date().toISOString()): PlacesDoc {
  return { schema: 1, updatedAt, places };
}
