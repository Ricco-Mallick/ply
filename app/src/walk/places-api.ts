// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Reading and writing a scene's places through the API (cloud/src/index.ts):
     GET  /v1/scenes/:id/places   public, never 404s: an unknown scene is an empty list
     PUT  /v1/scenes/:id/places   whole list, x-ply-key header, 400 names the index and field
   The team invite code the capture app already keeps on this device is the key. A save is reported, never thrown: the
   caller keeps the change locally and offers places.json when there is no server, no network, or a code that may look
   but not tag (a judges.* invite is refused with 403). */
import type { Place } from '../lab/places';
import { parsePlacesPayload } from '../lab/place-edit';

const KEY = 'ply-team-key';   // the capture app's key (capture/lib/backend.ts): entered once, kept on this device
export const API_BASE = ((import.meta.env.VITE_API as string | undefined) ?? '').replace(/\/$/, '');

export function savedCode(): string { try { return localStorage.getItem(KEY) ?? ''; } catch { return ''; } }
export function rememberCode(code: string): void { try { const c = code.trim(); if (c) localStorage.setItem(KEY, c); } catch { /* private mode */ } }

export interface PlacesTarget { url: string; sceneId: string }

/** Where this scene's places live. The scene folder names the id (`.../scenes/<id>/scene.json`). A scene served from
    this same site (the bundled demo, a public/scenes fixture) is a file with no API behind it: tagging there stays on
    the device and offers places.json. Everything else talks to VITE_API, or to the origin the scene came from. */
export function placesTarget(sceneBase: URL | string | null | undefined, apiBase = API_BASE): PlacesTarget | null {
  if (!sceneBase) return null;
  let base: URL;
  try { base = new URL(String(sceneBase), typeof location === 'undefined' ? undefined : location.href); } catch { return null; }
  const m = base.pathname.match(/\/scenes\/([\w-]+)(?:\/.*)?$/);
  if (!m) return null;
  const sceneId = m[1], configured = apiBase.trim().replace(/\/$/, ''), root = configured || base.origin;
  // A page-served scene has no API of its own and is treated as a file (local draft plus export), unless an API is
  // configured: then its id still resolves there, and tagging the bundled demo persists instead of staying on one
  // device, which is what a judge tagging a pin in the demo needs.
  if (!configured && typeof location !== 'undefined' && base.origin === location.origin) return null;
  return { url: `${root}${configured ? '/v1' : base.pathname.slice(0, m.index ?? 0)}/scenes/${sceneId}/places`, sceneId };
}

async function getJson(url: string, doFetch: typeof fetch, init?: RequestInit): Promise<{ ok: boolean; status: number; body: Record<string, unknown> | null }> {
  const res = await doFetch(url, init);
  const body = await res.json().catch(() => null) as Record<string, unknown> | null;
  return { ok: res.ok, status: res.status, body };
}

/** How the team set the twin to be shown (stored next to the places): a crop box in scene metres, and where on Earth
    the building is (for the "Where is this?" map). */
export interface SceneView { crop?: { min: [number, number, number]; max: [number, number, number] }; location?: { lat: number; lon: number } }
export interface PlacesRead { places: Place[]; source: 'api' | 'file'; view?: SceneView }

/** The view block of a places document: the parts of it that are sane. */
export function parseView(json: unknown): SceneView | undefined {
  const v = (json as { view?: { crop?: { min?: unknown; max?: unknown }; location?: { lat?: unknown; lon?: unknown } } } | null)?.view;
  const c = v?.crop, l = v?.location, out: SceneView = {};
  const pt = (q: unknown): q is [number, number, number] => Array.isArray(q) && q.length === 3 && q.every(x => typeof x === 'number' && Number.isFinite(x));
  if (c && pt(c.min) && pt(c.max) && c.min.every((m, k) => m < (c.max as number[])[k])) out.crop = { min: c.min, max: c.max };
  if (l && typeof l.lat === 'number' && typeof l.lon === 'number' && Math.abs(l.lat) <= 90 && Math.abs(l.lon) <= 180) out.location = { lat: l.lat, lon: l.lon };
  return out.crop || out.location ? out : undefined;
}

/** A scene's places: the API first (it never 404s, so an empty answer is a real empty list), and only when that is
    unreachable or answers badly, places.json next to the scene. A page-served scene has no API, so it reads the file.
    null when neither answered: the caller treats that as "no places here". */
export async function loadPlaces(base: URL | string | null, doFetch: typeof fetch = fetch): Promise<PlacesRead | null> {
  if (!base) return null;
  const url = base instanceof URL ? base : new URL(String(base), typeof location === 'undefined' ? undefined : location.href);
  const target = placesTarget(url);
  if (target) {
    try {
      const r = await getJson(target.url, doFetch, { cache: 'no-store' });
      if (r.ok) return { places: parsePlacesPayload(r.body), source: 'api', view: parseView(r.body) };
    } catch { /* no API from here: the file next to the scene is the fallback */ }
  }
  try {
    const r = await doFetch(new URL('places.json', url).href, { cache: 'no-store' });
    if (!r.ok) return null;
    const j = await r.json().catch(() => null);
    return { places: parsePlacesPayload(j), source: 'file', view: parseView(j) };
  } catch { return null; }
}

export type SaveResult =
  | { kind: 'saved'; updatedAt: string | null; count: number }
  | { kind: 'auth'; status: 401; detail: string }
  | { kind: 'forbidden'; status: 403; detail: string }
  | { kind: 'rejected'; status: 400; detail: string; index?: number; field?: string }
  | { kind: 'offline'; detail: string }
  | { kind: 'failed'; status?: number; detail: string };

/** PUT the whole list. 400 names the offending index and field; 401 is a missing or wrong code (ask for it); 403 is a
    code that may look but not tag (a judges invite): the work stays local and can still be exported. A request that
    never leaves the browser is 'offline'. */
export async function savePlaces(target: PlacesTarget, places: readonly Place[], code: string, doFetch: typeof fetch = fetch, view?: SceneView | null): Promise<SaveResult> {
  let res: Response;
  try {
    res = await doFetch(target.url, {
      method: 'PUT',
      headers: { 'content-type': 'application/json', 'x-ply-key': code.trim() },
      // no view: the server keeps the crop it has; null clears it
      body: JSON.stringify(view === undefined ? { places } : { places, view }),
    });
  } catch (err) {
    return { kind: 'offline', detail: err instanceof Error ? err.message : 'the request did not leave the browser' };
  }
  const body = await res.json().catch(() => null) as Record<string, unknown> | null;
  const detail = typeof body?.error === 'string' ? body.error : `the server answered ${res.status}`;
  if (res.status === 401) return { kind: 'auth', status: 401, detail };
  if (res.status === 403) return { kind: 'forbidden', status: 403, detail };
  if (res.ok) {
    return {
      kind: 'saved',
      updatedAt: typeof body?.updatedAt === 'string' ? body.updatedAt : null,
      count: typeof body?.count === 'number' ? body.count : places.length,
    };
  }
  if (res.status === 400) {
    return {
      kind: 'rejected', status: 400, detail,
      index: typeof body?.index === 'number' ? body.index : undefined,
      field: typeof body?.field === 'string' ? body.field : undefined,
    };
  }
  return { kind: 'failed', status: res.status, detail };
}
