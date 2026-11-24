// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The model page's hand-off to the walkthrough: walk.html?scene=<scene.json or folder, encoded>&place=<id>.
   Reading it back is deliberately strict about the place: an id comes from places.json (^[a-z0-9][a-z0-9-]{0,40}$,
   the same rule as the API), so a hand-edited or hostile link can only ask for an existing place, never for a route
   target of its own. Pure: test/deeplink.test.mts runs it under node. */

/** The id rule the API contract sets. place-edit.ts owns the copy the editors use; test/deeplink.test.mts asserts both
    agree, so the two cannot drift apart silently. */
export const PLACE_ID = /^[a-z0-9][a-z0-9-]{0,40}$/;
export const isPlaceId = (v: unknown): v is string => typeof v === 'string' && PLACE_ID.test(v);

/** A shared view: where the camera stood (metres, scene frame), where it looked (radians) and how it moved. */
export interface ViewPose { x: number; y: number; z: number; yaw: number; pitch: number; mode: 'walk' | 'fly' }
export interface DeepLink { scene: string | null; place: string | null; at: ViewPose | null; embed: boolean }

/** `at=x,y,z,yaw,pitch[,w|f]`: five finite numbers in sane ranges, or nothing. A link cannot put the camera 5 km away
    or pass anything but numbers into the engine. */
export function parseAt(v: string | null | undefined): ViewPose | null {
  const parts = String(v ?? '').split(',');
  if (parts.length < 5 || parts.length > 6) return null;
  const n = parts.slice(0, 5).map(Number);
  if (!n.every(Number.isFinite) || n.slice(0, 3).some(c => Math.abs(c) > 5000) || Math.abs(n[3]) > 100 || Math.abs(n[4]) > 1.6) return null;
  const m = parts[5] ?? 'w';
  if (m !== 'w' && m !== 'f') return null;
  return { x: n[0], y: n[1], z: n[2], yaw: n[3], pitch: n[4], mode: m === 'f' ? 'fly' : 'walk' };
}
export function formatAt(p: ViewPose): string {
  const r = (v: number, k = 2) => String(+v.toFixed(k));
  return [r(p.x), r(p.y), r(p.z), r(p.yaw, 3), r(p.pitch, 3), p.mode === 'fly' ? 'f' : 'w'].join(',');
}

/** The current page's link with the view in it (and nothing else of the page's state changed). */
export function shareHref(href: string, pose: ViewPose, extra: Record<string, string | null> = {}): string {
  const u = new URL(href);
  u.searchParams.set('at', formatAt(pose));
  u.searchParams.delete('place');
  for (const [k, v] of Object.entries(extra)) { if (v === null) u.searchParams.delete(k); else u.searchParams.set(k, v); }
  u.hash = '';
  return u.href;
}

/** The query of a search string, a "#fragment" or a whole URL. */
function params(search: string): URLSearchParams {
  const s = String(search ?? '').split('#')[0];
  const i = s.indexOf('?');
  return new URLSearchParams(i >= 0 ? s.slice(i + 1) : s);
}

/** What a walk link asks for. An unknown or malformed place id is null, not an error: the walkthrough then opens
    normally, and says so if it was asked for a place that is not in this scene. */
export function parseDeepLink(search: string): DeepLink {
  const q = params(search);
  const scene = (q.get('scene') ?? '').trim(), place = (q.get('place') ?? '').trim();
  const embed = q.get('embed');
  return { scene: scene || null, place: isPlaceId(place) ? place : null, at: parseAt(q.get('at')), embed: embed === '1' || embed === 'true' };
}

/** The link the model page's "Navigate" opens. `walkHref` is where walk.html lives ("walk.html", or "/ply/walk.html"
    when the page is not served from the site root); the scene is encoded whole, query string and all. */
export function buildWalkHref(sceneHref: string, placeId: string | null | undefined, walkHref = 'walk.html'): string {
  const q = new URLSearchParams();
  q.set('scene', String(sceneHref));
  if (isPlaceId(placeId)) q.set('place', placeId);
  return `${walkHref}?${q.toString()}`;
}
