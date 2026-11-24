// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The model page's hand-off to the walkthrough: walk.html?scene=<scene.json or folder, encoded>&place=<id>.
   Reading it back is deliberately strict about the place: an id comes from places.json (^[a-z0-9][a-z0-9-]{0,40}$,
   the same rule as the API), so a hand-edited or hostile link can only ask for an existing place, never for a route
   target of its own. Pure: test/deeplink.test.mts runs it under node. */

/** The id rule the API contract sets. place-edit.ts owns the copy the editors use; test/deeplink.test.mts asserts both
    agree, so the two cannot drift apart silently. */
export const PLACE_ID = /^[a-z0-9][a-z0-9-]{0,40}$/;
export const isPlaceId = (v: unknown): v is string => typeof v === 'string' && PLACE_ID.test(v);

export interface DeepLink { scene: string | null; place: string | null }

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
  return { scene: scene || null, place: isPlaceId(place) ? place : null };
}

/** The link the model page's "Navigate" opens. `walkHref` is where walk.html lives ("walk.html", or "/ply/walk.html"
    when the page is not served from the site root); the scene is encoded whole, query string and all. */
export function buildWalkHref(sceneHref: string, placeId: string | null | undefined, walkHref = 'walk.html'): string {
  const q = new URLSearchParams();
  q.set('scene', String(sceneHref));
  if (isPlaceId(placeId)) q.set('place', placeId);
  return `${walkHref}?${q.toString()}`;
}
