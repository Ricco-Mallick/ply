// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* An authored room module is script that runs on the site origin, where the team code is stored. Only this origin's own files qualify. */

/** `raw` resolved against `base` (the page URL), or null unless it is an http(s) URL on the same origin as `base`. */
export function sameOriginModuleUrl(raw: string, base: string): string | null {
  if (!raw.trim()) return null;
  try {
    const b = new URL(base), u = new URL(raw, b);
    // blob: URLs carry their creator's origin, so the origin test alone would let them through
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return null;
    return u.origin === b.origin ? u.href : null;
  } catch {
    return null;
  }
}
