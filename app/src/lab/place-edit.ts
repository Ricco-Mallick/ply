// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Places as data: the id/name/category/position rules from the API contract, the slug that names a new tag, the
   tolerant read of a places.json (the labelling pass writes one; a hand-pinned one is a different shape), and the
   file written back out when a save could not reach the server. Pure, so test/place-edit.test.mts covers it. */
import type { Category, Place, PlaceInfo } from './places';

/** The four categories the walkthrough's tag dialog offers, in the order it offers them. */
export const CATEGORIES: readonly Category[] = ['Furniture', 'Storage', 'Exits', 'Things'];
/** The API takes 500 places, names of 1..80 characters and ids of 1..41 matching ^[a-z0-9][a-z0-9-]{0,40}$. */
export const MAX_PLACES = 500;
export const NAME_MAX = 80;
/** The API's caps on a place's info card (cloud/src/places.ts). */
export const NOTE_MAX = 600, LINK_MAX = 400, PHOTO_MAX = 60_000;
const PHOTO_RE = /^data:image\/(jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/;
/** A link a card may open: http(s) only, so a hand-edited places.json cannot carry javascript: into an href. */
export function safeLink(v: unknown): string | null {
  if (typeof v !== 'string' || !v || v.length > LINK_MAX) return null;
  try { const u = new URL(v); return u.protocol === 'https:' || u.protocol === 'http:' ? u.href : null; } catch { return null; }
}
export const isPhoto = (v: unknown): v is string => typeof v === 'string' && v.length <= PHOTO_MAX && PHOTO_RE.test(v);
/** The card fields that pass the API's rules, or undefined when nothing is left. */
export function cleanInfo(v: unknown): PlaceInfo | undefined {
  if (!v || typeof v !== 'object') return undefined;
  const q = v as Record<string, unknown>, out: PlaceInfo = {};
  if (typeof q.note === 'string' && q.note.trim()) out.note = q.note.trim().slice(0, NOTE_MAX);
  const link = safeLink(q.link); if (link) out.link = link;
  if (isPhoto(q.photo)) out.photo = q.photo;
  return Object.keys(out).length ? out : undefined;
}
export const PLACE_ID = /^[a-z0-9][a-z0-9-]{0,40}$/;

export const isPlaceId = (v: unknown): v is string => typeof v === 'string' && PLACE_ID.test(v);
export const isCategory = (v: unknown): v is Category => CATEGORIES.includes(v as Category);

export interface PlaceIssue {
  /** -1 when the problem is the list itself, not one entry */
  index: number;
  field: 'places' | 'id' | 'name' | 'cat' | 'p' | 'info' | 'info.note' | 'info.link' | 'info.photo';
  message: string;
}
export interface PlaceCheck { ok: boolean; issues: PlaceIssue[] }

/** A name as it will be stored: one-line, no runs of spaces, 80 characters at most. */
export function cleanName(v: unknown): string {
  return String(v ?? '').replace(/\s+/g, ' ').trim().slice(0, NAME_MAX);
}

/** A position as a point, or null: three finite numbers. */
export function asPoint(v: unknown): [number, number, number] | null {
  if (!Array.isArray(v) || v.length !== 3) return null;
  const p = v.map(Number);
  return p.every(Number.isFinite) ? [p[0], p[1], p[2]] : null;
}

/** An id from a name: lowercase words joined by dashes, never longer than the contract allows, and unique against
    the ids already in the scene. A name of only punctuation still gets an id ("place"). */
export function slugId(name: string, taken: Iterable<string> = []): string {
  const used = new Set(taken);
  let base = cleanName(name).toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 41);
  if (!base) base = 'place';
  let id = base, n = 2;
  while (used.has(id)) { const suffix = `-${n++}`; id = `${base.slice(0, 41 - suffix.length)}${suffix}`; }
  return id;
}

/** Build the place the dialog saves: a cleaned name, a canonical category, a valid position, and an id that is the
    given one or a fresh slug unique against `taken`. */
export function normalisePlace(input: { name: string; cat: Category; p: readonly number[]; id?: string; ai?: Place['ai']; pinned?: boolean; info?: PlaceInfo }, taken: Iterable<string> = []): Place {
  const ai = input.ai && typeof input.ai === 'object' && typeof input.ai.cls === 'string'
    ? { cls: input.ai.cls, conf: Number.isFinite(input.ai.conf) ? input.ai.conf : 0, half: asPoint(input.ai.half) ?? [0.1, 0.1, 0.1] }
    : undefined;
  const info = cleanInfo(input.info);
  return {
    id: input.id && isPlaceId(input.id) ? input.id : slugId(input.name, taken),
    name: cleanName(input.name), cat: input.cat, p: asPoint(input.p) ?? [0, 0, 0],
    ...(ai ? { ai } : {}), ...(input.pinned ? { pinned: true } : {}),
    ...(info ? { info } : {}),
  };
}

/** Check a whole list the way PUT /v1/scenes/:id/places does, naming the entry and the field so the dialog can point
    at the row that has to change. Unknown fields are fine; they are carried through untouched. */
export function validatePlaces(input: unknown): PlaceCheck {
  if (!Array.isArray(input)) return { ok: false, issues: [{ index: -1, field: 'places', message: 'places must be a list' }] };
  const issues: PlaceIssue[] = [];
  if (input.length > MAX_PLACES) issues.push({ index: -1, field: 'places', message: `at most ${MAX_PLACES} places (this list has ${input.length})` });
  const seen = new Set<string>();
  input.forEach((raw, index) => {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) { issues.push({ index, field: 'places', message: 'each place must be an object' }); return; }
    const p = raw as Record<string, unknown>;
    if (!isPlaceId(p.id)) issues.push({ index, field: 'id', message: 'ids use a-z, 0-9 and dashes, start with a letter or digit, and stop at 41 characters' });
    else if (seen.has(p.id)) issues.push({ index, field: 'id', message: `“${p.id}” is used by another place` });
    else seen.add(p.id);
    if (typeof p.name !== 'string' || !p.name.trim()) issues.push({ index, field: 'name', message: 'a place needs a name' });
    else if (p.name.trim().length > NAME_MAX) issues.push({ index, field: 'name', message: `names stop at ${NAME_MAX} characters` });
    if (!isCategory(p.cat)) issues.push({ index, field: 'cat', message: `the category is one of ${CATEGORIES.join(', ')}` });
    if (!asPoint(p.p)) issues.push({ index, field: 'p', message: 'the position is three numbers [x, y, z]' });
    if (p.info !== undefined) {
      const i = p.info as Record<string, unknown> | null;
      if (!i || typeof i !== 'object') issues.push({ index, field: 'info', message: 'the card is an object' });
      else {
        if (i.note !== undefined && (typeof i.note !== 'string' || i.note.length > NOTE_MAX)) issues.push({ index, field: 'info.note', message: `notes stop at ${NOTE_MAX} characters` });
        if (i.link !== undefined && !safeLink(i.link)) issues.push({ index, field: 'info.link', message: 'the link must start with https:// or http://' });
        if (i.photo !== undefined && !isPhoto(i.photo)) issues.push({ index, field: 'info.photo', message: `the photo must be a JPEG under ${PHOTO_MAX / 1000} kB` });
      }
    }
  });
  return { ok: issues.length === 0, issues };
}

/** Read a places.json however it was written: the pipeline's `{version, source, places}`, an API payload, or a bare
    array. Entries that cannot be a place are dropped rather than crashing the viewer; the rest keep every field. */
export function parsePlacesPayload(json: unknown): Place[] {
  const list = Array.isArray(json) ? json
    : json && typeof json === 'object' && Array.isArray((json as { places?: unknown }).places) ? (json as { places: unknown[] }).places
      : [];
  return list.filter((p): p is Place => {
    if (!p || typeof p !== 'object' || Array.isArray(p)) return false;
    const q = p as Record<string, unknown>;
    return isPlaceId(q.id) && typeof q.name === 'string' && !!cleanName(q.name) && isCategory(q.cat) && !!asPoint(q.p);
  }).slice(0, MAX_PLACES);
}

/** The places.json a failed save offers as a download: the shape the pipeline writes (one-space indent, version and
    source), so the file can go straight back next to the scene or through the labelling pass. */
export function placesFile(places: readonly Place[], source = 'tagged in the walkthrough'): string {
  return `${JSON.stringify({ version: 1, source, places }, null, 1)}\n`;
}
