// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The search index behind the palette and the model page's place search.
   Names, categories and AI classes are tokenised once per place; a query is tokenised the same way and matched
   token by token, so "des" finds "Desk", "desks" finds it too, and "wc" finds the toilet. Scores fall in tiers —
   exact name, name prefix, whole token, token prefix, then substring — and a match through the synonym table scores
   below the direct word. Everything here is pure (no DOM, no React), so test/index.test.mts runs it under node. */

export interface PlaceLike {
  id: string;
  name: string;
  cat?: string;
  ai?: { cls?: string } | null;
}

interface Field { text: string; tokens: string[]; stems: string[]; weight: number }
interface Entry { id: string; name: string; cat: string; fields: Field[] }

export interface PlaceIndex { entries: Entry[] }
export interface Hit { id: string; score: number }
export interface SearchOptions {
  /** only places of this category */
  cat?: string | null;
  /** stop after this many hits (after ranking) */
  limit?: number;
}

/** What a match is worth, best first: an exact name beats a prefix, a whole token beats a token prefix, and any of
    those beats a substring somewhere inside a word. Field weight (name 1, category 0.8, AI class 0.7) scales a tier. */
export const TIERS = { exact: 1000, prefix: 700, token: 450, tokenPrefix: 350, substring: 150 } as const;
/** A word found through the synonym table ranks under the same word written on the place. */
export const SYNONYM_WEIGHT = 0.8;

/** The synonym table: any word in a group finds the others. Phrases are matched as a run of tokens in one field. */
export const SYNONYMS: Readonly<Record<string, readonly string[]>> = {
  wc: ['toilet'], toilet: ['wc'],
  exit: ['door'], door: ['exit'],
  sofa: ['couch'], couch: ['sofa'],
  dustbin: ['bin'], bin: ['dustbin'],
  ac: ['air conditioner'], 'air conditioner': ['ac'],
};

/** Lowercase, strip accents, and turn everything that is not a letter or digit into a separator. */
export function normalise(text: unknown): string {
  return String(text ?? '').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
}

/** Words of a piece of text, in order. */
export function tokenise(text: unknown): string[] {
  const n = normalise(text);
  return n ? n.split(' ') : [];
}

/** A light plural fold, so "desks" and "desk" meet: ies -> y, then a trailing s (but not "ss"). */
function stem(token: string): string {
  if (token.length > 4 && token.endsWith('ies')) return `${token.slice(0, -3)}y`;
  return token.length > 3 && token.endsWith('s') && !token.endsWith('ss') ? token.slice(0, -1) : token;
}

function field(text: unknown, weight: number): Field | null {
  if (typeof text !== 'string') return null;
  const tokens = tokenise(text);
  return tokens.length ? { text: normalise(text), tokens, stems: tokens.map(stem), weight } : null;
}

function entry(place: PlaceLike): Entry | null {
  if (!place || typeof place.id !== 'string' || !place.id || typeof place.name !== 'string') return null;
  const name = normalise(place.name);
  const fields = [field(place.name, 1), field(place.cat, 0.8), field(place.ai?.cls, 0.7)].filter((f): f is Field => !!f);
  return { id: place.id, name, cat: place.cat ?? '', fields };
}

/** Index a list of places. Duplicate ids are ignored (the first wins): the panes use them as keys. */
export function buildIndex(places: readonly PlaceLike[]): PlaceIndex {
  const entries: Entry[] = []; const seen = new Set<string>();
  for (const p of places) {
    if (!p || seen.has(p.id)) continue;
    const e = entry(p); if (!e) continue;
    seen.add(p.id); entries.push(e);
  }
  return { entries };
}

/** The best score one query word gets against a place, or null when it is not there at all. */
function matchTerm(e: Entry, term: string): number | null {
  const st = stem(term); let best = 0;
  for (const f of e.fields) {
    let tier = 0;
    if (f.text === term) tier = TIERS.exact;
    else if (f.tokens.includes(term) || f.stems.includes(st)) tier = TIERS.token;
    else if (f.tokens.some(t => t.startsWith(term) || stem(t).startsWith(st))) tier = TIERS.tokenPrefix;
    else if (f.text.includes(term) || (st !== term && f.text.includes(st))) tier = TIERS.substring;
    if (tier && tier * f.weight > best) best = tier * f.weight;
  }
  return best || null;
}

/** Score one whole query against one place: every word must match somewhere (a match-and search). */
function scoreQuery(e: Entry, query: string): number | null {
  const q = normalise(query), terms = tokenise(query);
  if (!terms.length) return 0;
  let score = 0;
  if (e.name === q) score += TIERS.exact;
  else if (e.name.startsWith(q)) score += TIERS.prefix;
  for (const t of terms) {
    const s = matchTerm(e, t);
    if (s === null) return null;
    score += s;
  }
  return score;
}

/** The words the table offers for a whole query ("wc" -> "toilet"), including phrase entries. */
export function synonymsOf(query: string): readonly string[] {
  return SYNONYMS[normalise(query)] ?? [];
}

/** Rank the places of an index for a query. An empty query returns everything at score 0 in index order, so the
    caller can still sort by distance. Ties keep the index order: sorting by score alone is stable. */
export function search(index: PlaceIndex, query: string, opts: SearchOptions = {}): Hit[] {
  const q = normalise(query), cat = opts.cat ? normalise(opts.cat) : '', aliases = q ? synonymsOf(q) : [];
  const hits: Hit[] = [];
  for (const e of index.entries) {
    if (cat && normalise(e.cat) !== cat) continue;
    let best: number | null = q ? null : 0;
    if (q) {
      const direct = scoreQuery(e, q);
      if (direct !== null) best = direct;
      for (const a of aliases) {
        const s = scoreQuery(e, a);
        if (s !== null) best = Math.max(best ?? 0, s * SYNONYM_WEIGHT);
      }
    }
    if (best === null) continue;
    hits.push({ id: e.id, score: best });
  }
  hits.sort((a, b) => b.score - a.score);   // Array#sort is stable: equal scores keep index order
  return opts.limit !== undefined && opts.limit >= 0 ? hits.slice(0, opts.limit) : hits;
}

/** Does this place match the query? The lab's filter boxes and the rail use this. */
export function matchPlace(place: PlaceLike, query: string): boolean {
  return search(buildIndex([place]), query).length > 0;
}
