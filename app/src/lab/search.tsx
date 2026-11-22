// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The command-palette search (the lab's pick): shared by the design lab and the walkthrough.
   Ranking comes from lab/index.ts, not a substring test: exact name, then prefix, then a whole word, then a word
   prefix, then a substring, with the synonym table behind it. A row can be routed to, edited (when the caller can
   edit), and the list filters by category and says when nothing matches. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Glass } from '../shared/Glass';
import { classColor } from './overlay';
import { CATEGORIES } from './place-edit';
import { buildIndex, matchPlace, search } from './index';
import type { Category, Place } from './places';

export const SearchIco = () => <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4.5 4.5" /></svg>;

/** The lab's sheet and rail filter with this; it goes through the index so they rank and match like the palette. */
export function match(p: Place, q: string) { return matchPlace(p, q); }

export function Palette({ places, pinned, dist, onPick, onClose, onEdit, onTag }: {
  places: Place[]; pinned: Place[]; dist: (p: Place) => number; onPick: (id: string) => void; onClose: () => void;
  /** when present, each row gets an edit button (the walkthrough's tag editor) */
  onEdit?: (id: string) => void;
  /** when present, "no results" offers to tag a place */
  onTag?: () => void;
}) {
  const [q, setQ] = useState(''); const [cat, setCat] = useState<Category | null>(null); const [i, setI] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const index = useMemo(() => buildIndex(places), [places]);
  const byId = useMemo(() => new Map(places.map(p => [p.id, p])), [places]);
  const pinnedIds = useMemo(() => new Set(pinned.map(p => p.id)), [pinned]);
  const hits = search(index, q, { cat });
  // score first, then the place you can reach soonest: stable, and a new tag lands where its score says it should
  const rows = hits.map(h => ({ p: byId.get(h.id)!, score: h.score })).filter(r => r.p).sort((a, b) => b.score - a.score || dist(a.p) - dist(b.p) || a.p.id.localeCompare(b.p.id));
  const others = cat ? search(index, q).length - hits.length : 0;
  const setQuery = (v: string) => { setQ(v); setI(0); };
  const setCategory = (c: Category | null) => { setCat(c); setI(0); };
  useEffect(() => { const el = listRef.current?.querySelector('.pal-i.on'); el?.scrollIntoView({ block: 'nearest' }); }, [i, q, cat]);
  return (
    <div className="pal-wrap" onClick={onClose}>
      <Glass className="pal" opts={{ radius: 22, shader: false }} onClick={(e: React.MouseEvent) => e.stopPropagation()}>
        <div className="pal-in"><SearchIco /><input autoFocus placeholder="Search this space: desk, door, wc…" value={q} onChange={e => setQuery(e.target.value)}
          aria-label="Search places" onKeyDown={e => {
            if (e.key === 'ArrowDown') { e.preventDefault(); setI(v => Math.min(rows.length - 1, v + 1)); }
            if (e.key === 'ArrowUp') { e.preventDefault(); setI(v => Math.max(0, v - 1)); }
            if (e.key === 'Enter' && rows[i]) onPick(rows[i].p.id);
            if (e.key === 'Escape') { e.preventDefault(); onClose(); }
          }} /><kbd>esc</kbd></div>
        <div className="pal-cats" role="group" aria-label="Filter by category">
          <button className={cat === null ? 'on' : ''} aria-pressed={cat === null} onClick={() => setCategory(null)}>All</button>
          {CATEGORIES.map(c => <button key={c} className={cat === c ? 'on' : ''} aria-pressed={cat === c} onClick={() => setCategory(cat === c ? null : c)}>{c}</button>)}
        </div>
        <div className="pal-list" ref={listRef} role="listbox" aria-label="Places">
          {rows.map(({ p }, k) => {
            const ai = p.ai && !pinnedIds.has(p.id) ? p.ai : null;
            return (
              <div className="pal-row" key={p.id}>
                <button className={`pal-i ${k === i ? 'on' : ''}`} role="option" aria-selected={k === i} onMouseEnter={() => setI(k)} onFocus={() => setI(k)} onClick={() => onPick(p.id)}>
                  <span className={`pal-ic ${ai ? 'ai' : ''}`} style={ai ? { color: classColor(ai.cls) } : undefined}>{ai ? '✦' : p.cat[0]}</span>
                  <span className="pal-n">{p.name}<em>{ai ? `${ai.cls} · ${Math.round(ai.conf * 100)}%` : p.cat}</em></span>
                  <span className="mono dim">{dist(p).toFixed(1)} m</span>
                  <span className="pal-go">Route ↵</span>
                </button>
                {onEdit && <button className="pal-edit" title={`Edit “${p.name}”`} aria-label={`Edit ${p.name}`} onClick={() => onEdit(p.id)}>✎</button>}
              </div>
            );
          })}
          {!rows.length && (
            <div className="pal-empty">
              <p>{q.trim() ? `No place here matches “${q.trim()}”${cat ? ` in ${cat}` : ''}.` : cat ? `Nothing is tagged ${cat} here yet.` : 'No places here yet.'}</p>
              {others > 0 && <p className="pal-else">{others} match{others === 1 ? '' : 'es'} in other categories. <button onClick={() => setCategory(null)}>Show {others === 1 ? 'it' : 'them'}</button></p>}
              {onTag && <button className="pal-tag" onClick={onTag}>＋ Tag a place</button>}
            </div>
          )}
        </div>
        <div className="pal-f mono"><span>↑↓ choose</span><span>↵ route</span>{onEdit && <span>✎ edit</span>}<span>esc close</span></div>
      </Glass>
    </div>
  );
}
