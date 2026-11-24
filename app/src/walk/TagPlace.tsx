// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The tag dialog: a name, one of four categories, Save. Opened with a picked floor point for a new place, or with an
   existing place to rename / recategorise / delete. It never talks to the API itself: Places.tsx owns the save so the
   optimistic list, the local fallback and the export all come from one place. */
import { useEffect, useRef, useState } from 'react';
import { Glass } from '../shared/Glass';
import { CATEGORIES, cleanName, NAME_MAX, normalisePlace, slugId } from '../lab/place-edit';
import type { Category, Place } from '../lab/places';

export function TagPlace({ initial, at, taken, busy, error, onSave, onDelete, onCancel }: {
  /** the place being edited, or null for a new tag */
  initial: Place | null;
  /** the floor point the tag sits on (the place's own position when editing) */
  at: [number, number, number];
  /** ids already in the scene, so a new tag's slug does not collide */
  taken: string[];
  busy: boolean;
  error: string | null;
  onSave: (place: Place) => void;
  onDelete: (id: string) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initial?.name ?? '');
  const [cat, setCat] = useState<Category>(initial?.cat ?? 'Things');
  const [confirm, setConfirm] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => { input.current?.focus(); input.current?.select(); }, []);
  const clean = cleanName(name), valid = clean.length > 0;
  const id = initial?.id ?? slugId(clean, taken);
  const submit = () => {
    if (!valid || busy) return;
    onSave(normalisePlace({ id, name: clean, cat, p: at, ai: initial?.ai, pinned: initial?.pinned ?? true }, taken));
  };
  return (
    <Glass className="tag-dlg" opts={{ radius: 20, shader: false }} role="dialog" aria-modal="false" aria-labelledby="tag-title">
      <form onSubmit={e => { e.preventDefault(); submit(); }}>
        <div className="tag-h">
          <span className="wide eyebrow">{initial ? 'Edit place' : 'New place'}</span>
          <button type="button" className="rc-x" onClick={onCancel} aria-label="Close">×</button>
        </div>
        <h2 id="tag-title">{initial ? initial.name : 'Tag this spot'}</h2>
        <label className="tag-field">Name
          <input ref={input} value={name} maxLength={NAME_MAX} onChange={e => setName(e.target.value)} placeholder="e.g. Fire exit, Bench 3, Water cooler"
            autoComplete="off" spellCheck={false} aria-describedby="tag-id" />
        </label>
        <div className="tag-cats" role="group" aria-label="Category">
          {CATEGORIES.map(c => <button key={c} type="button" className={cat === c ? 'on' : ''} aria-pressed={cat === c} onClick={() => setCat(c)}>{c}</button>)}
        </div>
        <p className="tag-at mono" id="tag-id">id <b>{id}</b> · floor {at[0].toFixed(2)}, {at[2].toFixed(2)} m{initial ? '' : ' · snapped to the nearest walkable spot'}</p>
        {error && <p className="tag-bad" role="alert">{error}</p>}
        <div className="tag-actions">
          <button className="btn" type="submit" disabled={!valid || busy}>{busy ? 'Saving…' : initial ? 'Save changes' : 'Save place'}</button>
          <button type="button" className="tag-plain" onClick={onCancel}>Cancel</button>
          {initial && (confirm
            ? <span className="tag-confirm">Delete it? <button type="button" className="tag-del on" onClick={() => onDelete(initial.id)} disabled={busy}>Delete</button><button type="button" className="tag-plain" onClick={() => setConfirm(false)}>Keep</button></span>
            : <button type="button" className="tag-del" onClick={() => setConfirm(true)}>Delete</button>)}
        </div>
      </form>
    </Glass>
  );
}
