/* The command-palette search (the lab's pick): shared by the design lab and the walkthrough. */
import { useState } from 'react';
import { Glass } from '../shared/Glass';
import { classColor } from './overlay';
import type { Place } from './places';

export const SearchIco = () => <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4.5 4.5" /></svg>;

export function match(p: Place, q: string) { q = q.trim().toLowerCase(); return !q || p.name.toLowerCase().includes(q) || p.cat.toLowerCase().includes(q) || !!p.ai?.cls.includes(q); }

export function Palette({ places, pinned, dist, onPick, onClose }: { places: Place[]; pinned: Place[]; dist: (p: Place) => number; onPick: (id: string) => void; onClose: () => void }) {
  const [q, setQ] = useState(''); const [i, setI] = useState(0);
  const pins = pinned.filter(p => match(p, q)).sort((a, b) => dist(a) - dist(b)), ai = places.filter(p => p.ai && !pinned.includes(p) && match(p, q));
  const all = [...pins, ...ai];
  return (
    <div className="pal-wrap" onClick={onClose}>
      <Glass className="pal" opts={{ radius: 22, shader: false }} onClick={(e: React.MouseEvent) => e.stopPropagation()}>
        <div className="pal-in"><SearchIco /><input autoFocus placeholder="Search this space: desk, door, chair…" value={q} onChange={e => { setQ(e.target.value); setI(0); }}
          onKeyDown={e => { if (e.key === 'ArrowDown') { e.preventDefault(); setI(v => Math.min(all.length - 1, v + 1)); } if (e.key === 'ArrowUp') { e.preventDefault(); setI(v => Math.max(0, v - 1)); } if (e.key === 'Enter' && all[i]) onPick(all[i].id); }} /><kbd>esc</kbd></div>
        {pins.length > 0 && <div className="pal-g mono">Places</div>}
        {pins.map((p, k) => <button key={p.id} className={`pal-i ${k === i ? 'on' : ''}`} onMouseEnter={() => setI(k)} onClick={() => onPick(p.id)}><span className="pal-ic">{p.cat[0]}</span><span className="pal-n">{p.name}<em>{p.cat}</em></span><span className="mono dim">{dist(p).toFixed(1)} m</span><span className="pal-go">Route ↵</span></button>)}
        {ai.length > 0 && <div className="pal-g mono">Found by AI</div>}
        {ai.map((p, k) => <button key={p.id} className={`pal-i ${pins.length + k === i ? 'on' : ''}`} onMouseEnter={() => setI(pins.length + k)} onClick={() => onPick(p.id)}><span className="pal-ic ai" style={{ color: classColor(p.ai!.cls) }}>✦</span><span className="pal-n">{p.name}<em>{p.ai!.cls} · {Math.round(p.ai!.conf * 100)}%</em></span><span className="mono dim">{dist(p).toFixed(1)} m</span><span className="pal-go">Route ↵</span></button>)}
        {!all.length && <p className="pal-empty">Nothing called “{q}” here yet.</p>}
        <div className="pal-f mono"><span>↑↓ choose</span><span>↵ route</span><span>esc close</span></div>
      </Glass>
    </div>
  );
}

