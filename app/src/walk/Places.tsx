// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Places, search and routes in the walkthrough (the design lab's picks: glass-tag pins, the command palette, the floor
   ribbon route). Places come from places.json next to the scene: hand-pinned for the demo, written by the labelling
   stage for pipeline twins. No places.json: nothing shows. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Glass } from '../shared/Glass';
import type { WalkEngine } from './engine';
import type { MapExtras } from './Walk';
import { Navigator, pathLength, steps, type Place } from '../lab/places';
import { Overlay } from '../lab/overlay';
import { Palette } from '../lab/search';

const MERGE_M = 2.5; // pins of the same kind closer than this share one badge
const MAX_STACK = 6; // ...up to six per badge, so a long row still shows where the benches are
/** 200 labelled places would mean 200 floating labels over the room. Same-kind neighbours merge into one pin
 *  ("Bench ×6") placed on the nearest real instance; the palette and routing still see every individual place. */
function clusterPlaces(places: Place[]): Place[] {
  const base = (n: string) => n.replace(/\s*\d+$/, '');
  const groups: { name: string; cat: string; members: Place[] }[] = [];
  for (const p of places) {
    const name = base(p.name), cat = (p as { cat?: string }).cat ?? '';
    const g = groups.find(q => q.name === name && q.cat === cat && q.members.length < MAX_STACK && q.members.some(m => Math.hypot(m.p[0] - p.p[0], m.p[2] - p.p[2]) <= MERGE_M));
    if (g) g.members.push(p); else groups.push({ name, cat, members: [p] });
  }
  const out: Place[] = [];
  for (const g of groups) {
    if (g.members.length === 1) { out.push(g.members[0]); continue; }
    const cx = g.members.reduce((a, m) => a + m.p[0], 0) / g.members.length, cz = g.members.reduce((a, m) => a + m.p[2], 0) / g.members.length;
    const at = g.members.reduce((b, m) => (Math.hypot(m.p[0] - cx, m.p[2] - cz) < Math.hypot(b.p[0] - cx, b.p[2] - cz) ? m : b), g.members[0]);
    out.push({ ...at, name: `${g.name} ×${g.members.length}` });
  }
  return out.slice(0, 400);
}

export function PlacesLayer({ engine, base, entered, onMap, onData, searchOpen, onSearch }: { engine: WalkEngine; base: URL | null; entered: boolean; onMap: (m: MapExtras) => void; onData?: (d: { places: Place[]; nav: Navigator | null }) => void; searchOpen: boolean; onSearch: (v: boolean) => void }) {
  const [places, setPlaces] = useState<Place[] | null>(null);
  const [ov, setOv] = useState<Overlay | null>(null); const [nav, setNav] = useState<Navigator | null>(null);
  const open = searchOpen, setOpen = onSearch; const [dest, setDest] = useState<string | null>(null); const [guide, setGuide] = useState(false); const [, tick] = useState(0);

  useEffect(() => { if (!base) return; fetch(new URL('places.json', base)).then(r => r.ok ? r.json() : null).then(j => setPlaces(j?.places ?? [])).catch(() => setPlaces([])); }, [base]);
  useEffect(() => {
    if (!engine.state.loaded || !places?.length || ov) return;
    const n = engine.grid ? new Navigator(engine.grid) : null, o = new Overlay(engine, n); o.allPlaces = places;
    setNav(n); setOv(o); onData?.({ places, nav: n }); return () => o.dispose();
  }, [engine.state.loaded, places]);
  // 200 labelled places means 200 floating labels over the room. Stack neighbours of the same kind into one pin
  // ("Bench ×6"); the palette and the routing still see every individual place.
  const shown = useMemo(() => clusterPlaces(places ?? []), [places]);
  // and only the pins near the visitor get a label: fourteen chips in a 10 m radius, not every desk in
  // the building. The map and the palette still list all 194.
  const [nearTick, setNearTick] = useState(0);
  const near = useMemo(() => {
    const p0 = engine.state.pos;
    return shown
      .map(p => ({ p, d: Math.hypot(p.p[0] - p0.x, p.p[2] - p0.z) }))
      .filter(x => x.d < 10)
      .sort((a, b) => a.d - b.d)
      .slice(0, 14)
      .map(x => x.p);
  }, [shown, nearTick]);
  useEffect(() => { ov?.setPlaces(near, 'tag', entered); }, [ov, near, entered]);
  useEffect(() => { const i = setInterval(() => { tick(n => n + 1); setNearTick(n => n + 1); }, 400); return () => clearInterval(i); }, []);

  const me = engine.state.pos;
  const path = useMemo(() => { if (!nav || !dest) return null; const p = places!.find(x => x.id === dest)!; return nav.route([me.x, me.z], [p.p[0], p.p[2]]); },
    [nav, dest, Math.round(me.x * 2), Math.round(me.z * 2)]);
  useEffect(() => { ov?.setRoute(path, 'ribbon', dest); }, [ov, path, dest]);
  useEffect(() => { onMap({ path, pins: shown.map(p => ({ x: p.p[0], z: p.p[2], dest: p.id === dest })) }); }, [path, shown, dest]);

  // a new destination: turn toward the route and look at the floor so it is in view
  const faced = useRef<string | null>(null);
  useEffect(() => {
    if (!path || path.length < 2 || !dest || faced.current === dest) return; faced.current = dest;
    const tgt = path.find(q => Math.hypot(q[0] - me.x, q[1] - me.z) > .6) ?? path[path.length - 1];
    let want = Math.atan2(-(tgt[0] - me.x), -(tgt[1] - me.z)), t = 0; const y0 = engine.state.yaw, p0 = engine.pitch;
    while (want - y0 > Math.PI) want -= 2 * Math.PI; while (want - y0 < -Math.PI) want += 2 * Math.PI;
    const hook = (dt: number) => { t = Math.min(1, t + dt / .9); const k = 1 - (1 - t) ** 3; engine.state.yaw = y0 + (want - y0) * k; engine.pitch = p0 + (-.4 - p0) * k; if (t >= 1) engine.frameHooks.delete(hook); };
    engine.frameHooks.add(hook); return () => { engine.frameHooks.delete(hook); };
  }, [path, dest]);
  useEffect(() => { if (!dest) faced.current = null; }, [dest]);

  // "Take me there": walk the route at 1.25 m/s, turning along it
  useEffect(() => {
    if (!guide || !path) return; const fixed = path.slice(), L = pathLength(fixed); let d = 0;
    const at = (s: number): [number, number] => { let acc = 0; for (let i = 1; i < fixed.length; i++) { const seg = Math.hypot(fixed[i][0] - fixed[i - 1][0], fixed[i][1] - fixed[i - 1][1]); if (acc + seg >= s) { const k = (s - acc) / seg; return [fixed[i - 1][0] + (fixed[i][0] - fixed[i - 1][0]) * k, fixed[i - 1][1] + (fixed[i][1] - fixed[i - 1][1]) * k]; } acc += seg; } return fixed[fixed.length - 1]; };
    engine.state.pos.y = engine.floorY;  // follow at eye height in either mode; do not switch the visitor into walk
    const hook = (dt: number) => {
      d = Math.min(L, d + dt * 1.25); const [x, z] = at(d), [ax, az] = at(Math.min(L, d + .6)); engine.state.pos.x = x; engine.state.pos.z = z;
      if (Math.hypot(ax - x, az - z) > .05) { const want = Math.atan2(-(ax - x), -(az - z)); let dy = want - engine.state.yaw; while (dy > Math.PI) dy -= 2 * Math.PI; while (dy < -Math.PI) dy += 2 * Math.PI; engine.state.yaw += dy * Math.min(1, dt * 3); }
      engine.pitch += (-.08 - engine.pitch) * Math.min(1, dt * 2);
      if (d >= L) { // arrived: turn to face the place itself
        const pl = places!.find(x => x.id === dest); setGuide(false);
        if (pl) { const ex = engine.state.pos, y0 = engine.state.yaw, p0 = engine.pitch; let want = Math.atan2(-(pl.p[0] - ex.x), -(pl.p[2] - ex.z));
          while (want - y0 > Math.PI) want -= 2 * Math.PI; while (want - y0 < -Math.PI) want += 2 * Math.PI;
          const look = Math.atan2(pl.p[1] - (engine.floorY + engine.eye), Math.max(.3, Math.hypot(pl.p[0] - ex.x, pl.p[2] - ex.z))); let t = 0;
          const turn = (dt2: number) => { t = Math.min(1, t + dt2 / .8); const k = 1 - (1 - t) ** 3; engine.state.yaw = y0 + (want - y0) * k; engine.pitch = p0 + (look - p0) * k; if (t >= 1) engine.frameHooks.delete(turn); };
          engine.frameHooks.add(turn); }
      }
    };
    engine.frameHooks.add(hook); return () => { engine.frameHooks.delete(hook); };
  }, [guide, dest]);

  // Ctrl/Cmd+K opens search, Esc closes it (the pointer lock is released so the palette can be used)
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); document.exitPointerLock?.(); setOpen(!searchOpen); } if (e.key === 'Escape') setOpen(false); };
    addEventListener('keydown', k); return () => removeEventListener('keydown', k);
  }, [searchOpen]);

  if (!places?.length || !entered) return null;
  const dist = (p: Place) => Math.hypot(p.p[0] - me.x, p.p[2] - me.z);
  const dp = places.find(p => p.id === dest) ?? null, L = path ? pathLength(path) : 0, turns = path && dp ? steps(path, dp.name) : [];
  return (
    <>
      {open && <Palette places={places} pinned={shown} dist={dist} onPick={id => { setDest(id); setGuide(false); setOpen(false); }} onClose={() => setOpen(false)} />}
      {dp && <Glass className="route-card rc-palette" opts={{ radius: 20, shader: false }}>
        <div className="rc-h"><span className="wide eyebrow">Route</span><button className="rc-x" onClick={() => { setDest(null); setGuide(false); }} aria-label="End route">×</button></div>
        <h2>{dp.name}</h2>
        {path ? <p className="mono rc-m"><b>{L.toFixed(1)} m</b> · {Math.max(1, Math.round(L / 1.2))} s walk · {turns.length - 1} turn{turns.length === 2 ? '' : 's'}</p> : <p className="mono rc-m bad">No walkable route from here</p>}
        {path && <ol className="rc-steps">{turns.map((t, i) => <li key={i}>{t.text}</li>)}</ol>}
        {path && <div className="rc-a"><button className="btn" onClick={() => setGuide(g => !g)}>{guide ? 'Pause' : 'Take me there'}</button></div>}
      </Glass>}
    </>
  );
}
