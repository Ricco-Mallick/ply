/* Places, search and routes in the walkthrough (the design lab's picks: glass-tag pins, the command palette, the floor
   ribbon route). Places come from places.json next to the scene: hand-pinned for the demo, written by the labelling
   stage for pipeline twins. No places.json: nothing shows. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Glass } from '../shared/Glass';
import type { WalkEngine } from './engine';
import type { MapExtras } from './Walk';
import { Navigator, pathLength, steps, type Place } from '../lab/places';
import { Overlay } from '../lab/overlay';
import { Palette, SearchIco } from '../lab/search';

export function PlacesLayer({ engine, base, entered, onMap, onData }: { engine: WalkEngine; base: URL | null; entered: boolean; onMap: (m: MapExtras) => void; onData?: (d: { places: Place[]; nav: Navigator | null }) => void }) {
  const [places, setPlaces] = useState<Place[] | null>(null);
  const [ov, setOv] = useState<Overlay | null>(null); const [nav, setNav] = useState<Navigator | null>(null);
  const [open, setOpen] = useState(false); const [dest, setDest] = useState<string | null>(null); const [guide, setGuide] = useState(false); const [, tick] = useState(0);

  useEffect(() => { if (!base) return; fetch(new URL('places.json', base)).then(r => r.ok ? r.json() : null).then(j => setPlaces(j?.places ?? [])).catch(() => setPlaces([])); }, [base]);
  useEffect(() => {
    if (!engine.state.loaded || !places?.length || ov) return;
    const n = engine.grid ? new Navigator(engine.grid) : null, o = new Overlay(engine, n); o.allPlaces = places;
    setNav(n); setOv(o); onData?.({ places, nav: n }); return () => o.dispose();
  }, [engine.state.loaded, places]);
  // hand-placed places first; AI-found ones (pipeline labelling) show as suggestions, dimmer and without the leader line
  const shown = useMemo(() => (places ?? []).slice(0, 400), [places]);
  useEffect(() => { ov?.setPlaces(shown, 'tag', entered); }, [ov, shown, entered]);
  useEffect(() => { const i = setInterval(() => tick(n => n + 1), 400); return () => clearInterval(i); }, []);

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
    if (engine.state.mode !== 'walk') engine.setMode('walk');
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
    const k = (e: KeyboardEvent) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); document.exitPointerLock?.(); setOpen(v => !v); } if (e.key === 'Escape') setOpen(false); };
    addEventListener('keydown', k); return () => removeEventListener('keydown', k);
  }, []);

  if (!places?.length || !entered) return null;
  const dist = (p: Place) => Math.hypot(p.p[0] - me.x, p.p[2] - me.z);
  const dp = places.find(p => p.id === dest) ?? null, L = path ? pathLength(path) : 0, turns = path && dp ? steps(path, dp.name) : [];
  return (
    <>
      <Glass as="button" className="walk-search pill" opts={{ shader: false, press: true }} onClick={() => { document.exitPointerLock?.(); setOpen(true); }}><SearchIco />Search places<kbd>⌘K</kbd></Glass>
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
