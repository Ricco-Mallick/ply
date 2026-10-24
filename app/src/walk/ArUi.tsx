// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* AR controls for the walkthrough: an "AR" menu (Android Chrome only) and the overlay shown over the camera during an
   AR session: two-tap alignment, destination picker and turn-by-turn for navigation; place / turn for the dollhouse. */
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Glass } from '../shared/Glass';
import { ArDollhouse, ArNavigator, arSupported, type NavPlace } from './ar';
import type { Navigator } from '../lab/places';

export function ArLauncher({ places, nav, splatUrl, bounds, floorY }: { places: NavPlace[]; nav: Navigator | null; splatUrl: string | null; bounds: { min: number[]; max: number[] } | null; floorY: number }) {
  const [ok, setOk] = useState(false); const [menu, setMenu] = useState(false); const [err, setErr] = useState<string | null>(null);
  const [run, setRun] = useState<ArNavigator | ArDollhouse | null>(null); const hud = useRef<HTMLDivElement>(null);
  const tick = useRef(0); useSyncExternalStore(cb => run ? run.on(() => { tick.current++; cb(); }) : () => {}, () => tick.current);
  useEffect(() => { arSupported().then(setOk); }, []);
  const drag = useRef<number | null>(null);
  if (!ok) return null;
  const start = async (kind: 'nav' | 'doll') => {
    setMenu(false); setErr(null); document.exitPointerLock?.();
    const r = kind === 'nav' ? new ArNavigator(places, nav!) : new ArDollhouse(splatUrl!, bounds, floorY);
    setRun(r);
    try { await r.start(hud.current!); r.session.addEventListener('end', () => setRun(null)); } catch (e) { setRun(null); setErr(`AR could not start: ${(e as Error).message}`); }
  };
  const canNav = places.length >= 2 && !!nav;
  const n = run instanceof ArNavigator ? run : null, d = run instanceof ArDollhouse ? run : null;
  return (
    <>
      <div className="views-wrap"><Glass as="button" className="tour-btn pill" opts={{ shader: false, press: true }} aria-expanded={menu} onClick={() => setMenu(v => !v)}>AR</Glass>
        {menu && <Glass className="views-menu" opts={{ radius: 14, shader: false }}>
          <button disabled={!canNav} onClick={() => start('nav')}><span className="mono">01</span>Navigate here{canNav ? '' : ' (needs places)'}</button>
          <button disabled={!splatUrl} onClick={() => start('doll')}><span className="mono">02</span>Dollhouse on a table</button>
        </Glass>}</div>
      {err && <div className="ar-err mono" role="alert" onClick={() => setErr(null)}>{err}</div>}
      {/* the DOM overlay: shown over the camera while the AR session runs */}
      <div ref={hud} className="arx" hidden={!run}
        onPointerDown={e => { if (d?.placed) drag.current = e.clientX; }} onPointerMove={e => { if (d && drag.current != null) { d.turn(e.clientX - drag.current); drag.current = e.clientX; } }} onPointerUp={() => { drag.current = null; }}>
        <div className="arx-top"><span className="arx-pill mono">{n ? 'AR navigation' : 'AR dollhouse'}</span><button className="arx-x" onClick={() => run?.stop()} aria-label="Leave AR">×</button></div>
        {n && (n.step === 'align1' || n.step === 'align2') && <div className="arx-card">
          <span className="mono">Line up · {n.step === 'align1' ? '1' : '2'} of 2</span>
          <b>Aim the ring at the floor right in front of <em>{n.anchors[n.step === 'align1' ? 0 : 1].name}</em>, then tap Here.</b>
          <button className="btn" onClick={() => n.tap()}>Here</button>
        </div>}
        {n?.step === 'pick' && <div className="arx-card"><span className="mono">Lined up{n.fit ? ` · scale ${n.fit.scale.toFixed(2)}` : ''}</span><b>Where to?</b>
          {n.warning && <p className="arx-warn">{n.warning}</p>}
          <div className="arx-picks">{n.places.map(p => <button key={p.id} onClick={() => n.go(p)}>{p.name}</button>)}</div>
          <button className="btn ghost" onClick={() => n.realign()}>Realign</button></div>}
        {n?.step === 'go' && <><div className="arx-instr"><b>{n.next || 'Planning…'}</b><span className="mono">Follow the arrows on the floor</span></div>
          <div className="arx-bottom"><button className="btn ghost" onClick={() => { n.step = 'pick'; n.emit(); }}>Change place</button><button className="btn ghost" onClick={() => n.realign()}>Realign</button></div></>}
        {d && !d.placed && <div className="arx-card"><b>Aim the ring at a table or the floor, then tap Place.</b><button className="btn" onClick={() => d.tap()}>Place</button></div>}
        {d?.placed && <div className="arx-bottom"><span className="arx-pill mono">Drag to turn</span><button className="btn ghost" onClick={() => d.reset()}>Move it</button></div>}
      </div>
    </>
  );
}
