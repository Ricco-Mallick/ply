// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Design lab: the next walkthrough features on the real twin, each with switchable designs, so they can be picked
   before the real build. Places + search, routes, AI labels, measure / share / screenshot, and AR mock-ups
   (navigation, capture guide, dollhouse) shown inside a phone frame. */
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import QRCode from 'qrcode';
import { WalkEngine } from '../walk/engine';
import { RADIUS } from '../walk/walker';
import { loadManifest, type SceneManifest } from '../walk/scene';
import { Glass } from '../shared/Glass';
import { Minimap } from '../walk/Walk';
import { placesFor, Navigator, pathLength, steps, type Place, type Category } from './places';
import { Palette, SearchIco, match } from './search';
import { Overlay, classColor, type PinStyle, type RouteStyle, type AiStyle, type ArMode, type ArNav, type ArCap } from './overlay';

const TOUCH = matchMedia('(pointer: coarse)').matches;
const BASE = import.meta.env.BASE_URL;
const MY_CAPTURE = 'https://ply-api.ricco-ply.workers.dev/v1/scenes/0ffc07fb-4d8d-49eb-9a40-22b9c6fddd76/';
// the bundled demo bedroom by default; ?scene=<MY_CAPTURE> switches to the first real capture
const SCENE = new URLSearchParams(location.search).get('scene');
const { places: PLACES, start: START } = placesFor(SCENE);

interface Opts { pins: PinStyle; search: 'palette' | 'sheet' | 'rail'; route: RouteStyle; ai: AiStyle; ar: ArMode; arNav: ArNav; arCap: ArCap; }
const DEFAULTS: Opts = { pins: 'tag', search: 'palette', route: 'ribbon', ai: 'off', ar: 'off', arNav: 'arrows', arCap: 'floor' };
const ROWS: { key: keyof Opts; label: string; opts: [string, string][]; show?: (o: Opts) => boolean }[] = [
  { key: 'pins', label: 'Place pins', opts: [['tag', 'Glass tag'], ['beacon', 'Beacon'], ['dot', 'Quiet dot']] },
  { key: 'search', label: 'Search', opts: [['palette', 'Palette'], ['sheet', 'Maps sheet'], ['rail', 'Side rail']] },
  { key: 'route', label: 'Route', opts: [['ribbon', 'Floor ribbon'], ['dots', 'Breadcrumbs'], ['tube', 'Light line']] },
  { key: 'ai', label: 'AI labels', opts: [['off', 'Off'], ['boxes', 'Boxes'], ['tint', 'Tint'], ['review', 'Review']] },
  { key: 'ar', label: 'AR mock', opts: [['off', 'Off'], ['nav', 'Navigate'], ['capture', 'Capture guide'], ['doll', 'Dollhouse']] },
  { key: 'arNav', label: 'AR nav style', opts: [['arrows', 'Floor arrows'], ['line', 'Path line'], ['compass', 'Compass']], show: o => o.ar === 'nav' },
  { key: 'arCap', label: 'Guide style', opts: [['floor', 'Painted floor'], ['trail', 'Floor + trail']], show: o => o.ar === 'capture' },
];

function useTick(engine: WalkEngine | null, hz = 6) {
  const [, set] = useState(0); useEffect(() => { const i = setInterval(() => set(n => n + 1), 1000 / hz); return () => clearInterval(i); }, [hz]);
  useSyncExternalStore(cb => engine ? engine.on(cb) : () => {}, () => engine?.state.loaded);
}

export function Lab() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [engine, setEngine] = useState<WalkEngine | null>(null); const [ov, setOv] = useState<Overlay | null>(null); const [nav, setNav] = useState<Navigator | null>(null);
  const [manifest, setManifest] = useState<SceneManifest | null>(null); const [err, setErr] = useState<string | null>(null); const [entered, setEntered] = useState(false);
  const [o, setO] = useState<Opts>(() => { try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('ply-lab') || '{}') }; } catch { return DEFAULTS; } });
  const [panel, setPanel] = useState(!TOUCH); const [dest, setDest] = useState<string | null>(null); const [guide, setGuide] = useState(false);
  const [tool, setTool] = useState<null | 'measure' | 'share' | 'shot'>(null); const [shot, setShot] = useState<string | null>(null);
  const [accepted, setAccepted] = useState<Set<string>>(new Set()); const [rejected, setRejected] = useState<Set<string>>(new Set());
  const [paletteOpen, setPaletteOpen] = useState(false);
  useTick(engine);
  const set = <K extends keyof Opts>(k: K, v: Opts[K]) => setO(p => { const n = { ...p, [k]: v }; try { localStorage.setItem('ply-lab', JSON.stringify(n)); } catch { /* ignore */ } return n; });

  // boot: engine, scene (the first real capture unless ?scene=), navigator and overlay once the grid exists
  useEffect(() => {
    const e = new WalkEngine(canvas.current!); setEngine(e); if (import.meta.env.DEV) (window as any).__ply = e;
    const q = new URLSearchParams(location.search);
    loadManifest().then(async ({ manifest, base }) => {
      setManifest(manifest); await e.load(manifest, base);
      const at = q.get('at')?.split(',').map(Number); if (at?.length === 4 && at.every(Number.isFinite)) { e.state.pos.x = at[0]; e.state.pos.z = at[1]; e.state.yaw = at[2]; e.pitch = at[3]; }
      else if (e.grid) { e.state.pos.x = START.x; e.state.pos.z = START.z; e.state.yaw = START.yaw; e.pitch = START.pitch; e.unstick(); }
      const n = e.grid ? new Navigator(e.grid, RADIUS) : null; setNav(n);
      const overlay = new Overlay(e, n); overlay.allPlaces = PLACES; setOv(overlay); if (import.meta.env.DEV) (window as any).__lab = overlay;
    }).catch(x => setErr(x.message));
    return () => e.dispose();
  }, []);

  const places = useMemo(() => PLACES.filter(p => p.pinned || accepted.has(p.id)), [accepted]);
  const me = engine?.state.pos;
  const path = useMemo(() => {
    if (!nav || !dest || !me) return null; const p = PLACES.find(x => x.id === dest)!;
    return nav.route([me.x, me.z], [p.p[0], p.p[2]]);
    // recompute when the destination changes, or when you have wandered off (rounded position)
  }, [nav, dest, me && Math.round(me.x * 2), me && Math.round(me.z * 2)]);

  useEffect(() => { ov?.setPlaces(places, o.pins, o.ar !== 'capture'); }, [ov, places, o.pins, o.ar]);
  useEffect(() => { ov?.setAi(o.ai, accepted); }, [ov, o.ai, accepted]);
  useEffect(() => { ov?.setAr(o.ar, o.arNav, o.arCap); }, [ov, o.ar, o.arNav, o.arCap]);
  useEffect(() => { if (ov && !(o.ar === 'capture' && o.arCap === 'trail')) ov.setRoute(path, o.route, dest); }, [ov, path, o.route, dest, o.ar, o.arNav, o.arCap]);
  // capture guide "trail": a breadcrumb route to the nearest floor nobody has filmed yet
  useEffect(() => {
    if (!ov || !nav || o.ar !== 'capture' || o.arCap !== 'trail') return;
    const i = setInterval(() => { const g = ov.nextGap(); const p = engine!.state.pos; ov.setRoute(g ? nav.route([p.x, p.z], g) : null, 'dots', null); }, 900);
    return () => { clearInterval(i); ov.setRoute(null, o.route, null); };
  }, [ov, nav, o.ar, o.arCap]);
  // AR framing: a phone's camera height and field of view
  useEffect(() => { if (!engine) return; engine.eye = o.ar === 'nav' || o.ar === 'capture' ? 1.4 : (manifest?.eyeHeight ?? 1.55); engine.camera.fov = o.ar === 'nav' || o.ar === 'capture' ? 62 : 70; engine.camera.updateProjectionMatrix(); }, [engine, o.ar, manifest]);

  // a new destination: turn to face the start of the route and look at the floor, so the route is in view
  const faced = useRef<string | null>(null);
  useEffect(() => {
    if (!engine || !path || path.length < 2 || !dest || faced.current === dest || o.ar === 'doll') return; faced.current = dest;
    const me = engine.state.pos, tgt = path.find(q => Math.hypot(q[0] - me.x, q[1] - me.z) > .6) ?? path[path.length - 1];
    let want = Math.atan2(-(tgt[0] - me.x), -(tgt[1] - me.z)), t = 0; const y0 = engine.state.yaw, p0 = engine.pitch;
    while (want - y0 > Math.PI) want -= 2 * Math.PI; while (want - y0 < -Math.PI) want += 2 * Math.PI;
    const hook = (dt: number) => { t = Math.min(1, t + dt / .9); const k = 1 - (1 - t) ** 3; engine.state.yaw = y0 + (want - y0) * k; engine.pitch = p0 + (-.42 - p0) * k; if (t >= 1) engine.frameHooks.delete(hook); };
    engine.frameHooks.add(hook); return () => { engine.frameHooks.delete(hook); };
  }, [engine, path, dest, o.ar]);
  useEffect(() => { if (!dest) faced.current = null; }, [dest]);

  // guided walk along the route ("Take me there")
  useEffect(() => {
    if (!engine || !guide || !path) return; let d = 0; const L = pathLength(path);
    const at = (s: number): [number, number] => { let acc = 0; for (let i = 1; i < path.length; i++) { const seg = Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]); if (acc + seg >= s) { const k = (s - acc) / seg; return [path[i - 1][0] + (path[i][0] - path[i - 1][0]) * k, path[i - 1][1] + (path[i][1] - path[i - 1][1]) * k]; } acc += seg; } return path[path.length - 1]; };
    const fixed = path.slice();
    const hook = (dt: number) => {
      d = Math.min(L, d + dt * 1.25); const [x, z] = at(d), [ax, az] = at(Math.min(L, d + .6)); engine.state.pos.x = x; engine.state.pos.z = z;
      if (Math.hypot(ax - x, az - z) > .05) { const want = Math.atan2(-(ax - x), -(az - z)); let dy = want - engine.state.yaw; while (dy > Math.PI) dy -= 2 * Math.PI; while (dy < -Math.PI) dy += 2 * Math.PI; engine.state.yaw += dy * Math.min(1, dt * 3); }
      engine.pitch += (-.08 - engine.pitch) * Math.min(1, dt * 2);
      if (d >= L) { setGuide(false); }
    };
    void fixed; engine.frameHooks.add(hook); return () => { engine.frameHooks.delete(hook); };
  }, [engine, guide, dest]);

  // keyboard: Ctrl/Cmd+K opens search, Esc closes things, M measure
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setPaletteOpen(v => !v); }
      if (e.key === 'Escape') { setPaletteOpen(false); setTool(null); }
    };
    addEventListener('keydown', k); return () => removeEventListener('keydown', k);
  }, []);

  // clicks on the stage: tape measure points, dollhouse orbit drags
  useEffect(() => {
    const c = canvas.current; if (!c || !ov) return; let down: { x: number; y: number } | null = null;
    const pd = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY }; };
    const pm = (e: PointerEvent) => { if (down && o.ar === 'doll' && e.buttons) { ov.dollDrag(e.movementX, e.movementY); } };
    const pu = (e: PointerEvent) => { if (!down) return; const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y); down = null; if (moved > 6) return;
      if (tool === 'measure') { const p = ov.pick(e.clientX, e.clientY, true); if (p) ov.addMeasure(p); } };
    c.addEventListener('pointerdown', pd); c.addEventListener('pointermove', pm); c.addEventListener('pointerup', pu);
    return () => { c.removeEventListener('pointerdown', pd); c.removeEventListener('pointermove', pm); c.removeEventListener('pointerup', pu); };
  }, [ov, tool, o.ar]);
  useEffect(() => { if (tool !== 'measure') ov?.clearMeasure(); }, [tool, ov]);

  const go = (id: string) => { setDest(id); setGuide(false); setPaletteOpen(false); };
  const end = () => { setDest(null); setGuide(false); };
  const takeShot = () => { if (!engine) return; engine.renderer.render(engine.scene, engine.camera); setShot(engine.canvas.toDataURL('image/jpeg', .9)); setTool('shot'); };
  const s = engine?.state; const ready = !!s?.loaded && !!ov;
  const destPlace = PLACES.find(p => p.id === dest) ?? null; const L = path ? pathLength(path) : 0; const turnList = path && destPlace ? steps(path, destPlace.name) : [];
  const dist = (p: Place) => me ? Math.hypot(p.p[0] - me.x, p.p[2] - me.z) : 0;
  const ar = o.ar !== 'off';

  return (
    <div className={`lab ${ar ? `ar ar-${o.ar}` : ''} ${tool === 'measure' ? 'measuring' : ''}`}>
      <canvas ref={canvas} className="stage" aria-label={manifest ? `3D twin of ${manifest.name}` : '3D twin'} />

      <header className="top">
        <Glass as="a" href={BASE} className="brand pill" opts={{ shader: false, press: true }}><span className="mark" aria-hidden="true" />PLY<span className="scene-name">Design lab</span></Glass>
        <div className="top-right">
          {o.search === 'palette' && !ar && <Glass as="button" className="lab-search-btn pill" opts={{ shader: false, press: true }} onClick={() => setPaletteOpen(true)}><SearchIco />Search places<kbd>⌘K</kbd></Glass>}
          <Glass as="button" className={`icon pill ${panel ? 'on' : ''}`} opts={{ shader: false, press: true }} aria-label="Design options" aria-pressed={panel} onClick={() => setPanel(v => !v)}><SlidersIco /></Glass>
        </div>
      </header>

      {panel && <Glass className="lab-panel" opts={{ radius: 18, shader: false }}>
        <div className="lab-panel-h"><span className="wide eyebrow">Design lab</span><span className="mono dim">pick one per row</span></div>
        {ROWS.filter(r => !r.show || r.show(o)).map(r => (
          <div className="lab-row" key={r.key}><span className="lab-row-l">{r.label}</span>
            <div className="lab-seg" role="radiogroup" aria-label={r.label}>{r.opts.map(([v, l]) => <button key={v} role="radio" aria-checked={o[r.key] === v} className={o[r.key] === v ? 'on' : ''} onClick={() => set(r.key, v as never)}>{l}</button>)}</div>
          </div>))}
        <div className="lab-row"><span className="lab-row-l">Scene</span><div className="lab-seg">
          <a className={!SCENE ? 'on' : ''} href={`${BASE}lab.html`}>Demo bedroom</a><a className={SCENE ? 'on' : ''} href={`${BASE}lab.html?scene=${encodeURIComponent(MY_CAPTURE)}`}>My capture</a></div></div>
        <p className="lab-note">Places and AI boxes are pinned by hand for this demo; the build fills them in automatically.</p>
      </Glass>}

      {/* ---------- search ---------- */}
      {!ar && o.search === 'palette' && paletteOpen && <Palette places={PLACES} pinned={places} dist={dist} onPick={go} onClose={() => setPaletteOpen(false)} />}
      {!ar && o.search === 'sheet' && !dest && <Sheet places={places} dist={dist} onPick={go} />}
      {!ar && o.search === 'rail' && <Rail places={places} dist={dist} dest={dest} onPick={go} />}

      {/* ---------- route card ---------- */}
      {!ar && destPlace && <Glass className={`route-card rc-${o.search}`} opts={{ radius: 20, shader: false }}>
        <div className="rc-h"><span className="wide eyebrow">Route</span><button className="rc-x" onClick={end} aria-label="End route">×</button></div>
        <h2>{destPlace.name}</h2>
        {path ? <p className="mono rc-m"><b>{L.toFixed(1)} m</b> · {Math.max(1, Math.round(L / 1.2))} s walk · {turnList.length - 1} turn{turnList.length === 2 ? '' : 's'}</p> : <p className="mono rc-m bad">No walkable route from here</p>}
        {path && <ol className="rc-steps">{turnList.map((t, i) => <li key={i}>{t.text}</li>)}</ol>}
        {path && <div className="rc-a"><button className="btn" onClick={() => setGuide(g => !g)}>{guide ? 'Pause' : 'Take me there'}</button><button className="btn ghost" onClick={() => set('ar', 'nav')}>Navigate in AR</button></div>}
      </Glass>}

      {/* ---------- AI review ---------- */}
      {!ar && o.ai === 'review' && <Glass className="ai-review" opts={{ radius: 18, shader: false }}>
        <div className="rc-h"><span className="wide eyebrow">AI found {PLACES.filter(p => p.ai).length} things</span></div>
        <ul>{PLACES.filter(p => p.ai && !p.pinned && !rejected.has(p.id)).map(p => (
          <li key={p.id} className={accepted.has(p.id) ? 'ok' : ''}><i style={{ background: classColor(p.ai!.cls) }} /><span><b>{p.name}</b><em className="mono">{p.ai!.cls} · {Math.round(p.ai!.conf * 100)}%</em></span>
            {accepted.has(p.id) ? <span className="mono dim">added</span> : <><button onClick={() => setAccepted(a => new Set(a).add(p.id))} aria-label={`Add ${p.name}`}>✓</button><button onClick={() => setRejected(r => new Set(r).add(p.id))} aria-label={`Dismiss ${p.name}`}>×</button></>}</li>))}</ul>
      </Glass>}

      {/* ---------- tools ---------- */}
      {!ar && <Glass className="tools pill" opts={{ shader: false }}>
        <button className={tool === 'measure' ? 'on' : ''} onClick={() => setTool(t => t === 'measure' ? null : 'measure')}><RulerIco />Measure</button>
        <button className={tool === 'share' ? 'on' : ''} onClick={() => setTool(t => t === 'share' ? null : 'share')}><ShareIco />Share</button>
        <button onClick={takeShot}><CamIco />Snapshot</button>
      </Glass>}
      {tool === 'measure' && !ar && <Glass className="tool-hint pill" opts={{ shader: false }}><span className="mono">{ov?.measured != null ? `${ov.measured.toFixed(2)} m · click again to start over` : `Click two points on anything${ov?.measurePts.length === 1 ? ' · one more' : ''}`}</span></Glass>}
      {tool === 'share' && engine && <ShareSheet engine={engine} onClose={() => setTool(null)} />}
      {tool === 'shot' && shot && <Glass className="shot-card" opts={{ radius: 18, shader: false }}><img src={shot} alt="Snapshot of the current view" /><div className="rc-a"><a className="btn" href={shot} download="ply-snapshot.jpg">Save image</a><button className="btn ghost" onClick={() => setTool(null)}>Close</button></div></Glass>}
      {tool === 'shot' && <div className="flash" key={shot} />}

      {!ar && engine && entered && <Minimap engine={engine} manifest={manifest} extras={{ path, pins: places.map(p => ({ x: p.p[0], z: p.p[2], dest: p.id === dest })) }} />}

      {/* ---------- AR mock-ups ---------- */}
      {ar && <ArChrome o={o} ov={ov} engine={engine} dest={destPlace} path={path} turns={turnList} onPick={go} places={places} dist={dist} onExit={() => set('ar', 'off')} />}

      <div className={`gate ${entered ? 'gone' : ''}`}>
        <div className="gate-inner">
          <div className="wide eyebrow">Design lab</div>
          <h1>The next features, on a real twin</h1>
          {err || s?.error ? <p className="err" role="alert">{err ?? s?.error}</p> : <>
            <div className="bar"><i style={{ transform: `scaleX(${s?.progress ?? 0})` }} /></div>
            <p className="mono status">{ready ? 'Ready' : `Streaming your twin · ${Math.round((s?.progress ?? 0) * 100)}%`}</p>
            <button className="btn" disabled={!ready} onClick={() => setEntered(true)}>Open the lab</button>
          </>}
        </div>
      </div>
    </div>
  );
}

/* ---------------- search: three designs ---------------- */
const CATS: Category[] = ['Furniture', 'Storage', 'Exits', 'Things'];
function Sheet({ places, dist, onPick }: { places: Place[]; dist: (p: Place) => number; onPick: (id: string) => void }) {
  const [q, setQ] = useState(''); const [cat, setCat] = useState<Category | null>(null); const [open, setOpen] = useState(true);
  const list = places.filter(p => match(p, q) && (!cat || p.cat === cat)).sort((a, b) => dist(a) - dist(b));
  return (
    <Glass className={`sheet ${open ? 'open' : ''}`} opts={{ radius: 26, shader: false }}>
      <button className="sheet-grip" onClick={() => setOpen(v => !v)} aria-label={open ? 'Collapse' : 'Expand'}><span /></button>
      <div className="sheet-in"><SearchIco /><input placeholder="Where to?" value={q} onFocus={() => setOpen(true)} onChange={e => setQ(e.target.value)} /></div>
      {open && <>
        <div className="chips">{CATS.map(c => <button key={c} className={cat === c ? 'on' : ''} onClick={() => setCat(v => v === c ? null : c)}>{c}</button>)}</div>
        <div className="cards">{list.map(p => <button key={p.id} className="card" onClick={() => onPick(p.id)}><span className="card-c mono">{p.cat}</span><b>{p.name}</b><span className="mono dim">{dist(p).toFixed(1)} m · {Math.max(1, Math.round(dist(p) / 1.2))} s</span></button>)}</div>
      </>}
    </Glass>
  );
}

function Rail({ places, dist, dest, onPick }: { places: Place[]; dist: (p: Place) => number; dest: string | null; onPick: (id: string) => void }) {
  const [q, setQ] = useState('');
  return (
    <Glass className="rail" opts={{ radius: 20, shader: false }}>
      <div className="rail-in"><SearchIco /><input placeholder="Filter places" value={q} onChange={e => setQ(e.target.value)} /></div>
      <ul>{CATS.map(c => { const l = places.filter(p => p.cat === c && match(p, q)); return l.length ? <li key={c}><span className="mono rail-c">{c}</span>
        {l.map(p => <button key={p.id} className={dest === p.id ? 'on' : ''} onClick={() => onPick(p.id)}><span>{p.name}</span><span className="mono dim">{dist(p).toFixed(1)} m</span></button>)}</li> : null; })}</ul>
    </Glass>
  );
}

function ShareSheet({ engine, onClose }: { engine: WalkEngine; onClose: () => void }) {
  const p = engine.state.pos, q = new URLSearchParams(location.search); q.set('at', [p.x, p.z, engine.state.yaw, engine.pitch].map(v => v.toFixed(2)).join(','));
  const url = `${location.origin}${location.pathname}?${q}`; const [qr, setQr] = useState(''); const [copied, setCopied] = useState(false);
  useEffect(() => { QRCode.toDataURL(url, { margin: 1, width: 360, color: { dark: '#0B1220', light: '#FFFFFF' } }).then(setQr); }, [url]);
  return (
    <Glass className="share" opts={{ radius: 22, shader: false }}>
      <div className="rc-h"><span className="wide eyebrow">Share this view</span><button className="rc-x" onClick={onClose} aria-label="Close">×</button></div>
      <p className="lead-s">Opens the twin right here, facing the same way.</p>
      {qr && <img className="qr" src={qr} alt="QR code for this view" />}
      <div className="share-url mono">{url.replace(/^https?:\/\//, '').slice(0, 64)}…</div>
      <button className="btn" onClick={() => navigator.clipboard?.writeText(url).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1400); })}>{copied ? 'Link copied' : 'Copy link'}</button>
    </Glass>
  );
}

/* ---------------- AR chrome (mock-ups of what the phone shows in Chrome's AR mode) ---------------- */
function ArChrome({ o, ov, engine, dest, path, turns, places, dist, onPick, onExit }: { o: Opts; ov: Overlay | null; engine: WalkEngine | null; dest: Place | null; path: [number, number][] | null; turns: { text: string; at: number }[]; places: Place[]; dist: (p: Place) => number; onPick: (id: string) => void; onExit: () => void }) {
  const left = path ? pathLength(path) : 0;
  // compass: bearing to the next corner relative to where the phone points
  let rel = 0; if (engine && path && path.length > 1) { const me = engine.state.pos, n = path.find(p => Math.hypot(p[0] - me.x, p[1] - me.z) > .5) ?? path[path.length - 1]; const want = Math.atan2(-(n[0] - me.x), -(n[1] - me.z)); rel = want - engine.state.yaw; while (rel > Math.PI) rel -= 2 * Math.PI; while (rel < -Math.PI) rel += 2 * Math.PI; }
  const next = turns.find(t => t.at > .3) ?? turns[turns.length - 1];
  return (
    <div className="arc">
      <div className="arc-frame" aria-hidden="true"><span className="arc-notch" /></div>
      <div className="arc-top">
        <span className="arc-pill mono"><i className="arc-live" />{o.ar === 'doll' ? 'AR · placed on table' : 'AR · floor tracked · 1:1 scale'}</span>
        <button className="arc-x" onClick={onExit} aria-label="Leave AR">×</button>
      </div>
      {o.ar === 'nav' && (dest ? <>
        <div className="arc-instr"><span className="arc-arrow" style={{ transform: `rotate(${-rel}rad)` }}>↑</span><div><b>{next?.text ?? 'Head along the arrows'}</b><span className="mono">{dest.name} · {left.toFixed(1)} m left</span></div></div>
        {o.arNav === 'compass' && <div className="arc-compass"><div className="arc-dial" style={{ transform: `rotate(${-rel}rad)` }}><span>▲</span></div><b className="mono">{left.toFixed(1)} m</b></div>}
        <div className="arc-bottom"><span className="mono">Walk with WASD to try it · the phone does this for real</span></div>
      </> : <div className="arc-pick"><b>Where to?</b><div className="arc-picks">{places.map(p => <button key={p.id} onClick={() => onPick(p.id)}>{p.name}<em className="mono">{dist(p).toFixed(1)} m</em></button>)}</div>
        <span className="mono dim">First the phone lines up with the twin: scan the Ply QR on the door, or tap two marked spots.</span></div>)}
      {o.ar === 'capture' && <>
        <div className="arc-cov"><svg viewBox="0 0 36 36" width="54" height="54"><circle cx="18" cy="18" r="15.5" fill="none" stroke="rgba(255,255,255,.2)" strokeWidth="3" /><circle cx="18" cy="18" r="15.5" fill="none" stroke="#3DDC97" strokeWidth="3" strokeDasharray={`${(ov?.coverage ?? 0) * 97.4} 97.4`} transform="rotate(-90 18 18)" strokeLinecap="round" /></svg>
          <div><b>{Math.round((ov?.coverage ?? 0) * 100)}% of the floor covered</b><span className="mono">{(ov?.coverage ?? 0) > .85 ? 'Great, finish with a slow loop' : o.arCap === 'trail' ? 'Follow the dots to the grey patch' : 'Walk until the grey floor turns green'}</span></div></div>
        <div className="arc-rec"><span className="arc-dot" />REC 01:24</div>
      </>}
      {o.ar === 'doll' && <div className="arc-bottom"><span className="mono">Drag to turn · the twin sits on your real table at 1:40</span></div>}
    </div>
  );
}

const RulerIco = () => <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><rect x="2.5" y="8" width="19" height="8" rx="1.5" /><path d="M6.5 8v3M10 8v4M13.5 8v3M17 8v4" /></svg>;
const ShareIco = () => <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M12 15V3m0 0L8 7m4-4 4 4" /><path d="M5 12v7a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-7" /></svg>;
const CamIco = () => <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M4 8h3l2-3h6l2 3h3v11H4z" /><circle cx="12" cy="13" r="3.5" /></svg>;
const SlidersIco = () => <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" aria-hidden="true"><path d="M4 7h9M17 7h3M4 17h3M11 17h9" /><circle cx="15" cy="7" r="2" /><circle cx="9" cy="17" r="2" /></svg>;
