// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { WalkEngine, type Mode } from './engine';
import { loadManifest, type SceneManifest } from './scene';
import { Glass } from '../shared/Glass';
import { Tour, autoTour, SHOT_NAMES } from './tour';
import { PlacesLayer } from './Places';
import { ArLauncher } from './ArUi';
import { MeasureLayer, useShell } from './Measure';
import type { Navigator, Place } from '../lab/places';

const TOUCH = matchMedia('(pointer: coarse)').matches;
const BASE = import.meta.env.BASE_URL;

function useEngine(engine: WalkEngine | null) {
  const snap = useRef(0);
  return useSyncExternalStore(cb => engine ? engine.on(() => { snap.current++; cb(); }) : () => {}, () => snap.current);
}

export function Walk() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [engine, setEngine] = useState<WalkEngine | null>(null);
  const [manifest, setManifest] = useState<SceneManifest | null>(null);
  const [entered, setEntered] = useState(false);
  const [help, setHelp] = useState(true);
  // display gain, because the splat's own colour has no tone curve: a capture that clipped will glare
  const [exposure, setExposure] = useState(() => { const v = +(localStorage.getItem('ply-exposure') ?? 1); return v >= .5 && v <= 1.6 ? v : 1; });
  useEffect(() => { const el = canvas.current; if (el) el.style.filter = exposure === 1 ? '' : `brightness(${exposure})`; try { localStorage.setItem('ply-exposure', String(exposure)); } catch { /* private mode */ } }, [exposure]);
  useEffect(() => { const k = (e: KeyboardEvent) => {
    if (e.key === '[') setExposure(v => Math.max(.5, +(v - .05).toFixed(2)));
    if (e.key === ']') setExposure(v => Math.min(1.6, +(v + .05).toFixed(2)));
    if (e.key === '\\') setExposure(1);
  }; addEventListener('keydown', k); return () => removeEventListener('keydown', k); }, []);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [base, setBase] = useState<URL | null>(null); const [mapX, setMapX] = useState<MapExtras>({});
  const [arData, setArData] = useState<{ places: Place[]; nav: Navigator | null }>({ places: [], nav: null });
  useEngine(engine);
  // cinematic tour: the pipeline's keyframes, or ones made from the walkable floor
  const [measured, setMeasured] = useState(false);
  const shell = useShell(base);
  const [tour, setTour] = useState<Tour | null>(null); const [, tourTick] = useState(0); const [views, setViews] = useState(false);
  useEffect(() => {
    if (!engine?.state.loaded || tour) return;
    const keys = manifest?.tour?.length ? manifest.tour : engine.grid ? autoTour(engine.grid, engine.floorY, engine.eye, engine.state.pos) : [];
    if (keys.length > 1) setTour(new Tour(engine, keys, () => tourTick(n => n + 1)));
  }, [engine?.state.loaded, manifest]);

  useEffect(() => {
    const e = new WalkEngine(canvas.current!); setEngine(e); if (import.meta.env.DEV) (window as any).__ply = e;
    loadManifest().then(({ manifest, base }) => { setManifest(manifest); setBase(base); document.title = `${manifest.name} · Ply`; return e.load(manifest, base); }).catch(err => setLoadErr(err.message));
    return () => e.dispose();
  }, []);
  // the controls hint steps aside once someone is moving, and comes back on ? or H
  useEffect(() => { if (engine?.state.moved) { const t = setTimeout(() => setHelp(false), 3500); return () => clearTimeout(t); } }, [engine?.state.moved]);
  useEffect(() => { const k = (e: KeyboardEvent) => { if (e.code === 'KeyH' || e.key === '?') setHelp(h => !h); }; addEventListener('keydown', k); return () => removeEventListener('keydown', k); }, []);

  const s = engine?.state; const err = loadErr ?? s?.error ?? null; const ready = !!s?.loaded;
  const enter = () => { if (!ready || !engine) return; setEntered(true); engine.lock(); };

  return (
    <>
      <canvas ref={canvas} className="stage" aria-label={manifest ? `3D walkthrough of ${manifest.name}` : '3D walkthrough'} onClick={() => { if (entered && !s?.locked) engine?.lock(); }} />

      <header className="top">
        <Glass as="a" href={BASE} className="brand pill" opts={{ shader: false, press: true }}><span className="mark" aria-hidden="true" />PLY<span className="scene-name">{manifest?.name ?? 'Loading'}</span></Glass>
        <div className="top-right">
          <Glass className="seg pill" opts={{ shader: false }} role="radiogroup" aria-label="Movement mode">
            {(['fly', 'walk'] as Mode[]).map(m => <button key={m} role="radio" aria-checked={s?.mode === m} className={s?.mode === m ? 'on' : ''} onClick={() => engine?.setMode(m)}>{m === 'fly' ? 'Free' : 'Walk'}</button>)}
          </Glass>
          {tour && entered && <Glass as="button" className={`tour-btn pill ${tour.playing ? 'on' : ''}`} opts={{ shader: false, press: true }} onClick={() => { if (tour.playing) tour.stop(); else { setViews(false); document.exitPointerLock?.(); tour.play(); } }}>{tour.playing ? '■ Stop tour' : '▶ Tour'}</Glass>}
          {tour && entered && <div className="views-wrap"><Glass as="button" className="tour-btn pill" opts={{ shader: false, press: true }} aria-expanded={views} onClick={() => setViews(v => !v)}>Views</Glass>
            {views && <Glass className="views-menu" opts={{ radius: 14, shader: false }}>{tour.keys.map((_, i) => <button key={i} onClick={() => { setViews(false); document.exitPointerLock?.(); tour.goTo(i); }}><span className="mono">{String(i + 1).padStart(2, '0')}</span>{SHOT_NAMES[i] ?? `View ${i + 1}`}</button>)}</Glass>}</div>}
          {entered && shell?.walls?.length && <Glass as="button" className={`tour-btn pill ${measured ? 'on' : ''}`} opts={{ shader: false, press: true }} aria-pressed={measured} title="Measured shell: walls and object boxes from the solve" onClick={() => setMeasured(v => !v)}>Measured</Glass>}
          {entered && manifest && base && <Glass as="a" className="tour-btn pill" opts={{ shader: false, press: true }} title="Open this twin in the editor (SuperSplat)"
            target="_blank" rel="noopener"
            href={`${BASE}editor/index.html?load=${encodeURIComponent(new URL(manifest.splat.replace(/scene\.spz$/, 'scene_lite.spz'), base).href)}&filename=${encodeURIComponent((manifest.name || 'twin').replace(/[^\w.-]+/g, '_'))}.spz`}>Edit</Glass>}
          {entered && engine && manifest && base && <ArLauncher places={arData.places.filter(p => p.pinned).map(p => ({ id: p.id, name: p.name, x: p.p[0], z: p.p[2] }))} nav={arData.nav}
            splatUrl={new URL(manifest.splat, base).href} bounds={manifest.bounds ?? null} floorY={engine.floorY} />}
          {entered && <Glass as="button" className="tour-btn pill" opts={{ shader: false, press: true }} title="Display exposure: [ dimmer, ] brighter, \\ reset" aria-label={`Exposure ${Math.round(exposure * 100)} percent`} onClick={() => setExposure(e => e >= 1.55 ? .7 : +(e + .1).toFixed(2))}>☀ {Math.round(exposure * 100)}%</Glass>}
          <Glass as="button" className="icon pill" opts={{ shader: false, press: true }} aria-label="Controls" aria-pressed={help} onClick={() => setHelp(h => !h)}>?</Glass>
          <Glass as="button" className="icon pill" opts={{ shader: false, press: true }} aria-label="Fullscreen" onClick={() => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.()}>⤢</Glass>
        </div>
      </header>

      {entered && s?.locked && <div className="cross" aria-hidden="true" />}

      <Glass className={`hint ${help && entered && !tour?.playing ? 'on' : ''}`} opts={{ radius: 16, vis: () => (help && entered && !tour?.playing ? 1 : 0) }} aria-hidden={!(help && entered && !tour?.playing)}>
        {TOUCH ? (
          <ul><li><b>Left thumb</b> move</li><li><b>Drag right side</b> look</li><li><b>Free</b> switch top right</li></ul>
        ) : (
          <ul>
            <li><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</li><li><kbd>Shift</kbd> run</li><li><kbd>F</kbd> free / walk</li>
            {s?.mode === 'fly' && <li><kbd>Space</kbd>/<kbd>C</kbd> up / down</li>}<li><kbd>Scroll</kbd> speed {s ? s.speed.toFixed(1) : '1.0'}×</li><li><kbd>[</kbd><kbd>]</kbd> exposure</li><li><kbd>Esc</kbd> free the mouse</li>
          </ul>
        )}
      </Glass>

      {TOUCH && entered && engine && <Joystick engine={engine} />}
      {engine && <MeasureLayer engine={engine} shell={shell} on={measured} showObjects={measured} />}
      {engine && <PlacesLayer engine={engine} base={base} entered={entered} onMap={setMapX} onData={setArData} />}
      {engine && <Minimap engine={engine} manifest={manifest} extras={mapX} />}

      <div className={`gate ${entered ? 'gone' : ''}`}>
        <div className="gate-inner">
          <div className="wide eyebrow">Walkthrough</div>
          <h1>{manifest?.name ?? 'Loading the space'}</h1>
          {err ? <p className="err" role="alert">{err}. Check the scene link or try again.</p> : (
            <>
              <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round((s?.progress ?? 0) * 100)} aria-label="Loading the space"><i style={{ transform: `scaleX(${s?.progress ?? 0})` }} /></div>
              <p className="mono status">{ready ? 'Ready' : `Streaming splats · ${Math.round((s?.progress ?? 0) * 100)}%`}</p>
              <button className="btn" disabled={!ready} onClick={enter}>{TOUCH ? 'Tap to explore' : 'Click to explore'}</button>
            </>
          )}
        </div>
      </div>
    </>
  );
}

/** Thumb stick for phones: writes a -1..1 vector into the engine. */
function Joystick({ engine }: { engine: WalkEngine }) {
  const knob = useRef<HTMLSpanElement>(null); const st = useRef<{ id: number; x: number; y: number } | null>(null);
  const move = (x: number, y: number) => { const s = st.current!; let dx = x - s.x, dy = y - s.y; const l = Math.hypot(dx, dy), R = 46; if (l > R) { dx *= R / l; dy *= R / l; }
    engine.joy.x = dx / R; engine.joy.y = dy / R; knob.current!.style.transform = `translate(${dx}px,${dy}px)`; };
  const end = () => { st.current = null; engine.joy.x = engine.joy.y = 0; if (knob.current) knob.current.style.transform = ''; };
  return (
    <Glass className="joy" opts={{ shader: false }} aria-label="Move"
      onPointerDown={(e: React.PointerEvent) => { const r = (e.currentTarget as HTMLElement).getBoundingClientRect(); st.current = { id: e.pointerId, x: r.left + r.width / 2, y: r.top + r.height / 2 }; (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); move(e.clientX, e.clientY); }}
      onPointerMove={(e: React.PointerEvent) => { if (st.current?.id === e.pointerId) move(e.clientX, e.clientY); }} onPointerUp={end} onPointerCancel={end}>
      <span className="knob" ref={knob} />
    </Glass>
  );
}

/** North-up map of the occupancy grid with the visitor's position and view cone. Hidden when the scene has no grid. */
/** Map extras the design lab draws: a route polyline (floor x, z) and place dots. */
export interface MapExtras { path?: [number, number][] | null; pins?: { x: number; z: number; dest?: boolean }[] }
export function Minimap({ engine, manifest, extras }: { engine: WalkEngine; manifest: SceneManifest | null; extras?: MapExtras }) {
  const ex = useRef(extras); ex.current = extras;
  const cv = useRef<HTMLCanvasElement>(null); const [img, setImg] = useState<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const g = engine.grid; if (!g) return; const { width: w, height: h } = g.meta; const c = document.createElement('canvas'); c.width = w; c.height = h; const x = c.getContext('2d')!; const d = x.createImageData(w, h);
    for (let i = 0; i < w * h; i++) { const b = g.data[i]; d.data.set(b ? [7, 11, 20, 120] : [236, 241, 250, 225], i * 4); /* walkable bright, blocked dark */ } x.putImageData(d, 0, 0); setImg(c);
  }, [engine.grid, manifest]);
  useEffect(() => {
    if (!img) return; let raf = 0; const S = 168;
    const draw = () => { raf = requestAnimationFrame(draw); const c = cv.current; if (!c || !engine.grid) return; const x = c.getContext('2d')!, dpr = Math.min(2, devicePixelRatio);
      if (c.width !== S * dpr) { c.width = c.height = S * dpr; } x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, S, S);
      const { origin, cell, width, height } = engine.grid.meta, sc = Math.min(S / width, S / height) * .92, ox = (S - width * sc) / 2, oy = (S - height * sc) / 2;
      x.imageSmoothingEnabled = false; x.globalAlpha = .9; x.drawImage(img, ox, oy, width * sc, height * sc); x.globalAlpha = 1;
      const p = engine.state.pos, px = ox + (p.x - origin[0]) / cell * sc, py = oy + (p.z - origin[1]) / cell * sc, yaw = engine.state.yaw;
      const toPx = (wx: number, wz: number) => [ox + (wx - origin[0]) / cell * sc, oy + (wz - origin[1]) / cell * sc] as const;
      for (const q of ex.current?.pins ?? []) { const [qx, qy] = toPx(q.x, q.z); x.fillStyle = q.dest ? '#FF6B00' : '#0B1220'; x.strokeStyle = q.dest ? '#fff' : 'rgba(11,18,32,.6)'; x.lineWidth = 1; x.beginPath(); x.arc(qx, qy, q.dest ? 4.5 : 2.6, 0, 7); x.fill(); x.stroke(); }
      const route = ex.current?.path; if (route && route.length > 1) {
        x.strokeStyle = '#FF6B00'; x.lineWidth = 2.6; x.lineCap = x.lineJoin = 'round'; x.setLineDash([5, 4]); x.lineDashOffset = -performance.now() / 60;
        x.beginPath(); route.forEach(([wx, wz], i) => { const [qx, qy] = toPx(wx, wz); if (i) x.lineTo(qx, qy); else x.moveTo(qx, qy); }); x.stroke(); x.setLineDash([]);
      }
      x.fillStyle = 'rgba(255,107,0,.22)'; x.beginPath(); x.moveTo(px, py); x.arc(px, py, 26, -Math.PI / 2 - yaw - .6, -Math.PI / 2 - yaw + .6); x.closePath(); x.fill();
      x.fillStyle = '#FF6B00'; x.beginPath(); x.arc(px, py, 4, 0, 7); x.fill(); x.strokeStyle = '#fff'; x.lineWidth = 1.5; x.stroke(); };
    draw(); return () => cancelAnimationFrame(raf);
  }, [img, engine]);
  if (!img) return null;
  return <Glass className="map" opts={{ radius: 14 }}><canvas ref={cv} style={{ width: 168, height: 168 }} aria-label="Map of the floor with your position" role="img" /><span className="wide map-l">Floor 01</span></Glass>;
}
