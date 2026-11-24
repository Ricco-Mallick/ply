// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Three more ways to look at a twin: a photo the training never saw against the twin's own render from the same
   camera (CompareView), where the building is on a map (LocationPanel), and everything about the twin in one zip
   (downloadPack). Written for Ply; the ideas come from looking at other tools, the code from nobody else. */
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { Glass } from '../shared/Glass';
import type { WalkEngine } from './engine';
import type { HeldOut } from './scene';
import type { PlacesApi } from './Places';
import { mergeView } from './ViewTools';
import { zip, type ZipFile } from './zip';

/* ---------------- photo vs twin ---------------- */

/** Where the photo sits on screen so the twin's render lines up with it, and the camera fov that makes it so. */
function fit(h: HeldOut, W: number, H: number) {
  const ap = h.width / h.height, aw = W / H, v = 2 * Math.atan(h.height / 2 / h.fy);
  if (aw >= ap) { const w = H * ap; return { left: (W - w) / 2, top: 0, width: w, height: H, fov: v }; }
  const rh = W / ap; return { left: 0, top: (H - rh) / 2, width: W, height: rh, fov: 2 * Math.atan(Math.tan(v / 2) * H / rh) };
}

export function CompareView({ engine, list, onClose }: { engine: WalkEngine; list: HeldOut[]; onClose: () => void }) {
  const [i, setI] = useState(0); const [split, setSplit] = useState(.5); const [size, setSize] = useState({ W: innerWidth, H: innerHeight });
  const [loaded, setLoaded] = useState(false);
  const saved = useRef({ fov: engine.camera.fov });
  useEffect(() => { const r = () => setSize({ W: innerWidth, H: innerHeight }); addEventListener('resize', r); return () => removeEventListener('resize', r); }, []);
  useEffect(() => {
    document.exitPointerLock?.(); engine.manual = true;
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); if (e.key === 'ArrowRight') setI(n => (n + 1) % list.length); if (e.key === 'ArrowLeft') setI(n => (n - 1 + list.length) % list.length); };
    addEventListener('keydown', k);
    return () => { removeEventListener('keydown', k); engine.manual = false; engine.camera.fov = saved.current.fov; engine.camera.updateProjectionMatrix(); };
  }, []);
  const h = list[i], f = useMemo(() => fit(h, size.W, size.H), [h, size]);
  useEffect(() => {
    const c = engine.camera; c.position.set(...h.position); c.quaternion.set(...h.quaternion); c.fov = THREE.MathUtils.radToDeg(f.fov); c.updateProjectionMatrix();
    setLoaded(false);
  }, [h, f]);
  const drag = (e: React.PointerEvent) => { if (e.buttons !== 1 && e.type !== 'pointerdown') return; setSplit(Math.min(1, Math.max(0, (e.clientX - f.left) / f.width))); };
  return (
    <>
      <div className="cmp" style={{ left: f.left, top: f.top, width: f.width, height: f.height }} onPointerDown={e => { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); drag(e); }} onPointerMove={drag}>
        <img src={h.image} alt="" draggable={false} onLoad={() => setLoaded(true)} style={{ clipPath: `inset(0 ${(1 - split) * 100}% 0 0)`, opacity: loaded ? 1 : 0 }} />
        <div className="cmp-line" style={{ left: `${split * 100}%` }}><i /></div>
        <span className="cmp-tag mono l">Real photo</span><span className="cmp-tag mono r">Twin</span>
      </div>
      <Glass className="cmp-bar" opts={{ radius: 16, shader: false }}>
        <div>
          <b>Photo vs twin</b>
          <p>Frame {i + 1} of {list.length}: a photo the training never saw, and the twin rendered from the same camera. Drag across to compare.</p>
        </div>
        <div className="cmp-a">
          <button className="tag-plain" onClick={() => setI(n => (n - 1 + list.length) % list.length)} aria-label="Previous photo">‹</button>
          <button className="tag-plain" onClick={() => setI(n => (n + 1) % list.length)} aria-label="Next photo">›</button>
          <button className="btn" onClick={onClose}>Done</button>
        </div>
      </Glass>
    </>
  );
}

/* ---------------- where is this ---------------- */

const Z = 17, TILE = 256;
function tileOf(lat: number, lon: number, z = Z) {
  const n = 2 ** z, r = lat * Math.PI / 180;
  return { x: (lon + 180) / 360 * n, y: (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * n };
}
/** "12.97, 77.59", or a map link that carries the point (Google Maps @lat,lon, OpenStreetMap mlat/mlon or #map=z/lat/lon). */
export function parseLatLon(s: string): { lat: number; lon: number } | null {
  const t = s.trim();
  const m = t.match(/@(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)/) ?? t.match(/[?&]mlat=(-?\d+(?:\.\d+)?)&mlon=(-?\d+(?:\.\d+)?)/)
    ?? t.match(/#map=\d+\/(-?\d+(?:\.\d+)?)\/(-?\d+(?:\.\d+)?)/) ?? t.match(/^(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)$/)
    ?? t.match(/[?&]q=(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)/);
  if (!m) return null;
  const lat = +m[1], lon = +m[2];
  return Math.abs(lat) <= 90 && Math.abs(lon) <= 180 ? { lat, lon } : null;
}

export function LocationPanel({ api, onClose }: { api: PlacesApi | null; onClose: () => void }) {
  const loc = api?.view?.location ?? null;
  const [edit, setEdit] = useState(!loc); const [text, setText] = useState(''); const [err, setErr] = useState<string | null>(null); const [busy, setBusy] = useState(false);
  const save = (l: { lat: number; lon: number } | null) => { if (!api) return; api.saveView(mergeView(api.view, { location: l ?? undefined })); setEdit(false); };
  const here = () => {
    if (!navigator.geolocation) { setErr('This browser cannot read its position.'); return; }
    setBusy(true); setErr(null);
    navigator.geolocation.getCurrentPosition(p => { setBusy(false); save({ lat: +p.coords.latitude.toFixed(6), lon: +p.coords.longitude.toFixed(6) }); },
      e => { setBusy(false); setErr(e.code === 1 ? 'Location permission was refused.' : 'Could not get a position fix. Paste the coordinates instead.'); }, { enableHighAccuracy: true, timeout: 15000 });
  };
  const W = 360, H = 220;
  return (
    <Glass className="tag-dlg loc-dlg" opts={{ radius: 20, shader: false }} role="dialog" aria-labelledby="loc-title">
      <div className="tag-h"><span className="wide eyebrow">Where is this?</span><button type="button" className="rc-x" onClick={onClose} aria-label="Close">×</button></div>
      {loc && !edit ? <>
        <div className="loc-map" style={{ width: '100%', aspectRatio: `${W} / ${H}` }}>
          {(() => {
            const t = tileOf(loc.lat, loc.lon), cx = Math.floor(t.x), cy = Math.floor(t.y), out = [];
            for (let dy = -1; dy <= 1; dy++) for (let dx = -2; dx <= 2; dx++) {
              const left = ((cx + dx - t.x) * TILE + W / 2) / W * 100, top = ((cy + dy - t.y) * TILE + H / 2) / H * 100;
              out.push(<img key={`${dx},${dy}`} alt="" src={`https://tile.openstreetmap.org/${Z}/${cx + dx}/${cy + dy}.png`} style={{ left: `${left}%`, top: `${top}%`, width: `${TILE / W * 100}%` }} />);
            }
            return out;
          })()}
          <span className="loc-pin" aria-hidden="true" />
        </div>
        <p className="loc-credit">Map © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap contributors</a></p>
        <p className="mono tag-at">{loc.lat.toFixed(5)}, {loc.lon.toFixed(5)}</p>
        <div className="tag-actions">
          <a className="btn" href={`https://www.openstreetmap.org/?mlat=${loc.lat}&mlon=${loc.lon}#map=18/${loc.lat}/${loc.lon}`} target="_blank" rel="noopener noreferrer">Open the map</a>
          <button className="tag-plain" onClick={() => setEdit(true)}>Change</button>
        </div>
      </> : <>
        <h2 id="loc-title">{loc ? 'Move the pin' : 'Pin this twin on a map'}</h2>
        <p className="tag-lead">Standing in the room with your phone? Use your position. Otherwise paste coordinates or a Google Maps or OpenStreetMap link. Saving needs the team code.</p>
        <div className="tag-actions" style={{ marginBottom: 10 }}>
          <button className="btn" disabled={!api || busy} onClick={here}>{busy ? 'Finding you…' : 'Use my position'}</button>
        </div>
        <form onSubmit={e => { e.preventDefault(); const l = parseLatLon(text); if (!l) { setErr('That is not a position: try "12.9716, 77.5946" or a map link.'); return; } setErr(null); save(l); }}>
          <label className="tag-field">Coordinates or map link
            <input value={text} onChange={e => setText(e.target.value)} placeholder="12.9716, 77.5946" autoComplete="off" spellCheck={false} />
          </label>
          {err && <p className="tag-bad" role="alert">{err}</p>}
          <div className="tag-actions" style={{ marginTop: 10 }}>
            <button className="btn" type="submit" disabled={!api || !text.trim()}>Save the pin</button>
            {loc && <button type="button" className="tag-plain" onClick={() => save(null)}>Remove it</button>}
            <button type="button" className="tag-plain" onClick={() => loc ? setEdit(false) : onClose()}>Cancel</button>
          </div>
        </form>
      </>}
    </Glass>
  );
}

/* ---------------- twin pack ---------------- */

const README = (name: string, page: string, files: string[]) => `${name} · a Ply digital twin
${'='.repeat(name.length + 22)}

Opened from: ${page}

Files
${files.map(f => `- ${f}`).join('\n')}

How to read them
- scene.json: the twin's frame (metres, +y up, floor at floorY), bounds, spawn, and the names of the other files.
- *.spz: the Gaussian splat (open it in SuperSplat, or any viewer that reads SPZ).
- occupancy.png and occupancy.json: the walking grid seen from above (white free, black blocked, grey unknown),
  with its origin and cell size in metres.
- places.json: tagged places with positions in the twin's frame, notes, links and photos.
- summary.json and summary.txt: the same facts, written for software and for people (and for an AI assistant).
- view.png: the frame you were looking at when you downloaded this.

Captured and reconstructed with Ply (https://ricco.codes/ply). Map tiles, if any, are © OpenStreetMap contributors.
`;

export async function downloadPack(engine: WalkEngine, base: URL, name: string, onToast: (t: string) => void) {
  const enc = new TextEncoder(), files: ZipFile[] = [];
  const get = async (rel: string, as = rel) => {
    try { const r = await fetch(new URL(rel, base)); if (r.ok) files.push({ name: as, data: new Uint8Array(await r.arrayBuffer()) }); } catch { /* skipped: listed as missing in no file */ }
  };
  onToast('Packing the twin…');
  const raw = await (await fetch(base)).json().catch(() => ({} as any));
  files.push({ name: 'scene.json', data: enc.encode(JSON.stringify(raw, null, 1)) });
  const want = new Set<string>([raw.splat?.file ?? 'scene.spz', raw.occupancy?.image, raw.occupancy?.meta, typeof raw.shell === 'string' ? raw.shell : null, raw.topview?.image,
    ...(Array.isArray(raw.heldOut) ? raw.heldOut.map((h: any) => h.image) : [])].filter((f): f is string => typeof f === 'string' && !f.includes('..') && !/^[a-z]+:/i.test(f)));
  let n = 0;
  for (const f of want) { await get(f); onToast(`Packing the twin… ${++n} of ${want.size}`); }
  // places and the summaries come from the API when the twin is served by it
  const m = base.pathname.match(/^(.*\/v1\/scenes\/[\w-]+)\//);
  if (m) {
    const root = new URL(m[1] + '/', base);
    for (const [rel, as] of [['places', 'places.json'], ['summary', 'summary.json'], ['summary.txt', 'summary.txt']] as const) {
      if (files.some(f => f.name === as)) continue;
      try { const r = await fetch(new URL(rel, root)); if (r.ok) files.push({ name: as, data: new Uint8Array(await r.arrayBuffer()) }); } catch { /* optional */ }
    }
  } else await get('places.json');
  const shot = engine.snapshot(1920, 'image/png');
  files.push({ name: 'view.png', data: Uint8Array.from(atob(shot.split(',')[1]), c => c.charCodeAt(0)) });
  files.unshift({ name: 'README.txt', data: enc.encode(README(name, location.href, files.map(f => f.name))) });
  const blob = new Blob([zip(files) as Uint8Array<ArrayBuffer>], { type: 'application/zip' }), url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = `${name.replace(/[^\w.-]+/g, '_') || 'twin'}-ply.zip`; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  onToast(`Twin pack saved · ${files.length} files · ${(blob.size / 1048576).toFixed(1)} MB`);
}
