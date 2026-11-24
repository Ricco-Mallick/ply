// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Viewer tools borrowed from scene editors: share the exact view as a link (or an iframe to embed), save the frame as
   a picture, crop the twin to a box, and click the floor to move there. Each reads and drives the engine only; the
   crop is saved through the places layer (PlacesApi) so it uses the same team-code prompt as a pin. */
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Glass } from '../shared/Glass';
import type { WalkEngine } from './engine';
import { shareHref } from './deeplink';
import type { PlacesApi } from './Places';
import type { SceneView } from './places-api';
import { RADIUS } from './walker';

type Box = { min: [number, number, number]; max: [number, number, number] };

/** One field of the saved view changed, the rest kept; null when nothing is left. */
export function mergeView(v: SceneView | undefined, patch: Partial<SceneView>): SceneView | null {
  const n: SceneView = { ...(v ?? {}), ...patch };
  for (const k of Object.keys(n) as (keyof SceneView)[]) if (n[k] === undefined) delete n[k];
  return n.crop || n.location ? n : null;
}

function useOutside(open: boolean, close: () => void) {
  const wrap = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const f = (e: PointerEvent) => { if (!wrap.current?.contains(e.target as Node)) close(); };
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') close(); };
    addEventListener('pointerdown', f); addEventListener('keydown', k);
    return () => { removeEventListener('pointerdown', f); removeEventListener('keydown', k); };
  }, [open]);
  return wrap;
}

async function copy(text: string) {
  try { await navigator.clipboard.writeText(text); return true; } catch {
    const t = document.createElement('textarea'); t.value = text; t.style.position = 'fixed'; t.style.opacity = '0'; document.body.append(t); t.select();
    const ok = document.execCommand('copy'); t.remove(); return ok;
  }
}

/** Share: the link to this exact spot and direction, the same as an iframe, or the frame as a PNG. */
export function ShareMenu({ engine, name, onToast, onPack }: { engine: WalkEngine; name: string; onToast: (t: string) => void; onPack?: () => void }) {
  const [open, setOpen] = useState(false); const wrap = useOutside(open, () => setOpen(false));
  const link = () => shareHref(location.href, engine.pose(), { ui: null, embed: null });
  const act = async (what: 'link' | 'embed' | 'image' | 'native') => {
    setOpen(false);
    if (what === 'image') {
      const url = engine.snapshot(2400, 'image/png');
      const a = document.createElement('a'); a.href = url; a.download = `${(name || 'twin').replace(/[^\w.-]+/g, '_')}-view.png`; document.body.append(a); a.click(); a.remove();
      onToast('Picture saved'); return;
    }
    if (what === 'native') { try { await navigator.share({ title: `${name} · Ply`, url: link() }); } catch { /* closed */ } return; }
    const href = what === 'link' ? link() : shareHref(location.href, engine.pose(), { ui: null, embed: '1' });
    const text = what === 'link' ? href : `<iframe src="${href.replace(/"/g, '&quot;')}" width="960" height="540" style="border:0;border-radius:12px" allow="fullscreen; xr-spatial-tracking" allowfullscreen loading="lazy" title="${name.replace(/"/g, '&quot;')} · Ply"></iframe>`;
    onToast((await copy(text)) ? (what === 'link' ? 'Link to this view copied' : 'Embed code copied') : 'Could not copy: your browser blocked the clipboard');
  };
  return (
    <div className="q-wrap" ref={wrap}>
      <Glass as="button" className={`tour-btn pill ${open ? 'on' : ''}`} opts={{ shader: false, press: true }} aria-expanded={open} aria-haspopup="true"
        title="Share this view" onClick={() => { document.exitPointerLock?.(); setOpen(o => !o); }}>Share</Glass>
      {open && <Glass className="views-menu share-menu" opts={{ radius: 14, shader: false }} role="menu" aria-label="Share">
        <button role="menuitem" onClick={() => act('link')}>Copy link to this view<small>opens where you stand, facing where you look</small></button>
        {'share' in navigator && matchMedia('(pointer: coarse)').matches && <button role="menuitem" onClick={() => act('native')}>Send…<small>through the phone's share sheet</small></button>}
        <button role="menuitem" onClick={() => act('embed')}>Copy embed code<small>an iframe for a site or a slide</small></button>
        <button role="menuitem" onClick={() => act('image')}>Save a picture<small>this frame as a PNG</small></button>
        {onPack && <button role="menuitem" onClick={() => { setOpen(false); onPack(); }}>Download the twin pack<small>splat, floor map, places and summary in one zip</small></button>}
      </Glass>}
    </div>
  );
}

const AXES = ['x', 'y', 'z'] as const;
const AXIS_NAME = { x: 'Left – right', y: 'Floor – ceiling', z: 'Front – back' };
const r2 = (v: number) => Math.round(v * 100) / 100;

/** Crop: six handles (two per axis) for a box; everything outside it is hidden at once, Save keeps it for everyone. */
export function CropPanel({ engine, api, onClose }: { engine: WalkEngine; api: PlacesApi | null; onClose: () => void }) {
  const ext = useRef<Box | null>(null);
  if (!ext.current) {
    const e = engine.extent();
    // a little past the splat's extent, so "no crop" is reachable with the handles
    ext.current = e ? { min: e.min.map(v => r2(v - .25)) as Box['min'], max: e.max.map(v => r2(v + .25)) as Box['max'] } : { min: [-10, -1, -10], max: [10, 4, 10] };
  }
  const full = ext.current; const saved = useRef(engine.crop);
  const [box, setBox] = useState<Box>(() => engine.crop ?? { min: [...full.min] as Box['min'], max: [...full.max] as Box['max'] });
  useEffect(() => { engine.setCrop(box); }, [box]);
  const cancel = () => { engine.setCrop(saved.current); onClose(); };
  const set = (side: 'min' | 'max', k: number, v: number) => setBox(b => {
    const n = { min: [...b.min] as Box['min'], max: [...b.max] as Box['max'] };
    n[side][k] = side === 'min' ? Math.min(v, n.max[k] - .2) : Math.max(v, n.min[k] + .2); return n;
  });
  const isFull = box.min.every((v, k) => v <= full.min[k] + .01) && box.max.every((v, k) => v >= full.max[k] - .01);
  return (
    <Glass className="tag-dlg crop-dlg" opts={{ radius: 20, shader: false }} role="dialog" aria-labelledby="crop-title">
      <div className="tag-h"><span className="wide eyebrow">Crop the twin</span><button type="button" className="rc-x" onClick={cancel} aria-label="Close">×</button></div>
      <h2 id="crop-title">Hide what is outside the room</h2>
      <p className="tag-lead">Drag the handles in until the fuzz outside the walls and the street through the windows disappear. Saved crops apply for everyone who opens this twin.</p>
      {AXES.map((a, k) => {
        const lo = full.min[k], hi = full.max[k], span = hi - lo, pct = (v: number) => ((v - lo) / span) * 100;
        return <div className="crop-row" key={a}>
          <span className="mono">{AXIS_NAME[a]}<b>{(box.max[k] - box.min[k]).toFixed(1)} m</b></span>
          <div className="crop-range" style={{ ['--a' as string]: `${pct(box.min[k])}%`, ['--b' as string]: `${pct(box.max[k])}%` }}>
            <input type="range" min={lo} max={hi} step={.05} value={box.min[k]} aria-label={`${AXIS_NAME[a]}: start`} onChange={e => set('min', k, +e.target.value)} />
            <input type="range" min={lo} max={hi} step={.05} value={box.max[k]} aria-label={`${AXIS_NAME[a]}: end`} onChange={e => set('max', k, +e.target.value)} />
          </div>
        </div>;
      })}
      <div className="tag-actions">
        <button className="btn" disabled={!api} onClick={() => { api?.saveView(mergeView(api.view, { crop: isFull ? undefined : { min: box.min.map(r2) as Box['min'], max: box.max.map(r2) as Box['max'] } })); saved.current = isFull ? null : box; onClose(); }}>{isFull ? 'Save without a crop' : 'Save crop'}</button>
        <button className="tag-plain" onClick={() => setBox({ min: [...full.min] as Box['min'], max: [...full.max] as Box['max'] })}>Reset</button>
        <button className="tag-plain" onClick={cancel}>Cancel</button>
      </div>
    </Glass>
  );
}

/** Click (or tap) the floor to glide there. A drag still looks around; with the mouse captured the click aims at the
    crosshair. A ring on the floor shows where a click would take you; nowhere to stand shows no ring. */
export function ClickToMove({ engine, enabled }: { engine: WalkEngine; enabled: boolean }) {
  const ring = useRef<THREE.Group | null>(null);
  useEffect(() => {
    if (!enabled) return;
    const g = new THREE.Group(), mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: .85, depthTest: false });
    const outer = new THREE.Mesh(new THREE.RingGeometry(.22, .26, 48), mat), inner = new THREE.Mesh(new THREE.CircleGeometry(.05, 24), mat);
    for (const m of [outer, inner]) { m.rotation.x = -Math.PI / 2; m.renderOrder = 999; g.add(m); }
    g.visible = false; engine.scene.add(g); ring.current = g;
    const canvas = engine.canvas; let down: { x: number; y: number; t: number; id: number } | null = null; let hoverAt: { x: number; y: number } | null = null;
    // where a click at (x, y) would take you: the surface hit (or the floor), stepped back towards the camera so a click
    // on a wall or a table lands in front of it, then the nearest spot the walker fits
    const target = (x: number, y: number, surface: boolean) => {
      const hit = engine.pick(x, y, surface); if (!hit) return null;
      const p = hit.point, cam = engine.camera.position, dx = cam.x - p.x, dz = cam.z - p.z, d = Math.hypot(dx, dz) || 1;
      let tx = p.x, tz = p.z;
      if (hit.surface && p.y - engine.floorY > .15) { tx += dx / d * .45; tz += dz / d * .45; }
      if (Math.hypot(tx - cam.x, tz - cam.z) > 40) return null;
      const c = engine.collider;
      if (c && !c.free(tx, tz, RADIUS)) { const q = c.nearestFree(tx, tz, RADIUS, .8); if (!q) return null; tx = q.x; tz = q.z; }
      return { x: tx, z: tz };
    };
    const show = (t: { x: number; z: number } | null) => { if (!t) { g.visible = false; return; } g.visible = true; g.position.set(t.x, engine.floorY + .02, t.z); };
    const centre = () => ({ x: innerWidth / 2, y: innerHeight / 2 });
    const onMove = (e: PointerEvent) => { if (e.pointerType === 'mouse') hoverAt = engine.state.locked ? centre() : { x: e.clientX, y: e.clientY }; };
    const onDown = (e: PointerEvent) => { if (e.target === canvas) down = { x: e.clientX, y: e.clientY, t: performance.now(), id: e.pointerId }; };
    const onUp = (e: PointerEvent) => {
      if (!down || down.id !== e.pointerId) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y), held = performance.now() - down.t; down = null;
      if (moved > 6 || held > 450 || engine.manual) return;                 // a drag to look, or a guided walk
      if (e.pointerType === 'touch' && e.clientX < innerWidth * .45) return;  // the joystick's half
      const at = engine.state.locked ? centre() : { x: e.clientX, y: e.clientY };
      const t = target(at.x, at.y, true);
      if (t && engine.glideTo(t.x, t.z)) { show(t); setTimeout(() => { if (!hoverAt) g.visible = false; }, 700); return; }
      if (!engine.state.locked && e.pointerType === 'mouse') engine.lock();  // clicked the sky: capture the mouse as before
    };
    // the hover ring follows the floor plane only (cheap); the click itself asks the splat
    const hook = () => { if (hoverAt && !engine.manual) show(target(hoverAt.x, hoverAt.y, false)); };
    const leave = () => { hoverAt = null; g.visible = false; };
    canvas.addEventListener('pointermove', onMove); addEventListener('mousemove', onMove as never); canvas.addEventListener('pointerleave', leave);
    addEventListener('pointerdown', onDown); addEventListener('pointerup', onUp); engine.frameHooks.add(hook);
    return () => {
      canvas.removeEventListener('pointermove', onMove); removeEventListener('mousemove', onMove as never); canvas.removeEventListener('pointerleave', leave);
      removeEventListener('pointerdown', onDown); removeEventListener('pointerup', onUp); engine.frameHooks.delete(hook);
      engine.scene.remove(g); outer.geometry.dispose(); inner.geometry.dispose(); mat.dispose(); ring.current = null;
    };
  }, [engine, enabled]);
  return null;
}
