// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Places, search, tags and routes in the walkthrough (the design lab's picks: glass-tag pins, the command palette, the
   floor ribbon route). Places come from places.json next to the scene: hand-pinned for the demo, written by the
   labelling stage for pipeline twins. No places.json: an empty list, so the first tag can be placed.
   A tag is picked on the splat (overlay.pick, then snapped to a walkable cell), named in TagPlace, and PUT to the API
   as the whole list. If that save needs a team code the dialog asks for it; if the network is out the change stays on
   this device, the note says so, and places.json can be downloaded from it. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Glass } from '../shared/Glass';
import type { WalkEngine } from './engine';
import type { MapExtras } from './Walk';
import { Navigator, pathLength, steps, type Place } from '../lab/places';
import { isPhoto, parsePlacesPayload, placesFile, safeLink, validatePlaces, type PlaceIssue } from '../lab/place-edit';
import { loadPlaces, placesTarget, rememberCode, savePlaces, savedCode, type SaveResult, type SceneView } from './places-api';
import { parseDeepLink } from './deeplink';
import { RADIUS } from './walker';
import { Overlay } from '../lab/overlay';
import { Palette } from '../lab/search';
import { TagPlace } from './TagPlace';
import { levelAt, routeAcross } from './levels';

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

/* An edit that did not reach the server lives here, keyed by the scene's id, so a reload does not lose it. The next
   successful save clears it; until then the note offers places.json and says the change is local. */
const DRAFT = 'ply-places-draft:';
const draftKeyFor = (base: URL) => DRAFT + (placesTarget(base)?.sceneId ?? base.href);
function readDraft(key: string): Place[] { try { return parsePlacesPayload(JSON.parse(localStorage.getItem(key) ?? 'null')); } catch { return []; } }
function writeDraft(key: string, places: Place[]) { try { localStorage.setItem(key, JSON.stringify(places)); } catch { /* private mode */ } }
function clearDraft(key: string) { try { localStorage.removeItem(key); } catch { /* private mode */ } }

const issueText = (issues: PlaceIssue[]) => issues.slice(0, 2).map(i => i.index < 0 ? i.message : `Place ${i.index + 1} · ${i.field}: ${i.message}`).join(' · ');
const failureText = (res: SaveResult) => res.kind === 'offline'
  ? 'Could not reach the server, so the change is kept on this device. Export places.json so it is not lost.'
  : res.kind === 'forbidden'
    ? 'That code can look but not tag places, so the change is kept on this device. Export places.json, or save with the team code.'
    : res.kind === 'rejected'
      ? `The server refused it${res.index !== undefined ? ` at place ${res.index + 1}${res.field ? ` (${res.field})` : ''}` : ''}: ${res.detail}. Kept on this device.`
      : res.kind === 'failed' ? `Saving failed${res.status ? ` (${res.status})` : ''}: ${res.detail}. Kept on this device.` : 'Kept on this device.';

/** What the walkthrough can ask of the places layer beyond pins: the team's saved view (crop) and saving a new one
    through the same save path (team code prompt, local fallback) the pins use. */
export interface PlacesApi { view: SceneView | undefined; saveView: (v: SceneView | null) => void }

export function PlacesLayer({ engine, base, entered, onMap, onData, searchOpen, onSearch, tagMode, onTagMode, snap, onApi }: {
  engine: WalkEngine; base: URL | null; entered: boolean; onMap: (m: MapExtras) => void;
  onData?: (d: { places: Place[]; nav: Navigator | null }) => void; searchOpen: boolean; onSearch: (v: boolean) => void;
  tagMode: boolean; onTagMode: (v: boolean) => void;
  snap?: () => string | null; onApi?: (api: PlacesApi) => void;
}) {
  const [places, setPlaces] = useState<Place[] | null>(null);
  const [ov, setOv] = useState<Overlay | null>(null); const [nav, setNav] = useState<Navigator | null>(null);
  const open = searchOpen, setOpen = onSearch; const [dest, setDest] = useState<string | null>(null); const [guide, setGuide] = useState(false); const [, tick] = useState(0);
  // tagging: the dialog's subject (null = a new place), the last save's news, and the team-code prompt
  const [draft, setDraft] = useState<{ place: Place | null; at: [number, number, number] } | null>(null);
  const [saving, setSaving] = useState(false); const [tagErr, setTagErr] = useState<string | null>(null);
  const [note, setNote] = useState<{ tone: 'ok' | 'local' | 'bad'; text: string } | null>(null);
  const [auth, setAuth] = useState<{ detail: string; forbidden?: boolean } | null>(null); const [pending, setPending] = useState<Place[] | null>(null);
  const [code, setCode] = useState(savedCode);
  const [localDraft, setLocalDraft] = useState(false);
  const draftKey = useRef(''), served = useRef<Place[]>([]);
  const [view, setView] = useState<SceneView | undefined>(undefined); const pendingView = useRef<SceneView | null | undefined>(undefined);
  const [card, setCard] = useState<string | null>(null);   // the place whose info card is open

  // places come from the API, then from places.json next to the scene; a draft left by a failed save wins until it is
  // saved or discarded, and the note says where the list came from
  useEffect(() => {
    if (!base) return;
    let live = true; draftKey.current = draftKeyFor(base);
    loadPlaces(base).then(read => {
      if (!live) return;
      served.current = read?.places ?? []; setView(read?.view);
      const kept = readDraft(draftKey.current);
      if (kept.length) {
        setPlaces(kept); setLocalDraft(true);
        setNote({ tone: 'local', text: `${kept.length} place${kept.length === 1 ? '' : 's'} here were kept on this device by a save that did not reach the server. Save again, or export places.json.` });
      } else setPlaces(served.current);
    }).catch(() => { if (live) setPlaces(readDraft(draftKey.current)); });
    return () => { live = false; };
  }, [base]);

  // one overlay, once the engine has a scene and the places have arrived: it draws the pins, the ribbon and the pick
  const made = useRef(false);
  useEffect(() => {
    if (made.current || !engine.state.loaded || places === null) return;
    const n = engine.grid ? new Navigator(engine.grid, RADIUS) : null, o = new Overlay(engine, n); o.allPlaces = places;
    o.onPin = id => { onTagMode(false); setCard(id); };
    made.current = true; setNav(n); setOv(o);
    return () => { o.dispose(); made.current = false; };
  }, [engine.state.loaded, places === null]);
  useEffect(() => { if (ov) ov.allPlaces = places ?? []; }, [ov, places]);
  // the saved crop applies once the splat is in; the walkthrough's crop panel saves through push() like a pin does
  useEffect(() => { if (engine.state.loaded) engine.setCrop(view?.crop ?? null); }, [engine.state.loaded, view]);
  const pushRef = useRef<(v: SceneView | null) => void>(() => {});
  pushRef.current = v => { void push(places ?? [], savedCode(), v); };
  useEffect(() => { onApi?.({ view, saveView: v => pushRef.current(v) }); }, [view, onApi]);
  useEffect(() => { if (places) onData?.({ places, nav }); }, [places, nav, onData]);
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
  // while the dialog is open its spot gets a pin too, so the tag can be seen where it will land
  const preview = useMemo<Place[]>(() => draft ? [{ id: draft.place?.id ?? '__draft', name: draft.place?.name ?? 'New place', cat: draft.place?.cat ?? 'Things', p: draft.at }] : [], [draft]);
  const pins = useMemo(() => preview.length ? [...near, ...preview] : near, [near, preview]);
  useEffect(() => { ov?.setPlaces(pins, 'tag', entered); }, [ov, pins, entered]);
  useEffect(() => { const i = setInterval(() => { tick(n => n + 1); setNearTick(n => n + 1); }, 400); return () => clearInterval(i); }, []);

  const me = engine.state.pos;
  const destPlace = places?.find(p => p.id === dest) ?? null;
  // several floors: one navigator per floor; a place belongs to the floor under it
  const navs = useMemo(() => engine.levels.length > 1 ? engine.levels.map(L => L.grid ? new Navigator(L.grid, RADIUS) : null) : null, [engine.state.loaded, engine.levels.length]);
  const full = useMemo(() => {
    if (!destPlace) return null;
    if (navs) {
      const to = levelAt(engine.levels, destPlace.p[1]);
      return routeAcross(l => navs[l] ? (a, b) => navs[l]!.route(a, b) : null, engine.stairs, { level: engine.level, at: [me.x, me.z] }, { level: to, at: [destPlace.p[0], destPlace.p[2]] });
    }
    if (!nav) return null;
    return nav.route([me.x, me.z], [destPlace.p[0], destPlace.p[2]])?.map(([x, z]) => ({ x, z, level: 0 })) ?? null;
    // recompute when the destination changes, a place is edited, the visitor changes floor or has moved far enough to matter
  }, [nav, navs, destPlace, engine.level, Math.round(me.x * 2), Math.round(me.z * 2)]);
  // what is drawn: the part of the route on the floor in view; what "Take me there" walks: all of it
  const path = useMemo<[number, number][] | null>(() => {
    if (!full) return null;
    const lv = navs ? engine.level : 0, here: [number, number][] = [];
    for (const q of full) { if (q.level === lv) here.push([q.x, q.z]); else if (here.length) break; }
    return here.length > 1 ? here : full.length > 1 ? null : null;
  }, [full]);
  const walkPath = useMemo<[number, number][] | null>(() => full && full.length > 1 ? full.map(q => [q.x, q.z] as [number, number]) : null, [full]);
  const otherFloor = !!(navs && destPlace && levelAt(engine.levels, destPlace.p[1]) !== engine.level);
  useEffect(() => { ov?.setRoute(path, 'ribbon', dest); }, [ov, path, dest]);
  useEffect(() => { onMap({ path, pins: shown.map(p => ({ x: p.p[0], z: p.p[2], dest: p.id === dest })) }); }, [path, shown, dest, onMap]);
  useEffect(() => { if (dest && places && !places.some(p => p.id === dest)) { setDest(null); setGuide(false); } }, [places, dest]);

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
    if (!guide || !walkPath) return; const fixed = walkPath.slice(), L = pathLength(fixed); let d = 0;
    const at = (s: number): [number, number] => { let acc = 0; for (let i = 1; i < fixed.length; i++) { const seg = Math.hypot(fixed[i][0] - fixed[i - 1][0], fixed[i][1] - fixed[i - 1][1]); if (acc + seg >= s) { const k = (s - acc) / seg; return [fixed[i - 1][0] + (fixed[i][0] - fixed[i - 1][0]) * k, fixed[i - 1][1] + (fixed[i][1] - fixed[i - 1][1]) * k]; } acc += seg; } return fixed[fixed.length - 1]; };
    engine.state.pos.y = engine.floorY;  // follow at eye height in either mode; do not switch the visitor into walk
    const hook = (dt: number) => {
      d = Math.min(L, d + dt * 1.25); const [x, z] = at(d), [ax, az] = at(Math.min(L, d + .6)); engine.state.pos.x = x; engine.state.pos.z = z;
      if (Math.hypot(ax - x, az - z) > .05) { const want = Math.atan2(-(ax - x), -(az - z)); let dy = want - engine.state.yaw; while (dy > Math.PI) dy -= 2 * Math.PI; while (dy < -Math.PI) dy += 2 * Math.PI; engine.state.yaw += dy * Math.min(1, dt * 3); }
      engine.pitch += (-.08 - engine.pitch) * Math.min(1, dt * 2);
      if (d >= L) { // arrived: turn to face the place itself
        const pl = places?.find(x => x.id === dest); setGuide(false);
        if (pl) { const ex = engine.state.pos, y0 = engine.state.yaw, p0 = engine.pitch; let want = Math.atan2(-(pl.p[0] - ex.x), -(pl.p[2] - ex.z));
          while (want - y0 > Math.PI) want -= 2 * Math.PI; while (want - y0 < -Math.PI) want += 2 * Math.PI;
          const look = Math.atan2(pl.p[1] - (engine.floorY + engine.eye), Math.max(.3, Math.hypot(pl.p[0] - ex.x, pl.p[2] - ex.z))); let t = 0;
          const turn = (dt2: number) => { t = Math.min(1, t + dt2 / .8); const k = 1 - (1 - t) ** 3; engine.state.yaw = y0 + (want - y0) * k; engine.pitch = p0 + (look - p0) * k; if (t >= 1) engine.frameHooks.delete(turn); };
          engine.frameHooks.add(turn); }
      }
    };
    engine.frameHooks.add(hook); return () => { engine.frameHooks.delete(hook); };
  }, [guide, dest]);

  // a link from the model page: preselect the place it names and let the route start from the spawn
  const wanted = useMemo(() => parseDeepLink(typeof location === 'undefined' ? '' : location.search).place, []);
  const preselected = useRef(false);
  useEffect(() => {
    if (!wanted || preselected.current || places === null) return;
    preselected.current = true;
    if (places.some(p => p.id === wanted)) { setDest(wanted); setGuide(false); }
    else setNote({ tone: 'bad', text: `This link asks for a place called “${wanted}”, which is not in this scene. It may have been renamed or deleted.` });
  }, [wanted, places]);

  useEffect(() => { if (!card) return; const k = (e: KeyboardEvent) => { if (e.key === 'Escape') setCard(null); }; addEventListener('keydown', k); return () => removeEventListener('keydown', k); }, [card]);
  // Ctrl/Cmd+K opens search, Esc closes it (the pointer lock is released so the palette can be used)
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); document.exitPointerLock?.(); setOpen(!searchOpen); } if (e.key === 'Escape') setOpen(false); };
    addEventListener('keydown', k); return () => removeEventListener('keydown', k);
  }, [searchOpen]);

  /* ---- tagging: pick a point, name it, save the whole list ---- */
  const list = places ?? [];

  // in tag mode a tap on the splat picks the spot: the surface point (so a tag on a desk sits on the desk), snapped to
  // the nearest cell the walker can stand in, clamped to the room's height
  useEffect(() => {
    if (!tagMode || !ov) return;
    document.exitPointerLock?.();
    const r3 = (v: number) => Math.round(v * 1000) / 1000;
    let down: { id: number; x: number; y: number; t: number } | null = null;
    const onDown = (e: PointerEvent) => { if (e.target instanceof HTMLCanvasElement) down = { id: e.pointerId, x: e.clientX, y: e.clientY, t: performance.now() }; };
    const onUp = (e: PointerEvent) => {
      if (!down || down.id !== e.pointerId) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y), held = performance.now() - down.t; down = null;
      if (moved > 8 || held > 900) return;   // a look-around drag is not a tap
      const surface = ov.pick(e.clientX, e.clientY, true), floor = ov.pick(e.clientX, e.clientY, false);
      if (!surface && !floor) return;        // pointing above the floor, at nothing
      const hit = surface ?? floor!; let x = hit.x, z = hit.z;
      const snapped = nav?.nearestFree(x, z, nav.need(0, Math.SQRT1_2, 1));
      if (snapped && nav) [x, z] = nav.pos(...snapped);
      const y = Math.min(engine.floorY + 2.3, Math.max(engine.floorY, surface ? hit.y : engine.floorY));
      setTagErr(null); setDraft({ place: null, at: [r3(x), r3(y), r3(z)] }); onTagMode(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onTagMode(false); };
    addEventListener('pointerdown', onDown); addEventListener('pointerup', onUp); addEventListener('keydown', onKey);
    return () => { removeEventListener('pointerdown', onDown); removeEventListener('pointerup', onUp); removeEventListener('keydown', onKey); };
  }, [tagMode, ov, nav]);

  const keepLocal = (next: Place[], text: string) => {
    if (draftKey.current) writeDraft(draftKey.current, next);
    setLocalDraft(true); setNote({ tone: 'local', text });
  };
  const push = async (next: Place[], codeUsed: string, v?: SceneView | null) => {
    const target = placesTarget(base);
    if (!target) {
      if (v !== undefined) { setNote({ tone: 'local', text: 'This scene is served as a file, so the change holds for this visit only.' }); return; }
      keepLocal(next, 'This scene is served as a file, so the tag stays on this device. Export places.json to move it onto the server.'); return;
    }
    setSaving(true); const res = await savePlaces(target, next, codeUsed, fetch, v); setSaving(false);
    if (res.kind === 'saved') {
      if (draftKey.current) clearDraft(draftKey.current);
      setLocalDraft(false); setPending(null); setAuth(null); setGuide(false); setTagErr(null); pendingView.current = undefined;
      if (v !== undefined) { setView(v ?? undefined); setNote({ tone: 'ok', text: 'Saved: everyone who opens this twin sees it.' }); return; }
      setNote({ tone: 'ok', text: `Saved · ${res.count} place${res.count === 1 ? '' : 's'} on the server${res.updatedAt ? ` · ${new Date(res.updatedAt).toLocaleTimeString()}` : ''}.` });
      return;
    }
    if (v !== undefined && res.kind !== 'auth') { setNote({ tone: 'bad', text: `Not saved: ${res.kind === 'forbidden' ? 'that code can look but not edit' : res.detail}.` }); return; }
    if (res.kind === 'auth') { setPending(next); pendingView.current = v; setAuth({ detail: res.detail }); setNote({ tone: 'bad', text: 'The server wants a team code before it will save.' }); return; }
    if (res.kind === 'forbidden') { keepLocal(next, failureText(res)); setPending(next); setAuth({ detail: res.detail, forbidden: true }); return; }
    keepLocal(next, failureText(res));
  };
  /** Check the whole list, show the change at once, then push it. False means nothing was saved and nothing changed,
      so the dialog that asked stays open with the field's error on it. */
  const commit = (next: Place[]): boolean => {
    const check = validatePlaces(next);
    if (!check.ok) { const text = issueText(check.issues); setTagErr(text); setNote({ tone: 'bad', text: `Not saved: ${text}` }); return false; }
    setPlaces(next);   // the palette and the plan see it at once, before the server answers
    void push(next, savedCode());
    return true;
  };
  const savePlace = (p: Place) => {
    const next = list.some(x => x.id === p.id) ? list.map(x => x.id === p.id ? p : x) : [...list, p];
    if (commit(next)) setDraft(null);
  };
  const deletePlace = (id: string) => {
    if (!commit(list.filter(x => x.id !== id))) return;
    setDraft(null); if (dest === id) { setDest(null); setGuide(false); }
  };
  const retrySave = () => {
    if (!pending) return;
    rememberCode(code); setAuth(null); void push(pending, code, pendingView.current);
  };
  const dismissAuth = () => { const next = pending; setAuth(null); setPending(null); if (next) keepLocal(next, 'Not saved: a team code is needed. The change is kept on this device; export places.json to keep it for good.'); };
  // Esc closes the tag dialog (its input usually has focus, but the scene can have it) or the code prompt
  useEffect(() => {
    if (!draft && !auth) return;
    const k = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (draft) { setDraft(null); setTagErr(null); return; }
      dismissAuth();
    };
    addEventListener('keydown', k); return () => removeEventListener('keydown', k);
  }, [draft, auth, pending]);
  const openEditor = (p: Place) => { setOpen(false); onTagMode(false); setTagErr(null); setDraft({ place: p, at: p.p }); };
  const discardLocal = () => { if (draftKey.current) clearDraft(draftKey.current); setPlaces(served.current); setLocalDraft(false); setNote({ tone: 'ok', text: 'Local copy dropped: this is the list the server has.' }); };
  const exportPlaces = () => {
    const url = URL.createObjectURL(new Blob([placesFile(list)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = 'places.json'; document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  if (places === null || !entered) return null;
  const dist = (p: Place) => Math.hypot(p.p[0] - me.x, p.p[2] - me.z);
  const cardPlace = card ? list.find(p => p.id === card) ?? null : null;
  const L = walkPath ? pathLength(walkPath) : 0, turns = walkPath && destPlace ? steps(walkPath, destPlace.name) : [];
  return (
    <>
      {tagMode && <div className="tag-hint mono" role="status">{ov ? <>Click a spot on the floor or an object to put a place there · <b>Esc</b> cancels</> : 'Loading the room before a tag can be placed…'}</div>}
      {open && <Palette places={list} pinned={shown} dist={dist} onPick={id => { setDest(id); setGuide(false); setOpen(false); }} onClose={() => setOpen(false)}
        onEdit={id => { const p = list.find(x => x.id === id); if (p) openEditor(p); }} onTag={() => { setOpen(false); onTagMode(true); }} />}
      {draft && <TagPlace initial={draft.place} at={draft.at} taken={list.map(p => p.id)} busy={saving} error={tagErr}
        onSave={savePlace} onDelete={deletePlace} onCancel={() => { setDraft(null); setTagErr(null); }} snap={snap} />}
      {cardPlace && !draft && <Glass className="route-card rc-palette info-card" opts={{ radius: 20, shader: false }} role="dialog" aria-labelledby="info-title"
        style={{ position: 'fixed', left: 18, bottom: 18, top: 'auto', right: 'auto' }}>
        <div className="rc-h"><span className="wide eyebrow">{cardPlace.cat} · {dist(cardPlace).toFixed(1)} m away</span><button className="rc-x" onClick={() => setCard(null)} aria-label="Close">×</button></div>
        <h2 id="info-title">{cardPlace.name}</h2>
        {isPhoto(cardPlace.info?.photo) && <img className="info-photo" src={cardPlace.info!.photo} alt={`The view at ${cardPlace.name}`} />}
        {cardPlace.info?.note ? <p className="info-note">{cardPlace.info.note}</p> : !cardPlace.info?.photo && <p className="info-note dim">No notes yet. Edit the place to add a note, a link or a photo.</p>}
        {safeLink(cardPlace.info?.link) && <a className="info-link" href={safeLink(cardPlace.info!.link)!} target="_blank" rel="noopener noreferrer">{new URL(safeLink(cardPlace.info!.link)!).hostname} ↗</a>}
        {cardPlace.ai && <p className="mono rc-m">AI found a {cardPlace.ai.cls} · {Math.round(cardPlace.ai.conf * 100)}%</p>}
        <div className="rc-a">
          <button className="btn" onClick={() => { setDest(cardPlace.id); setGuide(false); setCard(null); }}>Route here</button>
          <button className="rc-edit" onClick={() => { setCard(null); openEditor(cardPlace); }}>Edit place</button>
        </div>
      </Glass>}
      {auth && <Glass className="tag-dlg auth-dlg" opts={{ radius: 20, shader: false }} role="dialog" aria-labelledby="auth-title">
        <form onSubmit={e => { e.preventDefault(); retrySave(); }}>
          <div className="tag-h"><span className="wide eyebrow">Save places</span><button type="button" className="rc-x" onClick={dismissAuth} aria-label="Close">×</button></div>
          <h2 id="auth-title">Team code</h2>
          <p className="tag-lead">{auth.forbidden ? 'That code can look but not tag. The team code can: paste it here and the change is saved, or export places.json and keep it yourself.' : 'The same code the capture app uses. This browser remembers it after one save.'}</p>
          <label className="tag-field">Code
            <input value={code} onChange={e => setCode(e.target.value)} placeholder="team.…" autoComplete="off" spellCheck={false} autoFocus />
          </label>
          <p className="tag-bad" role="alert">{auth.detail}</p>
          <div className="tag-actions">
            <button className="btn" type="submit" disabled={!code.trim() || saving}>{saving ? 'Saving…' : 'Save places'}</button>
            <button type="button" className="tag-plain" onClick={dismissAuth}>Not now</button>
          </div>
        </form>
      </Glass>}
      {note && <div className={`place-note ${note.tone}`} role="status">
        <p>{note.text}</p>
        <div className="pn-a">
          {note.tone === 'local' && <button onClick={exportPlaces}>Export places.json</button>}
          {note.tone === 'local' && localDraft && <button onClick={discardLocal}>Use the server’s list</button>}
          <button className="pn-x" onClick={() => setNote(null)} aria-label="Dismiss">×</button>
        </div>
      </div>}
      {destPlace && !cardPlace && <Glass className="route-card rc-palette" opts={{ radius: 20, shader: false }} style={{ position: 'fixed', left: 18, bottom: 18, top: 'auto', right: 'auto' }}>
        <div className="rc-h"><span className="wide eyebrow">Route</span><button className="rc-x" onClick={() => { setDest(null); setGuide(false); }} aria-label="End route">×</button></div>
        <h2>{destPlace.name}</h2>
        {walkPath ? <p className="mono rc-m"><b>{L.toFixed(1)} m</b> · {Math.max(1, Math.round(L / 1.2))} s walk · {turns.length - 1} turn{turns.length === 2 ? '' : 's'}{otherFloor ? ` · by the stairs to ${engine.levels[levelAt(engine.levels, destPlace.p[1])]?.name}` : ''}</p> : <p className="mono rc-m bad">No walkable route from here</p>}
        {walkPath && <ol className="rc-steps">{turns.map((t, i) => <li key={i}>{t.text}</li>)}</ol>}
        <div className="rc-a">
          {walkPath && <button className="btn" onClick={() => setGuide(g => !g)}>{guide ? 'Pause' : 'Take me there'}</button>}
          <button className="rc-edit" onClick={() => openEditor(destPlace)}>Edit place</button>
        </div>
      </Glass>}
    </>
  );
}
