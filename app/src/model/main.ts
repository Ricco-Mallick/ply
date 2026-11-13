// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The semantic viewer: a captured room rebuilt as editable Three.js geometry rather than splats.
   Walk it, look at it like a dollhouse, read the plan, wire it up, pick any object and change it — then take the
   scene out as GLB. The splat twin stays the photoreal layer; this is the one you can measure and modify. */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { buildObjects, buildRoom, drawable } from './build';
import { sameOriginModuleUrl } from './module-url';
import { factorFrom, fmtScale, LIMITS, measure as tapDistance, rescaleScene } from './rescale';
import type { Obj, SceneDocument, Shell } from './types';

const BASE = '/ply/';
const params = new URLSearchParams(location.search);
const sceneBase = new URL(params.get('scene') ?? `${BASE}scenes/demo/`, location.href).href;
const lidarUrl = params.get('lidar');   // a coloured point cloud (PLY) to look at on its own

type View = 'walk' | 'doll' | 'plan';
const el = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;
const canvas = el<HTMLCanvasElement>('#stage');
const show = (text: string) => { el('#status').textContent = text; };
const fmt = (n: number) => `${n.toFixed(2)} m`;
const cell = (tag: string, text: string) => { const e = document.createElement(tag); e.textContent = text; return e; };   // object names come from shell.json: text, never HTML

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;

const scene = new THREE.Scene();
scene.background = new THREE.Color('#0b1020');
scene.fog = new THREE.Fog('#0b1020', 30, 90);

const camera = new THREE.PerspectiveCamera(65, innerWidth / innerHeight, 0.05, 300);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495;

// lights: a soft sky, a key with shadows, a cool fill from the opposite side
scene.add(new THREE.HemisphereLight('#f4ecdd', '#101724', 1.25));  // warm daylight, not a blue studio
const key = new THREE.DirectionalLight('#fff2df', 2.4);
key.position.set(9, 14, 7);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -18; key.shadow.camera.right = 18; key.shadow.camera.top = 18; key.shadow.camera.bottom = -18;
key.shadow.bias = -0.0006;
scene.add(key);
const fill = new THREE.DirectionalLight('#8fb6ff', 0.7);
fill.position.set(-10, 8, -6);
scene.add(fill);
scene.add(new THREE.AmbientLight('#6a6f7d', 0.55));

// grid on the ground, a bench plate under the room
const grid = new THREE.GridHelper(60, 60, 0x243352, 0x18233a);
grid.position.y = -0.02;
scene.add(grid);

let room: THREE.Group | null = null;
let objectsRoot: THREE.Group | null = null;
let shell: Shell | null = null;
let view: View = (params.get('view') as View) ?? 'walk';
let wire = false;
let meshLayer: THREE.Mesh | null = null;
let meshLoading = false;
let selected: THREE.Object3D | null = null;
let hidden = new Set<string>();
let deleted = new Set<string>();

// reference measurement (plan section 9, 2r): the scene manifest, the total factor applied since load, and the two taps
let sceneDoc: SceneDocument | null = null;
let hasSceneJson = false;
let appliedScale = 1;
const measure = { on: false, stage: 'idle' as 'idle' | 'first' | 'asking' | 'applied', distance: 0, points: [] as THREE.Vector3[] };

// ---------------------------------------------------------------- load
async function load() {
  show('loading measured shell…');
  const tryFetch = async (name: string) => {
    const res = await fetch(new URL(name, sceneBase));
    return res.ok ? res.json().catch(() => null) : null;
  };
  const shellP: Promise<Shell> = tryFetch('shell.json');
  const docP: Promise<SceneDocument | null> = tryFetch('scene.json').catch(() => null);   // absent for a hand-written scene
  const [s, doc] = await Promise.all([shellP, docP]);
  hasSceneJson = !!doc && typeof doc === 'object' && !Array.isArray(doc);
  sceneDoc = hasSceneJson ? doc : null;
  paintScale();
  const requested = params.get('module');   // an authored room module (see .agents/skills/ply-room-model)
  const moduleUrl = requested ? sameOriginModuleUrl(requested, location.href) : null;
  const refused = requested && !moduleUrl ? 'module refused (not from this site) · ' : '';   // first, so the hint pill never covers it
  if (!s) { show(refused + 'no shell.json next to this scene — run a capture, then open the model view'); return; }
  shell = s;
  if (moduleUrl) {
    // a hand-modelled room: the agent's module gets THREE and the measurements, and returns real geometry
    const mod = await import(/* @vite-ignore */ moduleUrl);
    room = mod.buildRoom(THREE, s) ?? buildRoom(s);
    objectsRoot = mod.buildObjects(THREE, s) ?? buildObjects(s);
    show(`authored module · ${(s.walls ?? []).length} walls · ${objectsRoot ? objectsRoot.children.length : 0} objects`);
  } else {
    room = buildRoom(s);
    objectsRoot = buildObjects(s);
  }
  if (room) scene.add(room);
  if (objectsRoot) scene.add(objectsRoot);
  const walls = s.walls?.length ?? 0;
  const objects = s.objects?.length ?? 0;
  show(`${refused}${walls} walls · ${(s.openings ?? []).length} openings · ${objects} objects`);
  buildOutliner(s);
  const pose = params.get('pose');
  if (pose) {
    const [x, y, z, yw, pt] = pose.split(',').map(Number);
    view = 'walk';
    document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach((b) => b.classList.toggle('on', false));
    walked.set(x, 0, z);
    shell && (shell.floorY = y - EYE);
    yawPitch.yaw = yaw = yw || 0; yawPitch.pitch = pitch = pt || 0;
    document.body.classList.add('posing');
  } else setView(view);
}

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}

// ---------------------------------------------------------------- views
function bounds() {
  const box = new THREE.Box3().setFromObject(room ?? new THREE.Object3D());
  return box.isEmpty() ? new THREE.Box3(new THREE.Vector3(-5, 0, -5), new THREE.Vector3(5, 3, 5)) : box;
}

/** The floor in world units: a reference measurement may have rescaled the model since it was built. */
const floorWorld = () => (shell?.floorY ?? 0) * appliedScale;

function setView(next: View) {
  view = next;
  document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === next));
  grid.visible = next !== 'walk';
  if (!room) return;
  const ghost = next === "doll";
  // the ceiling (and anything else named so) lifts off in the dollhouse view, or you are looking at a lid
  room.traverse((o) => {
    if (/^ceiling/i.test(o.name)) o.visible = !ghost;
  });
  room.traverse((o) => {
    if (!(o instanceof THREE.Mesh)) return;
    const m = o.material as THREE.MeshStandardMaterial;
    if (o.name.startsWith('wall-') || o.name.startsWith('opening-')) {
      m.transparent = ghost; m.opacity = ghost ? 0.28 : 1; m.depthWrite = !ghost; m.needsUpdate = true;
    }
    if (o.name === 'ceiling') o.visible = !ghost;
  });
  if (!room) return;
  const b = bounds();
  const size = b.getSize(new THREE.Vector3());
  const centre = b.getCenter(new THREE.Vector3());
  if (next === 'walk') {
    controls.enabled = false;
    camera.position.set(centre.x, floorWorld() + EYE, centre.z);
    yaw = 0; pitch = 0;
    walked.set(centre.x, 0, centre.z);
  } else if (next === 'doll') {
    controls.enabled = true;
    controls.target.copy(centre).setY(floorWorld() + size.y * 0.35);
    camera.position.copy(centre).add(new THREE.Vector3(size.x * 0.75, size.x * 0.85, size.z * 0.9));
    controls.update();
  } else {
    controls.enabled = true;
    controls.target.copy(centre);
    camera.position.set(centre.x, centre.y + Math.max(size.x, size.z) * 1.25, centre.z + 0.01);
    camera.up.set(0, 0, -1);
    camera.lookAt(controls.target);
    controls.update();
    camera.up.set(0, 1, 0);
  }
}
document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view as View)));

// ---------------------------------------------------------------- walking (with real collision)
const yawPitch = { yaw: 0, pitch: 0 };
let yaw = 0, pitch = 0;
const walked = new THREE.Vector3();
const keys = new Set<string>();
const EYE = 1.62, RADIUS = 0.3;

const inField = (e: KeyboardEvent) => e.target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
const inControl = (e: KeyboardEvent) => e.target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT|BUTTON|A)$/.test(e.target.tagName);
addEventListener('keydown', (e) => {
  if (inField(e)) return;   // the measure field keeps its own Enter and Escape, and typing must not walk the camera
  if (measure.on && e.code.startsWith('Arrow')) { e.preventDefault(); moveCursor(e); return; }   // the keyboard's tap point
  keys.add(e.code);
  if (e.code === 'KeyX') toggleWire();
  if (e.code === 'KeyM') setMeasure(!measure.on);
  if (e.code === 'Space' && measure.on && !inControl(e)) { e.preventDefault(); placePoint(); }   // the crosshair takes the tap
  if (e.code === 'Enter' && measure.stage === 'asking' && !inControl(e)) applyMeasurement();
  if (e.code === 'Escape') {
    if (measure.on) { if (!cancelMeasurement()) setMeasure(false); }
    else select(null);
  }
});
addEventListener('keyup', (e) => keys.delete(e.code));
canvas.addEventListener('click', () => { if (!measure.on && view === 'walk' && !document.pointerLockElement) canvas.requestPointerLock(); });
addEventListener('mousemove', (e) => {
  if (document.pointerLockElement !== canvas) return;
  yawPitch.yaw -= e.movementX * 0.0022;
  yawPitch.pitch = THREE.MathUtils.clamp(yawPitch.pitch - e.movementY * 0.0022, -1.2, 1.2);
});

function blocked(x: number, z: number) {
  const k = appliedScale;   // the shell keeps the units the model was built in; a reference measurement moves the room, not the walker
  const floorY = (shell?.floorY ?? 0) * k;
  for (const w of shell?.walls ?? []) {
    const ax = w.a[0] * k, az = w.a[1] * k, bx = w.b[0] * k, bz = w.b[1] * k;
    const dx = bx - ax, dz = bz - az;
    const t = THREE.MathUtils.clamp(((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz || 1), 0, 1);
    const px = ax + dx * t, pz = az + dz * t;
    if (Math.hypot(x - px, z - pz) < RADIUS + 0.07) return true;
  }
  for (const o of shell?.objects ?? []) {
    if (deleted.has(o.id) || hidden.has(o.id)) continue;
    const cx = o.center[0] * k, cy = o.center[1] * k, cz = o.center[2] * k;
    const hx = o.half[0] * k, hy = o.half[1] * k, hz = o.half[2] * k;
    if (cy - hy > floorY + 1.7 || cy + hy < floorY + 0.2) continue; // above head or below the knee: walk under/over
    if (Math.abs(x - cx) < hx + RADIUS && Math.abs(z - cz) < hz + RADIUS) return true;
  }
  return false;
}

// ---------------------------------------------------------------- picking + editing
const ray = new THREE.Raycaster();
const pointer = new THREE.Vector2();
canvas.addEventListener('pointerdown', (e) => {
  if (measure.on) return;   // measuring: a tap places a point (see the measure section instead)
  if (document.pointerLockElement === canvas) return;
  pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(pointer, camera);
  const hits = ray.intersectObjects([objectsRoot, room].filter(Boolean) as THREE.Object3D[], true);
  const hit = hits.find((h) => h.object.type === 'Mesh');
  select(hit ? hit.object : null);
});

function rootObject(obj: THREE.Object3D | null): THREE.Object3D | null {
  let cur: THREE.Object3D | null = obj;
  while (cur && !String(cur.name).match(/^(wall-\d+|.+-\d+)$/)) cur = cur.parent;
  return cur ?? obj;
}

function select(object: THREE.Object3D | null) {
  selected = object ? rootObject(object) : null;
  document.querySelectorAll('.sel').forEach((n) => n.classList.remove('sel'));
  const panel = el<HTMLDivElement>('#panel');
  if (!selected) { panel.hidden = true; return; }
  const data = selected.userData as { kind?: string; label?: string; type?: string; size?: number[]; id?: string; length?: number };
  panel.hidden = false;
  el('#panel-name').textContent = data.label ?? selected.name;
  el('#panel-kind').textContent = data.kind === 'wall' ? 'wall' : `${data.type ?? 'object'}`;
  paintPanelSize(selected);
  selected.traverse((o) => { if (o instanceof THREE.Mesh) o.material = (o.material as THREE.MeshStandardMaterial).clone(); });
  el('#panel').dataset.target = selected.name;
}

/** The selection panel's size line, in the model's current units (a reference measurement rescales it). */
function paintPanelSize(target: THREE.Object3D) {
  const data = target.userData as { kind?: string; length?: number };
  target.updateWorldMatrix(true, true);   // a rescale just moved the parent, and the render loop may not have run since
  const box = new THREE.Box3().setFromObject(target);
  const s = box.getSize(new THREE.Vector3());
  el('#panel-size').textContent = data.kind === 'wall' && data.length
    ? `length ${fmt(data.length * appliedScale)}`
    : `${fmt(s.x)} × ${fmt(s.y)} × ${fmt(s.z)}`;
}

const sizeText = (o: Obj) => o.half.map((v) => (v * 2 * appliedScale).toFixed(1)).join('×');

el('#hide').addEventListener('click', () => { if (!selected) return; selected.visible = false; hidden.add(selected.name); select(null); refreshOutliner(); });
el('#delete').addEventListener('click', () => { if (!selected) return; deleted.add(selected.name); selected.removeFromParent(); select(null); refreshOutliner(); });
el('#grow').addEventListener('click', () => scaleSelected(1.12));
el('#shrink').addEventListener('click', () => scaleSelected(1 / 1.12));
el('#focus').addEventListener('click', () => { if (!selected) return; const b = new THREE.Box3().setFromObject(selected); const c = b.getCenter(new THREE.Vector3()); controls.target.copy(c); camera.lookAt(c); });
function scaleSelected(k: number) { if (!selected) return; selected.scale.multiplyScalar(k); select(selected); }
el('#export').addEventListener('click', async () => {
  if (!room || !objectsRoot) return;
  const group = new THREE.Group();
  group.add(room.clone(true), objectsRoot.clone(true));
  const glb = await new GLTFExporter().parseAsync(group, { binary: true });
  const url = URL.createObjectURL(new Blob([glb as ArrayBuffer], { type: 'model/gltf-binary' }));
  const a = document.createElement('a');
  a.href = url; a.download = `${(shell as unknown as { name?: string })?.name ?? 'ply-model'}.glb`; a.click();
  URL.revokeObjectURL(url);
});

async function toggleMesh() {
  const btn = document.querySelector<HTMLButtonElement>('[data-mesh]');
  if (meshLayer) { meshLayer.visible = !meshLayer.visible; btn?.classList.toggle('on', meshLayer.visible); return; }
  if (meshLoading) return;
  meshLoading = true;
  show('loading dense mesh…');
  const url = new URL('mesh.ply', sceneBase).href;
  try {
    const geo = await new PLYLoader().loadAsync(url);
    geo.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0.02, side: THREE.DoubleSide });
    meshLayer = new THREE.Mesh(geo, material);
    meshLayer.name = 'dense-mesh';
    meshLayer.position.y = 0;
    meshLayer.scale.setScalar(appliedScale);   // a correction may already be applied to the room
    meshLayer.castShadow = meshLayer.receiveShadow = true;
    scene.add(meshLayer);
    // show the surface on its own: the proxies would z-fight with it
    if (room) room.visible = false;
    if (objectsRoot) objectsRoot.visible = false;
    btn?.classList.add('on');
    show(`dense mesh · ${(geo.attributes.position.count / 1e6).toFixed(2)}M vertices`);
  } catch (e) {
    show('no dense mesh for this job yet (the stage runs after export)');
  } finally { meshLoading = false; }
}
document.querySelector<HTMLButtonElement>('[data-mesh]')?.addEventListener('click', toggleMesh);

// ---------------------------------------------------------------- reference measurement (plan section 9, 2r)
/* Two taps give a length in the model's units; the true length typed in gives F = true / measured. Everything metric
   moves by F about the scene origin (rescale.py's convention), and the local manifest takes that module's fields:
   `scale`, `scaleSource: "reference"` and a `reference` block with both lengths. Nothing about the twin's files is
   written from here, and nothing here pretends otherwise. A rescale is one atomic move of scene.spz, shell.json,
   occupancy.json, places.json and scene.json, with the transform, bounds, camera path and topview rewritten together;
   scaling scene.json alone would leave the splat at the old scale. The API has no route for it either (`cloud/src/
   index.ts`: PUT /v1/files is limited to captures/, scenes/ is written only by the worker token), so the correction
   lives in this tab and the panel hands over the exact `plypipe rescale` command that makes it permanent. */
const hintEl = el<HTMLDivElement>('#hint');
const HINT_DEFAULT = hintEl.innerHTML;
const HINT_MEASURE = 'Measure: tap two points · <b>Space</b> taps the crosshair (arrow keys move it) · <b>Esc</b> cancels';
const measurePanel = el<HTMLElement>('#measure-panel');
const measureInput = el<HTMLInputElement>('#measure-real');
const measureNote = el<HTMLParagraphElement>('#measure-note');
const measureCommand = el<HTMLElement>('#measure-command');
const measureField = el<HTMLElement>('#measure-field');
const measureActions = el<HTMLElement>('#measure-actions');
const measureDone = el<HTMLElement>('#measure-done');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');   // read live: the marker pulse is the only motion here
const measureRoot = new THREE.Group();
measureRoot.name = 'ply-measure';
scene.add(measureRoot);
let pulse: THREE.Object3D | null = null;
// where a keyboard tap lands: the crosshair, moved with the arrow keys or by the last pointer tap
const cursor = { x: innerWidth / 2, y: innerHeight / 2 };

const note = (text: string) => { measureNote.textContent = text; };
const setHint = (html: string) => { hintEl.innerHTML = html; };
const crosshair = el<HTMLDivElement>('#crosshair');

function paintCursor() {
  crosshair.style.left = `${cursor.x}px`;
  crosshair.style.top = `${cursor.y}px`;
}

function moveCursor(e: KeyboardEvent) {
  const step = e.shiftKey ? 6 : 28;
  if (e.code === 'ArrowLeft') cursor.x -= step;
  if (e.code === 'ArrowRight') cursor.x += step;
  if (e.code === 'ArrowUp') cursor.y -= step;
  if (e.code === 'ArrowDown') cursor.y += step;
  cursor.x = THREE.MathUtils.clamp(cursor.x, 8, innerWidth - 8);
  cursor.y = THREE.MathUtils.clamp(cursor.y, 8, innerHeight - 8);
  paintCursor();
}

/** The scale readout by the status line: what scene.json says, and what this tab has corrected since. */
function paintScale() {
  const out = el<HTMLDivElement>('#scale');
  const now = typeof sceneDoc?.scale === 'number' ? sceneDoc.scale : null;
  if (now === null && appliedScale === 1) { out.hidden = true; return; }
  out.hidden = false;
  out.textContent = now !== null
    ? `scale ${fmtScale(now)} · ${sceneDoc?.scaleSource ?? 'unknown source'}`
    : `scale ×${fmtScale(appliedScale)} · reference`;
}

function setMeasure(on: boolean) {
  measure.on = on;
  const btn = el<HTMLButtonElement>('#measure');
  btn.classList.toggle('on', on);
  btn.setAttribute('aria-pressed', String(on));
  document.body.classList.toggle('measuring', on);
  crosshair.hidden = !on;
  hintEl.classList.toggle('measure', on);
  if (on) { cursor.x = innerWidth / 2; cursor.y = innerHeight / 2; paintCursor(); }
  clearMeasurement();
  setHint(on ? HINT_MEASURE : HINT_DEFAULT);
}

function cancelMeasurement(): boolean {
  const had = measure.stage !== 'idle' || measure.points.length > 0;
  clearMeasurement();
  if (measure.on) setHint(HINT_MEASURE);
  return had;
}

function clearMarks() {
  measure.points = [];
  pulse = null;
  measureRoot.traverse((o) => {
    if (o instanceof THREE.Mesh) {
      o.geometry.dispose();
      (o.material as THREE.Material).dispose();
    }
  });
  measureRoot.clear();
}

function clearMeasurement() {
  measure.stage = 'idle';
  measure.distance = 0;
  clearMarks();
  resetPanel();
}

function resetPanel() {
  measurePanel.hidden = true;
  measure.stage = 'idle';
  el('#measure-title').textContent = 'Real distance';
  el('#measure-kind').textContent = 'reference measurement';
  el('#measure-distance').textContent = '—';
  measureField.hidden = false;
  measureActions.hidden = false;
  measureDone.hidden = true;
  measureCommand.hidden = true;
  measureCommand.textContent = '';
  measureInput.value = '';
  note('');
}

/** The first visible mesh under a screen point: a pointer tap, or the crosshair a keyboard tap uses. */
function pick(clientX: number, clientY: number): THREE.Vector3 | null {
  pointer.set((clientX / innerWidth) * 2 - 1, -(clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(pointer, camera);
  const targets: THREE.Object3D[] = [];
  for (const o of [room, objectsRoot, meshLayer]) if (o) targets.push(o);
  const hit = ray.intersectObjects(targets, true).find((h) => h.object instanceof THREE.Mesh && shown(h.object));
  return hit ? hit.point.clone() : null;
}

/** Visible up the chain: the ceiling lifts off in the dollhouse view and must not take a tap. */
function shown(o: THREE.Object3D) {
  for (let n: THREE.Object3D | null = o; n; n = n.parent) if (!n.visible) return false;
  return true;
}

function addMarker(p: THREE.Vector3) {
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(Math.max(0.02, camera.position.distanceTo(p) * 0.012), 16, 12),
    new THREE.MeshBasicMaterial({ color: 0x6ea8ff, depthTest: false, depthWrite: false, transparent: true, opacity: 0.95, fog: false }),
  );
  marker.position.copy(p);
  marker.renderOrder = 31;
  measureRoot.add(marker);
  return marker;
}

function addLine(a: THREE.Vector3, b: THREE.Vector3) {
  // a thin bar, not a 1 px Line: at room distance the GL line is barely visible, and this reads as a tape
  const dir = new THREE.Vector3().subVectors(b, a);
  const radius = Math.max(0.012, camera.position.distanceTo(a) * 0.0035);
  const bar = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, dir.length(), 6),
    new THREE.MeshBasicMaterial({ color: 0x9ec8ff, depthTest: false, depthWrite: false, transparent: true, opacity: 0.95, fog: false }),
  );
  bar.position.copy(a).lerp(b, 0.5);
  bar.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
  bar.renderOrder = 30;
  measureRoot.add(bar);
}

function placePoint(clientX?: number, clientY?: number) {
  if (!measure.on) return;
  if (clientX !== undefined && clientY !== undefined) { cursor.x = clientX; cursor.y = clientY; paintCursor(); }
  const p = pick(cursor.x, cursor.y);
  if (!p) { setHint('nothing under that tap — aim at a wall, the floor or an object'); return; }
  if (measure.stage === 'asking' || measure.stage === 'applied') clearMeasurement();   // a new pair replaces the last one
  measure.points.push(p);
  const marker = addMarker(p);
  if (measure.points.length === 1) {
    measure.stage = 'first';
    pulse = marker;
    setHint('first point · tap the other end of the same thing');
    return;
  }
  const [a, b] = measure.points;
  measure.distance = tapDistance(a, b);
  if (measure.distance < 0.02) {
    clearMeasurement();
    setHint('those two taps are the same point — tap the ends of something longer');
    return;
  }
  if (pulse) pulse.scale.setScalar(1);   // the first point stops breathing once the pair is complete
  pulse = null;
  addLine(a, b);
  openPanel(measure.distance);
}

function openPanel(d: number) {
  measure.stage = 'asking';
  measurePanel.hidden = false;
  el('#measure-title').textContent = 'Real distance';
  el('#measure-kind').textContent = 'reference measurement';
  el('#measure-distance').textContent = `two taps · ${d.toFixed(2)} m apart`;
  measureField.hidden = false;
  measureActions.hidden = false;
  measureDone.hidden = true;
  measureCommand.hidden = true;
  measureInput.value = d.toFixed(2);   // the default is the measured distance rounded to 2 decimals
  note(typeof sceneDoc?.scale === 'number' ? `current scale ${fmtScale(sceneDoc.scale)} · ${sceneDoc.scaleSource ?? 'unknown source'}` : 'this scene carries no scale yet');
  measureInput.focus({ preventScroll: true });
  measureInput.select();
  setHint('type the real length · <b>Enter</b> applies · <b>Esc</b> cancels');
}

function applyMeasurement() {
  if (measure.stage !== 'asking') return;
  const d = measure.distance;
  const real = Number(measureInput.value);
  const f = factorFrom(real, d);
  if (f === null) { note('type the real length in metres'); measureInput.focus(); return; }
  if (f < LIMITS[0] || f > LIMITS[1]) { note(`×${f.toFixed(2)} is outside ${LIMITS[0]} to ${LIMITS[1]}: check the length`); measureInput.focus(); return; }
  const before = typeof sceneDoc?.scale === 'number' ? sceneDoc.scale : null;
  sceneDoc = rescaleScene(sceneDoc ?? {}, f, { knownM: real, measuredM: d });
  clearMarks();   // the taps were in the old units: the line must not stay behind at the old size
  applyScale(f);
  paintScale();
  measure.stage = 'applied';
  const after = typeof sceneDoc.scale === 'number' ? sceneDoc.scale : null;
  el('#measure-title').textContent = 'Scale corrected';
  el('#measure-kind').textContent = before === null || after === null
    ? `scale ×${fmtScale(appliedScale)}`
    : `scale ${fmtScale(before)} → ${fmtScale(after)}`;
  el('#measure-distance').textContent = `${d.toFixed(2)} m measured · ${real} m entered · ×${fmtScale(f)}`;
  measureField.hidden = true;
  measureActions.hidden = true;
  measureDone.hidden = false;
  note(hasSceneJson ? 'kept in this tab’s manifest — scene.json was not written' : 'kept in this tab — there is no scene.json next to this scene');
  measureCommand.textContent = `plypipe rescale <folder> --known ${real} --measured ${d.toFixed(4)}`;
  measureCommand.hidden = false;
  setHint('corrected for this visit only · the twin’s files need the command in the panel');
}

/** Move the model by f. The camera and the walker move too, so the view holds and the fixed 1 m grid shows the change. */
function applyScale(f: number) {
  appliedScale *= f;
  for (const group of [room, objectsRoot, meshLayer]) if (group) group.scale.setScalar(appliedScale);
  camera.position.multiplyScalar(f);
  controls.target.multiplyScalar(f);
  controls.update();
  walked.x *= f;
  walked.z *= f;
  if (selected) paintPanelSize(selected);
  refreshOutlinerSizes();
}

function refreshOutlinerSizes() {
  if (!shell) return;
  const by = new Map<string, Obj>(drawable(shell).map((o) => [o.id, o] as [string, Obj]));
  document.querySelectorAll<HTMLButtonElement>('#outliner-list .item').forEach((b) => {
    const o = by.get(b.dataset.id ?? '');
    const em = o && b.querySelector('em');
    if (em) em.textContent = sizeText(o);
  });
}

el('#measure').addEventListener('click', () => setMeasure(!measure.on));
el('#measure-apply').addEventListener('click', applyMeasurement);
el('#measure-cancel').addEventListener('click', () => { cancelMeasurement(); });
el('#measure-again').addEventListener('click', () => { cancelMeasurement(); });
el('#measure-exit').addEventListener('click', () => setMeasure(false));
measureInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); applyMeasurement(); }
  if (e.key === 'Escape') { e.preventDefault(); cancelMeasurement(); }
});

// a tap is a pointerdown and a pointerup in the same place: an orbit drag and a hold are not taps
let tap: { id: number; x: number; y: number; t: number } | null = null;
canvas.addEventListener('pointerdown', (e) => {
  if (!measure.on || document.pointerLockElement === canvas) return;
  tap = { id: e.pointerId, x: e.clientX, y: e.clientY, t: performance.now() };
});
canvas.addEventListener('pointerup', (e) => {
  if (!tap || tap.id !== e.pointerId) return;
  const moved = Math.hypot(e.clientX - tap.x, e.clientY - tap.y);
  const held = performance.now() - tap.t;
  tap = null;
  if (!measure.on || moved > 8 || held > 800) return;
  placePoint(e.clientX, e.clientY);
});
canvas.addEventListener('pointercancel', () => { tap = null; });

// ---------------------------------------------------------------- outliner: every object on its own
function buildOutliner(s: Shell) {
  const list = el<HTMLDivElement>('#outliner-list');
  list.innerHTML = '';
  const groups = new Map<string, Obj[]>();
  for (const o of drawable(s)) {
    const k = (o.kind ?? 'object').toLowerCase();
    (groups.get(k) ?? groups.set(k, []).get(k)!).push(o);
  }
  for (const [kind, items] of [...groups.entries()].sort((a, b) => b[1].length - a[1].length)) {
    const row = document.createElement('details');
    row.open = items.length <= 6;
    const head = document.createElement('summary'); head.append(cell('span', kind), cell('em', String(items.length))); row.append(head);
    for (const o of items) {
      const item = document.createElement('button');
      item.className = 'item';
      item.dataset.id = o.id;
      item.append(cell('span', o.name), cell('em', sizeText(o)));
      item.addEventListener('click', () => {
        const target = objectsRoot?.getObjectByName(o.id);
        if (target) { select(target); const b = new THREE.Box3().setFromObject(target); const c = b.getCenter(new THREE.Vector3()); controls.target.copy(c); }
      });
      row.appendChild(item);
    }
    list.appendChild(row);
  }
  el('#outliner-count').textContent = `${drawable(s).length}`;
}

function refreshOutliner() {
  document.querySelectorAll<HTMLButtonElement>('#outliner-list .item').forEach((b) => {
    const id = b.dataset.id!;
    b.classList.toggle('gone', hidden.has(id) || deleted.has(id));
  });
}

function toggleWire() {
  wire = !wire;
  scene.traverse((o) => { if (o instanceof THREE.Mesh) (o.material as THREE.MeshStandardMaterial).wireframe = wire; });
}

// ---------------------------------------------------------------- loop
function loop() {
  const dt = 1 / 60;
  if (view === 'walk') {
    const speed = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 4.2 : 2.0;
    const dir = new THREE.Vector3();
    if (keys.has('KeyW') || keys.has('ArrowUp')) dir.z -= 1;
    if (keys.has('KeyS') || keys.has('ArrowDown')) dir.z += 1;
    if (keys.has('KeyA') || keys.has('ArrowLeft')) dir.x -= 1;
    if (keys.has('KeyD') || keys.has('ArrowRight')) dir.x += 1;
    if (dir.lengthSq() > 0) {
      dir.normalize().applyAxisAngle(new THREE.Vector3(0, 1, 0), yawPitch.yaw);
      const nx = walked.x + dir.x * speed * dt;
      const nz = walked.z + dir.z * speed * dt;
      if (!blocked(nx, walked.z)) walked.x = nx;
      if (!blocked(walked.x, nz)) walked.z = nz;
    }
    yaw += (yawPitch.yaw - yaw) * 0.4;
    pitch += (yawPitch.pitch - pitch) * 0.4;
    camera.position.set(walked.x, floorWorld() + EYE, walked.z);
    camera.rotation.set(pitch, yaw, 0, 'YXZ');
  } else {
    controls.update();
  }
  // the first point breathes so it reads as pending; with reduced motion it just sits there
  if (pulse && !reducedMotion.matches) pulse.scale.setScalar(1 + 0.1 * Math.sin(performance.now() / 300));
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

canvas.addEventListener('pointerlockchange', () => {
  document.body.classList.toggle('locked', document.pointerLockElement === canvas);
});

async function loadLidar() {
  if (!lidarUrl) return false;
  show('loading point cloud…');
  const geo = await new PLYLoader().loadAsync(lidarUrl);
  geo.computeBoundingBox();
  const mat = new THREE.PointsMaterial({ size: 0.055, vertexColors: true, sizeAttenuation: true });
  const cloud = new THREE.Points(geo, mat);
  cloud.frustumCulled = false;   // the scan sits far from the origin, so a stale bound culled the whole cloud
  controls.minDistance = 0.2;
  controls.maxDistance = 400;
  setView('doll');   // orbit, not walk: the walk controller parks the camera at the origin and hides the cloud
  cloud.name = 'lidar-cloud';
  scene.add(cloud);
  const b = geo.boundingBox!;
  const c = b.getCenter(new THREE.Vector3()), sz = b.getSize(new THREE.Vector3());
  camera.position.copy(c).add(new THREE.Vector3(sz.x * 0.7, sz.y * 0.55, sz.z * 0.7));
  controls.target.copy(c);
  controls.enabled = true; controls.update();
  show(`lidar cloud · ${(geo.attributes.position.count / 1e6).toFixed(2)}M points · ${sz.x.toFixed(1)} × ${sz.y.toFixed(1)} × ${sz.z.toFixed(1)} m`);
  return true;
}

// the loop and the resize handler belong to the page, not to one loader: a lidar-only visit used to
// load the cloud and then never draw it
addEventListener('resize', resize);
resize();
loop();
loadLidar().then((done) => { if (!done) load().catch((e) => show(`could not load: ${e}`)); });
