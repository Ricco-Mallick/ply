// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The semantic viewer: a captured room rebuilt as editable Three.js geometry rather than splats.
   Walk it, look at it like a dollhouse, read the plan, wire it up, pick any object and change it — then take the
   scene out as GLB. The splat twin stays the photoreal layer; this is the one you can measure and modify. */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { buildObjects, buildRoom, drawable } from './build';
import type { Obj, Shell } from './types';

const BASE = '/ply/';
const params = new URLSearchParams(location.search);
const sceneBase = params.get('scene') ?? `${BASE}scenes/demo/`;

type View = 'walk' | 'doll' | 'plan';
const el = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;
const canvas = el<HTMLCanvasElement>('#stage');
const show = (text: string) => { el('#status').textContent = text; };
const fmt = (n: number) => `${n.toFixed(2)} m`;

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
scene.add(new THREE.HemisphereLight('#cfe0ff', '#0a0f1a', 1.1));
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
scene.add(new THREE.AmbientLight('#4b5875', 0.5));

// grid on the ground, a bench plate under the room
const grid = new THREE.GridHelper(60, 60, 0x243352, 0x18233a);
grid.position.y = -0.02;
scene.add(grid);

let room: THREE.Group | null = null;
let objectsRoot: THREE.Group | null = null;
let shell: Shell | null = null;
let view: View = (params.get('view') as View) ?? 'walk';
let wire = false;
let selected: THREE.Object3D | null = null;
let hidden = new Set<string>();
let deleted = new Set<string>();


// ---------------------------------------------------------------- load
async function load() {
  show('loading measured shell…');
  const tryFetch = async (name: string) => {
    const res = await fetch(new URL(name, sceneBase));
    return res.ok ? res.json() : null;
  };
  const s: Shell = await tryFetch('shell.json');
  if (!s) { show('no shell.json next to this scene — run a capture, then open the model view'); return; }
  shell = s;
  room = buildRoom(s);
  objectsRoot = buildObjects(s);
  scene.add(room, objectsRoot);
  const walls = s.walls?.length ?? 0;
  const objects = s.objects?.length ?? 0;
  show(`${walls} walls · ${(s.openings ?? []).length} openings · ${objects} objects`);
  buildOutliner(s);
  setView(view);
  addEventListener('resize', resize);
  resize();
  loop();
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

function setView(next: View) {
  view = next;
  document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === next));
  grid.visible = next !== 'walk';
  if (!room) return;
  const ghost = next === "doll";
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
    camera.position.set(centre.x, (shell?.floorY ?? 0) + 1.62, centre.z);
    yaw = 0; pitch = 0;
    walked.set(centre.x, 0, centre.z);
  } else if (next === 'doll') {
    controls.enabled = true;
    controls.target.copy(centre).setY((shell?.floorY ?? 0) + size.y * 0.35);
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

addEventListener('keydown', (e) => {
  keys.add(e.code);
  if (e.code === 'KeyX') toggleWire();
  if (e.code === 'Escape') select(null);
});
addEventListener('keyup', (e) => keys.delete(e.code));
canvas.addEventListener('click', () => { if (view === 'walk' && !document.pointerLockElement) canvas.requestPointerLock(); });
addEventListener('mousemove', (e) => {
  if (document.pointerLockElement !== canvas) return;
  yawPitch.yaw -= e.movementX * 0.0022;
  yawPitch.pitch = THREE.MathUtils.clamp(yawPitch.pitch - e.movementY * 0.0022, -1.2, 1.2);
});

function blocked(x: number, z: number) {
  const floorY = shell?.floorY ?? 0;
  for (const w of shell?.walls ?? []) {
    const [ax, az] = w.a, [bx, bz] = w.b;
    const dx = bx - ax, dz = bz - az;
    const t = THREE.MathUtils.clamp(((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz || 1), 0, 1);
    const px = ax + dx * t, pz = az + dz * t;
    if (Math.hypot(x - px, z - pz) < RADIUS + 0.07) return true;
  }
  for (const o of shell?.objects ?? []) {
    if (deleted.has(o.id) || hidden.has(o.id)) continue;
    const [cx, cy, cz] = o.center;
    const [hx, hy, hz] = o.half;
    if (cy - hy > floorY + 1.7 || cy + hy < floorY + 0.2) continue; // above head or below the knee: walk under/over
    if (Math.abs(x - cx) < hx + RADIUS && Math.abs(z - cz) < hz + RADIUS) return true;
  }
  return false;
}

// ---------------------------------------------------------------- picking + editing
const ray = new THREE.Raycaster();
const pointer = new THREE.Vector2();
canvas.addEventListener('pointerdown', (e) => {
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
  const box = new THREE.Box3().setFromObject(selected);
  const s = box.getSize(new THREE.Vector3());
  el('#panel-size').textContent = data.kind === 'wall' && data.length ? `length ${fmt(data.length)}` : `${fmt(s.x)} × ${fmt(s.y)} × ${fmt(s.z)}`;
  selected.traverse((o) => { if (o instanceof THREE.Mesh) o.material = (o.material as THREE.MeshStandardMaterial).clone(); });
  el('#panel').dataset.target = selected.name;
}

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
    row.innerHTML = `<summary><span>${kind}</span><em>${items.length}</em></summary>`;
    for (const o of items) {
      const item = document.createElement('button');
      item.className = 'item';
      item.dataset.id = o.id;
      item.innerHTML = `<span>${o.name}</span><em>${o.half.map((v) => (v * 2).toFixed(1)).join('×')}</em>`;
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
    camera.position.set(walked.x, (shell?.floorY ?? 0) + EYE, walked.z);
    camera.rotation.set(pitch, yaw, 0, 'YXZ');
  } else {
    controls.update();
  }
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

canvas.addEventListener('pointerlockchange', () => {
  document.body.classList.toggle('locked', document.pointerLockElement === canvas);
});

load().catch((e) => show(`could not load: ${e}`));
