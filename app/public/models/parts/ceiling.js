/**
 * ceiling.js — the ceiling of the captured classroom, as procedural three.js.
 *
 * Contents
 *   · flat plaster ceiling plane at y = 3.10 m over the whole shell footprint,
 *     with a procedurally stained/peeling lime-wash texture (no external assets)
 *   · shallow perimeter beams, two longitudinal main beams and three cross joists
 *   · two long-bladed dark-wood ceiling fans (1.4 m tip to tip, 3 straight blades)
 *   · white projector on an articulated dark ceiling bracket, aimed at the screen
 *   · ten slim wall-mounted fluorescent battens with bright emissive faces
 *
 * Units: metres, +Y up. Room floor y = 0, ceiling y = 3.10.
 * No imports, no assets, no DOM work at module load. The canvas texture helper
 * is guarded so this also evaluates in a headless/Node renderer.
 *
 * Export: buildCeiling(THREE, shell) -> THREE.Group
 */

const CEILING_Y = 3.1;

// Bound by buildCeiling before any helper runs; the module receives THREE as a
// parameter rather than importing it, so the shared helpers need this handle.
let T = null;

/* ------------------------------------------------------------------ */
/* small deterministic helpers                                         */
/* ------------------------------------------------------------------ */

function makeRng(seed) {
  let s = seed >>> 0;
  return function rng() {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Room extents from the wall lines, with a sane fallback. */
function roomExtents(shell) {
  const xs = [];
  const zs = [];
  const walls = shell && Array.isArray(shell.walls) ? shell.walls : [];
  for (let i = 0; i < walls.length; i++) {
    const w = walls[i];
    if (!w || !Array.isArray(w.a) || !Array.isArray(w.b)) continue;
    xs.push(w.a[0], w.b[0]);
    zs.push(w.a[1], w.b[1]);
  }
  let minX = xs.length ? Math.min.apply(null, xs) : -5.4;
  let maxX = xs.length ? Math.max.apply(null, xs) : 5.4;
  let minZ = zs.length ? Math.min.apply(null, zs) : -5.0;
  let maxZ = zs.length ? Math.max.apply(null, zs) : 5.0;
  // Guard against degenerate solves.
  if (!isFinite(minX) || !isFinite(maxX) || maxX - minX < 4) {
    minX = -5.4;
    maxX = 5.4;
  }
  if (!isFinite(minZ) || !isFinite(maxZ) || maxZ - minZ < 4) {
    minZ = -5.0;
    maxZ = 5.0;
  }
  return {
    minX,
    maxX,
    minZ,
    maxZ,
    cx: (minX + maxX) / 2,
    cz: (minZ + maxZ) / 2,
    sx: maxX - minX,
    sz: maxZ - minZ,
  };
}

/** One box mesh, fully shadow-participating, with a name. */
function box(parent, name, w, h, d, x, y, z, material) {
  const mesh = new T.Mesh(new T.BoxGeometry(w, h, d), material);
  mesh.name = name;
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

/** One cylinder mesh, axis Y unless a rotation is supplied. */
function cylinder(parent, name, rTop, rBottom, height, x, y, z, material, segments) {
  const geo = new T.CylinderGeometry(rTop, rBottom, height, segments || 18, 1);
  const mesh = new T.Mesh(geo, material);
  mesh.name = name;
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

/** Straight round bar between two local points (bracket braces, rods). */
function barBetween(parent, name, a, b, radius, material, segments) {
  const from = new T.Vector3(a[0], a[1], a[2]);
  const to = new T.Vector3(b[0], b[1], b[2]);
  const dir = to.clone().sub(from);
  const len = dir.length();
  const geo = new T.CylinderGeometry(radius, radius, len, segments || 10, 1);
  const mesh = new T.Mesh(geo, material);
  mesh.name = name;
  mesh.position.copy(from).add(to).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), dir.normalize());
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

/* ------------------------------------------------------------------ */
/* procedural plaster canvas                                           */
/* ------------------------------------------------------------------ */

/**
 * Aged lime-plaster map, generated on a canvas.
 *
 * A seamless periodic value-noise field supplies the broad warm/cool unevenness
 * and the fine plaster grain; soft, irregular damp blooms and chalky paint-loss
 * patches are then layered on. All overlays are repeated at the nine wrapped
 * offsets so the texture tiles. Browser-only: returns null in headless contexts,
 * in which case the materials fall back to a flat plaster colour.
 */
function plasterTexture(THREE, opts) {
  const o = opts || {};
  if (typeof document === 'undefined' || !document.createElement) return null;
  let canvas;
  try {
    canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
  } catch (e) {
    return null;
  }
  const ctx = canvas.getContext && canvas.getContext('2d');
  if (!ctx) return null;

  const S = 512;
  const seed = o.seed || 7;
  const rng = makeRng(seed);
  const dirty = o.dirty === undefined ? 1 : o.dirty;
  const baseRGB = o.baseRGB || [217, 214, 203];
  const grimeRGB = o.grimeRGB || [112, 116, 107];
  const grime = grimeRGB.join(', ');
  const clamp8 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

  // Tiling value noise: the integer lattice wraps on `period`, so opposite
  // canvas edges agree exactly.
  const hash = (ix, iy, hs) => {
    let h = Math.imul(ix, 374761393) + Math.imul(iy, 668265263) + Math.imul(hs, 1442695041);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  };
  const vnoise = (x, y, period, hs) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const xf = x - xi;
    const yf = y - yi;
    const u = xf * xf * (3 - 2 * xf);
    const v = yf * yf * (3 - 2 * yf);
    const x0 = ((xi % period) + period) % period;
    const y0 = ((yi % period) + period) % period;
    const x1 = (x0 + 1) % period;
    const y1 = (y0 + 1) % period;
    const a = hash(x0, y0, hs);
    const b = hash(x1, y0, hs);
    const c = hash(x0, y1, hs);
    const d = hash(x1, y1, hs);
    const ab = a + (b - a) * u;
    const cd = c + (d - c) * u;
    return ab + (cd - ab) * v;
  };
  const fbm = (u, v, hs, period, octaves) => {
    let sum = 0;
    let amp = 1;
    let norm = 0;
    let p = period;
    for (let k = 0; k < octaves; k++) {
      sum += amp * vnoise(u * p, v * p, p, hs + k * 131);
      norm += amp;
      amp *= 0.5;
      p *= 2;
    }
    return sum / norm;
  };

  // Base field: cool grey-green plaster with broad unevenness and fine grain.
  const img = ctx.createImageData(S, S);
  const data = img.data;
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const u = x / S;
      const v = y / S;
      const n = fbm(u, v, seed, 4, 4);
      const broad = vnoise(u * 2, v * 2, 2, seed + 97);
      const grain = (rng() - 0.5) * 3.6;
      const shade = (n - 0.5) * 23 + (broad - 0.5) * 15;
      const damp = Math.max(0, 0.54 - n);
      const i = (y * S + x) * 4;
      data[i] = clamp8(baseRGB[0] + shade + grain - damp * 5);
      data[i + 1] = clamp8(baseRGB[1] + shade * 0.98 + grain - damp * 4);
      data[i + 2] = clamp8(baseRGB[2] + shade * 0.9 + grain - damp * 1);
      data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  const wrapped = (draw) => {
    for (let dx = -S; dx <= S; dx += S) {
      for (let dy = -S; dy <= S; dy += S) draw(dx, dy);
    }
  };

  /** Soft elliptical blob; no hard outline, so it reads as staining rather than a shape. */
  const blob = (cx, cy, rx, ry, rot, inner, mid, outer) => {
    wrapped((dx, dy) => {
      ctx.save();
      ctx.translate(cx + dx, cy + dy);
      ctx.rotate(rot);
      ctx.scale(1, ry / rx);
      const g = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
      g.addColorStop(0, inner);
      g.addColorStop(0.55, mid);
      g.addColorStop(1, outer);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(0, 0, rx, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  };

  // Large damp blooms / water damage.
  const blooms = Math.round(22 * dirty);
  for (let i = 0; i < blooms; i++) {
    const cx = rng() * S;
    const cy = rng() * S;
    const rx = 24 + rng() * 120;
    const ry = 18 + rng() * 85;
    const a = 0.03 + rng() * 0.075;
    const a2 = a * (0.5 + rng() * 0.4);
    blob(
      cx,
      cy,
      rx,
      ry,
      rng() * Math.PI,
      'rgba(' + grime + ',' + (a * 1.2).toFixed(3) + ')',
      'rgba(' + grime + ',' + a2.toFixed(3) + ')',
      'rgba(' + grime + ',0)'
    );
    // a smaller offset lobe makes the outline irregular
    if (dirty > 0.5) {
      blob(
        cx + (rng() - 0.5) * rx * 0.9,
        cy + (rng() - 0.5) * ry * 0.9,
        rx * (0.35 + rng() * 0.35),
        ry * (0.3 + rng() * 0.3),
        rng() * Math.PI,
        'rgba(' + grime + ',' + (a * 0.75).toFixed(3) + ')',
        'rgba(' + grime + ',0)',
        'rgba(' + grime + ',0)'
      );
    }
  }

  // Long, faint damp runs / streaks.
  for (let i = 0; i < Math.round(7 * dirty); i++) {
    blob(
      rng() * S,
      rng() * S,
      18 + rng() * 30,
      55 + rng() * 110,
      -0.35 + rng() * 0.7,
      'rgba(' + grime + ',' + (0.035 + rng() * 0.04).toFixed(3) + ')',
      'rgba(' + grime + ',0.012)',
      'rgba(' + grime + ',0)'
    );
  }

  // Chalky paint loss: a soft pale patch with a faint grey halo.
  for (let i = 0; i < Math.round(13 * dirty); i++) {
    const cx = rng() * S;
    const cy = rng() * S;
    const rx = 13 + rng() * 48;
    const ry = 10 + rng() * 34;
    const rot = rng() * Math.PI;
    // halo
    blob(
      cx,
      cy,
      rx * 1.22,
      ry * 1.25,
      rot,
      'rgba(126,127,117,' + (0.04 + rng() * 0.05).toFixed(3) + ')',
      'rgba(126,127,117,0.02)',
      'rgba(126,127,117,0)'
    );
    // chalk
    blob(
      cx + (rng() - 0.5) * 5,
      cy + (rng() - 0.5) * 5,
      rx,
      ry,
      rot,
      'rgba(237,234,224,' + (0.12 + rng() * 0.10).toFixed(3) + ')',
      'rgba(232,229,218,0.06)',
      'rgba(232,229,218,0)'
    );
  }

  // A few hairline cracks.
  for (let i = 0; i < 8; i++) {
    let x = rng() * S;
    let y = rng() * S;
    const steps = 4 + Math.floor(rng() * 5);
    const pts = [[x, y]];
    for (let k = 0; k < steps; k++) {
      x += (rng() - 0.5) * 42;
      y += (rng() - 0.5) * 42;
      pts.push([x, y]);
    }
    const width = 0.6 + rng() * 0.7;
    const alpha = 0.05 + rng() * 0.05;
    wrapped((dx, dy) => {
      ctx.beginPath();
      ctx.moveTo(pts[0][0] + dx, pts[0][1] + dy);
      for (let k = 1; k < pts.length; k++) ctx.lineTo(pts[k][0] + dx, pts[k][1] + dy);
      ctx.strokeStyle = 'rgba(105,103,95,' + alpha.toFixed(3) + ')';
      ctx.lineWidth = width;
      ctx.stroke();
    });
  }

  // Soot / grit.
  for (let i = 0; i < 300; i++) {
    const x = rng() * S;
    const y = rng() * S;
    const r = 0.3 + rng() * 1.15;
    ctx.fillStyle = 'rgba(101,100,92,' + (0.04 + rng() * 0.11).toFixed(3) + ')';
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(o.repeatX || 3, o.repeatY || 3);
  if (THREE.SRGBColorSpace) tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

/* ------------------------------------------------------------------ */
/* materials                                                           */
/* ------------------------------------------------------------------ */

function makeMaterials(THREE) {
  const ceilingMap = plasterTexture(THREE, {
    seed: 23,
    dirty: 1.35,
    baseRGB: [210, 208, 197],
    grimeRGB: [109, 113, 105],
    repeatX: 3,
    repeatY: 3,
  });
  const beamMap = plasterTexture(THREE, {
    seed: 51,
    dirty: 0.3,
    baseRGB: [229, 226, 215],
    grimeRGB: [139, 138, 127],
    repeatX: 2,
    repeatY: 1,
  });

  const m = {};

  m.ceiling = new THREE.MeshStandardMaterial({
    color: ceilingMap ? 0xf4f1e9 : 0xd9d4c5,
    map: ceilingMap,
    roughness: 0.97,
    metalness: 0,
    dithering: true,
  });

  m.beam = new THREE.MeshStandardMaterial({
    color: beamMap ? 0xffffff : 0xe9e5d7,
    map: beamMap,
    roughness: 0.92,
    metalness: 0,
    dithering: true,
  });

  m.beamSide = new THREE.MeshStandardMaterial({
    color: 0xd9d4c4,
    roughness: 0.95,
    metalness: 0,
  });

  m.fanWood = new THREE.MeshPhysicalMaterial({
    color: 0x24170f,
    roughness: 0.7,
    metalness: 0.02,
    clearcoat: 0.16,
    clearcoatRoughness: 0.62,
  });

  m.fanMetal = new THREE.MeshStandardMaterial({
    color: 0x3d2a21,
    roughness: 0.46,
    metalness: 0.38,
  });

  m.fanHub = new THREE.MeshStandardMaterial({
    color: 0x4a2e23,
    roughness: 0.38,
    metalness: 0.5,
  });

  m.brass = new THREE.MeshStandardMaterial({
    color: 0x9a7845,
    roughness: 0.36,
    metalness: 0.72,
  });

  m.ceilingWhite = new THREE.MeshStandardMaterial({
    color: 0xefece2,
    roughness: 0.8,
    metalness: 0,
  });

  m.projWhite = new THREE.MeshPhysicalMaterial({
    color: 0xf2f2ed,
    roughness: 0.42,
    metalness: 0.04,
    clearcoat: 0.22,
    clearcoatRoughness: 0.4,
  });

  m.projGrey = new THREE.MeshStandardMaterial({
    color: 0xc6c7c2,
    roughness: 0.55,
    metalness: 0.12,
  });

  m.darkMetal = new THREE.MeshStandardMaterial({
    color: 0x3b3d3c,
    roughness: 0.44,
    metalness: 0.55,
  });

  m.lensGlass = new THREE.MeshPhysicalMaterial({
    color: 0x081217,
    roughness: 0.06,
    metalness: 0.35,
    clearcoat: 1,
    clearcoatRoughness: 0.05,
  });

  m.lensTrim = new THREE.MeshStandardMaterial({
    color: 0x767a78,
    roughness: 0.34,
    metalness: 0.7,
  });

  m.batten = new THREE.MeshStandardMaterial({
    color: 0xd9dad4,
    roughness: 0.5,
    metalness: 0.28,
  });

  m.battenEnd = new THREE.MeshStandardMaterial({
    color: 0x747a7a,
    roughness: 0.55,
    metalness: 0.3,
  });

  m.tubeGlow = new THREE.MeshStandardMaterial({
    color: 0xf8fdff,
    emissive: 0xe4f6ff,
    emissiveIntensity: 4.2,
    roughness: 0.22,
    metalness: 0,
  });
  // Keep the lamps reading as sources even in a bright room / tonemapped view.
  m.tubeGlow.toneMapped = false;

  m.led = new THREE.MeshStandardMaterial({
    color: 0x9ddc8a,
    emissive: 0x5fbf4c,
    emissiveIntensity: 1.4,
    roughness: 0.4,
    metalness: 0,
  });

  return m;
}

/* ------------------------------------------------------------------ */
/* ceiling slab, beams and cornice                                     */
/* ------------------------------------------------------------------ */

function buildCeilingShell(THREE, root, ext, m) {
  const group = new THREE.Group();
  group.name = 'ceiling_shell';
  root.add(group);

  // One shallow slab; its underside is the visible ceiling at exactly 3.10 m.
  const slab = box(
    group,
    'ceiling_plaster_slab',
    ext.sx + 0.10,
    0.12,
    ext.sz + 0.10,
    ext.cx,
    CEILING_Y + 0.06,
    ext.cz,
    m.ceiling
  );
  slab.castShadow = true;
  slab.receiveShadow = true;

  // Cornice/edge trim where the ceiling meets the walls — a small cleaned band.
  const coveInset = 0.13;
  const coveW = 0.16;
  const coveH = 0.075;
  const coveY = CEILING_Y - coveH / 2;
  box(group, 'ceiling_cove_north', ext.sx - 0.02, coveH, coveW,
    ext.cx, coveY, ext.maxZ - coveInset, m.beam);
  box(group, 'ceiling_cove_south', ext.sx - 0.02, coveH, coveW,
    ext.cx, coveY, ext.minZ + coveInset, m.beam);
  box(group, 'ceiling_cove_east', coveW, coveH, ext.sz - 0.02,
    ext.maxX - coveInset, coveY, ext.cz, m.beam);
  box(group, 'ceiling_cove_west', coveW, coveH, ext.sz - 0.02,
    ext.minX + coveInset, coveY, ext.cz, m.beam);

  const beams = new THREE.Group();
  beams.name = 'ceiling_beams';
  group.add(beams);

  // Perimeter beams, dropped 140 mm below the plaster.
  const pInset = 0.24;
  const pW = 0.30;
  const pH = 0.16;
  const pY = CEILING_Y - pH / 2;
  box(beams, 'beam_perimeter_north', ext.sx - 0.02, pH, pW,
    ext.cx, pY, ext.maxZ - pInset, m.beam);
  box(beams, 'beam_perimeter_south', ext.sx - 0.02, pH, pW,
    ext.cx, pY, ext.minZ + pInset, m.beam);
  box(beams, 'beam_perimeter_east', pW, pH, ext.sz - 0.02,
    ext.maxX - pInset, pY, ext.cz, m.beam);
  box(beams, 'beam_perimeter_west', pW, pH, ext.sz - 0.02,
    ext.minX + pInset, pY, ext.cz, m.beam);

  // Two long main beams running the 10.8 m way; they frame the big hall coffer.
  const mainZ = 2.85;
  const mainW = 0.38;
  const mainH = 0.20;
  const mainY = CEILING_Y - mainH / 2;
  const mainLen = ext.sx - 0.42;
  box(beams, 'beam_main_north', mainLen, mainH, mainW,
    ext.cx, mainY, ext.cz + mainZ, m.beam);
  box(beams, 'beam_main_south', mainLen, mainH, mainW,
    ext.cx, mainY, ext.cz - mainZ, m.beam);

  // Three shallower cross joists.
  const joistX = 3.95;
  const joistW = 0.22;
  const joistH = 0.10;
  const joistY = CEILING_Y - joistH / 2;
  const joistLen = ext.sz - 0.42;
  box(beams, 'joist_west', joistW, joistH, joistLen,
    ext.cx - joistX, joistY, ext.cz, m.beam);
  box(beams, 'joist_center', joistW, joistH, joistLen,
    ext.cx, joistY, ext.cz, m.beam);
  box(beams, 'joist_east', joistW, joistH, joistLen,
    ext.cx + joistX, joistY, ext.cz, m.beam);

  // A slightly darker soffit lip on the two main beams reads as a shadow line.
  const lipW = 0.035;
  box(beams, 'beam_main_north_lip', mainLen, 0.024, lipW,
    ext.cx, CEILING_Y - mainH - 0.012, ext.cz + mainZ + mainW / 2 - lipW / 2, m.beamSide);
  box(beams, 'beam_main_south_lip', mainLen, 0.024, lipW,
    ext.cx, CEILING_Y - mainH - 0.012, ext.cz - mainZ + mainW / 2 - lipW / 2, m.beamSide);
}

/* ------------------------------------------------------------------ */
/* ceiling fan                                                         */
/* ------------------------------------------------------------------ */

/** Tapered, straight three-blade fan paddle, plan view built in XZ. */
function fanBladeGeometry(THREE) {
  const rootX = 0.048;
  const tipX = 0.700;
  const t = 0.011;
  const shape = new THREE.Shape();
  shape.moveTo(rootX, -0.034);
  shape.bezierCurveTo(0.20, -0.043, 0.46, -0.048, 0.635, -0.050);
  shape.quadraticCurveTo(0.688, -0.048, tipX, 0);
  shape.quadraticCurveTo(0.688, 0.048, 0.635, 0.050);
  shape.bezierCurveTo(0.46, 0.048, 0.20, 0.043, rootX, 0.034);
  shape.quadraticCurveTo(0.032, 0, rootX, -0.034);

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: t,
    steps: 1,
    curveSegments: 16,
    bevelEnabled: true,
    bevelThickness: 0.0025,
    bevelSize: 0.0025,
    bevelSegments: 2,
  });
  geo.translate(0, 0, -t / 2);
  geo.rotateX(-Math.PI / 2); // plan in XZ, thickness along Y
  return geo;
}

function buildFan(THREE, mats, bladeGeo, name, x, z, phase) {
  const group = new THREE.Group();
  group.name = name;
  group.position.set(x, CEILING_Y, z);

  // Flush ceiling canopy and a long dark down-rod.
  cylinder(group, name + '_canopy', 0.050, 0.078, 0.055, 0, -0.0275, 0, mats.ceilingWhite, 20);
  cylinder(group, name + '_canopyRing', 0.059, 0.059, 0.011, 0, -0.055, 0, mats.fanMetal, 20);
  cylinder(group, name + '_downrod', 0.0115, 0.0115, 0.275, 0, -0.1975, 0, mats.fanMetal, 12);
  cylinder(group, name + '_rodCoupler', 0.022, 0.022, 0.042, 0, -0.351, 0, mats.fanHub, 14);

  // Motor shell: a small lathed bowl, dark maroon metal with a brass band.
  const profile = [
    new THREE.Vector2(0.0001, -0.367),
    new THREE.Vector2(0.030, -0.370),
    new THREE.Vector2(0.070, -0.382),
    new THREE.Vector2(0.086, -0.405),
    new THREE.Vector2(0.085, -0.440),
    new THREE.Vector2(0.062, -0.462),
    new THREE.Vector2(0.028, -0.470),
    new THREE.Vector2(0.0001, -0.472),
  ];
  const motor = new THREE.Mesh(new THREE.LatheGeometry(profile, 24), mats.fanMetal);
  motor.name = name + '_motorHousing';
  motor.castShadow = true;
  motor.receiveShadow = true;
  group.add(motor);

  const band = new THREE.Mesh(new THREE.TorusGeometry(0.073, 0.006, 8, 28), mats.brass);
  band.name = name + '_brassBand';
  band.position.y = -0.422;
  band.rotation.x = Math.PI / 2;
  band.castShadow = true;
  band.receiveShadow = true;
  group.add(band);

  cylinder(group, name + '_hub', 0.043, 0.047, 0.028, 0, -0.485, 0, mats.fanHub, 18);
  cylinder(group, name + '_hubCap', 0.021, 0.032, 0.022, 0, -0.503, 0, mats.brass, 16);

  // A short pull chain, visible on the fans in f022/f040.
  cylinder(group, name + '_pullChain', 0.0018, 0.0018, 0.20, 0.030, -0.600, 0.012, mats.fanMetal, 6);
  const bead = new THREE.Mesh(new THREE.SphereGeometry(0.008, 8, 6), mats.brass);
  bead.name = name + '_pullChainBead';
  bead.position.set(0.030, -0.705, 0.012);
  bead.castShadow = true;
  bead.receiveShadow = true;
  group.add(bead);

  const blades = new THREE.Group();
  blades.name = name + '_blades';
  group.add(blades);

  for (let i = 0; i < 3; i++) {
    const leg = new THREE.Group();
    leg.name = name + '_blade' + (i + 1);
    leg.rotation.y = phase + (i * Math.PI * 2) / 3;
    blades.add(leg);

    const blade = new THREE.Mesh(bladeGeo, mats.fanWood);
    blade.name = name + '_blade' + (i + 1) + '_paddle';
    blade.position.y = -0.498;
    blade.rotation.x = 0.085; // slight pitch
    blade.castShadow = true;
    blade.receiveShadow = true;
    leg.add(blade);

    const iron = box(leg, name + '_blade' + (i + 1) + '_iron',
      0.155, 0.012, 0.032, 0.115, -0.512, 0, mats.fanMetal);
    iron.rotation.x = 0.02;
  }

  return group;
}

/* ------------------------------------------------------------------ */
/* projector                                                           */
/* ------------------------------------------------------------------ */

function buildProjector(THREE, mats, root, x, z, target) {
  const group = new THREE.Group();
  group.name = 'projector_assembly';
  group.position.set(x, 0, z);
  root.add(group);

  const bodyY = 2.44;

  // Ceiling plate and the dark articulated bracket.
  box(group, 'projector_ceilingPlate', 0.24, 0.018, 0.24,
    0, CEILING_Y - 0.010, 0, mats.ceilingWhite);
  cylinder(group, 'projector_bracketStem', 0.017, 0.017, 0.50,
    0, 2.83, 0, mats.darkMetal, 12);
  barBetween(group, 'projector_bracketBrace',
    [0, CEILING_Y - 0.05, -0.105], [0, 2.55, -0.012], 0.0075, mats.darkMetal, 8);
  const knuckle = new THREE.Mesh(new THREE.SphereGeometry(0.033, 14, 10), mats.darkMetal);
  knuckle.name = 'projector_bracketKnuckle';
  knuckle.position.set(0, 2.555, 0);
  knuckle.castShadow = true;
  knuckle.receiveShadow = true;
  group.add(knuckle);
  box(group, 'projector_bracketArm', 0.036, 0.036, 0.16,
    0, 2.555, 0.055, mats.darkMetal);

  // Body frame: +Z is the lens direction.
  const body = new THREE.Group();
  body.name = 'projector_body';
  body.position.set(0, bodyY, 0.065);

  const dir = new THREE.Vector3(
    target[0] - x,
    target[1] - bodyY,
    target[2] - (z + 0.065)
  ).normalize();
  const yaw = Math.atan2(dir.x, dir.z);
  const pitch = Math.asin(-dir.y);
  body.rotation.set(pitch, yaw, 0, 'YXZ');
  group.add(body);

  // Mount ears connect the tilting body back to the knuckle.
  box(body, 'projector_mountEarLeft', 0.018, 0.075, 0.13,
    -0.098, 0.078, -0.025, mats.darkMetal);
  box(body, 'projector_mountEarRight', 0.018, 0.075, 0.13,
    0.098, 0.078, -0.025, mats.darkMetal);
  box(body, 'projector_mountBar', 0.205, 0.020, 0.045,
    0, 0.105, -0.055, mats.darkMetal);

  // Main white housing, slightly stepped lid.
  box(body, 'projector_housing', 0.370, 0.115, 0.275, 0, 0, 0, mats.projWhite);
  box(body, 'projector_topLid', 0.336, 0.014, 0.246, 0, 0.064, -0.004, mats.projGrey);
  box(body, 'projector_frontBezel', 0.338, 0.070, 0.012, 0, -0.014, 0.143, mats.projGrey);
  box(body, 'projector_bottomPlate', 0.300, 0.010, 0.220, 0, -0.062, -0.012, mats.projGrey);
  for (let f = 0; f < 3; f++) {
    box(body, 'projector_undersideVent' + (f + 1), 0.100, 0.005, 0.012,
      0.062, -0.069, 0.055 - f * 0.031, mats.darkMetal);
  }

  // Lens barrel, glass and silver trim.
  const lens = cylinder(body, 'projector_lensBarrel', 0.045, 0.052, 0.044,
    0.020, -0.010, 0.164, mats.darkMetal, 22);
  lens.rotation.x = Math.PI / 2;
  const glass = cylinder(body, 'projector_lensGlass', 0.034, 0.034, 0.009,
    0.020, -0.010, 0.187, mats.lensGlass, 22);
  glass.rotation.x = Math.PI / 2;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.047, 0.0045, 8, 26), mats.lensTrim);
  ring.name = 'projector_lensTrim';
  ring.position.set(0.020, -0.010, 0.185);
  ring.castShadow = true;
  ring.receiveShadow = true;
  body.add(ring);

  // Side ventilation slots and a small status LED.
  for (let i = 0; i < 3; i++) {
    box(body, 'projector_vent' + (i + 1), 0.006, 0.036, 0.095,
      0.188, -0.008 + (i - 1) * 0.030, -0.055, mats.darkMetal);
  }
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.0065, 8, 6), mats.led);
  led.name = 'projector_statusLed';
  led.position.set(-0.148, -0.040, 0.150);
  led.castShadow = true;
  led.receiveShadow = true;
  body.add(led);

  // Cable drop from the ceiling rose into the body.
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.055, CEILING_Y - 0.02, -0.055),
    new THREE.Vector3(0.095, 2.98, -0.085),
    new THREE.Vector3(0.115, 2.82, -0.035),
    new THREE.Vector3(0.075, 2.64, 0.010),
    new THREE.Vector3(0.020, 2.555, 0.035),
  ]);
  const cable = new THREE.Mesh(new THREE.TubeGeometry(curve, 24, 0.0065, 6, false), mats.darkMetal);
  cable.name = 'projector_cableLoop';
  cable.castShadow = true;
  cable.receiveShadow = true;
  group.add(cable);
}

/* ------------------------------------------------------------------ */
/* wall-mounted fluorescent batten                                     */
/* ------------------------------------------------------------------ */

function buildTubeLight(THREE, mats, name, point, normal) {
  const group = new THREE.Group();
  group.name = name;
  // These sit a little lower than the perimeter beam soffit, as in f040.
  group.position.set(point[0], 2.845, point[1]);
  group.rotation.y = Math.atan2(normal[0], normal[1]);
  // local +Z faces the room, local X runs along the wall

  // Two short wall brackets rather than a full backboard.
  box(group, name + '_wallPlateLeft', 0.16, 0.075, 0.022,
    -0.38, 0, -0.115, mats.darkMetal);
  box(group, name + '_wallPlateRight', 0.16, 0.075, 0.022,
    0.38, 0, -0.115, mats.darkMetal);
  box(group, name + '_armLeft', 0.020, 0.020, 0.110,
    -0.38, 0.004, -0.055, mats.darkMetal);
  box(group, name + '_armRight', 0.020, 0.020, 0.110,
    0.38, 0.004, -0.055, mats.darkMetal);

  // Slim painted-steel batten with an exposed glowing tube and opal face.
  box(group, name + '_battenHousing', 1.24, 0.058, 0.095,
    0, 0.016, 0, mats.batten);
  box(group, name + '_diffuserFace', 1.135, 0.036, 0.014,
    0, -0.014, 0.055, mats.tubeGlow);

  const tube = new THREE.Mesh(new THREE.CylinderGeometry(0.023, 0.023, 1.125, 14, 1), mats.tubeGlow);
  tube.name = name + '_lampTube';
  tube.rotation.z = Math.PI / 2;
  tube.position.set(0, -0.029, 0.042);
  tube.castShadow = true;
  tube.receiveShadow = true;
  group.add(tube);

  box(group, name + '_endCapLeft', 0.040, 0.064, 0.101,
    -0.638, 0.012, 0, mats.battenEnd);
  box(group, name + '_endCapRight', 0.040, 0.064, 0.101,
    0.638, 0.012, 0, mats.battenEnd);

  return group;
}

/* ------------------------------------------------------------------ */
/* entry point                                                         */
/* ------------------------------------------------------------------ */

export function buildCeiling(THREE, shell) {
  T = THREE;
  const root = new THREE.Group();
  root.name = 'ceiling';

  // Accept either shell.json itself or a scene object that carries a .shell field.
  const sh = shell && shell.shell && Array.isArray(shell.shell.walls) ? shell.shell : shell;

  const ext = roomExtents(sh);
  const mats = makeMaterials(THREE);

  buildCeilingShell(THREE, root, ext, mats);

  // The two long-bladed fans read between the columns in f005/f015/f026:
  // their hubs sit either side of the ceiling centreline, x ≈ -0.87 m,
  // z ≈ -1.05 / +0.90 m (shell detections ceiling-fan-4 and ceiling-fan-6).
  const fanGroup = new THREE.Group();
  fanGroup.name = 'ceiling_fans';
  root.add(fanGroup);
  const bladeGeo = fanBladeGeometry(THREE);
  fanGroup.add(buildFan(THREE, mats, bladeGeo, 'ceilingFan_01', -0.868, 0.896, 0.32));
  fanGroup.add(buildFan(THREE, mats, bladeGeo, 'ceilingFan_02', -0.868, -1.048, 1.28));

  // Screen centre from the shell, used as the throw target; fallback to the
  // west wall centre when the object is absent.
  let target = [-5.13, 1.99, 0.0];
  const objs = sh && Array.isArray(sh.objects) ? sh.objects : [];
  for (let i = 0; i < objs.length; i++) {
    const o = objs[i];
    if (o && o.kind === 'projector screen' && Array.isArray(o.center)) {
      target = [o.center[0], o.center[1], o.center[2]];
      break;
    }
  }
  // Mid-hall between the two fans, as in f005/f026.
  buildProjector(THREE, mats, root, -1.55, -0.08, target);

  // Ten battens around the four walls, 0.23 m below the ceiling.
  const tubeGroup = new THREE.Group();
  tubeGroup.name = 'ceiling_fluorescent_lights';
  root.add(tubeGroup);

  const walls = sh && Array.isArray(sh.walls) ? sh.walls : [];
  // Fallback wall lines (same order as the shell: 0 north, 1 west, 2 east,
  // 3 south) so the battens still exist if this part is built shell-less.
  const fallbackWalls = [
    { a: [ext.maxX, ext.maxZ], b: [ext.minX, ext.maxZ], normal: [0, -1] },
    { a: [ext.minX, ext.maxZ], b: [ext.minX, ext.minZ], normal: [1, 0] },
    { a: [ext.maxX, ext.minZ], b: [ext.maxX, ext.maxZ], normal: [-1, 0] },
    { a: [ext.minX, ext.minZ], b: [ext.maxX, ext.minZ], normal: [0, 1] },
  ];
  const specs = [
    { wall: 0, us: [0.18, 0.50, 0.82] },
    { wall: 3, us: [0.20, 0.50, 0.80] },
    { wall: 1, us: [0.27, 0.73] },
    { wall: 2, us: [0.27, 0.73] },
  ];
  let lightIndex = 0;
  for (let s = 0; s < specs.length; s++) {
    const spec = specs[s];
    const wall =
      walls[spec.wall] && Array.isArray(walls[spec.wall].a) && Array.isArray(walls[spec.wall].b)
        ? walls[spec.wall]
        : fallbackWalls[spec.wall];
    if (!wall) continue;
    const ax = wall.a[0];
    const az = wall.a[1];
    const bx = wall.b[0];
    const bz = wall.b[1];
    let nx = Array.isArray(wall.normal) ? wall.normal[0] : 0;
    let nz = Array.isArray(wall.normal) ? wall.normal[1] : 0;
    const nLen = Math.hypot(nx, nz) || 1;
    nx /= nLen;
    nz /= nLen;
    // Make sure the normal points into the room.
    const mx = (ax + bx) / 2;
    const mz = (az + bz) / 2;
    if ((ext.cx - mx) * nx + (ext.cz - mz) * nz < 0) {
      nx = -nx;
      nz = -nz;
    }
    for (let u = 0; u < spec.us.length; u++) {
      const t = spec.us[u];
      const px = ax + (bx - ax) * t + nx * 0.18;
      const pz = az + (bz - az) * t + nz * 0.18;
      lightIndex += 1;
      const name = 'fluorescent_' + String(lightIndex).padStart(2, '0');
      tubeGroup.add(buildTubeLight(THREE, mats, name, [px, pz], [nx, nz]));
    }
  }

  return root;
}
