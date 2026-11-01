/**
 * walls.js — procedural shell for the captured classroom.
 *
 * export buildWalls(THREE, shell) -> THREE.Group
 *
 * Room: 10.80 m (X) x 10.00 m (Z) inside faces, 3.10 m clear height,
 * 0.16 m thick masonry, floor y = 0.  The south-east corner is replaced by
 * a 1.20 m chamfer wall (shell.walls[4] measures 1.174 m).
 *
 *          +Z (north)
 *   ┌──────────────────────────┐
 *   │   [north side]  [main]   │   two arched double doors
 *   │                          │
 * W │                          │ E   [east door] in the east wall
 *   │                          │
 *   │                     ╱    │   chamfer at the SE corner
 *   └──────────────────╱───────┘
 *          -Z (south)
 *
 * The shell's five wall lines are skewed by less than 0.3 deg and their
 * detected endpoints do not close at the corners, so they are squared up to
 * the stated 10.8 x 10.0 m room.  The two north doorways are placed from the
 * captured door positions (door-1/door-2 at x ≈ +1.06 m, door-3 at x ≈ -3.30 m);
 * the east doorway comes from shell.openings (wall 2, u = 2.836 m).
 *
 * Finish: cream-white plaster above a grey-green 1.20 m dado, a dark skirting,
 * a 1.2 m chamfer, and three shallow-arched openings.  The openings carry dark
 * panelled timber leaves and a carved lattice transom with a quatrefoil centre.
 *
 * No imports, no textures, no external assets.  Core three.js geometry only.
 * Every solid mesh casts and receives shadows.
 */

export function buildWalls(THREE, shell) {
  if (!THREE || !THREE.Group) {
    throw new Error('buildWalls(THREE, shell): THREE must be supplied');
  }

  // ---------------------------------------------------------------------------
  // Dimensions and layout
  // ---------------------------------------------------------------------------
  const ROOM = {
    width: 10.8, // X, inside face to inside face
    depth: 10.0, // Z, inside face to inside face
    height: 3.1,
    wallT: 0.16,
    dadoH: 1.2,
    skirtH: 0.14,
    chamfer: 1.2,
    archLeg: 1.2 / Math.SQRT2,
  };
  const HX = ROOM.width / 2; // 5.40
  const HZ = ROOM.depth / 2; // 5.00
  const T = ROOM.wallT;
  const H = ROOM.height;
  const LEG = ROOM.archLeg; // 0.8485

  // SE chamfer: inside face from B=(HX-LEG, -HZ) to A=(HX, -HZ+LEG).
  const CH_MX = HX - LEG / 2;
  const CH_MZ = -HZ + LEG / 2;

  // ---------------------------------------------------------------------------
  // Placements taken from shell.json (with frame-derived fallbacks)
  // ---------------------------------------------------------------------------
  const shellWalls = shell && Array.isArray(shell.walls) ? shell.walls : [];
  const shellOpenings = shell && Array.isArray(shell.openings) ? shell.openings : [];
  const shellObjects = shell && Array.isArray(shell.objects) ? shell.objects : [];
  const number = (value) => (typeof value === 'number' && isFinite(value) ? value : null);

  // Measured lines, kept for provenance only: the detected segments do not close
  // at their ends, so the built room uses the stated 10.80 x 10.00 m inside box.
  const shellMeasured = {
    width: shellWalls[0] ? number(shellWalls[0].length) : null,
    depth: shellWalls[1] ? number(shellWalls[1].length) : null,
    chamfer: shellWalls[4] ? number(shellWalls[4].length) : null,
  };

  // Main north door: the stained-glass transom object in shell.json sits on the
  // same centre line.  Side north door: the north-wall door object at x ≈ -3.3.
  let mainDoorX = 1.06;
  let sideDoorX = -3.3;
  for (const o of shellObjects) {
    if (!o || !Array.isArray(o.center)) continue;
    const x = number(o.center[0]);
    const y = number(o.center[1]);
    const z = number(o.center[2]);
    if (o.kind === 'stained glass window' && x !== null && z !== null && z > 0 && x > -1 && x < 3) {
      mainDoorX = x;
    }
    if (o.kind === 'door' && x !== null && y !== null && z !== null && z > 0 && y > 0.8 && y < 2.0 && x < -2) {
      sideDoorX = x;
    }
  }

  // East door: shell.openings gives u = 2.836 m along wall 2.  The room is
  // squared up, so the position is transferred as a z offset from the south end.
  let eastDoorCx = 2.22;
  const wall2 = shellWalls[2];
  const eastOpening = shellOpenings.find((o) => o && o.wall === 2 && o.kind === 'door');
  if (wall2 && eastOpening && Array.isArray(wall2.a) && Array.isArray(wall2.b)) {
    const u = number(eastOpening.u);
    const dz = wall2.b[1] - wall2.a[1];
    const dx = wall2.b[0] - wall2.a[0];
    const len = Math.hypot(dx, dz) || number(wall2.length);
    if (u !== null && len) {
      const t = Math.min(1, Math.max(0, u / len));
      const z = wall2.a[1] + dz * t;
      if (isFinite(z)) eastDoorCx = -z;
    }
  }
  // Keep the east opening clear of the chamfer and of the north corner.
  eastDoorCx = Math.min(HZ - LEG - 0.70, Math.max(-HZ + 0.70, eastDoorCx));

  // East door local X = -world Z (the east wall's local +X runs north).

  // Local X ranges of each wall (local +X runs along the wall face).
  // Wall groups sit on their inside face; local +Z points out of the room.
  const WALLS = [
    {
      name: 'Wall_North',
      position: [0, 0, HZ],
      rotationY: 0,
      range: [-HX - T, HX + T],
      interior: [-HX, HX],
      openings: [
        { name: 'NorthMainDoor', cx: mainDoorX, width: 1.78, spring: 2.04, rise: 0.5 },
        { name: 'NorthSideDoor', cx: sideDoorX, width: 1.5, spring: 2.02, rise: 0.46 },
      ],
    },
    {
      name: 'Wall_West',
      position: [-HX, 0, 0],
      rotationY: -Math.PI / 2,
      range: [-HZ - T, HZ + T],
      interior: [-HZ, HZ],
      openings: [],
    },
    {
      name: 'Wall_East',
      position: [HX, 0, 0],
      rotationY: Math.PI / 2,
      range: [-(HZ + T), HZ - LEG + 0.2],
      interior: [-HZ, HZ - LEG],
      openings: [
        // shell.openings wall 2: u = 2.836 m from a z=-5.057 -> z≈-2.22
        { name: 'EastDoor', cx: eastDoorCx, width: 1.16, spring: 2.0, rise: 0.42 },
      ],
    },
    {
      name: 'Wall_South',
      position: [0, 0, -HZ],
      rotationY: Math.PI,
      range: [-(HX - LEG + 0.2), HX + T],
      interior: [-(HX - LEG), HX],
      openings: [],
    },
  ];

  // ---------------------------------------------------------------------------
  // Materials — MeshStandardMaterial only
  // ---------------------------------------------------------------------------
  const MAT = {
    plaster: new THREE.MeshStandardMaterial({
      name: 'walls_plaster',
      color: 0xdcd6c6,
      roughness: 0.94,
      metalness: 0.0,
    }),
    trim: new THREE.MeshStandardMaterial({
      name: 'walls_plasterTrim',
      color: 0xd2ccbb,
      roughness: 0.9,
      metalness: 0.0,
    }),
    dado: new THREE.MeshStandardMaterial({
      name: 'walls_dadoGreyGreen',
      color: 0x878876,
      roughness: 0.88,
      metalness: 0.0,
    }),
    skirting: new THREE.MeshStandardMaterial({
      name: 'walls_skirting',
      color: 0x6a6b5d,
      roughness: 0.82,
      metalness: 0.0,
    }),
    timber: new THREE.MeshStandardMaterial({
      name: 'walls_darkTimber',
      color: 0x120b07,
      roughness: 0.62,
      metalness: 0.03,
    }),
    timberTrim: new THREE.MeshStandardMaterial({
      name: 'walls_darkTimberPanel',
      color: 0x1d120b,
      roughness: 0.58,
      metalness: 0.03,
    }),
    brass: new THREE.MeshStandardMaterial({
      name: 'walls_agedBrass',
      color: 0x9b7d47,
      roughness: 0.36,
      metalness: 0.72,
    }),
    stone: new THREE.MeshStandardMaterial({
      name: 'walls_thresholdStone',
      color: 0x4b4a43,
      roughness: 0.82,
      metalness: 0.02,
    }),
    glow: new THREE.MeshStandardMaterial({
      name: 'walls_transomDaylight',
      color: 0x6e6249,
      emissive: 0xd4b476,
      emissiveIntensity: 0.62,
      roughness: 1.0,
      metalness: 0.0,
      side: THREE.DoubleSide,
    }),
  };

  // ---------------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------------
  const root = new THREE.Group();
  root.name = 'Walls';
  root.userData = {
    kind: 'walls',
    units: 'm',
    roomInside: { width: ROOM.width, depth: ROOM.depth, height: ROOM.height },
    wallThickness: T,
    dadoHeight: ROOM.dadoH,
    chamferLength: ROOM.chamfer,
    openings: ['NorthMainDoor', 'NorthSideDoor', 'EastDoor'],
    doorCentresX: { northMain: mainDoorX, northSide: sideDoorX, eastLocal: eastDoorCx },
    shellMeasured,
    source: 'frames f007/f012/f014/f019/f022/f030/f038 + shell.json',
  };

  function addMesh(parent, geometry, material, name) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = name;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  }

  function addBox(parent, name, width, height, depth, x, y, z, material) {
    const mesh = addMesh(parent, new THREE.BoxGeometry(width, height, depth), material, name);
    mesh.position.set(x, y, z);
    return mesh;
  }

  function addGroup(parent, name) {
    const group = new THREE.Group();
    group.name = name;
    parent.add(group);
    return group;
  }

  /**
   * Shallow basket/segmental arch as sampled points, left springing to right
   * springing.  The sin^0.9 lift keeps the shoulders steep and the crown full,
   * which is closer to the four-centred door heads in the photos than a plain
   * semicircle.  (cx, spring) is the springing line; rise is measured from it.
   */
  function archPoints(cx, halfWidth, spring, rise, steps) {
    const count = steps || 18;
    const pts = [];
    for (let i = 0; i <= count; i++) {
      const theta = Math.PI - (Math.PI * i) / count;
      const x = cx + halfWidth * Math.cos(theta);
      const y = spring + rise * Math.pow(Math.max(0, Math.sin(theta)), 0.9);
      pts.push({ x, y });
    }
    return pts;
  }

  /** Height of the arch at world/local X (0 outside the arch). */
  function archHeightAt(x, cx, halfWidth, spring, rise) {
    const u = Math.min(1, Math.abs((x - cx) / halfWidth));
    const theta = Math.acos(u);
    return spring + rise * Math.pow(Math.sin(theta), 0.9);
  }

  /** Extrude a Shape in the wall plane.  `z` is the back of the extrusion. */
  function extrudeShape(parent, name, shape, depth, z, material) {
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth,
      steps: 1,
      curveSegments: 10,
      bevelEnabled: false,
    });
    const mesh = addMesh(parent, geometry, material, name);
    mesh.position.z = z;
    return mesh;
  }

  /** Subtract closed intervals from a range; used to stop bands at openings. */
  function subtractRanges(range, cuts) {
    const a = range[0];
    const b = range[1];
    const clipped = cuts
      .map((c) => [Math.max(a, c[0]), Math.min(b, c[1])])
      .filter((c) => c[1] - c[0] > 0.0005)
      .sort((c0, c1) => c0[0] - c1[0]);
    const out = [];
    let cursor = a;
    for (const c of clipped) {
      if (c[0] > cursor + 0.0005) out.push([cursor, c[0]]);
      cursor = Math.max(cursor, c[1]);
    }
    if (b > cursor + 0.0005) out.push([cursor, b]);
    return out;
  }

  /** Full-height plaster for one solid stretch of wall. */
  function addWallPiece(parent, name, x0, x1) {
    const w = x1 - x0;
    if (w < 0.002) return;
    const mesh = addMesh(parent, new THREE.BoxGeometry(w, H, T), MAT.plaster, name);
    mesh.position.set((x0 + x1) / 2, H / 2, T / 2);
  }

  /** Plaster above one arched opening: rectangle with the arch as its bottom. */
  function addSpandrel(parent, name, opening) {
    const cx = opening.cx;
    const halfW = opening.width / 2;
    const pts = archPoints(cx, halfW, opening.spring, opening.rise, 24);
    const shape = new THREE.Shape();
    shape.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) shape.lineTo(pts[i].x, pts[i].y);
    shape.lineTo(cx + halfW, H);
    shape.lineTo(cx - halfW, H);
    shape.closePath();
    extrudeShape(parent, name, shape, T, 0, MAT.plaster);
  }

  // ---------------------------------------------------------------------------
  // Dado, top bead and skirting
  // ---------------------------------------------------------------------------
  function addBands(dadoParent, beadParent, skirtParent, wallName, a, b, tag) {
    const width = b - a;
    if (width < 0.002) return;
    const mid = (a + b) / 2;
    const suffix = tag ? '_' + tag : '';

    // Grey-green dado, 1.20 m high overall, sitting on the skirting.
    const dadoH = ROOM.dadoH - ROOM.skirtH;
    addBox(
      dadoParent,
      wallName + '_Dado' + suffix,
      width,
      dadoH,
      0.014,
      mid,
      ROOM.skirtH + dadoH / 2,
      -0.007,
      MAT.dado
    );

    // Narrow cream bead at the top of the dado.
    addBox(
      beadParent,
      wallName + '_DadoBead' + suffix,
      width,
      0.028,
      0.02,
      mid,
      ROOM.dadoH + 0.014,
      -0.01,
      MAT.trim
    );

    // Skirting, slightly darker and a touch more proud than the dado.
    addBox(
      skirtParent,
      wallName + '_Skirting' + suffix,
      width,
      ROOM.skirtH,
      0.024,
      mid,
      ROOM.skirtH / 2,
      -0.012,
      MAT.skirting
    );
  }

  // ---------------------------------------------------------------------------
  // Doors, leaves, architrave and lattice transom
  // ---------------------------------------------------------------------------

  /** U-shaped plaster architrave around an opening, open at the floor. */
  function addArchitrave(parent, name, opening) {
    const cx = opening.cx;
    const halfW = opening.width / 2;
    const spring = opening.spring;
    const rise = opening.rise;
    const m = 0.085; // face width of the surround
    const inner = archPoints(cx, halfW, spring, rise, 18);
    const outer = archPoints(cx, halfW + m, spring, rise + m, 18);
    const xL = cx - halfW;
    const xR = cx + halfW;
    const xLo = cx - halfW - m;
    const xRo = cx + halfW + m;

    const shape = new THREE.Shape();
    shape.moveTo(xLo, 0);
    shape.lineTo(xLo, spring);
    for (let i = 0; i < outer.length; i++) shape.lineTo(outer[i].x, outer[i].y);
    shape.lineTo(xRo, 0);
    shape.lineTo(xR, 0);
    shape.lineTo(xR, spring);
    for (let i = inner.length - 1; i >= 0; i--) shape.lineTo(inner[i].x, inner[i].y);
    shape.lineTo(xL, 0);
    shape.closePath();

    extrudeShape(parent, name, shape, 0.028, -0.028, MAT.trim);
  }

  /** One panelled leaf, hanging in a wall-local group. */
  function addLeaf(parent, name, cxLocal, width, height, zCenter) {
    const group = addGroup(parent, name);
    group.position.set(cxLocal, 0.025 + height / 2, zCenter);

    const depth = 0.055;
    addBox(group, name + '_Slab', width, height, depth, 0, 0, 0, MAT.timber);

    // Raised stiles/rails laid over the slab face, 14 mm proud, giving the
    // 2-column x 4-row panel grid seen in f012/f014.
    const faceZ = -depth / 2 - 0.007;
    const railD = 0.014;

    const topRailH = 0.072;
    const botRailH = 0.095;
    const sideStileW = 0.052;
    addBox(group, name + '_RailTop', width - 0.06, topRailH, railD, 0, height / 2 - topRailH / 2 - 0.03, faceZ, MAT.timberTrim);
    addBox(group, name + '_RailBottom', width - 0.06, botRailH, railD, 0, -height / 2 + botRailH / 2 + 0.03, faceZ, MAT.timberTrim);
    addBox(group, name + '_StileL', sideStileW, height - 0.2, railD, -width / 2 + sideStileW / 2 + 0.03, 0, faceZ, MAT.timberTrim);
    addBox(group, name + '_StileR', sideStileW, height - 0.2, railD, width / 2 - sideStileW / 2 - 0.03, 0, faceZ, MAT.timberTrim);
    addBox(group, name + '_Muntin', 0.028, height - 0.2, railD, 0, 0, faceZ, MAT.timberTrim);

    const panelTop = height / 2 - 0.11;
    const panelBottom = -height / 2 + 0.16;
    for (let i = 1; i <= 3; i++) {
      const y = panelBottom + ((panelTop - panelBottom) * i) / 4;
      addBox(group, name + '_RailH' + i, width - 0.13, 0.026, railD, 0, y, faceZ, MAT.timberTrim);
    }

    return group;
  }

  /** Ring pull and its backplate.  The local XY plane is the door face. */
  function addHandle(parent, name, x, y, z, faceMaterial) {
    const plate = addMesh(
      parent,
      new THREE.CylinderGeometry(0.024, 0.024, 0.007, 14),
      faceMaterial,
      name + '_Plate'
    );
    plate.position.set(x, y, z);
    plate.rotation.x = Math.PI / 2;

    const ring = addMesh(
      parent,
      new THREE.TorusGeometry(0.038, 0.0065, 6, 22),
      faceMaterial,
      name + '_Ring'
    );
    ring.position.set(x, y, z - 0.009);
  }

  /** Carved lattice transom: clipped grid bars, arched border and quatrefoil. */
  function addTransom(parent, name, opening) {
    const group = addGroup(parent, name);
    const cx = opening.cx;
    const halfW = opening.width / 2;
    const spring = opening.spring;
    const rise = opening.rise;
    const y0 = spring + 0.005;
    const zBars = 0.11;
    const barD = 0.044;
    const barT = 0.022;

    // Warm, back-lit plaster/louvre behind the carving: in the capture the
    // cells glow with daylight from the verandah beyond.
    const backShape = new THREE.Shape();
    backShape.moveTo(cx - halfW + 0.012, y0);
    backShape.lineTo(cx + halfW - 0.012, y0);
    backShape.lineTo(cx + halfW - 0.012, spring);
    const top = archPoints(cx, halfW - 0.012, spring, rise - 0.012, 20);
    for (let i = top.length - 1; i >= 0; i--) backShape.lineTo(top[i].x, top[i].y);
    backShape.lineTo(cx - halfW + 0.012, spring);
    backShape.closePath();
    const backing = addMesh(group, new THREE.ShapeGeometry(backShape), MAT.glow, name + '_Backing');
    backing.position.z = 0.154;
    backing.castShadow = true;
    backing.receiveShadow = true;

    // Bottom rail / transom sill.
    addBox(group, name + '_Sill', opening.width - 0.02, 0.06, 0.09, cx, spring - 0.01, zBars, MAT.timber);

    // Vertical bars, clipped by the arched head.
    const barCount = Math.max(5, Math.round(opening.width / 0.17));
    for (let i = 1; i <= barCount; i++) {
      const x = cx - halfW + (opening.width * i) / (barCount + 1);
      const topY = archHeightAt(x, cx, halfW, spring, rise) - 0.026;
      const bottomY = y0 + 0.015;
      const h = topY - bottomY;
      if (h < 0.045) continue;
      addBox(group, name + '_BarV' + String(i).padStart(2, '0'), barT, h, barD, x, bottomY + h / 2, zBars, MAT.timber);
    }

    // Horizontal bars, shortened to the arch width at their height.
    const rows = 3;
    for (let r = 1; r <= rows; r++) {
      const y = spring + (rise * r) / (rows + 1.1);
      const s = Math.max(0, Math.min(1, (y - spring) / rise));
      const theta = Math.asin(Math.pow(s, 1 / 0.9));
      const half = halfW * Math.cos(theta) - 0.03;
      if (half < 0.05) continue;
      addBox(group, name + '_BarH' + r, half * 2, barT, barD, cx, y, zBars, MAT.timber);
    }

    // Carved arched border following the plaster reveal.
    const borderPts = archPoints(cx, halfW - 0.035, spring, rise - 0.025, 12);
    for (let i = 0; i < borderPts.length - 1; i++) {
      const p = borderPts[i];
      const q = borderPts[i + 1];
      const dx = q.x - p.x;
      const dy = q.y - p.y;
      const len = Math.hypot(dx, dy) * 1.16;
      const bar = addMesh(
        group,
        new THREE.BoxGeometry(len, 0.03, 0.05),
        MAT.timber,
        name + '_ArchBar' + String(i + 1).padStart(2, '0')
      );
      bar.position.set((p.x + q.x) / 2, (p.y + q.y) / 2, zBars + 0.004);
      bar.rotation.z = Math.atan2(dy, dx);
    }

    // Quatrefoil medallion at the centre of the fanlight.
    const ringY = spring + rise * 0.55;
    const ring = addMesh(
      group,
      new THREE.TorusGeometry(0.105, 0.010, 7, 28),
      MAT.timber,
      name + '_QuatrefoilRing'
    );
    ring.position.set(cx, ringY, 0.092);

    const petalR = 0.078;
    for (let k = 0; k < 4; k++) {
      const angle = (k * Math.PI) / 2;
      const petal = addMesh(
        group,
        new THREE.TorusGeometry(0.038, 0.008, 6, 18),
        MAT.timber,
        name + '_QuatrefoilPetal' + (k + 1)
      );
      petal.position.set(cx + Math.cos(angle) * petalR, ringY + Math.sin(angle) * petalR, 0.092);
    }

    addMesh(
      group,
      new THREE.SphereGeometry(0.023, 12, 8),
      MAT.timber,
      name + '_QuatrefoilBoss'
    ).position.set(cx, ringY, 0.092);

    return group;
  }

  /** Complete doorway: threshold, two leaves, handles, transom and surround. */
  function addDoor(parent, opening) {
    const door = addGroup(parent, opening.name);
    const cx = opening.cx;
    const width = opening.width;
    const spring = opening.spring;
    const leafW = (width - 0.06) / 2;
    const leafH = spring - 0.075;

    // Stone threshold, just proud of the wall face.
    addBox(
      door,
      opening.name + '_Threshold',
      width + 0.16,
      0.028,
      0.22,
      cx,
      0.014,
      0.08,
      MAT.stone
    );

    const leaves = addGroup(door, opening.name + '_Leaves');
    const zLeaf = 0.105;
    for (const side of [-1, 1]) {
      const leafCx = cx + side * (leafW / 2 + 0.002);
      const label = side < 0 ? 'L' : 'R';
      addLeaf(leaves, opening.name + '_Leaf' + label, leafCx, leafW, leafH, zLeaf);
    }

    // Two ring pulls either side of the meeting stiles.
    addHandle(door, opening.name + '_HandleL', cx - 0.082, 1.02, 0.068, MAT.brass);
    addHandle(door, opening.name + '_HandleR', cx + 0.082, 1.02, 0.068, MAT.brass);

    addTransom(door, opening.name + '_Transom', opening);
    addArchitrave(parent, opening.name + '_Architrave', opening);

    return door;
  }

  // ---------------------------------------------------------------------------
  // Build the four walls
  // ---------------------------------------------------------------------------
  for (const wall of WALLS) {
    const wallGroup = addGroup(root, wall.name);
    wallGroup.position.set(wall.position[0], wall.position[1], wall.position[2]);
    wallGroup.rotation.y = wall.rotationY;
    wallGroup.userData = {
      kind: 'wall',
      localX: 'along the wall face',
      localZ: 'outward',
      interior: wall.interior,
    };

    const plasterG = addGroup(wallGroup, wall.name + '_Plaster');
    const dadoG = addGroup(wallGroup, wall.name + '_Dado');
    const trimG = addGroup(wallGroup, wall.name + '_Bead');
    const skirtG = addGroup(wallGroup, wall.name + '_Skirting');

    const openings = wall.openings.slice().sort((a, b) => a.cx - b.cx);
    const cuts = openings.map((o) => [o.cx - o.width / 2, o.cx + o.width / 2]);

    // Plaster: solid stretches plus one arched spandrel over each opening.
    if (openings.length === 0) {
      addWallPiece(plasterG, wall.name + '_Solid', wall.range[0], wall.range[1]);
    } else {
      let cursor = wall.range[0];
      let pierIndex = 1;
      for (const op of openings) {
        const left = op.cx - op.width / 2;
        const right = op.cx + op.width / 2;
        if (left > cursor) addWallPiece(plasterG, wall.name + '_Pier' + pierIndex++, cursor, left);
        cursor = Math.max(cursor, right);
      }
      if (wall.range[1] > cursor) addWallPiece(plasterG, wall.name + '_Pier' + pierIndex++, cursor, wall.range[1]);
      for (const op of openings) addSpandrel(plasterG, wall.name + '_Spandrel_' + op.name, op);
    }

    // Finish bands stop at every opening.
    const bandRanges = subtractRanges(wall.interior, cuts);
    bandRanges.forEach(([a, b], i) => {
      addBands(dadoG, trimG, skirtG, wall.name, a, b, String(i + 1).padStart(2, '0'));
    });

    for (const op of wall.openings) addDoor(wallGroup, op);
  }

  // ---------------------------------------------------------------------------
  // SE chamfer wall (1.20 m face length)
  // ---------------------------------------------------------------------------
  const chamferG = addGroup(root, 'Wall_Chamfer');
  chamferG.position.set(CH_MX, 0, CH_MZ);
  chamferG.rotation.y = (3 * Math.PI) / 4;
  chamferG.userData = {
    kind: 'wall',
    faceLength: ROOM.chamfer,
    note: 'cuts the south-east corner, 45 deg, squared from shell.walls[4] (1.174 m)',
  };

  const chamferPlaster = addGroup(chamferG, 'Wall_Chamfer_Plaster');
  const chamferDado = addGroup(chamferG, 'Wall_Chamfer_Dado');
  const chamferBead = addGroup(chamferG, 'Wall_Chamfer_Bead');
  const chamferSkirt = addGroup(chamferG, 'Wall_Chamfer_Skirting');

  // 1.44 m core so the ends bury into the east and south walls.
  addWallPiece(chamferPlaster, 'Wall_Chamfer_Solid', -0.72, 0.72);
  addBands(
    chamferDado,
    chamferBead,
    chamferSkirt,
    'Wall_Chamfer',
    -ROOM.chamfer / 2,
    ROOM.chamfer / 2,
    '01'
  );

  return root;
}
