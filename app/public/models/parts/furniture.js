/**
 * Classroom — fixed furniture, built as plain procedural Three.js.
 *
 * Exports: buildFurniture(THREE, shell) -> THREE.Group
 *
 * This part owns:
 *   - the long dark timber bench in front of the whiteboard wall
 *     (3.60 x 0.50 x 0.78 m legs, aprons and twin stretchers);
 *   - the whiteboard assembly on the west wall (4.00 x 1.20 m white face,
 *     aluminium frame, dark-green backing / chalk strip and chalk tray,
 *     screen casing above);
 *   - two small dark-teak wall shelves (door wall and window wall);
 *   - the electricity distribution panel with four switch boxes, conduits,
 *     labels and the small teak wall cabinet under it;
 *   - the three dark-timber arched doorways on the +Z wall: single west,
 *     double centre and single east.  Each has panelled leaves, an arched
 *     frame and a pointed lattice transom above;
 *   - the light-painted arched service door on the east wall.
 *
 * Coordinates are metres, +Y up, floor at y = 0.  Placements come from the
 * matching boxes in shell.objects (projector-screen-1 for the whiteboard,
 * shelf-1/shelf-2, door-1..door-4 and the transom windows).  Furniture sits
 * on the facade planes implied by the captured objects (whiteboard wall
 * x ~= -5.17, door wall z ~= 3.10, window wall z ~= -3.16); see the note in
 * the delivery summary about shell.walls extending well beyond these planes.
 *
 * No imports, no textures, no external assets.  Only THREE is used.
 */

const WHITEBOARD = {
  width: 4.0,
  faceHeight: 1.2,
  faceCenterY: 2.1,
  backingWidth: 4.45,
  backingHeight: 1.66,
  backingCenterY: 2.03,
};

const BENCH = {
  length: 3.6,
  depth: 0.5,
  height: 0.78,
  topThickness: 0.06,
  wallClearance: 0.25,
};

const DOOR_WALL_Z = 3.1;
const WINDOW_WALL_Z = -3.16;
const EAST_WALL_X = 5.38; // falls back to door-4's plane when shell has it
const ARCH_CTRL_X = 0.26; // fractions of width / rise for the pointed-arch quadratics
const ARCH_CTRL_Y = 0.8;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

function makeMaterials(THREE) {
  const std = (color, roughness, metalness) =>
    new THREE.MeshStandardMaterial({ color, roughness: roughness ?? 0.8, metalness: metalness ?? 0 });
  return {
    woodBenchTop: std(0x6f4630, 0.6),
    woodBenchTop2: std(0x67402c, 0.63),
    woodBenchFrame: std(0x4a2d20, 0.7),
    woodDoor: std(0x2a1a13, 0.62),
    woodDoorDark: std(0x1e120d, 0.68),
    woodDoorPanel: std(0x372018, 0.55),
    woodTrim: std(0x3b2820, 0.68),
    woodShelf: std(0x54392a, 0.62),
    woodShelfEdge: std(0x3d281d, 0.72),
    woodCabinet: std(0x3b241b, 0.68),
    boardBacking: std(0x1b2b25, 0.86),
    boardChalk: std(0x14211d, 0.9),
    whiteFace: std(0xf2f0e7, 0.42, 0.02),
    aluminium: std(0xb1b5b2, 0.34, 0.68),
    steelPanel: std(0xaeb3af, 0.48, 0.28),
    steelDoor: std(0xbcc0ba, 0.42, 0.32),
    switchWhite: std(0xe7e5da, 0.62),
    blackPlastic: std(0x1d1c1a, 0.5),
    brass: std(0x94703b, 0.38, 0.72),
    iron: std(0x27221d, 0.5, 0.45),
    stone: std(0x5d574b, 0.85),
    creamPaint: std(0xd8d2bd, 0.7),
    creamDoor: std(0xcac6b2, 0.65),
    glassWarm: new THREE.MeshStandardMaterial({
      color: 0xf2e7cc,
      roughness: 0.44,
      metalness: 0.0,
      emissive: 0xc7b389,
      emissiveIntensity: 0.9,
    }),
    glassCool: new THREE.MeshPhysicalMaterial({
      color: 0xbfd1cf,
      roughness: 0.16,
      metalness: 0.0,
      transparent: true,
      opacity: 0.46,
      side: THREE.DoubleSide,
    }),
    paper: std(0xe9e3d1, 0.92),
    slateLabel: std(0x3e4641, 0.8),
  };
}

function nameAndShadow(mesh, name, castShadow = true, receiveShadow = true) {
  mesh.name = name;
  mesh.castShadow = castShadow;
  mesh.receiveShadow = receiveShadow;
  return mesh;
}

function addBox(parent, THREE, material, name, w, h, d, x, y, z, rotX, rotY, rotZ) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  mesh.position.set(x ?? 0, y ?? 0, z ?? 0);
  if (rotX) mesh.rotation.x = rotX;
  if (rotY) mesh.rotation.y = rotY;
  if (rotZ) mesh.rotation.z = rotZ;
  nameAndShadow(mesh, name, true, true);
  parent.add(mesh);
  return mesh;
}

function addCylinder(parent, THREE, material, name, radius, height, x, y, z, segments) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, height, segments ?? 10), material);
  mesh.position.set(x ?? 0, y ?? 0, z ?? 0);
  nameAndShadow(mesh, name, true, true);
  parent.add(mesh);
  return mesh;
}

function addGroup(parent, THREE, name) {
  const g = new THREE.Group();
  g.name = name;
  parent.add(g);
  return g;
}

function quadratic(a, c, b, t) {
  const u = 1 - t;
  return [u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1]];
}

/**
 * Outline of a pointed (two-quadratic) arch, open at the bottom.
 * Points run CCW in local XY: bottom-left -> bottom-right -> up the right
 * jamb -> right shoulder -> apex -> left shoulder -> down the left jamb.
 */
function archPoints(width, bottom, spring, apex, segments = 18) {
  const w = Math.max(0.2, width);
  const rise = Math.max(0.06, apex - spring);
  const pts = [
    [-w / 2, bottom],
    [w / 2, bottom],
    [w / 2, spring],
  ];
  const rightControl = [w * ARCH_CTRL_X, spring + rise * ARCH_CTRL_Y];
  const leftControl = [-w * ARCH_CTRL_X, spring + rise * ARCH_CTRL_Y];
  for (let i = 1; i <= segments; i++) {
    pts.push(quadratic([w / 2, spring], rightControl, [0, apex], i / segments));
  }
  for (let i = 1; i <= segments; i++) {
    pts.push(quadratic([0, apex], leftControl, [-w / 2, spring], i / segments));
  }
  return pts;
}

function shapeFromPoints(THREE, points) {
  return new THREE.Shape(points.map((p) => new THREE.Vector2(p[0], p[1])));
}

function archShape(THREE, width, bottom, spring, apex) {
  return shapeFromPoints(THREE, archPoints(width, bottom, spring, apex));
}

function archHalfWidthAtY(width, spring, apex, y) {
  const half = width / 2;
  if (y <= spring) return half;
  if (y >= apex) return 0;
  const rise = Math.max(0.06, apex - spring);
  const control = [width * ARCH_CTRL_X, spring + rise * ARCH_CTRL_Y];
  // right shoulder -> apex is a quadratic; y is monotone over this segment
  let best = 0;
  let prev = [half, spring];
  for (let i = 1; i <= 60; i++) {
    const p = quadratic([half, spring], control, [0, apex], i / 60);
    if (p[1] >= y) {
      const dy = p[1] - prev[1];
      const t = dy < 1e-6 ? 0 : (y - prev[1]) / dy;
      return Math.abs(prev[0] + (p[0] - prev[0]) * t);
    }
    prev = p;
    best = p[0];
  }
  return Math.max(0, Math.abs(best));
}

function archTopAtX(width, spring, apex, x) {
  const half = width / 2;
  const ax = Math.min(half, Math.abs(x));
  if (ax >= half - 1e-6) return spring;
  const rise = Math.max(0.06, apex - spring);
  const control = [width * ARCH_CTRL_X, spring + rise * ARCH_CTRL_Y];
  let prev = [half, spring];
  for (let i = 1; i <= 80; i++) {
    const p = quadratic([half, spring], control, [0, apex], i / 80);
    if (p[0] <= ax) {
      const dx = p[0] - prev[0];
      const t = Math.abs(dx) < 1e-6 ? 1 : (ax - prev[0]) / dx;
      return prev[1] + (p[1] - prev[1]) * t;
    }
    prev = p;
  }
  return apex;
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------

function buildBench(THREE, M, root, x, z) {
  const g = addGroup(root, THREE, 'Bench_Long_DarkWood');
  g.position.set(x, 0, z);
  g.rotation.y = Math.PI / 2; // length along world Z, wall on the world -X side
  const L = BENCH.length;
  const D = BENCH.depth;
  const H = BENCH.height;

  // three-plank top, with 2 mm shadow gaps between the boards
  const plankDepth = (D - 0.004 * 2) / 3;
  for (let i = 0; i < 3; i++) {
    addBox(
      g, THREE, i === 1 ? M.woodBenchTop : M.woodBenchTop2,
      `Bench_Top_Plank_${i + 1}`,
      L, BENCH.topThickness, plankDepth,
      0, H - BENCH.topThickness / 2, -D / 2 + plankDepth / 2 + i * (plankDepth + 0.004),
    );
  }

  // aprons
  for (const side of [-1, 1]) {
    addBox(g, THREE, M.woodBenchFrame, `Bench_Apron_${side < 0 ? 'Back' : 'Front'}`,
      L - 0.16, 0.1, 0.04, 0, H - BENCH.topThickness - 0.055, side * (D / 2 - 0.05));
  }
  for (const end of [-1, 1]) {
    addBox(g, THREE, M.woodBenchFrame, `Bench_Apron_End_${end < 0 ? 'L' : 'R'}`,
      0.04, 0.1, D - 0.1, end * (L / 2 - 0.09), H - BENCH.topThickness - 0.055, 0);
  }

  // four square legs, inset at the ends; the photo shows a plain square leg
  const legH = H - BENCH.topThickness;
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      addBox(
        g, THREE, M.woodBenchFrame, `Bench_Leg_${sx < 0 ? 'L' : 'R'}${sz < 0 ? 'B' : 'F'}`,
        0.07, legH, 0.07,
        sx * (L / 2 - 0.19), legH / 2, sz * (D / 2 - 0.09),
      );
    }
  }

  // twin long stretchers and cross stretchers near the floor
  for (const side of [-1, 1]) {
    addBox(g, THREE, M.woodBenchFrame, `Bench_Stretcher_${side < 0 ? 'Back' : 'Front'}`,
      L - 0.44, 0.055, 0.05, 0, 0.2, side * (D / 2 - 0.09));
  }
  for (const end of [-1, 1]) {
    addBox(g, THREE, M.woodBenchFrame, `Bench_CrossStretcher_${end < 0 ? 'L' : 'R'}`,
      0.05, 0.055, D - 0.18, end * (L / 2 - 0.19), 0.2, 0);
  }
  addBox(g, THREE, M.woodBenchFrame, 'Bench_MidStretcher', 0.05, 0.05, D - 0.18, 0, 0.2, 0);

  // small black rubber feet, as seen under the legs
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      addBox(
        g, THREE, M.blackPlastic, `Bench_Foot_${sx < 0 ? 'L' : 'R'}${sz < 0 ? 'B' : 'F'}`,
        0.08, 0.018, 0.08,
        sx * (L / 2 - 0.19), 0.009, sz * (D / 2 - 0.09),
      );
    }
  }
  return g;
}

// ---------------------------------------------------------------------------
// whiteboard
// ---------------------------------------------------------------------------

function buildWhiteboard(THREE, M, root, x, z) {
  // The group origin is the wall face; local +Z points into the room.
  const g = addGroup(root, THREE, 'Whiteboard_Main');
  g.position.set(x, 0, z);
  g.rotation.y = Math.PI / 2; // local +Z -> world +X, local +X -> world -Z
  const W = WHITEBOARD;
  const z0 = 0.0;

  addBox(g, THREE, M.boardBacking, 'Whiteboard_DarkBacking', W.backingWidth, W.backingHeight, 0.035,
    0, W.backingCenterY, z0 + 0.018);
  addBox(g, THREE, M.whiteFace, 'Whiteboard_WhiteFace', W.width, W.faceHeight, 0.02,
    0, W.faceCenterY, z0 + 0.052);

  // aluminium frame, 35 mm, proud of the white face
  const f = 0.035;
  const frameTop = W.faceCenterY + W.faceHeight / 2;
  const frameBottom = W.faceCenterY - W.faceHeight / 2;
  addBox(g, THREE, M.aluminium, 'Whiteboard_Frame_Top', W.width + 2 * f, f, 0.04, 0, frameTop + f / 2, z0 + 0.058);
  addBox(g, THREE, M.aluminium, 'Whiteboard_Frame_Bottom', W.width + 2 * f, f, 0.04, 0, frameBottom - f / 2, z0 + 0.058);
  for (const side of [-1, 1]) {
    addBox(g, THREE, M.aluminium, `Whiteboard_Frame_${side < 0 ? 'Left' : 'Right'}`,
      f, W.faceHeight + 2 * f, 0.04, side * (W.width / 2 + f / 2), W.faceCenterY, z0 + 0.058);
  }

  // dark chalk strip under the face and its slim aluminium chalk ledge
  addBox(g, THREE, M.boardChalk, 'Whiteboard_ChalkStrip', W.backingWidth - 0.14, 0.18, 0.016,
    0, 1.345, z0 + 0.048);
  addBox(g, THREE, M.aluminium, 'Whiteboard_ChalkTray', W.backingWidth - 0.08, 0.035, 0.095,
    0, 1.245, z0 + 0.073);
  addBox(g, THREE, M.aluminium, 'Whiteboard_ChalkTrayLip', W.backingWidth - 0.08, 0.02, 0.02,
    0, 1.273, z0 + 0.116);

  // the white screen casing above the board
  addBox(g, THREE, M.aluminium, 'Whiteboard_ScreenCasing', W.backingWidth + 0.06, 0.08, 0.11,
    0, 2.885, z0 + 0.055);
  addBox(g, THREE, M.steelPanel, 'Whiteboard_ScreenCasingCap', W.backingWidth + 0.02, 0.02, 0.13,
    0, 2.932, z0 + 0.06);

  // two hidden wall brackets, visible as small lugs from the side
  for (const side of [-1, 1]) {
    addBox(g, THREE, M.steelPanel, `Whiteboard_Bracket_${side < 0 ? 'L' : 'R'}`,
      0.05, 0.11, 0.035, side * 1.72, 2.79, z0 + 0.002);
  }
  return g;
}

// ---------------------------------------------------------------------------
// wall shelf
// ---------------------------------------------------------------------------

function buildShelf(THREE, M, root, name, x, z, yaw) {
  const g = addGroup(root, THREE, name);
  g.position.set(x, 2.16, z);
  g.rotation.y = yaw;
  addBox(g, THREE, M.woodShelf, `${name}_Top`, 0.52, 0.035, 0.245, 0, 0, 0.122);
  addBox(g, THREE, M.woodShelfEdge, `${name}_FrontEdge`, 0.52, 0.055, 0.018, 0, -0.012, 0.238);
  for (const side of [-1, 1]) {
    const bx = side * 0.19;
    addBox(g, THREE, M.woodShelfEdge, `${name}_Bracket_${side < 0 ? 'L' : 'R'}_WallPlate`,
      0.035, 0.2, 0.045, bx, -0.13, 0.022);
    const brace = addBox(g, THREE, M.woodShelfEdge, `${name}_Bracket_${side < 0 ? 'L' : 'R'}_Brace`,
      0.032, 0.27, 0.032, bx, -0.12, 0.11);
    brace.rotation.x = 0.53;
  }
  return g;
}

// ---------------------------------------------------------------------------
// electricity panel, switch boxes and the small teak wall cabinet
// ---------------------------------------------------------------------------

function buildElectricalPanel(THREE, M, root, x, z) {
  const g = addGroup(root, THREE, 'ElectricalPanel_Assembly');
  g.position.set(x, 0, z);
  g.rotation.y = Math.PI; // local +Z -> world -Z, into the room

  // steel enclosure with a hinged door
  addBox(g, THREE, M.steelPanel, 'Panel_Enclosure', 0.52, 0.66, 0.1, 0, 1.67, 0.06);
  addBox(g, THREE, M.steelPanel, 'Panel_EnclosureRim', 0.56, 0.7, 0.035, 0, 1.67, 0.025);
  addBox(g, THREE, M.steelDoor, 'Panel_HingedDoor', 0.46, 0.6, 0.025, 0.015, 1.67, 0.125);
  addBox(g, THREE, M.steelPanel, 'Panel_Handle', 0.04, 0.1, 0.014, 0.19, 1.67, 0.143);
  for (const side of [-1, 1]) {
    addCylinder(g, THREE, M.brass, `Panel_Hinge_${side < 0 ? 'Lower' : 'Upper'}`, 0.009, 0.07,
      -0.225, 1.67 + side * 0.21, 0.135, 8);
  }
  addBox(g, THREE, M.paper, 'Panel_Label', 0.14, 0.085, 0.006, -0.03, 1.86, 0.14);
  addBox(g, THREE, M.slateLabel, 'Panel_LabelStripe', 0.11, 0.012, 0.002, -0.03, 1.852, 0.144);

  // four white switch boxes in a 2 x 2 block beneath the steel enclosure
  const boxYs = [1.22, 0.98];
  const boxXs = [-0.13, 0.13];
  let boxIndex = 0;
  for (const by of boxYs) {
    for (const bx of boxXs) {
      boxIndex++;
      addBox(g, THREE, M.switchWhite, `SwitchBox_${boxIndex}_Case`, 0.24, 0.17, 0.075, bx, by, 0.05);
      addBox(g, THREE, M.steelPanel, `SwitchBox_${boxIndex}_Lid`, 0.215, 0.145, 0.012, bx, by, 0.09);
      for (let i = 0; i < 3; i++) {
        addBox(g, THREE, M.blackPlastic, `SwitchBox_${boxIndex}_Toggle_${i + 1}`,
          0.018, 0.048, 0.012, bx - 0.065 + i * 0.065, by, 0.1);
      }
      addBox(g, THREE, M.brass, `SwitchBox_${boxIndex}_Screw`, 0.012, 0.012, 0.006, bx + 0.08, by + 0.06, 0.098);
    }
  }

  // vertical conduits from the panel up towards the ceiling, plus a feeder
  addCylinder(g, THREE, M.steelPanel, 'Panel_Conduit_Left', 0.012, 0.95, -0.215, 2.45, 0.035, 8);
  addCylinder(g, THREE, M.steelPanel, 'Panel_Conduit_Right', 0.012, 1.05, 0.12, 2.4, 0.035, 8);
  addCylinder(g, THREE, M.steelPanel, 'Panel_Conduit_Horizontal', 0.011, 0.4, -0.215, 2.93, 0.035, 8)
    .rotation.z = Math.PI / 2;
  addCylinder(g, THREE, M.steelPanel, 'Panel_Feeder', 0.012, 0.55, -0.29, 0.58, 0.03, 8);
  addCylinder(g, THREE, M.steelPanel, 'Panel_Feeder_Elbow', 0.012, 0.16, -0.235, 0.85, 0.03, 8)
    .rotation.z = Math.PI / 2;

  // taped paper notice under the switch gear
  addBox(g, THREE, M.paper, 'Panel_NoticeSheet', 0.28, 0.36, 0.004, -0.02, 0.66, 0.02);
  addBox(g, THREE, M.slateLabel, 'Panel_NoticeLine_1', 0.2, 0.008, 0.002, -0.02, 0.74, 0.024);
  addBox(g, THREE, M.slateLabel, 'Panel_NoticeLine_2', 0.16, 0.008, 0.002, -0.04, 0.71, 0.024);

  // small dark-teak wall cabinet seen below the shelf on the same wall
  const c = addGroup(root, THREE, 'WallCabinet_Teak');
  c.position.set(-4.86, 0, z);
  c.rotation.y = Math.PI;
  addBox(c, THREE, M.woodCabinet, 'WallCabinet_Body', 0.5, 0.56, 0.19, 0, 1.0, 0.095);
  addBox(c, THREE, M.woodShelfEdge, 'WallCabinet_Face', 0.44, 0.5, 0.018, 0, 1.0, 0.199);
  addBox(c, THREE, M.glassCool, 'WallCabinet_Glass', 0.3, 0.34, 0.006, -0.03, 1.03, 0.209);
  addBox(c, THREE, M.woodShelfEdge, 'WallCabinet_GlassFrame_Top', 0.34, 0.02, 0.012, -0.03, 1.21, 0.208);
  addBox(c, THREE, M.woodShelfEdge, 'WallCabinet_GlassFrame_Bottom', 0.34, 0.02, 0.012, -0.03, 0.85, 0.208);
  addBox(c, THREE, M.woodShelfEdge, 'WallCabinet_GlassFrame_Left', 0.02, 0.38, 0.012, -0.19, 1.03, 0.208);
  addBox(c, THREE, M.woodShelfEdge, 'WallCabinet_GlassFrame_Right', 0.02, 0.38, 0.012, 0.13, 1.03, 0.208);
  addCylinder(c, THREE, M.brass, 'WallCabinet_Knob', 0.012, 0.025, 0.18, 1.0, 0.216, 8)
    .rotation.x = Math.PI / 2;
  return g;
}

// ---------------------------------------------------------------------------
// doors
// ---------------------------------------------------------------------------

function buildDarkDoorway(THREE, M, root, spec) {
  const g = addGroup(root, THREE, spec.name);
  g.position.set(spec.x, 0, DOOR_WALL_Z);
  g.rotation.y = Math.PI; // local +Z faces into the room (-Z world)

  const clearW = spec.clearWidth;
  const trim = 0.12;
  const frameDepth = 0.15;
  const spring = spec.springY;
  const apex = spec.apexY;

  // one-piece arched timber door frame with a pointed head
  const outer = archShape(THREE, clearW + trim * 2, 0.0, spring, apex + trim);
  const inner = new THREE.Path(
    archPoints(clearW, 0.02, spring, apex).slice().reverse().map((p) => new THREE.Vector2(p[0], p[1])),
  );
  outer.holes.push(inner);
  const frame = new THREE.Mesh(
    new THREE.ExtrudeGeometry(outer, { depth: frameDepth, bevelEnabled: false, curveSegments: 18 }),
    M.woodTrim,
  );
  frame.position.z = 0;
  nameAndShadow(frame, `${spec.name}_CarvedFrame`, true, true);
  g.add(frame);

  // threshold and a sill under the leaves
  addBox(g, THREE, M.stone, `${spec.name}_Threshold`, clearW + 0.22, 0.035, 0.22, 0, 0.0175, 0.12);
  addBox(g, THREE, M.woodTrim, `${spec.name}_TransomSill`, clearW - 0.02, 0.065, 0.1, 0, spec.transomSill + 0.005, 0.09);

  // door leaves: base slab + 8 raised panels + hinges + handles
  const leafCount = spec.leaves;
  const gap = 0.012;
  const leafW = (clearW - gap * (leafCount - 1)) / leafCount - 0.006;
  const leafBottom = 0.03;
  const leafH = spec.transomSill - 0.035 - leafBottom;
  const leafFaceZ = 0.075;
  const leafDepth = 0.06;
  for (let li = 0; li < leafCount; li++) {
    const lx = -clearW / 2 + leafW / 2 + 0.003 + li * (leafW + gap);
    const leafName = `${spec.name}_Leaf_${li + 1}`;
    addBox(g, THREE, M.woodDoor, `${leafName}_Slab`, leafW, leafH, leafDepth, lx, leafBottom + leafH / 2, leafFaceZ);
    // raised panels, 2 columns x 4 rows
    const cols = 2;
    const rows = 4;
    const sideMargin = 0.07;
    const rowGap = 0.055;
    const pW = (leafW - sideMargin * 2 - 0.035) / cols;
    const usableH = leafH - 0.16 - rowGap * (rows - 1);
    const pH = usableH / rows;
    let pi = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        pi++;
        const px = lx - leafW / 2 + sideMargin + pW / 2 + c * (pW + 0.035);
        const py = leafBottom + 0.08 + pH / 2 + r * (pH + rowGap);
        addBox(g, THREE, M.woodDoorDark, `${leafName}_PanelBorder_${pi}`,
          pW + 0.024, pH + 0.024, 0.007, px, py, leafFaceZ + leafDepth / 2 + 0.002);
        addBox(g, THREE, M.woodDoorPanel, `${leafName}_Panel_${pi}`,
          pW, pH, 0.014, px, py, leafFaceZ + leafDepth / 2 + 0.006);
      }
    }
    // small shadow strip between the two panel columns
    addBox(g, THREE, M.woodDoorDark, `${leafName}_PanelGroove`, 0.014, leafH - 0.18, 0.006,
      lx, leafBottom + leafH / 2, leafFaceZ + leafDepth / 2 + 0.01);
    // brass hinges on the hanging edge
    const hingeSide = leafCount === 1 ? 1 : (li === 0 ? -1 : 1);
    for (let hi = 0; hi < 3; hi++) {
      addBox(g, THREE, M.brass, `${leafName}_Hinge_${hi + 1}`, 0.024, 0.1, 0.016,
        lx + hingeSide * (leafW / 2 - 0.012), 0.45 + hi * 0.63, leafFaceZ + leafDepth / 2 + 0.005);
    }
    // handle: for the double door both leaves get one at the meeting stiles
    const handleSide = leafCount === 1 ? 1 : (li === 0 ? 1 : -1);
    const hx = lx + handleSide * (leafW / 2 - 0.075);
    addBox(g, THREE, M.iron, `${leafName}_HandlePlate`, 0.035, 0.17, 0.014, hx, 1.04, leafFaceZ + leafDepth / 2 + 0.005);
    addBox(g, THREE, M.iron, `${leafName}_HandleLever`, 0.12, 0.025, 0.025, hx - handleSide * 0.045, 1.07, leafFaceZ + leafDepth / 2 + 0.025);
    addBox(g, THREE, M.brass, `${leafName}_Keyhole`, 0.022, 0.032, 0.008, hx, 0.95, leafFaceZ + leafDepth / 2 + 0.006);
  }

  // transom: warm backlit glass, pointed outline, clipped timber lattice
  const transomGlass = new THREE.Mesh(
    new THREE.ShapeGeometry(archShape(THREE, clearW, spec.transomSill, spring, apex), 18),
    M.glassWarm,
  );
  transomGlass.position.z = 0.035;
  nameAndShadow(transomGlass, `${spec.name}_TransomGlass`, true, true);
  g.add(transomGlass);

  const latticeMat = M.woodDoorDark;
  const barZ = 0.08;
  const verticalCount = Math.max(5, Math.round((clearW - 0.1) / 0.1));
  for (let i = 0; i < verticalCount; i++) {
    const bx = -clearW / 2 + 0.06 + (i * (clearW - 0.12)) / (verticalCount - 1);
    const bottom = spec.transomSill + 0.03;
    const top = archTopAtX(clearW, spring, apex, bx) - 0.02;
    if (top - bottom < 0.05) continue;
    addBox(g, THREE, latticeMat, `${spec.name}_TransomBar_V_${i + 1}`, 0.014, top - bottom, 0.018,
      bx, (top + bottom) / 2, barZ);
  }
  const horizontalYs = [
    spec.transomSill + 0.07,
    spring - 0.005,
    spring + (apex - spring) * 0.28,
    spring + (apex - spring) * 0.53,
  ];
  let hi = 0;
  for (const hy of horizontalYs) {
    if (hy >= apex - 0.04) continue;
    const half = archHalfWidthAtY(clearW, spring, apex, hy) - 0.03;
    if (half < 0.06) continue;
    hi++;
    addBox(g, THREE, latticeMat, `${spec.name}_TransomBar_H_${hi}`, half * 2, 0.013, 0.018,
      0, hy, barZ + 0.002);
  }
  // small circular tracery motif near the apex
  const ringY = spring + (apex - spring) * 0.47;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.06, 0.011, 8, 20), latticeMat);
  ring.position.set(0, ringY, barZ + 0.004);
  nameAndShadow(ring, `${spec.name}_TransomRosette`, true, true);
  g.add(ring);
  const petalOffsets = [
    [0.095, 0], [-0.095, 0], [0, 0.095], [0, -0.095],
  ];
  petalOffsets.forEach(([px, py], i) => {
    addBox(g, THREE, latticeMat, `${spec.name}_TransomPetal_${i + 1}`, 0.034, 0.034, 0.017,
      px, ringY + py, barZ + 0.002);
  });

  return g;
}

function buildEastLightDoor(THREE, M, root, z, wallX) {
  const g = addGroup(root, THREE, 'Door_East_Light');
  g.position.set(wallX, 0, z);
  g.rotation.y = -Math.PI / 2; // local +Z -> world -X (into the room)
  const clearW = 0.9;
  const spring = 1.98;
  const apex = 2.5;
  const trim = 0.1;

  const outer = archShape(THREE, clearW + trim * 2, 0.0, spring, apex + trim);
  const inner = new THREE.Path(
    archPoints(clearW, 0.02, spring, apex).slice().reverse().map((p) => new THREE.Vector2(p[0], p[1])),
  );
  outer.holes.push(inner);
  const frame = new THREE.Mesh(
    new THREE.ExtrudeGeometry(outer, { depth: 0.14, bevelEnabled: false, curveSegments: 18 }),
    M.creamPaint,
  );
  nameAndShadow(frame, 'Door_East_Light_ArchedFrame', true, true);
  g.add(frame);

  // solid cream-painted arched leaf
  const leafShape = archShape(THREE, clearW - 0.025, 0.025, spring, apex - 0.015);
  const leaf = new THREE.Mesh(
    new THREE.ExtrudeGeometry(leafShape, { depth: 0.055, bevelEnabled: false, curveSegments: 18 }),
    M.creamDoor,
  );
  leaf.position.z = 0.06;
  nameAndShadow(leaf, 'Door_East_Light_Leaf', true, true);
  g.add(leaf);

  // glazed upper half with a white stile around it
  addBox(g, THREE, M.glassCool, 'Door_East_Light_UpperGlass', 0.5, 0.56, 0.008, 0, 1.72, 0.122);
  addBox(g, THREE, M.creamPaint, 'Door_East_Light_GlassLintel', 0.6, 0.035, 0.02, 0, 2.02, 0.122);
  addBox(g, THREE, M.creamPaint, 'Door_East_Light_GlassSill', 0.6, 0.045, 0.025, 0, 1.42, 0.124);
  for (const side of [-1, 1]) {
    addBox(g, THREE, M.creamPaint, `Door_East_Light_GlassStile_${side < 0 ? 'L' : 'R'}`,
      0.04, 0.62, 0.02, side * 0.28, 1.72, 0.122);
  }
  addBox(g, THREE, M.creamPaint, 'Door_East_Light_UpperRail', 0.72, 0.055, 0.024, 0, 1.25, 0.124);

  // lower raised panels
  for (const side of [-1, 1]) {
    addBox(g, THREE, M.creamDoor, `Door_East_Light_LowerPanel_${side < 0 ? 'L' : 'R'}`,
      0.24, 0.62, 0.014, side * 0.16, 0.8, 0.122);
  }
  addBox(g, THREE, M.creamDoor, 'Door_East_Light_MidRail', 0.8, 0.07, 0.02, 0, 1.15, 0.121);

  // ironmongery
  addBox(g, THREE, M.iron, 'Door_East_Light_HandlePlate', 0.035, 0.16, 0.014, -0.33, 1.04, 0.132);
  addBox(g, THREE, M.iron, 'Door_East_Light_HandleLever', 0.13, 0.025, 0.025, -0.285, 1.07, 0.15);
  for (let i = 0; i < 3; i++) {
    addCylinder(g, THREE, M.iron, `Door_East_Light_Hinge_${i + 1}`, 0.014, 0.11, 0.385, 0.45 + i * 0.65, 0.13, 8);
  }
  addBox(g, THREE, M.stone, 'Door_East_Light_Threshold', clearW + 0.16, 0.03, 0.18, 0, 0.015, 0.13);
  return g;
}

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------

export function buildFurniture(THREE, shell) {
  if (!THREE || !THREE.Group) throw new Error('buildFurniture requires THREE');
  const M = makeMaterials(THREE);
  const root = new THREE.Group();
  root.name = 'Furniture_Fixed';
  root.userData = {
    part: 'furniture',
    description: 'Fixed classroom furniture: bench, whiteboard, shelves, electrical panel, timber doors.',
    units: 'metres',
  };

  const objects = shell && Array.isArray(shell.objects) ? shell.objects : [];
  const byId = (id) => objects.find((o) => o && o.id === id) || null;
  const byKind = (kind) => objects.filter((o) => o && String(o.kind).toLowerCase() === kind);
  const num = (v, fallback) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);

  // --- measured anchors -----------------------------------------------------
  const screen = byId('projector-screen-1') || byKind('projector screen')[0] || null;
  const boardX = num(screen && screen.center && screen.center[0], -5.13);
  const boardZ = num(screen && screen.center && screen.center[2], 0.0);

  const shelves = objects.filter((o) => o && String(o.kind).toLowerCase() === 'shelf');
  const shelfDoor = shelves.find((o) => o.center && o.center[2] > 0) || byId('shelf-1');
  const shelfWindow = shelves.find((o) => o.center && o.center[2] < 0) || byId('shelf-2');

  const doors = byKind('door');
  const doorWest = byId('door-3') || doors.find((o) => o.center && o.center[0] < -1) || null;
  const doorCentreLeafA = byId('door-1') || null;
  const doorCentreLeafB = byId('door-2') || null;

  // derive the centre double-door extents from the two detected leaves
  let centreX = 1.0;
  let centreWidth = 1.24;
  if (doorCentreLeafA && doorCentreLeafB) {
    const ax = doorCentreLeafA.center[0];
    const bx = doorCentreLeafB.center[0];
    const ah = doorCentreLeafA.half[0];
    const bh = doorCentreLeafB.half[0];
    const min = Math.min(ax - ah, bx - bh);
    const max = Math.max(ax + ah, bx + bh);
    centreX = (min + max) / 2;
    centreWidth = Math.max(1.1, Math.min(1.45, max - min));
  } else if (doorCentreLeafA) {
    centreX = doorCentreLeafA.center[0];
    centreWidth = doorCentreLeafA.half[0] * 2;
  }
  // the stained-glass transom marks the centre door axis
  const stained = byId('stained-glass-window-1');
  if (stained && stained.center) centreX = (centreX + stained.center[0]) / 2;

  const westX = num(doorWest && doorWest.center && doorWest.center[0], -3.3);
  const westDoorW = num(doorWest && doorWest.half && doorWest.half[0] * 2, 0.95);
  const eastTransom = byId('window-5');
  const eastDoorObj = byId('door-4');
  const eastWallX = num(eastDoorObj && eastDoorObj.center && eastDoorObj.center[0], EAST_WALL_X);
  // the easternmost door sits close to the corner: keep its jamb inside the east wall
  const eastDoorRawX = num(eastTransom && eastTransom.center && eastTransom.center[0], 5.0);
  const eastDoorX = Math.min(eastDoorRawX, eastWallX - 0.45 - 0.12 - 0.01);

  // --- place the pieces -----------------------------------------------------
  const benchX = boardX + BENCH.wallClearance + BENCH.depth / 2;
  buildBench(THREE, M, root, benchX, boardZ);
  buildWhiteboard(THREE, M, root, boardX - 0.04, boardZ);

  const shelfTopY = 2.16;
  if (shelfDoor && shelfDoor.center) {
    buildShelf(THREE, M, root, 'Shelf_DoorWall', shelfDoor.center[0], DOOR_WALL_Z, Math.PI);
  } else {
    buildShelf(THREE, M, root, 'Shelf_DoorWall', -4.8, DOOR_WALL_Z, Math.PI);
  }
  if (shelfWindow && shelfWindow.center) {
    buildShelf(THREE, M, root, 'Shelf_WindowWall', shelfWindow.center[0], WINDOW_WALL_Z, 0);
  } else {
    buildShelf(THREE, M, root, 'Shelf_WindowWall', -4.75, WINDOW_WALL_Z, 0);
  }
  // keep the object-measured shelf height
  root.children.forEach((child) => {
    if (child.name === 'Shelf_DoorWall') child.position.y = num(shelfDoor && shelfDoor.center && shelfDoor.center[1], shelfTopY);
    if (child.name === 'Shelf_WindowWall') child.position.y = num(shelfWindow && shelfWindow.center && shelfWindow.center[1], shelfTopY);
  });

  // the switch gear sits in the clear strip between the west door and the corner
  const panelX = Math.min(westX - westDoorW / 2 - 0.48, -4.16);
  buildElectricalPanel(THREE, M, root, panelX, DOOR_WALL_Z);

  const doorsGroup = addGroup(root, THREE, 'Doors_DarkTimber');
  doorsGroup.userData = { note: 'leaves, arched frames and lattice transoms; walls.js owns the wall openings' };

  buildDarkDoorway(THREE, M, doorsGroup, {
    name: 'Door_WestSingle',
    x: westX,
    clearWidth: Math.max(0.9, westDoorW),
    leaves: 1,
    transomSill: 2.18,
    springY: 2.36,
    apexY: 2.9,
  });
  buildDarkDoorway(THREE, M, doorsGroup, {
    name: 'Door_CentreDouble',
    x: centreX,
    clearWidth: Math.max(1.18, centreWidth + 0.04),
    leaves: 2,
    transomSill: 2.14,
    springY: 2.34,
    apexY: 2.92,
  });
  buildDarkDoorway(THREE, M, doorsGroup, {
    name: 'Door_EastSingle',
    x: eastDoorX,
    clearWidth: 0.9,
    leaves: 1,
    transomSill: 2.18,
    springY: 2.36,
    apexY: 2.9,
  });
  const eastDoorZ = num(eastDoorObj && eastDoorObj.center && eastDoorObj.center[2], -2.22);
  buildEastLightDoor(THREE, M, root, eastDoorZ, eastWallX);

  // object-box versions of the fixed pieces, so downstream tooling can relate
  // the authored meshes back to the captured ids
  root.userData.shellAnchors = {
    whiteboard: screen ? screen.id : null,
    shelves: [shelfDoor ? shelfDoor.id : null, shelfWindow ? shelfWindow.id : null],
    doors: [doorWest ? doorWest.id : null, doorCentreLeafA ? doorCentreLeafA.id : null, doorCentreLeafB ? doorCentreLeafB.id : null],
    eastDoor: byId('door-4') ? 'door-4' : null,
  };
  root.userData.planes = {
    whiteboardWallX: boardX - 0.04,
    doorWallZ: DOOR_WALL_Z,
    windowWallZ: WINDOW_WALL_Z,
    eastWallX,
    floorY: 0,
    ceilingY: 3.1,
    note: 'wall-mounted pieces are authored with their backs on these planes',
  };
  root.userData.dimensions = {
    bench: [BENCH.length, BENCH.depth, BENCH.height],
    whiteboard: [WHITEBOARD.width, WHITEBOARD.faceHeight],
    wallShelf: [0.52, 0.245],
    doorClearWidths: [westDoorW, centreWidth, 0.9, 0.9],
  };

  return root;
}
