/**
 * lectern.js — the white painted lectern/podium from f001, plus the two pieces
 * of wall dressing that belong over the bench: a school clock and a cork
 * notice board.
 *
 * export buildLectern(THREE) -> THREE.Group
 *
 * The lectern is the pale, slightly wedge-shaped unit standing in front of the
 * tall arched recess on the window side of the room.  Its shell detection
 * (podium-1) is a 0.56 x 0.38 m dark plate at (-3.901, 0.903, -1.557), which
 * reads as the black front panel; the body is 0.55 m deep and sits behind it.
 * That detection fixes the placement below.
 *
 *   · white bent-steel body, 0.68 W x 0.55 D x 1.15 H, with a 14.8 deg sloped
 *     reading top, black bumper lip, black rear rail and black corner trims
 *   · recessed black front panel with a louvred vent, pull and key switch
 *   · two round fan grilles low on each side, rear service inlet and vent
 *   · small dark control plate on the sloped top (screen, buttons, switch)
 *   · black plinth base and a five-cable bundle trailing onto the floor
 *
 * The clock (0.34 m dia, white dial, black bezel) and the cork notice board
 * (0.90 x 0.60 m cork in a 0.94 x 0.64 m teak frame) hang on the west wall.
 * The existing 4.45 m whiteboard assembly leaves only the 0.94 m clear strip
 * between the window-wall corner and the board's north edge; both pieces are
 * fitted into that strip, stacked over the head of the bench, rather than
 * hanging across the whiteboard.  See ROOTNOTE in the returned userData.
 *
 * Coordinates are metres, +Y up, floor y = 0.  No imports, no assets, no DOM
 * access: every surface is a MeshStandardMaterial and every mesh casts and
 * receives shadows.  Three.js core geometry only.
 */

const LECTERN = {
  width: 0.68, // overall X
  depth: 0.55, // overall Z at the top
  height: 1.15, // highest point of the sloped top
  bodyWidth: 0.66,
  bodyDepth: 0.52,
  plinthHeight: 0.055,
  bodyBottom: 0.069,
  bodyTop: 0.995,
  topWidth: 0.68,
  topDepth: 0.55,
  topBottom: 0.985,
  topFrontY: 1.005,
  topRearY: 1.15,
  frontZ: 0.26,
  rearZ: -0.26,
};

const TOP_SLOPE = Math.atan2(
  LECTERN.topRearY - LECTERN.topFrontY,
  LECTERN.topDepth
); // ~14.8 deg
const TOP_MID_Y = (LECTERN.topFrontY + LECTERN.topRearY) / 2;
const SURFACE_FALL = (LECTERN.topRearY - LECTERN.topFrontY) / LECTERN.topDepth;

// podium-1 in shell.json: front panel centred at x = -3.901, z = -1.557.
// Moving back half the body depth puts the body centre at z = -1.832.
const LECTERN_AT = { x: -3.901, y: 0, z: -1.832 };

// The west wall plane, and the clear strip north of the whiteboard backing
// (whiteboard backing spans z = -2.217 .. +2.233, so this strip is 0.943 m).
const WEST_WALL_X = -5.17;
const WALL_ITEM_X = WEST_WALL_X + 0.002;
const WALL_ITEM_Z = -2.6885; // board outer edges: -3.1585 .. -2.2185
const NOTICE_BOARD = { corkW: 0.9, corkH: 0.6, frame: 0.02, centerY: 1.78 };
const CLOCK = { diameter: 0.34, centerY: 2.4 };

/* -------------------------------------------------------------------------- */
/* small deterministic helpers                                                */
/* -------------------------------------------------------------------------- */

function addMesh(parent, THREE, geometry, material, name) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = name;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function addBox(parent, THREE, material, name, w, h, d, x, y, z) {
  const mesh = addMesh(parent, THREE, new THREE.BoxGeometry(w, h, d), material, name);
  mesh.position.set(x, y, z);
  return mesh;
}

/** Cylinder with its axis along X, Y or Z. */
function addCylinder(parent, THREE, material, name, radius, height, x, y, z, axis, segments) {
  const geometry = new THREE.CylinderGeometry(radius, radius, height, segments || 16);
  const mesh = addMesh(parent, THREE, geometry, material, name);
  if (axis === 'x') mesh.rotation.z = Math.PI / 2;
  if (axis === 'z') mesh.rotation.x = Math.PI / 2;
  mesh.position.set(x, y, z);
  return mesh;
}

function addTorus(parent, THREE, material, name, radius, tube, x, y, z, axis) {
  const geometry = new THREE.TorusGeometry(radius, tube, 10, 40);
  const mesh = addMesh(parent, THREE, geometry, material, name);
  if (axis === 'x') mesh.rotation.y = Math.PI / 2;
  mesh.position.set(x, y, z);
  return mesh;
}

function addTube(parent, THREE, material, name, points, radius) {
  const curve = new THREE.CatmullRomCurve3(
    points.map((p) => new THREE.Vector3(p[0], p[1], p[2])),
    false,
    'catmullrom',
    0.35
  );
  const geometry = new THREE.TubeGeometry(curve, 32, radius, 6, false);
  return addMesh(parent, THREE, geometry, material, name);
}

/**
 * Closed wedge/right prism with a flat bottom and a single falling top:
 * width along X, depth along Z, top height yFront at +Z and yRear at -Z.
 * Unshared triangle vertices give flat face normals on a Standard material.
 */
function wedgeGeometry(THREE, width, depth, yBottom, yFront, yRear) {
  const x0 = -width / 2;
  const x1 = width / 2;
  const zf = depth / 2;
  const zr = -depth / 2;
  const v = [
    [x0, yBottom, zf], // 0 front-left, bottom
    [x1, yBottom, zf], // 1 front-right, bottom
    [x1, yBottom, zr], // 2 rear-right, bottom
    [x0, yBottom, zr], // 3 rear-left, bottom
    [x0, yFront, zf], // 4 front-left, top
    [x1, yFront, zf], // 5 front-right, top
    [x1, yRear, zr], // 6 rear-right, top
    [x0, yRear, zr], // 7 rear-left, top
  ];
  const faces = [
    [0, 1, 5], [0, 5, 4], // +Z front
    [1, 2, 6], [1, 6, 5], // +X right
    [2, 3, 7], [2, 7, 6], // -Z back
    [3, 0, 4], [3, 4, 7], // -X left
    [0, 2, 1], [0, 3, 2], // bottom
    [4, 5, 6], [4, 6, 7], // top
  ];
  const positions = new Float32Array(faces.length * 9);
  let o = 0;
  for (const face of faces) {
    for (const index of face) {
      positions[o++] = v[index][0];
      positions[o++] = v[index][1];
      positions[o++] = v[index][2];
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}

/* -------------------------------------------------------------------------- */
/* materials                                                                  */
/* -------------------------------------------------------------------------- */

function makeMaterials(THREE) {
  const std = (name, color, roughness, metalness, extra) =>
    new THREE.MeshStandardMaterial({
      name,
      color,
      roughness,
      metalness,
      ...(extra || {}),
    });
  return {
    enamel: std('lectern_enamelWhite', 0xe5e5dd, 0.46, 0.04),
    enamelShade: std('lectern_enamelShade', 0xd2d3ca, 0.55, 0.03),
    pad: std('lectern_readingPad', 0xb4b9b1, 0.66, 0.03),
    trim: std('lectern_blackTrim', 0x121414, 0.4, 0.18),
    panel: std('lectern_frontPanel', 0x232727, 0.5, 0.14),
    cavity: std('lectern_ventCavity', 0x050606, 0.92, 0.0),
    slat: std('lectern_ventSlat', 0x5a625f, 0.4, 0.45),
    steel: std('lectern_steel', 0x9ea4a0, 0.34, 0.62),
    rubber: std('lectern_rubber', 0x111211, 0.88, 0.0),
    screen: std('lectern_controlScreen', 0x162224, 0.22, 0.1, {
      emissive: 0x1b3034,
      emissiveIntensity: 0.32,
    }),
    ledGreen: std('lectern_ledGreen', 0x1d5a25, 0.34, 0.0, {
      emissive: 0x2c8a39,
      emissiveIntensity: 0.65,
    }),
    cork: std('noticeBoard_cork', 0xb08352, 0.97, 0.0),
    timber: std('noticeBoard_teak', 0x4d3523, 0.63, 0.02),
    timberEdge: std('noticeBoard_teakEdge', 0x3b281b, 0.66, 0.02),
    hardboard: std('noticeBoard_backing', 0x2b2118, 0.92, 0.0),
    brass: std('noticeBoard_brassScrew', 0x96743d, 0.38, 0.7),
    paper: std('noticeBoard_paper', 0xf3f0e4, 0.92, 0.0),
    paperBlue: std('noticeBoard_paperBlue', 0xccdce4, 0.92, 0.0),
    paperYellow: std('noticeBoard_paperYellow', 0xefe6c4, 0.92, 0.0),
    pinRed: std('noticeBoard_pinRed', 0xa6322a, 0.45, 0.05),
    pinBlue: std('noticeBoard_pinBlue', 0x2d4d7a, 0.45, 0.05),
    pinBrass: std('noticeBoard_pinBrass', 0xa98546, 0.4, 0.55),
    clockCase: std('clock_blackCase', 0x141619, 0.4, 0.2),
    clockFace: std('clock_ivoryFace', 0xf2efe3, 0.55, 0.0),
    clockInk: std('clock_blackInk', 0x1c1d1d, 0.52, 0.1),
    clockRed: std('clock_secondHandRed', 0xb0392a, 0.46, 0.05),
  };
}

/* -------------------------------------------------------------------------- */
/* lectern body                                                               */
/* -------------------------------------------------------------------------- */

function buildLecternBody(THREE, M) {
  const L = LECTERN;
  const g = new THREE.Group();
  g.name = 'Lectern_White';

  // ---- black plinth base ---------------------------------------------------
  addBox(g, THREE, M.trim, 'Lectern_Plinth', 0.68, 0.055, 0.55, 0, 0.0275, 0);
  addBox(g, THREE, M.trim, 'Lectern_PlinthReveal', 0.646, 0.016, 0.516, 0, 0.063, 0);

  // ---- sheet-steel carcass -------------------------------------------------
  const bodyHeight = L.bodyTop - L.bodyBottom;
  addBox(
    g,
    THREE,
    M.enamel,
    'Lectern_Body',
    L.bodyWidth,
    bodyHeight,
    L.bodyDepth,
    0,
    L.bodyBottom + bodyHeight / 2,
    0
  );

  // a shallow shadow line where the body meets the plinth
  addBox(g, THREE, M.enamelShade, 'Lectern_BodyShadowLine', 0.662, 0.008, 0.522, 0, 0.076, 0);

  // ---- sloped top ----------------------------------------------------------
  const top = addMesh(
    g,
    THREE,
    wedgeGeometry(THREE, L.topWidth, L.topDepth, L.topBottom, L.topFrontY, L.topRearY),
    M.enamel,
    'Lectern_SlopedTop'
  );
  top.position.z = 0;

  // inset writing pad on the slope; its plane matches the wedge top
  const pad = addBox(g, THREE, M.pad, 'Lectern_ReadingPad', 0.6, 0.005, 0.44, 0, TOP_MID_Y + 0.003, 0.0005);
  pad.rotation.x = TOP_SLOPE;

  // thick rounded black bumper that hangs off the low front edge
  addCylinder(g, THREE, M.trim, 'Lectern_FrontBumper', 0.018, 0.68, 0, 0.993, 0.257, 'x', 16);
  addBox(g, THREE, M.trim, 'Lectern_FrontBumperGasket', 0.68, 0.012, 0.02, 0, 1.003, 0.263);

  // black band along the high rear edge + slim black inlays on the slope's sides
  addBox(g, THREE, M.trim, 'Lectern_TopRearRail', 0.68, 0.03, 0.032, 0, 1.131, -0.259);
  for (const side of [-1, 1]) {
    const tag = side < 0 ? 'L' : 'R';
    const edge = addBox(
      g,
      THREE,
      M.trim,
      'Lectern_TopSideTrim_' + tag,
      0.034,
      0.004,
      0.52,
      side * 0.323,
      TOP_MID_Y + 0.0021,
      0
    );
    edge.rotation.x = TOP_SLOPE;
  }

  // black vertical trims on the two rear corners (as in f037)
  for (const side of [-1, 1]) {
    addBox(
      g,
      THREE,
      M.trim,
      'Lectern_RearCornerTrim_' + (side < 0 ? 'L' : 'R'),
      0.026,
      bodyHeight + 0.008,
      0.026,
      side * 0.33,
      L.bodyBottom + bodyHeight / 2,
      -0.254
    );
  }

  // ---- recessed black front panel -----------------------------------------
  const front = new THREE.Group();
  front.name = 'Lectern_FrontAssembly';
  g.add(front);

  // white raised bevel around the recessed panel
  addBox(front, THREE, M.enamel, 'Lectern_FrontBevel_Top', 0.546, 0.016, 0.02, 0, 0.799, 0.266);
  addBox(front, THREE, M.enamel, 'Lectern_FrontBevel_Bottom', 0.546, 0.016, 0.02, 0, 0.231, 0.266);
  for (const side of [-1, 1]) {
    addBox(
      front,
      THREE,
      M.enamel,
      'Lectern_FrontBevel_' + (side < 0 ? 'Left' : 'Right'),
      0.016,
      0.596,
      0.02,
      side * 0.258,
      0.515,
      0.266
    );
  }

  addBox(front, THREE, M.panel, 'Lectern_FrontBlackPanel', 0.5, 0.56, 0.018, 0, 0.515, 0.266);

  // louvred vent near the bottom of the black panel
  addBox(front, THREE, M.cavity, 'Lectern_FrontVentRecess', 0.32, 0.13, 0.005, 0, 0.32, 0.2765);
  for (let i = 0; i < 6; i++) {
    addBox(
      front,
      THREE,
      M.slat,
      'Lectern_FrontVentSlat_' + (i + 1),
      0.29,
      0.0095,
      0.008,
      0,
      0.272 + i * 0.019,
      0.2805
    );
  }
  for (const side of [-1, 1]) {
    addBox(
      front,
      THREE,
      M.steel,
      'Lectern_FrontVentPost_' + (side < 0 ? 'L' : 'R'),
      0.006,
      0.13,
      0.011,
      side * 0.16,
      0.32,
      0.28
    );
  }

  // recessed pull and keyed switch, echoing the cabinet door in the frames
  addBox(front, THREE, M.cavity, 'Lectern_FrontPullRecess', 0.15, 0.045, 0.009, 0, 0.70, 0.2785);
  addBox(front, THREE, M.steel, 'Lectern_FrontPullBar', 0.122, 0.014, 0.012, 0, 0.70, 0.282);
  addCylinder(front, THREE, M.steel, 'Lectern_FrontKeyEscutcheon', 0.016, 0.007, 0.175, 0.747, 0.279, 'z', 18);
  addBox(front, THREE, M.cavity, 'Lectern_FrontKeyhole', 0.01, 0.004, 0.004, 0.175, 0.747, 0.2835);

  // pale upper band: perforated speaker vent and a power LED
  addCylinder(front, THREE, M.cavity, 'Lectern_SpeakerGrille', 0.038, 0.006, 0.215, 0.905, 0.2615, 'z', 26);
  addTorus(front, THREE, M.steel, 'Lectern_SpeakerRing', 0.043, 0.004, 0.215, 0.905, 0.2645, 'z');
  addCylinder(front, THREE, M.ledGreen, 'Lectern_PowerLED', 0.0055, 0.004, -0.225, 0.905, 0.2625, 'z', 14);

  // subtle white seams across the upper front (access door joints in the frames)
  addBox(front, THREE, M.enamelShade, 'Lectern_UpperDoorSeam_Top', 0.54, 0.0025, 0.003, 0, 0.94, 0.262);
  addBox(front, THREE, M.enamelShade, 'Lectern_UpperDoorSeam_Bottom', 0.54, 0.0025, 0.003, 0, 0.86, 0.262);

  // ---- side seams and round fan grilles -----------------------------------
  for (const side of [-1, 1]) {
    const tag = side < 0 ? 'L' : 'R';
    const sx = side * 0.3315;
    addBox(g, THREE, M.enamelShade, 'Lectern_SideSeam_Top_' + tag, 0.003, 0.004, 0.35, sx, 0.835, -0.022);
    addBox(g, THREE, M.enamelShade, 'Lectern_SideSeam_Bottom_' + tag, 0.003, 0.004, 0.35, sx, 0.258, -0.022);
    addBox(g, THREE, M.enamelShade, 'Lectern_SideSeam_Front_' + tag, 0.003, 0.577, 0.004, sx, 0.547, 0.153);
    addBox(g, THREE, M.enamelShade, 'Lectern_SideSeam_Rear_' + tag, 0.003, 0.577, 0.004, sx, 0.547, -0.197);

    const vents = [
      { name: 'LowFront', y: 0.175, z: 0.125 },
      { name: 'LowRear', y: 0.275, z: -0.105 },
    ];
    for (const vent of vents) {
      addCylinder(
        g,
        THREE,
        M.cavity,
        'Lectern_SideGrille_' + tag + '_' + vent.name,
        0.047,
        0.007,
        sx + side * 0.001,
        vent.y,
        vent.z,
        'x',
        26
      );
      addTorus(
        g,
        THREE,
        M.steel,
        'Lectern_SideGrilleRing_' + tag + '_' + vent.name,
        0.048,
        0.0035,
        sx + side * 0.0045,
        vent.y,
        vent.z,
        'x'
      );
    }
  }

  // ---- rear face: vent, inlet, switch, cable gland -------------------------
  addBox(g, THREE, M.cavity, 'Lectern_RearVentRecess', 0.26, 0.13, 0.006, 0, 0.6, -0.2635);
  for (let i = 0; i < 5; i++) {
    addBox(g, THREE, M.slat, 'Lectern_RearVentSlat_' + (i + 1), 0.23, 0.009, 0.007, 0, 0.558 + i * 0.021, -0.266);
  }

  addBox(g, THREE, M.trim, 'Lectern_RearInletPlate', 0.1, 0.065, 0.008, -0.19, 0.35, -0.264);
  addBox(g, THREE, M.cavity, 'Lectern_RearIECSocket', 0.055, 0.035, 0.006, -0.19, 0.35, -0.268);
  addBox(g, THREE, M.trim, 'Lectern_RearPowerSwitch', 0.026, 0.018, 0.014, -0.09, 0.35, -0.269);

  addCylinder(g, THREE, M.rubber, 'Lectern_CableGland', 0.021, 0.028, 0.17, 0.155, -0.27, 'z', 18);
  addCylinder(g, THREE, M.rubber, 'Lectern_CableGlandBoot', 0.026, 0.012, 0.17, 0.155, -0.284, 'z', 18);

  // ---- control plate on the sloped top ------------------------------------
  const control = new THREE.Group();
  control.name = 'Lectern_ControlPanel';
  control.position.set(0.205, TOP_MID_Y + 0.1 * SURFACE_FALL + 0.0015, -0.1);
  control.rotation.x = TOP_SLOPE;
  g.add(control);

  addBox(control, THREE, M.trim, 'Lectern_Control_Base', 0.18, 0.022, 0.11, 0, 0.011, 0);
  addBox(control, THREE, M.panel, 'Lectern_Control_Face', 0.168, 0.007, 0.098, 0, 0.0255, 0);
  addBox(control, THREE, M.steel, 'Lectern_Control_ScreenBezel', 0.086, 0.002, 0.047, -0.025, 0.0305, 0.008);
  addBox(control, THREE, M.screen, 'Lectern_Control_Screen', 0.078, 0.002, 0.039, -0.025, 0.0315, 0.008);
  for (let i = 0; i < 3; i++) {
    addCylinder(
      control,
      THREE,
      M.steel,
      'Lectern_Control_Button_' + (i + 1),
      0.0062,
      0.005,
      0.038 + i * 0.017,
      0.032,
      0.018,
      'y',
      12
    );
  }
  addCylinder(control, THREE, M.ledGreen, 'Lectern_Control_LED', 0.004, 0.003, 0.038, 0.031, -0.029, 'y', 10);
  addBox(control, THREE, M.trim, 'Lectern_Control_Rocker', 0.018, 0.005, 0.012, 0.065, 0.0305, -0.024);

  // ---- cable bundle onto the floor ----------------------------------------
  const cables = new THREE.Group();
  cables.name = 'Lectern_Cables';
  g.add(cables);

  const cableSpecs = [
    { name: 'A', radius: 0.0068, dx: -0.02, tail: -0.62, loop: false },
    { name: 'B', radius: 0.0078, dx: -0.007, tail: -0.642, loop: false },
    { name: 'C', radius: 0.0058, dx: 0.006, tail: -0.606, loop: false },
    { name: 'D', radius: 0.0064, dx: 0.019, tail: -0.63, loop: false },
    { name: 'E', radius: 0.0046, dx: 0.004, tail: -0.56, loop: true },
  ];

  for (const spec of cableSpecs) {
    const x0 = 0.17 + spec.dx * 0.4;
    const x1 = 0.17 + spec.dx * 0.8;
    const x2 = 0.178 + spec.dx;
    const x3 = 0.193 + spec.dx * 1.2;
    const x4 = 0.215 + spec.dx * 1.4;
    const x5 = 0.245 + spec.dx * 1.6;
    const x6 = 0.275 + spec.dx * 1.8;
    const x7 = 0.293 + spec.dx * 2;
    const points = spec.loop
      ? [
          [0.17, 0.155, -0.279],
          [0.158, 0.11, -0.314],
          [0.125, 0.055, -0.336],
          [0.08, 0.014, -0.374],
          [0.055, 0.0075, -0.435],
          [0.085, 0.0075, -0.49],
          [0.155, 0.0075, -0.53],
          [0.235, 0.0075, -0.56],
        ]
      : [
          [x0, 0.155, -0.279],
          [x1, 0.11, -0.316],
          [x2, 0.052, -0.345],
          [x3, 0.014, -0.397],
          [x4, 0.0075, -0.462],
          [x5, 0.0075, -0.53],
          [x6, 0.0075, -0.59],
          [x7, 0.0075, spec.tail],
        ];
    addTube(cables, THREE, M.rubber, 'Lectern_Cable_' + spec.name, points, spec.radius);

    const endPoint = points[points.length - 1];
    const end = new THREE.Mesh(new THREE.SphereGeometry(spec.radius, 8, 6), M.rubber);
    end.name = 'Lectern_CableEnd_' + spec.name;
    end.position.set(endPoint[0], endPoint[1], endPoint[2]);
    end.castShadow = true;
    end.receiveShadow = true;
    cables.add(end);
  }

  // short rubber wrap where the five leads leave the gland
  const cableWrap = addCylinder(cables, THREE, M.rubber, 'Lectern_CableWrap', 0.022, 0.085, 0.172, 0.127, -0.3, 'y', 14);
  cableWrap.rotation.x = 0.62;
  cableWrap.rotation.z = 0.04;

  // slim black floor cable clip where the bundle meets the terrazzo
  const clip = addBox(cables, THREE, M.trim, 'Lectern_FloorCableClip', 0.14, 0.007, 0.045, 0.245, 0.004, -0.575);
  clip.rotation.y = 1.02;

  g.userData = {
    part: 'lectern',
    description: 'White painted lectern with sloped top, black front panel, control plate and cable bundle.',
    units: 'metres',
    facing: '+Z (front face looks into the room)',
    dimensions: {
      overall: [L.width, L.height, L.depth],
      body: [L.bodyWidth, L.bodyTop - L.bodyBottom, L.bodyDepth],
      slopedTop: [L.topWidth, L.topDepth],
      topSlopeDeg: Number(((TOP_SLOPE * 180) / Math.PI).toFixed(1)),
      plinth: [0.68, 0.055, 0.55],
    },
    placement: { x: LECTERN_AT.x, z: LECTERN_AT.z },
    reference: 'shell.json podium-1: dark front plate at (-3.901, 0.903, -1.557)',
  };
  return g;
}

/* -------------------------------------------------------------------------- */
/* wall clock                                                                 */
/* -------------------------------------------------------------------------- */

function addClockHand(parent, THREE, material, name, width, length, tail, angle, z) {
  const geometry = new THREE.BoxGeometry(width, length, 0.004);
  geometry.translate(0, length / 2 - tail, 0);
  const mesh = addMesh(parent, THREE, geometry, material, name);
  mesh.position.z = z;
  mesh.rotation.z = angle;
  return mesh;
}

function buildClock(THREE, M) {
  const g = new THREE.Group();
  g.name = 'Wall_Clock';

  // black case; its back sits on the wall plane (local z = 0)
  addCylinder(g, THREE, M.clockCase, 'Clock_Case', 0.168, 0.028, 0, 0, 0.014, 'z', 44);
  addCylinder(g, THREE, M.clockFace, 'Clock_Face', 0.146, 0.012, 0, 0, 0.032, 'z', 44);
  addTorus(g, THREE, M.clockCase, 'Clock_BlackRim', 0.156, 0.014, 0, 0, 0.039, 'z');

  // fine inner minute track
  addTorus(g, THREE, M.clockInk, 'Clock_MinuteTrack', 0.137, 0.0014, 0, 0, 0.0385, 'z');

  for (let i = 0; i < 12; i++) {
    const angle = (i * Math.PI) / 6;
    const quarter = i % 3 === 0;
    const tick = new THREE.Mesh(
      new THREE.BoxGeometry(quarter ? 0.006 : 0.0038, quarter ? 0.025 : 0.015, 0.003),
      M.clockInk
    );
    tick.name = 'Clock_Tick_' + String(i + 1).padStart(2, '0');
    tick.position.set(Math.sin(angle) * 0.121, Math.cos(angle) * 0.121, 0.0415);
    tick.rotation.z = -angle;
    tick.castShadow = true;
    tick.receiveShadow = true;
    g.add(tick);
  }

  // fixed at ten past ten for a legible, symmetrical reading
  addClockHand(g, THREE, M.clockInk, 'Clock_HourHand', 0.014, 0.085, 0.012, -5.323, 0.0445);
  addClockHand(g, THREE, M.clockInk, 'Clock_MinuteHand', 0.009, 0.118, 0.014, -1.103, 0.0465);
  addClockHand(g, THREE, M.clockRed, 'Clock_SecondHand', 0.0032, 0.132, 0.022, -3.351, 0.049);
  addCylinder(g, THREE, M.clockCase, 'Clock_CenterCap', 0.012, 0.01, 0, 0, 0.053, 'z', 18);

  g.userData = {
    part: 'clock',
    description: 'School wall clock, white dial and black rim.',
    units: 'metres',
    diameter: 0.34,
    depth: 0.058,
    facing: '+X (into the room)',
  };
  return g;
}

/* -------------------------------------------------------------------------- */
/* cork notice board                                                          */
/* -------------------------------------------------------------------------- */

function addNoticePaper(THREE, parent, name, w, h, x, y, rotation, paperMaterial, pinMaterial) {
  const paperGroup = new THREE.Group();
  paperGroup.name = name;
  paperGroup.position.set(x, y, 0);
  paperGroup.rotation.z = rotation;
  parent.add(paperGroup);

  addBox(paperGroup, THREE, paperMaterial, name + '_Sheet', w, h, 0.0025, 0, 0, 0.0215);
  const pin = new THREE.Mesh(new THREE.SphereGeometry(0.0065, 8, 6), pinMaterial);
  pin.name = name + '_Pin';
  pin.position.set(0, h / 2 - 0.017, 0.0255);
  pin.scale.z = 0.55;
  pin.castShadow = true;
  pin.receiveShadow = true;
  paperGroup.add(pin);
}

function buildNoticeBoard(THREE, M) {
  const B = NOTICE_BOARD;
  const g = new THREE.Group();
  g.name = 'Wall_NoticeBoard';

  // hardboard backing and the cork field, both flush on the wall
  addBox(g, THREE, M.hardboard, 'NoticeBoard_Backing', 0.94, 0.64, 0.008, 0, 0, 0.004);
  addBox(g, THREE, M.cork, 'NoticeBoard_Cork', B.corkW, B.corkH, 0.012, 0, 0, 0.014);

  // teak frame: 20 mm face, 34 mm deep, simply butted at the corners
  addBox(g, THREE, M.timber, 'NoticeBoard_Frame_Top', 0.94, 0.02, 0.034, 0, 0.31, 0.021);
  addBox(g, THREE, M.timber, 'NoticeBoard_Frame_Bottom', 0.94, 0.02, 0.034, 0, -0.31, 0.021);
  for (const side of [-1, 1]) {
    addBox(
      g,
      THREE,
      M.timber,
      'NoticeBoard_Frame_' + (side < 0 ? 'Left' : 'Right'),
      0.02,
      0.6,
      0.034,
      side * 0.46,
      0,
      0.021
    );
  }

  // slim inner lip so the cork reads as recessed behind the frame
  addBox(g, THREE, M.timberEdge, 'NoticeBoard_InnerLip_Top', 0.9, 0.004, 0.006, 0, 0.298, 0.025);
  addBox(g, THREE, M.timberEdge, 'NoticeBoard_InnerLip_Bottom', 0.9, 0.004, 0.006, 0, -0.298, 0.025);

  // brass fixing screws in the top and bottom rails
  for (const side of [-1, 1]) {
    for (const top of [-1, 1]) {
      addCylinder(
        g,
        THREE,
        M.brass,
        'NoticeBoard_Screw_' + (top < 0 ? 'Bottom' : 'Top') + (side < 0 ? 'Left' : 'Right'),
        0.0055,
        0.004,
        side * 0.4,
        top * 0.31,
        0.0395,
        'z',
        12
      );
    }
  }

  // pinned notices; deliberately a little untidy
  addNoticePaper(THREE, g, 'NoticeBoard_Notice_1', 0.16, 0.22, -0.245, 0.07, 0.055, M.paper, M.pinRed);
  addNoticePaper(THREE, g, 'NoticeBoard_Notice_2', 0.19, 0.12, 0.14, 0.14, -0.035, M.paperBlue, M.pinBlue);
  addNoticePaper(THREE, g, 'NoticeBoard_Notice_3', 0.14, 0.18, 0.205, -0.105, 0.085, M.paper, M.pinBrass);
  addNoticePaper(THREE, g, 'NoticeBoard_Notice_4', 0.12, 0.13, -0.085, -0.155, -0.11, M.paperYellow, M.pinRed);

  g.userData = {
    part: 'notice-board',
    description: 'Cork notice board in a dark teak frame, four pinned notices.',
    units: 'metres',
    cork: [B.corkW, B.corkH],
    outer: [B.corkW + 2 * B.frame, B.corkH + 2 * B.frame],
    depth: 0.042,
    facing: '+X (into the room)',
  };
  return g;
}

/* -------------------------------------------------------------------------- */
/* entry point                                                                */
/* -------------------------------------------------------------------------- */

export function buildLectern(THREE) {
  if (!THREE || !THREE.Group || !THREE.Mesh) {
    throw new Error('buildLectern(THREE): THREE must be supplied');
  }

  const M = makeMaterials(THREE);
  const root = new THREE.Group();
  root.name = 'Lectern_Set';

  const lectern = buildLecternBody(THREE, M);
  lectern.position.set(LECTERN_AT.x, LECTERN_AT.y, LECTERN_AT.z);
  root.add(lectern);

  const clock = buildClock(THREE, M);
  clock.position.set(WALL_ITEM_X, CLOCK.centerY, WALL_ITEM_Z);
  clock.rotation.y = Math.PI / 2; // local +Z faces into the room (+X)
  root.add(clock);

  const noticeBoard = buildNoticeBoard(THREE, M);
  noticeBoard.position.set(WALL_ITEM_X, NOTICE_BOARD.centerY, WALL_ITEM_Z);
  noticeBoard.rotation.y = Math.PI / 2;
  root.add(noticeBoard);

  let meshCount = 0;
  root.traverse((child) => {
    if (child && child.isMesh) meshCount += 1;
  });

  root.userData = {
    part: 'lectern',
    description:
      'White classroom lectern with sloped top and cable bundle; wall clock and cork notice board above the head of the bench.',
    units: 'metres',
    floorY: 0,
    meshCount,
    placements: {
      lectern: { position: [LECTERN_AT.x, 0, LECTERN_AT.z], yaw: 0, facing: '+Z' },
      clock: { position: [WALL_ITEM_X, CLOCK.centerY, WALL_ITEM_Z], yaw: Math.PI / 2, diameter: CLOCK.diameter },
      noticeBoard: {
        position: [WALL_ITEM_X, NOTICE_BOARD.centerY, WALL_ITEM_Z],
        yaw: Math.PI / 2,
        outer: [0.94, 0.64],
      },
    },
    dimensions: {
      lectern: [LECTERN.width, LECTERN.height, LECTERN.depth],
      clockDiameter: CLOCK.diameter,
      noticeBoardCork: [NOTICE_BOARD.corkW, NOTICE_BOARD.corkH],
      noticeBoardOuter: [0.94, 0.64],
    },
    ROOTNOTE:
      'The whiteboard assembly occupies the west wall above the bench (backing 4.45 x 1.66 m, ' +
      'z = -2.217 .. +2.233).  Clock and notice board are therefore hung in the 0.943 m clear strip ' +
      'between the window-wall corner (z = -3.16) and the whiteboard edge, above the north head of ' +
      'the bench.  Shift both wall groups together if the bench run moves.',
  };

  return root;
}
