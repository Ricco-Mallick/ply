/**
 * windows.js — the grilled window recesses and the split ACs of the captured classroom.
 *
 * export buildWindows(THREE) -> THREE.Group
 *
 * The capture frames show one long window wall (f001/f002/f003/f005/f007/
 * f008/f022/f030/f040): three tall, shallow-arched plaster relief panels with a
 * barred window in the upper half, a blank panel below and a dark stone sill;
 * white split-AC boxes sit high on the wall between the panels with a white
 * conduit running up to the ceiling.
 *
 * Envelope and placement
 *   · window (south) wall inside face ........ z = -3.16
 *   · relief panel face plane ................ z = -3.10  (60 mm proud of the wall)
 *   · window bays ............................ x = -3.20, +0.20, +3.40
 *   · split AC units ......................... x = -2.00, +1.40, +4.60, centre y = 2.35
 *   · floor y = 0, ceiling y = 3.10, metres, +Y up
 *
 * The window wall is solid in the captured shell (no openings were detected in
 * it), so the windows are authored as relief: the arched panel projects 60 mm
 * and the glazing sits 25 mm behind the panel field.  All of the window
 * geometry therefore lives between the wall face and z = -3.055 (the sill
 * nose) — nothing is buried in the masonry and no wall patch is needed.
 *
 * Window unit, dimensions taken from the frames and the brief
 *   panel ................. 0.95 w x 2.40 h, segmental head springing at 2.20
 *                          with a 0.20 rise, raised architrave 60 mm wide
 *   window ................ 0.75 w x 1.15 h, sill 1.05, head 2.20
 *   frame ................. 25 mm painted face, 698 x 1098 mm clear opening
 *   grille ................ dark metal, 7 vertical x 8 horizontal bars + frame
 *   glazing ............... 742 x 1142 mm, 25 mm behind the panel field
 *   sill .................. stone, 0.90 x 0.05 x 0.103, 45 mm proud of the face
 *   lower panel ........... 0.76 x 0.90 raised moulding, 0.72 x 0.86 field,
 *                          small louvred vent at y = 0.40
 *
 * AC unit
 *   shell ................. 0.90 w x 0.30 h x 0.25 d, rounded white body
 *   louvre slot ........... 0.76 x 0.072 dark outlet with 3 blades and a flap
 *   brand strip ........... 140 mm silver strip + 36 mm blue badge, status LED
 *   conduit ............... 45 mm white trunking from the shell top (2.50) to
 *                          the ceiling, then 0.35 m along the wall at 3.09
 *
 * Plain ES module: no imports, no textures, no external assets, no DOM access.
 * Only MeshStandardMaterial is used.  Every mesh is named and participates in
 * shadows.  Mesh budget: 32 per window x 3 + 16 per AC x 3 = 144.
 */

const WALL_Z = -3.16;
const FACE_Z = -3.1;
const WINDOW_XS = [-3.2, 0.2, 3.4];
const AC_XS = [-2.0, 1.4, 4.6];

const WINDOW = {
  panelWidth: 0.95,
  panelHeight: 2.4,
  springY: 2.2,
  archRise: 0.2,
  borderFace: 0.06,
  windowWidth: 0.75,
  windowHeight: 1.15,
  sillY: 1.05,
  sillWidth: 0.9,
  sillThickness: 0.05,
  sillDepth: 0.103,
  sillFrontZ: 0.045,
  panelThickness: 0.027,
  panelBackZ: -0.058,
  fieldZ: -0.031,
  borderDepth: 0.03,
  frameFace: 0.025,
  frameZ: -0.026,
  grilleZ: -0.045,
  glassZ: -0.0555,
};

const AC = {
  width: 0.9,
  height: 0.3,
  depth: 0.25,
  centreY: 2.35,
  corner: 0.042,
  bevel: 0.006,
  bodyZ: 0.006,
  louvreY: -0.095,
  conduitX: 0.36,
  conduitTop: 3.09,
};

/* ------------------------------------------------------------------------- */
/* materials                                                                 */
/* ------------------------------------------------------------------------- */

function createMaterials(THREE) {
  const standard = (name, color, roughness, metalness, extra) => {
    const material = new THREE.MeshStandardMaterial(
      Object.assign({ color, roughness, metalness: metalness || 0 }, extra || {})
    );
    material.name = name;
    return material;
  };

  return {
    // pale cream plaster, a touch whiter than walls_plaster (0xdcd6c6)
    panel: standard('windows_panelPlaster', 0xe0dbc9, 0.95, 0.0),
    border: standard('windows_architravePlaster', 0xe8e3d3, 0.9, 0.0),
    lowerField: standard('windows_lowerPanelField', 0xd8d2c1, 0.96, 0.0),
    frame: standard('windows_paintedFrame', 0xe9e6da, 0.56, 0.02),
    stone: standard('windows_sillStone', 0x747065, 0.74, 0.02),
    sillShadow: standard('windows_sillUndercut', 0x4f4c44, 0.88, 0.0),
    grille: standard('windows_grilleIron', 0x484338, 0.34, 0.58),
    glazing: standard('windows_glazing', 0x0d1416, 0.18, 0.35, {
      emissive: 0x0a1113,
      emissiveIntensity: 0.5,
    }),
    vent: standard('windows_ventIron', 0x3b3c37, 0.5, 0.35),
    ventSlat: standard('windows_ventSlat', 0x5b5b52, 0.55, 0.3),

    acShell: standard('windows_acShell', 0xf2f0e8, 0.5, 0.01),
    acPlate: standard('windows_acWallPlate', 0xd7d4c9, 0.7, 0.02),
    acSlot: standard('windows_acOutlet', 0x232524, 0.55, 0.2),
    acBlade: standard('windows_acLouvre', 0xd9d6ca, 0.5, 0.03),
    acBrand: standard('windows_acBrandStrip', 0xaeb2b4, 0.36, 0.55),
    acBadge: standard('windows_acBadge', 0x2b5f96, 0.4, 0.15),
    acLed: standard('windows_acLed', 0x2c3a2a, 0.35, 0.0, {
      emissive: 0x7ed46a,
      emissiveIntensity: 1.5,
    }),
    conduit: standard('windows_conduit', 0xebe8de, 0.62, 0.01),
    conduitClip: standard('windows_conduitClip', 0xd2cfc5, 0.6, 0.05),
  };
}

/* ------------------------------------------------------------------------- */
/* geometry helpers                                                          */
/* ------------------------------------------------------------------------- */

function addMesh(THREE, parent, geometry, material, name) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = name;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function addBox(THREE, parent, material, name, w, h, d, x, y, z) {
  const mesh = addMesh(
    THREE,
    parent,
    new THREE.BoxGeometry(w, h, d),
    material,
    name
  );
  mesh.position.set(x, y, z);
  return mesh;
}

/**
 * Arch outline points, clockwise, from the bottom-left corner up the jamb,
 * across a three-centred-ish segmental head (ellipse sampled) and down the
 * right jamb.  Sampling 32 points keeps the shallow head smooth.
 */
function archPoints(halfWidth, spring, rise, baseY, steps) {
  const count = steps || 32;
  const pts = [
    { x: -halfWidth, y: baseY },
    { x: -halfWidth, y: spring },
  ];
  for (let i = 1; i < count; i++) {
    const theta = Math.PI * (1 - i / count);
    pts.push({
      x: halfWidth * Math.cos(theta),
      y: spring + rise * Math.sin(theta),
    });
  }
  pts.push({ x: halfWidth, y: spring });
  pts.push({ x: halfWidth, y: baseY });
  return pts;
}

function shapeFromPoints(THREE, pts) {
  const shape = new THREE.Shape();
  shape.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) shape.lineTo(pts[i].x, pts[i].y);
  shape.closePath();
  return shape;
}

function pathFromPoints(THREE, pts) {
  const path = new THREE.Path();
  path.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) path.lineTo(pts[i].x, pts[i].y);
  path.closePath();
  return path;
}

/** Extrude a shape from `backZ` towards +Z by `depth`. */
function extrudeShape(THREE, parent, material, name, shape, depth, backZ) {
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth,
    steps: 1,
    curveSegments: 8,
    bevelEnabled: false,
  });
  const mesh = addMesh(THREE, parent, geometry, material, name);
  mesh.position.z = backZ;
  return mesh;
}

/** Rectangular picture frame, centred on the origin, as one extruded ring. */
function rectRing(THREE, outerWidth, outerHeight, innerWidth, innerHeight) {
  const ow = outerWidth / 2;
  const oh = outerHeight / 2;
  const iw = innerWidth / 2;
  const ih = innerHeight / 2;
  const shape = new THREE.Shape();
  shape.moveTo(-ow, -oh);
  shape.lineTo(ow, -oh);
  shape.lineTo(ow, oh);
  shape.lineTo(-ow, oh);
  shape.closePath();

  // opposite winding, so the hole is subtracted cleanly
  const hole = new THREE.Path();
  hole.moveTo(-iw, -ih);
  hole.lineTo(-iw, ih);
  hole.lineTo(iw, ih);
  hole.lineTo(iw, -ih);
  hole.closePath();
  shape.holes.push(hole);
  return shape;
}

function roundedRect(THREE, width, height, radius) {
  const hw = width / 2;
  const hh = height / 2;
  const r = Math.max(0.001, Math.min(radius, hw - 0.002, hh - 0.002));
  const shape = new THREE.Shape();
  shape.moveTo(-hw + r, -hh);
  shape.lineTo(hw - r, -hh);
  shape.quadraticCurveTo(hw, -hh, hw, -hh + r);
  shape.lineTo(hw, hh - r);
  shape.quadraticCurveTo(hw, hh, hw - r, hh);
  shape.lineTo(-hw + r, hh);
  shape.quadraticCurveTo(-hw, hh, -hw, hh - r);
  shape.lineTo(-hw, -hh + r);
  shape.quadraticCurveTo(-hw, -hh, -hw + r, -hh);
  shape.closePath();
  return shape;
}

/* ------------------------------------------------------------------------- */
/* one window recess                                                         */
/* ------------------------------------------------------------------------- */

function buildWindow(THREE, MAT, name, x) {
  const group = new THREE.Group();
  group.name = name;
  group.position.set(x, 0, FACE_Z);
  group.userData = {
    kind: 'windowRecess',
    bayCentreX: x,
    wallZ: WALL_Z,
    faceZ: FACE_Z,
    panelWidth: WINDOW.panelWidth,
    panelHeight: WINDOW.panelHeight,
    windowWidth: WINDOW.windowWidth,
    windowHeight: WINDOW.windowHeight,
    sillY: WINDOW.sillY,
    note: 'arched plaster relief, dark grille over glazing, stone sill, blank lower panel',
  };

  const half = WINDOW.panelWidth / 2;
  const halfWin = WINDOW.windowWidth / 2;
  const winTop = WINDOW.sillY + WINDOW.windowHeight;

  /* --- arched plaster slab, with the window opening cut through it ------- */
  const slab = shapeFromPoints(
    THREE,
    archPoints(half, WINDOW.springY, WINDOW.archRise, 0, 34)
  );
  slab.holes.push(
    pathFromPoints(THREE, [
      { x: -halfWin, y: WINDOW.sillY },
      { x: -halfWin, y: winTop },
      { x: halfWin, y: winTop },
      { x: halfWin, y: WINDOW.sillY },
    ])
  );
  extrudeShape(
    THREE,
    group,
    MAT.panel,
    name + '_Panel',
    slab,
    WINDOW.panelThickness,
    WINDOW.panelBackZ
  );

  /* --- raised architrave ring (60 mm face, 30 mm proud) ------------------ */
  const border = shapeFromPoints(
    THREE,
    archPoints(half, WINDOW.springY, WINDOW.archRise, 0, 34)
  );
  const inner = archPoints(
    half - WINDOW.borderFace,
    WINDOW.springY,
    WINDOW.archRise - WINDOW.borderFace,
    0.085,
    34
  );
  // reverse so the hole winds against the outline
  border.holes.push(pathFromPoints(THREE, inner.reverse()));
  extrudeShape(
    THREE,
    group,
    MAT.border,
    name + '_Architrave',
    border,
    WINDOW.borderDepth,
    -WINDOW.borderDepth
  );

  /* --- projecting plaster plinth at the foot of the panel ---------------- */
  addBox(
    THREE,
    group,
    MAT.border,
    name + '_BasePlinth',
    0.96,
    0.11,
    0.043,
    0,
    0.055,
    -0.011
  );

  /* --- stone sill, 45 mm proud of the panel face ------------------------- */
  addBox(
    THREE,
    group,
    MAT.stone,
    name + '_StoneSill',
    WINDOW.sillWidth,
    WINDOW.sillThickness,
    WINDOW.sillDepth,
    0,
    WINDOW.sillY - WINDOW.sillThickness / 2,
    WINDOW.sillFrontZ - WINDOW.sillDepth / 2
  );
  addBox(
    THREE,
    group,
    MAT.sillShadow,
    name + '_SillUndercut',
    0.86,
    0.008,
    0.016,
    0,
    WINDOW.sillY - WINDOW.sillThickness - 0.004,
    WINDOW.sillFrontZ - 0.018
  );

  /* --- painted window frame, recessed just behind the panel face --------- */
  const frame = extrudeShape(
    THREE,
    group,
    MAT.frame,
    name + '_Frame',
    rectRing(THREE, 0.748, 1.148, 0.698, 1.098),
    0.022,
    -0.048
  );
  frame.position.y = WINDOW.sillY + WINDOW.windowHeight / 2;

  /* --- blank lower panel: raised moulding over a slightly darker field --- */
  const lowerFrame = extrudeShape(
    THREE,
    group,
    MAT.border,
    name + '_LowerPanelFrame',
    rectRing(THREE, 0.76, 0.9, 0.72, 0.86),
    0.01,
    WINDOW.fieldZ
  );
  lowerFrame.position.y = 0.57;

  addBox(
    THREE,
    group,
    MAT.lowerField,
    name + '_LowerPanelField',
    0.72,
    0.86,
    0.003,
    0,
    0.57,
    -0.0285
  );

  /* --- small louvred vent at the foot of the blank panel ----------------- */
  addBox(
    THREE,
    group,
    MAT.vent,
    name + '_Vent',
    0.16,
    0.11,
    0.012,
    0,
    0.4,
    -0.0255
  );
  for (let i = 0; i < 3; i++) {
    addBox(
      THREE,
      group,
      MAT.ventSlat,
      name + '_VentSlat_0' + (i + 1),
      0.13,
      0.007,
      0.005,
      0,
      0.37 + i * 0.03,
      -0.0178
    );
  }

  /* --- dark glazing, 25 mm behind the panel field ------------------------ */
  addBox(
    THREE,
    group,
    MAT.glazing,
    name + '_Glazing',
    0.742,
    1.142,
    0.004,
    0,
    WINDOW.sillY + WINDOW.windowHeight / 2,
    WINDOW.glassZ
  );

  /* --- dark metal grille: perimeter + 7 verticals + 8 horizontals -------- */
  const gx = 0.3475; // grille half width
  const gy0 = 1.074;
  const gy1 = 2.172;
  const gyMid = (gy0 + gy1) / 2;
  const gh = gy1 - gy0;
  const barT = 0.016;
  const barZ = WINDOW.grilleZ;
  const barD = 0.014;

  addBox(THREE, group, MAT.grille, name + '_GrilleFrame_L', barT, gh, barD, -gx + barT / 2, gyMid, barZ);
  addBox(THREE, group, MAT.grille, name + '_GrilleFrame_R', barT, gh, barD, gx - barT / 2, gyMid, barZ);
  addBox(THREE, group, MAT.grille, name + '_GrilleFrame_T', gx * 2, barT, barD, 0, gy1 - barT / 2, barZ);
  addBox(THREE, group, MAT.grille, name + '_GrilleFrame_B', gx * 2, barT, barD, 0, gy0 + barT / 2, barZ);

  for (let i = 0; i < 7; i++) {
    const bx = -0.26 + i * (0.52 / 6);
    addBox(
      THREE,
      group,
      MAT.grille,
      name + '_GrilleV_' + String(i + 1).padStart(2, '0'),
      Math.abs(bx) < 0.001 ? 0.016 : 0.011,
      gh - 0.014,
      barD,
      bx,
      gyMid,
      barZ
    );
  }
  for (let i = 0; i < 8; i++) {
    addBox(
      THREE,
      group,
      MAT.grille,
      name + '_GrilleH_' + String(i + 1).padStart(2, '0'),
      gx * 2 - 0.017,
      0.011,
      barD,
      0,
      1.16 + i * 0.14,
      barZ
    );
  }

  return group;
}

/* ------------------------------------------------------------------------- */
/* one split AC unit                                                         */
/* ------------------------------------------------------------------------- */

function buildAC(THREE, MAT, name, x) {
  const group = new THREE.Group();
  group.name = name;
  group.position.set(x, AC.centreY, WALL_Z);
  group.userData = {
    kind: 'splitAC',
    centreX: x,
    centreY: AC.centreY,
    wallZ: WALL_Z,
    width: AC.width,
    height: AC.height,
    depth: AC.depth,
    note: 'white wall split with bottom louvre slot, brand strip and conduit to ceiling',
  };

  /* --- wall plate, just visible above and below the shell ---------------- */
  addBox(
    THREE,
    group,
    MAT.acPlate,
    name + '_WallPlate',
    0.88,
    0.34,
    0.018,
    0,
    0,
    0.009
  );

  /* --- rounded white shell: 0.888 + 2 x 6 mm bevel = 0.90 m -------------- */
  const bodyShape = roundedRect(
    THREE,
    AC.width - AC.bevel * 2,
    AC.height - AC.bevel * 2,
    AC.corner
  );
  const bodyGeometry = new THREE.ExtrudeGeometry(bodyShape, {
    depth: AC.depth - AC.bevel * 2,
    steps: 1,
    curveSegments: 10,
    bevelEnabled: true,
    bevelThickness: AC.bevel,
    bevelSize: AC.bevel,
    bevelSegments: 2,
  });
  const body = addMesh(
    THREE,
    group,
    bodyGeometry,
    MAT.acShell,
    name + '_Shell'
  );
  body.position.z = AC.bodyZ;

  /* --- bottom outlet: recessed dark slot, three blades, hinged flap ------ */
  addBox(
    THREE,
    group,
    MAT.acSlot,
    name + '_LouvreRecess',
    0.76,
    0.072,
    0.012,
    0,
    AC.louvreY,
    0.25
  );
  for (let i = 0; i < 3; i++) {
    addBox(
      THREE,
      group,
      MAT.acBlade,
      name + '_LouvreBlade_0' + (i + 1),
      0.73,
      0.01,
      0.008,
      0,
      AC.louvreY - 0.019 + i * 0.019,
      0.262
    );
  }
  addBox(
    THREE,
    group,
    MAT.acBlade,
    name + '_OutletFlap',
    0.78,
    0.034,
    0.018,
    0,
    -0.13,
    0.254
  );

  /* --- brand strip, badge, status LED and the shrink seam ---------------- */
  addBox(
    THREE,
    group,
    MAT.acBrand,
    name + '_BrandStrip',
    0.14,
    0.018,
    0.004,
    0.21,
    0.012,
    0.254
  );
  addBox(
    THREE,
    group,
    MAT.acBadge,
    name + '_BrandBadge',
    0.036,
    0.009,
    0.0015,
    0.21,
    0.012,
    0.2575
  );
  const led = addMesh(
    THREE,
    group,
    new THREE.CylinderGeometry(0.006, 0.006, 0.004, 12),
    MAT.acLed,
    name + '_StatusLed'
  );
  led.rotation.x = Math.PI / 2;
  led.position.set(-0.3, -0.06, 0.254);

  addBox(
    THREE,
    group,
    MAT.acPlate,
    name + '_ShellSeam',
    0.8,
    0.006,
    0.002,
    0,
    0.112,
    0.253
  );

  /* --- white conduit: shell top to ceiling, then 0.35 m along the wall --- */
  addBox(
    THREE,
    group,
    MAT.conduit,
    name + '_ConduitRiser',
    0.045,
    0.594,
    0.03,
    AC.conduitX,
    0.437,
    0.017
  );
  addBox(
    THREE,
    group,
    MAT.conduit,
    name + '_ConduitCeilingRun',
    0.35,
    0.045,
    0.03,
    AC.conduitX + 0.175,
    0.7175,
    0.017
  );
  addBox(
    THREE,
    group,
    MAT.conduit,
    name + '_ConduitElbow',
    0.052,
    0.052,
    0.036,
    AC.conduitX,
    0.7125,
    0.019
  );
  for (let i = 0; i < 2; i++) {
    addBox(
      THREE,
      group,
      MAT.conduitClip,
      name + '_ConduitClip_0' + (i + 1),
      0.055,
      0.019,
      0.04,
      AC.conduitX,
      0.3 + i * 0.25,
      0.022
    );
  }

  return group;
}

/* ------------------------------------------------------------------------- */
/* assembly                                                                  */
/* ------------------------------------------------------------------------- */

export function buildWindows(THREE) {
  if (!THREE || !THREE.Group || !THREE.Mesh || !THREE.ExtrudeGeometry) {
    throw new Error('buildWindows(THREE): pass the THREE namespace as the first argument');
  }

  const MAT = createMaterials(THREE);
  const root = new THREE.Group();
  root.name = 'Windows';

  const pad = (n) => String(n).padStart(2, '0');

  WINDOW_XS.forEach((x, i) => {
    root.add(buildWindow(THREE, MAT, 'WindowRecess_' + pad(i + 1), x));
  });
  AC_XS.forEach((x, i) => {
    root.add(buildAC(THREE, MAT, 'SplitAC_' + pad(i + 1), x));
  });

  let meshCount = 0;
  root.traverse((child) => {
    if (child.isMesh) meshCount += 1;
  });

  root.userData = {
    kind: 'windows',
    units: 'm',
    floorY: 0,
    ceilingY: 3.1,
    wallZ: WALL_Z,
    faceZ: FACE_Z,
    panelProjection: FACE_Z - WALL_Z,
    windowXs: WINDOW_XS.slice(),
    acXs: AC_XS.slice(),
    window: {
      panelWidth: WINDOW.panelWidth,
      panelHeight: WINDOW.panelHeight,
      springY: WINDOW.springY,
      archRise: WINDOW.archRise,
      windowWidth: WINDOW.windowWidth,
      windowHeight: WINDOW.windowHeight,
      sillY: WINDOW.sillY,
      grille: '7 vertical + 8 horizontal dark iron bars',
      recessDepth: -(WINDOW.glassZ - WINDOW.fieldZ),
    },
    splitAC: {
      width: AC.width,
      height: AC.height,
      depth: AC.depth,
      centreY: AC.centreY,
      conduitToY: AC.conduitTop,
    },
    meshCount,
    source: 'frames f001/f002/f003/f005/f007/f008/f022/f030/f040',
  };

  return root;
}
