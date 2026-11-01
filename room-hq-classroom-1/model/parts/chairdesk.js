/**
 * chairdesk.js — the two seating/table props of the captured classroom.
 *
 *   chairDesk(THREE)  — cobalt-blue moulded polypropylene chair with a dark
 *                       writing tablet clamped to its right arm (the detector
 *                       labelled these "tables"; they are chair-desks).
 *   benchTable(THREE) — the long dark mahogany bench-table used along the
 *                       whiteboard / blackboard wall.
 *
 * Units are metres, +Y up, floor at y = 0, matching the room shell.
 * Chair local frame: +Z is the direction the sitter faces, +X is the
 * sitter's LEFT, so the right-arm tablet sits at local -X. The group's
 * origin is the floor point under the seat centre.
 * Bench local frame: the 2.65 m length runs along local X, depth along Z,
 * origin on the floor under the centre of the top.
 *
 * No imports, no assets, core three.js only. Every mesh casts/receives.
 * Mesh budget: chair 12, bench 13.
 */

/** Rounded-rectangle outline, centred on the shape origin. */
function roundedRectShape(THREE, width, height, radius) {
  const hw = width / 2;
  const hh = height / 2;
  const r = Math.max(0.001, Math.min(radius, hw - 0.0005, hh - 0.0005));
  const s = new THREE.Shape();
  s.moveTo(-hw + r, -hh);
  s.lineTo(hw - r, -hh);
  s.quadraticCurveTo(hw, -hh, hw, -hh + r);
  s.lineTo(hw, hh - r);
  s.quadraticCurveTo(hw, hh, hw - r, hh);
  s.lineTo(-hw + r, hh);
  s.quadraticCurveTo(-hw, hh, -hw, hh - r);
  s.lineTo(-hw, -hh + r);
  s.quadraticCurveTo(-hw, -hh, -hw + r, -hh);
  s.closePath();
  return s;
}

/**
 * Horizontal rounded slab. The outline is width (X) x depth (Z), extruded
 * thickness along Y, with a small machined bevel on the edges. Geometry is
 * centred on the mesh origin; `y` means the solid's mid-plane.
 */
function roundedSlab(THREE, name, width, depth, thickness, corner, material) {
  const shape = roundedRectShape(THREE, width, depth, corner);
  const bevel = Math.min(0.0035, thickness * 0.16);
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: thickness,
    steps: 1,
    curveSegments: 10,
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 2,
  });
  geo.translate(0, 0, -thickness / 2);
  geo.rotateX(-Math.PI / 2); // outline XZ, thickness along Y
  const mesh = new THREE.Mesh(geo, material);
  mesh.name = name;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/** Straight round tube between two points (chair legs). */
function legTube(THREE, name, a, b, radius, material, radialSegments) {
  const from = new THREE.Vector3(a[0], a[1], a[2]);
  const to = new THREE.Vector3(b[0], b[1], b[2]);
  const dir = to.clone().sub(from);
  const length = dir.length();
  const geo = new THREE.CylinderGeometry(radius, radius * 0.94, length, radialSegments || 10, 1);
  const mesh = new THREE.Mesh(geo, material);
  mesh.name = name;
  mesh.position.copy(from).add(to).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/** Smooth bent tube through control points (arms, seat frame, tablet stay). */
function bentTube(THREE, name, pts, radius, material, closed, segments) {
  const points = pts.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
  const curve = new THREE.CatmullRomCurve3(points, !!closed, 'catmullrom', 0.28);
  const geo = new THREE.TubeGeometry(curve, segments || 56, radius, 8, !!closed);
  const mesh = new THREE.Mesh(geo, material);
  mesh.name = name;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/**
 * Blue one-piece seat/back shell.
 * Photos f005/f008/f015/f022/f028/f036: tall rounded back, slightly narrower at
 * the shoulders, concave in plan, blue shell carried by a thin black tube
 * frame. The back is bent procedurally so the sitter's back wraps into it.
 */
function chairBackGeometry(THREE) {
  const w = 0.44;
  const h = 0.38;
  const t = 0.034;
  const shape = roundedRectShape(THREE, w, h, 0.068);
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: t,
    steps: 1,
    curveSegments: 12,
    bevelEnabled: true,
    bevelThickness: 0.005,
    bevelSize: 0.005,
    bevelSegments: 3,
  });
  geo.translate(0, 0, -t / 2);
  const pos = geo.attributes.position;
  const halfW = w / 2;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const v = y / (h / 2);
    // shoulders a touch narrower than the lumbar area
    const taper = 1 - 0.045 * Math.max(0, v);
    // concave plan: centre recessed, side edges wrap forward around the back
    const u = x / halfW;
    const wrap = 0.036 * u * u;
    pos.setXYZ(i, x * taper, y, z + wrap);
  }
  geo.computeVertexNormals();
  return geo;
}

/** Blue shell chair with a 0.50 x 0.35 m writing tablet on the right arm. */
export function chairDesk(THREE) {
  const group = new THREE.Group();
  group.name = 'chairDesk';
  group.userData = {
    kind: 'chair-desk',
    units: 'm',
    facing: '+Z (tablet on the sitter\'s right, local -X)',
    source: 'frames f005/f008/f015/f022/f028/f036',
  };

  // Moulded cobalt-blue polypropylene, with the soft satin sheen seen in f028.
  const blueMat = new THREE.MeshStandardMaterial({
    name: 'chairDesk_bluePoly',
    color: 0x2053cd,
    roughness: 0.40,
    metalness: 0.02,
  });
  // Thin powder-coated black steel tube frame.
  const frameMat = new THREE.MeshStandardMaterial({
    name: 'chairDesk_blackSteel',
    color: 0x151619,
    roughness: 0.43,
    metalness: 0.38,
  });
  // Dark grey-brown laminate writing tablet, scuffed pale on top in the frames.
  const tabletMat = new THREE.MeshStandardMaterial({
    name: 'chairDesk_tabletLaminate',
    color: 0x49443c,
    roughness: 0.55,
    metalness: 0.06,
  });

  // ---- shell --------------------------------------------------------------
  // Seat: 0.44 x 0.435 m, top around 0.455 m (nominal classroom chair height),
  // 2 deg of waterfall (front edge up).
  const seat = roundedSlab(THREE, 'chairDesk_seat', 0.44, 0.435, 0.038, 0.065, blueMat);
  seat.position.set(0, 0.434, 0.015);
  seat.rotation.x = -0.035;
  group.add(seat);

  // Back: 0.44 m wide, 0.38 m tall, bottom y=0.43 m, top y=0.825 m, raked 8 deg.
  const back = new THREE.Mesh(chairBackGeometry(THREE), blueMat);
  back.name = 'chairDesk_back';
  back.position.set(0, 0.624, -0.192);
  back.rotation.x = -0.14;
  back.castShadow = true;
  back.receiveShadow = true;
  group.add(back);

  // ---- black tube frame ---------------------------------------------------
  // Perimeter rail under the seat (one closed tube loop).
  const seatFrame = bentTube(
    THREE,
    'chairDesk_seatFrame',
    [
      [-0.185, 0.412, -0.172],
      [-0.197, 0.412, 0.03],
      [-0.19, 0.412, 0.183],
      [-0.12, 0.412, 0.192],
      [0, 0.412, 0.194],
      [0.12, 0.412, 0.192],
      [0.19, 0.412, 0.183],
      [0.197, 0.412, 0.03],
      [0.185, 0.412, -0.172],
      [0.10, 0.412, -0.178],
      [0, 0.412, -0.18],
      [-0.10, 0.412, -0.178],
    ],
    0.0105,
    frameMat,
    true,
    88
  );
  group.add(seatFrame);

  // Four 21 mm round legs, splayed outward about 8 deg at the rear.
  group.add(
    legTube(THREE, 'chairDesk_legFrontLeft', [-0.18, 0.416, 0.17], [-0.225, 0.004, 0.225], 0.0105, frameMat)
  );
  group.add(
    legTube(THREE, 'chairDesk_legFrontRight', [0.18, 0.416, 0.17], [0.225, 0.004, 0.225], 0.0105, frameMat)
  );
  group.add(
    legTube(THREE, 'chairDesk_legRearLeft', [-0.18, 0.416, -0.165], [-0.25, 0.004, -0.235], 0.0105, frameMat)
  );
  group.add(
    legTube(THREE, 'chairDesk_legRearRight', [0.18, 0.416, -0.165], [0.25, 0.004, -0.235], 0.0105, frameMat)
  );

  // Arms: a continuous black tube each side, rising from the seat rail at the
  // back, running level at ~0.633 m, then dropping into the front seat corner.
  const leftArmPts = [
    [0.20, 0.418, -0.17],
    [0.222, 0.505, -0.185],
    [0.228, 0.605, -0.155],
    [0.230, 0.632, -0.07],
    [0.230, 0.634, 0.06],
    [0.228, 0.625, 0.145],
    [0.218, 0.555, 0.198],
    [0.20, 0.445, 0.20],
  ];
  const rightArmPts = leftArmPts.map((p) => [-p[0], p[1], p[2]]);
  group.add(bentTube(THREE, 'chairDesk_armLeft', leftArmPts, 0.0115, frameMat, false, 48));
  group.add(bentTube(THREE, 'chairDesk_armRight', rightArmPts, 0.0115, frameMat, false, 48));

  // ---- right-arm writing tablet ------------------------------------------
  // 0.50 x 0.35 m, top at 0.68 m, long side across the sitter's lap, 3.4 deg
  // tilted down toward the sitter. Clamped over the right arm at -X.
  const tabletAssembly = new THREE.Group();
  tabletAssembly.name = 'chairDesk_tabletAssembly';
  tabletAssembly.position.set(-0.03, 0.669, 0.06);
  tabletAssembly.rotation.x = -0.06;
  group.add(tabletAssembly);

  const tablet = roundedSlab(THREE, 'chairDesk_tablet', 0.50, 0.35, 0.018, 0.030, tabletMat);
  tabletAssembly.add(tablet);

  // Rubber-lined clamp block gripping the right arm.
  const clampGeo = new THREE.BoxGeometry(0.048, 0.058, 0.072);
  const clamp = new THREE.Mesh(clampGeo, frameMat);
  clamp.name = 'chairDesk_tabletClamp';
  clamp.position.set(-0.228, 0.632, -0.055);
  clamp.castShadow = true;
  clamp.receiveShadow = true;
  group.add(clamp);

  // One-piece steel stay from the clamp, under the tablet, toward its centre.
  const tabletStay = bentTube(
    THREE,
    'chairDesk_tabletStay',
    [
      [-0.228, 0.638, -0.05],
      [-0.224, 0.653, -0.01],
      [-0.165, 0.660, 0.02],
      [-0.075, 0.661, 0.045],
    ],
    0.010,
    frameMat,
    false,
    24
  );
  group.add(tabletStay);

  return group;
}

/**
 * Long dark mahogany bench-table, 2.65 x 0.55 m top at 0.795 m. Square
 * tapered legs, slim aprons and a low box stretcher — matching the console
 * under the whiteboard in f012/f022/f028 and the table in f040.
 */
export function benchTable(THREE) {
  const group = new THREE.Group();
  group.name = 'benchTable';
  group.userData = {
    kind: 'bench-table',
    units: 'm',
    lengthAxis: 'X',
    source: 'frames f012/f022/f028/f040',
  };

  const woodTopMat = new THREE.MeshStandardMaterial({
    name: 'benchTable_mahoganyTop',
    color: 0x5a3828,
    roughness: 0.44,
    metalness: 0.03,
  });
  const woodFrameMat = new THREE.MeshStandardMaterial({
    name: 'benchTable_darkWoodFrame',
    color: 0x3c251a,
    roughness: 0.58,
    metalness: 0.02,
  });

  const L = 2.65;
  const D = 0.55;
  const H = 0.795;
  const topT = 0.052;
  const legX = L / 2 - 0.125;
  const legZ = D / 2 - 0.065;
  const legTop = H - topT - 0.007; // meets the underside of the bevelled top

  // Thick edge-grained top, 52 mm, overhanging the frame all round.
  const top = roundedSlab(THREE, 'benchTable_top', L, D, topT, 0.022, woodTopMat);
  top.position.y = H - topT / 2 - 0.0035;
  group.add(top);

  // Aprons just inside the top edge.
  const apronH = 0.072;
  const apronT = 0.026;
  const apronY = legTop - apronH / 2;
  const apronGeoX = new THREE.BoxGeometry(L - 0.26, apronH, apronT);
  const apronGeoZ = new THREE.BoxGeometry(apronT, apronH, D - 0.12);
  for (const sign of [-1, 1]) {
    const sideApron = new THREE.Mesh(apronGeoX, woodFrameMat);
    sideApron.name = sign < 0 ? 'benchTable_apronBack' : 'benchTable_apronFront';
    sideApron.position.set(0, apronY, sign * (D / 2 - 0.045));
    sideApron.castShadow = true;
    sideApron.receiveShadow = true;
    group.add(sideApron);

    const endApron = new THREE.Mesh(apronGeoZ, woodFrameMat);
    endApron.name = sign < 0 ? 'benchTable_apronLeft' : 'benchTable_apronRight';
    endApron.position.set(sign * (L / 2 - 0.13), apronY, 0);
    endApron.castShadow = true;
    endApron.receiveShadow = true;
    group.add(endApron);
  }

  // Four square tapered legs (52 mm at the top, 38 mm at the floor).
  // A 4-sided cylinder gives a light taper with only four sides.
  const legGeo = new THREE.CylinderGeometry(0.0365, 0.0268, legTop, 4, 1);
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const leg = new THREE.Mesh(legGeo, woodFrameMat);
      leg.name =
        'benchTable_leg' + (sz < 0 ? 'Rear' : 'Front') + (sx < 0 ? 'Left' : 'Right');
      leg.position.set(sx * legX, legTop / 2, sz * legZ);
      leg.rotation.y = Math.PI / 4; // align the four-sided cylinder as a square
      leg.castShadow = true;
      leg.receiveShadow = true;
      group.add(leg);
    }
  }

  // Low box stretcher, about 0.135 m off the floor, as in f012 and f040.
  const stretcherY = 0.135;
  const stretcherT = 0.038;
  const longStretcherGeo = new THREE.BoxGeometry(legX * 2, stretcherT, stretcherT);
  const shortStretcherGeo = new THREE.BoxGeometry(stretcherT, stretcherT, legZ * 2);
  for (const sign of [-1, 1]) {
    const longRail = new THREE.Mesh(longStretcherGeo, woodFrameMat);
    longRail.name = sign < 0 ? 'benchTable_stretcherBack' : 'benchTable_stretcherFront';
    longRail.position.set(0, stretcherY, sign * legZ);
    longRail.castShadow = true;
    longRail.receiveShadow = true;
    group.add(longRail);

    const shortRail = new THREE.Mesh(shortStretcherGeo, woodFrameMat);
    shortRail.name = sign < 0 ? 'benchTable_stretcherLeft' : 'benchTable_stretcherRight';
    shortRail.position.set(sign * legX, stretcherY, 0);
    shortRail.castShadow = true;
    shortRail.receiveShadow = true;
    group.add(shortRail);
  }

  return group;
}
