/* The viewer's scale correction against rescale.py's fields and transform.
   Run: node --experimental-strip-types test/rescale.test.mts */
import { LIMITS, factorFrom, fmtScale, measure, rescaleScene } from '../src/model/rescale.ts';
import type { SceneDocument } from '../src/model/types.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
const near = (a: number | undefined, b: number, eps = 1e-6) => typeof a === 'number' && Math.abs(a - b) < eps;

const SCENE: SceneDocument = {
  version: 1, scale: 0.349, scaleSource: 'camera-height-prior', cameraHeight: 1.45,
  transform: [[1, 0, 0, -0.2], [0, 1, 0, 1.45], [0, 0, 1, 0], [0, 0, 0, 1]],
  bounds: { min: [-5, -0.2, -3], max: [5, 3, 3] },
  cameraPath: [[-2.6, 1.09, 2.647]],
  spawn: { position: [1, 1.6, 2], lookAt: [1.1, 1.6, 2] },
  tour: [{ pos: [0, 1.7, 0], target: [1, 1.6, 0], fov: 60, dur: 4 }],
  topview: { image: 'topview.webp', origin: [-5, -3], metersPerPixel: 0.02, width: 1010, height: 441, cutY: 1.8 },
};

check('the factor is true / measured', factorFrom(2.04, 1.85) === 2.04 / 1.85);
check('a zero, negative or non-finite length has no factor',
  factorFrom(0, 1) === null && factorFrom(1, 0) === null && factorFrom(-2, 1) === null && factorFrom(NaN, 1) === null && factorFrom(1, Infinity) === null);
check('the viewer takes the pipeline limits (0.2 to 5)', LIMITS[0] === 0.2 && LIMITS[1] === 5);
check('measure is the distance between the taps', measure({ x: 0, y: 0, z: 0 }, { x: 3, y: 4, z: 0 }) === 5);
check('fmtScale prints a scale without trailing zeros', fmtScale(1.0253644971256997) === '1.0254' && fmtScale(0.349) === '0.349');

const f = factorFrom(2.04, 1.85)!;
const s1 = rescaleScene(SCENE, f, { knownM: 2.04, measuredM: 1.85 });
check('a twin distance of 1.85 typed as 2.04 m becomes exactly 2.04 m after the correction',
  near(measure({ x: 0, y: 0, z: 0 }, { x: 1.85 * f, y: 0, z: 0 }), 2.04));
check('scale moves by the factor', near(s1.scale, 0.349 * f));
check('scaleSource becomes reference', s1.scaleSource === 'reference');
check('the reference records both lengths, the factor and what it replaced',
  near(s1.reference?.factor, f) && near(s1.reference?.appliedFactor, f) && s1.reference?.knownM === 2.04 && s1.reference?.measuredM === 1.85 &&
  s1.reference?.previousSource === 'camera-height-prior' && near(s1.reference?.previousScale, 0.349) && near(s1.reference?.scale, 0.349 * f));
check('the transform rows 0-2 gain the factor and row 3 stays', near(s1.transform?.[0][0], f) && near(s1.transform?.[0][3], -0.2 * f) && near(s1.transform?.[3][3], 1));
check('bounds, cameraPath and the topview scale about the origin',
  near(s1.bounds?.min[0], -5 * f) && near(s1.bounds?.max[1], 3 * f) && near(s1.cameraPath?.[0][2], 2.647 * f) &&
  near(s1.topview?.metersPerPixel as number | undefined, 0.02 * f) && near((s1.topview?.origin as number[])[0], -5 * f));
check('the eye height in spawn and tour does not scale',
  near((s1.spawn?.position as number[])[1], 1.6) && near((s1.tour?.[0].pos as number[])[1], 1.7) &&
  near((s1.tour?.[0].target as number[])[1], 1.6) && near((s1.tour?.[0].target as number[])[0], f));
check('the manifest handed in is not mutated', SCENE.scale === 0.349 && SCENE.scaleSource === 'camera-height-prior' && !SCENE.reference);

const s2 = rescaleScene(s1, 1 / f, { knownM: 1.85, measuredM: 2.04 });
check('a correction and its inverse give the first scale back, and the reference keeps the first source',
  near(s2.scale, 0.349, 1e-5) && near(s2.reference?.factor, 1) && s2.reference?.previousSource === 'camera-height-prior' && near(s2.reference?.previousScale, 0.349, 1e-5));

const plain: SceneDocument = { scaleSource: 'none' };
const s3 = rescaleScene(plain, 1.1, { knownM: 2.2, measuredM: 2 });
check('a manifest with no scale still records the reference and does not invent a scale',
  s3.scale === undefined && near(s3.reference?.factor, 1.1) && s3.reference?.previousSource === 'none' && s3.scaleSource === 'reference');

process.exit(fail ? 1 : 0);
