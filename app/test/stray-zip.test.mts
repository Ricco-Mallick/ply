/* The capture page's check of an iPhone LiDAR zip before upload. Run:
   node --experimental-strip-types test/stray-zip.test.mts  (small zips in test/fixtures/lidar) */
import { readFileSync } from 'node:fs';
import { inspectLidarZip } from '../src/capture/lib/stray-zip.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
const DIR = process.env.ZIPS ?? new URL('./fixtures/lidar', import.meta.url).pathname;
const file = (n: string) => new File([readFileSync(`${DIR}/${n}`)], n, { type: 'application/zip' });


for (const n of ['deflate.zip', 'stored.zip', 'z64.zip']) {
  const i = await inspectLidarZip(file(n));
  check(`${n}: the scan, its frames and depth are read`, i.scan === 'ab12cd34' && i.frames === 300 && i.depthMaps === 150 && i.video === 'rgb.mp4');
  check(`${n}: length and distance come from the poses`, Math.abs(i.seconds - 299 / 30) < 1e-3 && Math.abs(i.walkedM - 2.99) < .05 && i.path.length > 10);
}
let msg = ''; try { await inspectLidarZip(file('notscan.zip')); } catch (e) { msg = (e as Error).message; }
check('a zip with no scan is refused with a reason', /odometry\.csv/.test(msg));
msg = ''; try { await inspectLidarZip(new File([new Uint8Array(100)], 'x.zip')); } catch (e) { msg = (e as Error).message; }
check('a file that is not a zip is refused', /not a zip/.test(msg));
process.exit(fail ? 1 : 0);
