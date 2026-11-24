// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* iPhone LiDAR scans from Stray Scanner, uploaded as a zip (Files → long-press the scan folder → Compress).
   Before anything is uploaded, the zip's own directory is read from its last bytes (File.slice, no full read) and
   odometry.csv is inflated to check it really is a scan: video, poses, depth, and how long and how far it went. */

export interface LidarInfo { scan: string; frames: number; seconds: number; walkedM: number; depthMaps: number; video: string; path: { x: number; z: number; t: number }[]; }

interface Entry { name: string; method: number; csize: number; size: number; offset: number; }

const u16 = (d: DataView, o: number) => d.getUint16(o, true);
const u32 = (d: DataView, o: number) => d.getUint32(o, true);
const u64 = (d: DataView, o: number) => Number(d.getBigUint64(o, true));

async function bytes(f: Blob, from: number, to: number) { return new DataView(await f.slice(from, to).arrayBuffer()); }

/** The central directory of a zip (zip64 included), read from the end of the file. */
export async function zipEntries(f: Blob): Promise<Entry[]> {
  const tail = Math.min(f.size, 65557 + 20);
  const t = await bytes(f, f.size - tail, f.size);
  let e = -1;
  for (let i = t.byteLength - 22; i >= 0; i--) if (u32(t, i) === 0x06054b50) { e = i; break; }
  if (e < 0) throw new Error('not a zip file');
  let count = u16(t, e + 10), cdSize = u32(t, e + 12), cdOff = u32(t, e + 16);
  if (cdOff === 0xffffffff || count === 0xffff) {  // zip64: the locator sits just before the end record
    const loc = e - 20;
    if (loc < 0 || u32(t, loc) !== 0x07064b50) throw new Error('zip64 locator missing');
    const z = await bytes(f, u64(t, loc + 8), u64(t, loc + 8) + 56);
    count = u64(z, 32); cdSize = u64(z, 40); cdOff = u64(z, 48);
  }
  const cd = await bytes(f, cdOff, cdOff + cdSize), dec = new TextDecoder(), out: Entry[] = [];
  for (let p = 0, n = 0; n < count && p + 46 <= cd.byteLength; n++) {
    if (u32(cd, p) !== 0x02014b50) break;
    const method = u16(cd, p + 10), nl = u16(cd, p + 28), xl = u16(cd, p + 30), cl = u16(cd, p + 32);
    let csize = u32(cd, p + 20), size = u32(cd, p + 24), offset = u32(cd, p + 42);
    const name = dec.decode(new Uint8Array(cd.buffer, cd.byteOffset + p + 46, nl));
    for (let x = p + 46 + nl; x < p + 46 + nl + xl;) {  // zip64 extra field: the 64-bit sizes and offset, in order
      const id = u16(cd, x), len = u16(cd, x + 2); let q = x + 4;
      if (id === 1) { if (size === 0xffffffff) { size = u64(cd, q); q += 8; } if (csize === 0xffffffff) { csize = u64(cd, q); q += 8; } if (offset === 0xffffffff) offset = u64(cd, q); }
      x += 4 + len;
    }
    out.push({ name, method, csize, size, offset });
    p += 46 + nl + xl + cl;
  }
  return out;
}

async function readEntry(f: Blob, e: Entry): Promise<string> {
  const h = await bytes(f, e.offset, e.offset + 30);
  if (u32(h, 0) !== 0x04034b50) throw new Error('bad local header');
  const start = e.offset + 30 + u16(h, 26) + u16(h, 28), raw = f.slice(start, start + e.csize);
  if (e.method === 0) return raw.text();
  if (e.method !== 8 || typeof DecompressionStream === 'undefined') throw new Error('cannot inflate');
  return new Response(raw.stream().pipeThrough(new DecompressionStream('deflate-raw'))).text();
}

/** What a Stray Scanner zip holds, or an error that says what is missing. */
export async function inspectLidarZip(f: File): Promise<LidarInfo> {
  const all = (await zipEntries(f)).filter(e => !e.name.startsWith('__MACOSX/') && !e.name.split('/').pop()!.startsWith('._'));
  const odo = all.find(e => /(^|\/)odometry\.csv$/i.test(e.name));
  const video = all.find(e => /(^|\/)rgb\.(mp4|mov)$/i.test(e.name));
  if (!odo || !video) throw new Error(`this zip is not a Stray Scanner scan (it needs ${!odo ? 'odometry.csv' : 'rgb.mp4'}). Compress the scan's folder itself, not a file inside it`);
  const dir = odo.name.slice(0, odo.name.length - 'odometry.csv'.length);
  const depthMaps = all.filter(e => e.name.startsWith(`${dir}depth/`) && /\.(png|npy)$/i.test(e.name)).length;
  let frames = 0, seconds = 0, walkedM = 0; const path: LidarInfo['path'] = [];
  try {
    const rows = (await readEntry(f, odo)).trim().split(/\r?\n/).slice(1).map(l => l.split(',').map(s => +s.trim()));
    frames = rows.length;
    if (rows.length > 1) {
      seconds = rows[rows.length - 1][0] - rows[0][0];
      const every = Math.max(1, Math.floor(rows.length / 400));
      for (let i = 0; i < rows.length; i++) {
        if (i) walkedM += Math.hypot(rows[i][2] - rows[i - 1][2], rows[i][3] - rows[i - 1][3], rows[i][4] - rows[i - 1][4]);
        if (i % every === 0) path.push({ x: rows[i][2], z: rows[i][4], t: rows[i][0] - rows[0][0] });
      }
    }
  } catch { /* an old browser that cannot inflate: the pipeline reads it anyway */ }
  return { scan: dir.replace(/\/$/, '').split('/').pop() || f.name.replace(/\.zip$/i, ''), frames, seconds, walkedM: +walkedM.toFixed(1), depthMaps, video: video.name.split('/').pop()!, path };
}
