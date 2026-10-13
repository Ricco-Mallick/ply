// node --experimental-strip-types scripts/tar-check.mjs out.tar  -> writes a tar with the app's writer (for Python's tarfile to read)
import fs from 'node:fs';
import { tarHeader, tarPad } from '../src/capture/lib/tar.ts';
const files = [['frames/00000.jpg', new Uint8Array(1000).fill(7)], ['poses.json', new TextEncoder().encode(JSON.stringify({ keyframes: [{ file: 'frames/00000.jpg' }] }))]];
const parts = [];
for (const [n, d] of files) parts.push(tarHeader(n, d.length), d, tarPad(d.length));
parts.push(new Uint8Array(1024));
fs.writeFileSync(process.argv[2], Buffer.concat(parts.map(p => Buffer.from(p))));
