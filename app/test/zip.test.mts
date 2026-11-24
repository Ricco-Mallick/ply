/* The twin pack's zip writer, read back by Python's zipfile. Run: node --experimental-strip-types test/zip.test.mts */
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { crc32, zip } from '../src/walk/zip.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
check('crc32 of "123456789" is the standard check value', crc32(new TextEncoder().encode('123456789')) === 0xcbf43926);
const big = new Uint8Array(300_000).map((_, i) => (i * 7) & 255);
const z = zip([{ name: 'README.txt', data: new TextEncoder().encode('hello twin\n') }, { name: 'data/scene.spz', data: big }]);
const p = join(mkdtempSync(join(tmpdir(), 'zip-')), 'pack.zip'); writeFileSync(p, z);
const out = execFileSync('python3', ['-I', '-c', `import zipfile,sys,hashlib
z=zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None
print(z.namelist(), z.read('README.txt').decode().strip(), len(z.read('data/scene.spz')), hashlib.md5(z.read('data/scene.spz')).hexdigest())`, p]).toString();
const md5 = execFileSync('python3', ['-I', '-c', 'import sys,hashlib; print(hashlib.md5(bytes((i*7)&255 for i in range(300000))).hexdigest())']).toString().trim();
check('python reads both files back intact', out.includes("['README.txt', 'data/scene.spz'] hello twin 300000") && out.includes(md5));
process.exit(fail ? 1 : 0);
