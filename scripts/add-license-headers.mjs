// Prepend the copyright line to every source file we wrote (not vendored code), once.
//   node scripts/add-license-headers.mjs        (run from the repo root)
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join, extname } from 'node:path';

const LINE = {
  ts: '// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.',
  tsx: '// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.',
  js: '// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.',
  mjs: '// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.',
  py: '# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.',
  css: '/* Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE. */',
};
const ROOTS = ['app/src', 'cloud/src', 'pipeline/plypipe', 'scripts'];
const SKIP = [/node_modules/, /\/dist\//, /\.venv/, /app\/figures\/kit\//, /app\/public\/editor\//];
const seen = new Set();
function walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (SKIP.some((re) => re.test(p))) continue;
    const st = statSync(p);
    if (st.isDirectory()) walk(p);
    else if (LINE[extname(p).slice(1)] && !seen.has(p)) { seen.add(p); add(p); }
  }
}
function add(file) {
  const src = readFileSync(file, 'utf8');
  const line = LINE[extname(file).slice(1)];
  if (src.includes('Ricco Marcelino. All rights reserved')) return;
  // keep shebangs and Python module docstrings first: a header above them would change behaviour
  const header = line + '\n';
  if (src.startsWith('#!')) {
    const [first, ...rest] = src.split('\n');
    writeFileSync(file, `${first}\n${header}${rest.join('\n')}`);
  } else {
    writeFileSync(file, header + src);
  }
}
for (const r of ROOTS) walk(r);
console.log(`headers added to ${seen.size} files`);
