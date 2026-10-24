// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
// Render the model from a captured camera pose, so it can be compared with the photograph taken there.
//   node scripts/render_pose.mjs <room-dir-or-scene-url> <pose-index> [out.png]
// Reads <room>/evidence/poses.json, opens /ply/model.html?scene=…&pose=…, screenshots with headless Chrome.
import { readFileSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join, resolve } from 'node:path';

const [, , target, indexArg = '0', outArg] = process.argv;
if (!target) { console.error('usage: node scripts/render_pose.mjs <room-dir|scene-url> <pose-index> [out.png]'); process.exit(2); }
const room = resolve(target);
let poses = [];
try { poses = JSON.parse(readFileSync(join(room, 'evidence', 'poses.json'), 'utf8')).frames ?? []; } catch {}
const pose = poses[Number(indexArg)];
if (!pose) { console.error(`no pose ${indexArg} in ${room}/evidence/poses.json`); process.exit(2); }
const [x, y, z] = pose.position ?? [0, 1.6, 0];
const sceneUrl = process.env.PLY_SCENE ?? `https://ply-api.ricco-ply.workers.dev/v1/scenes/${room.split('/').pop()}/`;
const url = `https://ricco.codes/ply/model.html?scene=${encodeURIComponent(sceneUrl)}&pose=${x},${y},${z},${pose.yaw ?? 0},0`;
const out = outArg ?? join(room, 'compare', `pose-${String(indexArg).padStart(3, '0')}-render.png`);
mkdirSync(join(room, 'compare'), { recursive: true });
execFileSync('google-chrome', ['--headless=new', '--disable-gpu', '--hide-scrollbars', '--virtual-time-budget=12000',
  '--window-size=1280,720', `--screenshot=${out}`, url], { stdio: 'inherit' });
console.log(`rendered ${out}`);
console.log(`compare with ${join(room, 'evidence', 'frames', String(pose.index).padStart(3, '0') + '.jpg')}`);
