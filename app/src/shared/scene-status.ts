// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Which pipeline produced a twin, said plainly on the page rather than in a footnote.
   Keyed by scene id (the job id in the scene URL); unknown ids get no badge. */

export interface SceneStatus {
  /** short chip text */
  label: string;
  /** one line of honest context */
  text: string;
  /** 'old' | 'mid' | 'new' | 'dev' — drives the chip colour */
  tone: 'old' | 'mid' | 'new' | 'dev';
}

/** Early runs the app still links or ships: say so plainly. Ids not listed in STATUS get no badge. */
const EARLY: SceneStatus = {
  label: 'Bad capture · very old pipeline',
  text: 'Bad capture output, produced by a very old development pipeline. Kept as a record of the work, not as a demo of what Ply produces now.',
  tone: 'old',
};

const STATUS: Record<string, SceneStatus> = {
  // the first real capture (the design lab's "My capture") and the pipeline contract-test fixture in public/scenes
  '0ffc07fb-4d8d-49eb-9a40-22b9c6fddd76': EARLY,
  'synth-test': EARLY,
  'hq-chapel-1': {
    label: 'Older pipeline',
    text: 'Captured before the current solver and trainer. Detail and alignment are below what the pipeline produces now.',
    tone: 'old',
  },
  'hq-classroom-1': {
    label: 'One version behind',
    text: 'Produced by the previous pipeline. The current one keeps more frames and trains a denser model.',
    tone: 'mid',
  },
  'lidar-office-1': {
    label: 'Under development',
    text: 'The LiDAR path: a scanner cloud and a measured shell. Working, but not a shipping feature yet.',
    tone: 'dev',
  },
  'chapel-high-2': {
    label: 'Latest pipeline',
    text: 'The chapel through the current pipeline at the high preset: 1,236 frames, mismatched poses dropped, the floor levelled from the cameras.',
    tone: 'new',
  },
  'chapel-test-quick': {
    label: 'Test run · quick',
    text: 'An end-to-end test of the current pipeline at its quickest setting (7k training steps), from before the floor and pose fixes. Not a finished twin.',
    tone: 'dev',
  },
  'c09a4c82-ca08-47b3-9850-e8e30a54d064': {
    label: 'Latest pipeline',
    text: 'Current solver, trainer and export: this is what Ply produces today.',
    tone: 'new',
  },
};

export function sceneStatus(sceneUrl: string | null | undefined, base: string | undefined = typeof location === 'undefined' ? undefined : location.href): SceneStatus | null {
  if (!sceneUrl) return null;
  try {
    const id = new URL(sceneUrl, base).pathname.match(/\/scenes\/([\w-]+)\//)?.[1];
    return id && Object.hasOwn(STATUS, id) ? STATUS[id] : null;
  } catch {
    return null;
  }
}
