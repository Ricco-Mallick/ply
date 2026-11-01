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

const STATUS: Record<string, SceneStatus> = {
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
  'c09a4c82-ca08-47b3-9850-e8e30a54d064': {
    label: 'Latest pipeline',
    text: 'Current solver, trainer and export: this is what Ply produces today.',
    tone: 'new',
  },
};

/** Anything not named above is a demo-run or an early experiment: say so plainly. */
const DEFAULT: SceneStatus = {
  label: 'Bad capture · very old pipeline',
  text: 'Bad capture output, produced by a very old development pipeline. Kept as a record of the work, not as a demo of what Ply produces now.',
  tone: 'old',
};

export function sceneStatus(sceneUrl: string | null | undefined): SceneStatus | null {
  if (!sceneUrl) return null;
  try {
    const id = new URL(sceneUrl, location.href).pathname.match(/\/scenes\/([\w-]+)\//)?.[1];
    return id ? STATUS[id] ?? DEFAULT : null;
  } catch {
    return null;
  }
}
