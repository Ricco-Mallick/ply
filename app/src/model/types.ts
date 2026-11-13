// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Semantic model: build a parametric Three.js room from the measured shell (walls, openings, objects) instead of
   splats. This is the "LEGO" side of a Ply capture — every wall, door and chair is a real object you can select,
   measure, move or delete, and the whole thing exports as GLB/IFC-ready JSON. */

export type Wall = { a: [number, number]; b: [number, number]; length: number; normal?: [number, number] };
export type Opening = { kind: string; wall: number; u: number; width: number; sill: number; height: number; name?: string };
export type Obj = { id: string; name: string; kind?: string; center: [number, number, number]; half: [number, number, number] };
export type Shell = { floorY?: number; ceilingY?: number; walls?: Wall[]; openings?: Opening[]; objects?: Obj[] };

/** The reference block `plypipe rescale` records in scene.json (pipeline/plypipe/rescale.py). */
export type Reference = {
  factor?: number; appliedFactor?: number; previousSource?: string; previousScale?: number;
  scale?: number; knownM?: number; measuredM?: number;
};
/** The scene.json fields the viewer reads and a rescale rewrites. Everything else in the file passes through untouched. */
export type SceneDocument = {
  scale?: number;
  scaleSource?: string;
  reference?: Reference;
  transform?: number[][];
  bounds?: Record<string, number[]>;
  cameraPath?: number[][];
  spawn?: Record<string, unknown>;
  tour?: Record<string, unknown>[];
  topview?: Record<string, unknown>;
  [key: string]: unknown;
};

export const PALETTE = {
  floor: 0x2a3448,
  wall: 0xd8d2c6,
  wallSide: 0xb9b3a7,
  ceiling: 0xf0eee9,
  frame: 0x8f8a80,
  glass: 0x9fd8ff,
  wood: 0xb98a5e,
  fabric: 0x5b6d8c,
  metal: 0x9aa3b2,
  dark: 0x2b3240,
  accent: 0x6ea8ff,
};
