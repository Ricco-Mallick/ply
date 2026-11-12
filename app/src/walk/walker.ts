// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The walker: its size, its pace, and one frame of Walk on the floor plane. Pure like collider.ts, so the engine and
   the tests run the same code. */
import type { Collider, Point } from './collider';

/** Half the walker's width, metres. */
export const RADIUS = .22;
/** Metres per second, on foot and in Free alike; Shift multiplies it by RUN. */
export const SPEED = 2.2, RUN = 2.2;
/** The wheel multiplies the pace by .25 to 4 in Free. Walk stops at 2: 4.4 m/s is a jog and 9.7 m/s with Shift a
    sprint, the fastest anyone is on foot, and a wheel burst lands there, not at a pace no doorway can be steered
    through. At a 50 ms frame that is 0.48 m, 20 of a move's 64 sub-steps on 5 cm cells. */
export const WHEEL_MIN = .25, WHEEL = 4, WHEEL_WALK = 2;
/** How far around a walker that is inside a wall a place for it is looked for, before it goes back to the spawn. */
export const REACH = 6;

/** Metres per second for a mode, Shift and the wheel's multiplier. */
export const pace = (walk: boolean, run: boolean, wheel: number) => SPEED * (run ? RUN : 1) * (walk ? Math.min(wheel, WHEEL_WALK) : wheel);
/** The share of the gap to the wanted velocity closed in dt: quicker to start than to stop. */
export const ease = (dt: number, go: boolean) => 1 - Math.exp(-dt * (go ? 10 : 7));

/** The walker on the floor plane: where it is and how fast it moves, metres and metres per second. */
export interface Walker extends Point { vx: number; vz: number; }

/** Somewhere the walker fits when it is inside a wall: the nearest such place within REACH, else the spawn point (the
    nearest place to it, if the spawn is blocked too). Null when it fits nowhere. */
export function recover(c: Collider, x: number, z: number, spawn: Point): Point | null {
  return c.nearestFree(x, z, RADIUS, REACH) ?? c.nearestFree(spawn.x, spawn.z, RADIUS);
}

/** One frame of Walk: ease the velocity towards (wx, wz), move, and keep only the velocity the walls allowed, so the
    next frame pushes along a wall and not into it. True when something was touched. */
export function walkFrame(c: Collider, w: Walker, wx: number, wz: number, dt: number, spawn: Point): boolean {
  const go = wx !== 0 || wz !== 0, a = ease(dt, go);
  w.vx += (wx - w.vx) * a; w.vz += (wz - w.vz) * a;
  // a guided route or a jump to a place can leave the walker inside a wall: it goes back to where it fits before it
  // takes a step, never through the wall. Not while it stands still, because a guide may be carrying it.
  if (go && !c.free(w.x, w.z, RADIUS)) { const q = recover(c, w.x, w.z, spawn); if (q) { w.x = q.x; w.z = q.z; } }
  const m = c.move(w.x, w.z, w.vx * dt, w.vz * dt, RADIUS);
  if (m.hit && dt > 0) { w.vx = (m.x - w.x) / dt; w.vz = (m.z - w.z) / dt; }
  w.x = m.x; w.z = m.z; return m.hit;
}
