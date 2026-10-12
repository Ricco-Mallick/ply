/* Motion sensors: turn rate (blur risk), tilt (framing), heading and steps (a dead-reckoned path for the coverage map).
   iOS asks for permission from a tap; Android grants it silently on https. Absent sensors (desktop) leave `available` false
   and the guidance falls back to what the video alone can tell. */

export interface SensorSample { t: number; heading: number | null; pitch: number; roll: number; turn: number; accel: number; }

export async function requestMotionPermission(): Promise<'granted' | 'denied' | 'not-needed'> {
  const DM = (window as any).DeviceMotionEvent, DO = (window as any).DeviceOrientationEvent;
  try {
    const asks = [DM?.requestPermission, DO?.requestPermission].filter(Boolean);
    if (!asks.length) return 'not-needed';
    const res = await Promise.all([DM?.requestPermission?.(), DO?.requestPermission?.()].filter(Boolean));
    return res.every((r: string) => r === 'granted') ? 'granted' : 'denied';
  } catch { return 'denied'; }
}

export class Sensors {
  available = false; heading: number | null = null; pitch = 0; roll = 0; turn = 0; accel = 9.81;
  steps = 0; pos = { x: 0, z: 0 }; path: { x: number; z: number; t: number }[] = [{ x: 0, z: 0, t: 0 }];
  private lastStep = 0; private peak = false; private hpAccel = 0; private stepLen = .62; private t0 = performance.now();
  private onOri = (e: DeviceOrientationEvent) => {
    if (e.beta == null) return; this.available = true;
    // phone upright in portrait: beta ~90 when vertical; pitch = how far it tips down from vertical
    this.pitch = 90 - e.beta; this.roll = e.gamma ?? 0;
    const abs = (e as any).webkitCompassHeading ?? (e.absolute && e.alpha != null ? 360 - e.alpha : e.alpha != null ? 360 - e.alpha : null);
    if (abs != null) this.heading = this.heading == null ? abs : this.smoothAngle(this.heading, abs, .25);
  };
  private onMotion = (e: DeviceMotionEvent) => {
    const r = e.rotationRate; if (r) { this.available = true; this.turn = Math.hypot(r.alpha ?? 0, r.beta ?? 0, r.gamma ?? 0); }
    const a = e.accelerationIncludingGravity; if (!a) return;
    const m = Math.hypot(a.x ?? 0, a.y ?? 0, a.z ?? 0); this.accel = m;
    // step detector: high-passed magnitude crossing +1.1 m/s² with a 280 ms refractory window
    this.hpAccel = .85 * this.hpAccel + .15 * (m - 9.81);
    const now = performance.now();
    if (!this.peak && this.hpAccel > 1.1 && now - this.lastStep > 280) { this.peak = true; this.lastStep = now; this.step(now); }
    if (this.peak && this.hpAccel < .2) this.peak = false;
  };
  private step(now: number) {
    this.steps++; if (this.heading == null) return; const h = this.heading * Math.PI / 180;
    this.pos = { x: this.pos.x + Math.sin(h) * this.stepLen, z: this.pos.z - Math.cos(h) * this.stepLen };
    this.path.push({ ...this.pos, t: now - this.t0 });
  }
  private smoothAngle(a: number, b: number, k: number) { const d = ((b - a + 540) % 360) - 180; return (a + d * k + 360) % 360; }

  start() {
    addEventListener('deviceorientationabsolute' as any, this.onOri as any); addEventListener('deviceorientation', this.onOri);
    addEventListener('devicemotion', this.onMotion);
  }
  stop() { removeEventListener('deviceorientationabsolute' as any, this.onOri as any); removeEventListener('deviceorientation', this.onOri); removeEventListener('devicemotion', this.onMotion); }
  resetPath() { this.steps = 0; this.pos = { x: 0, z: 0 }; this.path = [{ x: 0, z: 0, t: 0 }]; this.t0 = performance.now(); }
  sample(t: number): SensorSample { return { t, heading: this.heading, pitch: this.pitch, roll: this.roll, turn: this.turn, accel: this.accel }; }
  /** Back within a metre of where recording began, after a real walk. */
  get loopClosed() { const d = Math.hypot(this.pos.x, this.pos.z); return this.steps > 24 && d < 1.2; }
}
