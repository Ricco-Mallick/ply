/* Camera: which rear lens, and how exposure behaves.
   Chrome on Android lists every rear sensor as its own device ("camera2 0, facing back" is the main one on Samsung and
   most others) and exposes exposure / focus / white balance as MediaTrack constraints (Image Capture spec). Safari and
   desktop mostly don't, so everything here is best-effort and reports what it could do.

   Exposure: Auto by default. The pipeline's per-frame appearance model (PPISP) evens out auto-exposure drift, so a bad
   lock is worse than none. Lock is opt-in: it meters on the point you tap, judges brightness by the median of the frame
   with blown highlights ignored (a lamp in view doesn't darken the room), pins the shutter at 1/100 s against 50 Hz
   flicker, then nudges ISO until the room sits at a normal brightness. */

export interface CamInfo { width: number; height: number; fps: number; label: string; deviceId: string; }
export interface Lens { deviceId: string; label: string; name: string; index: number; }
export interface LockReport { exposure?: string; focus?: string; whiteBalance?: string; unsupported: string[]; note?: string; }
export interface CamControls { zoom: { min: number; max: number; step: number } | null; ev: { min: number; max: number; step: number } | null; canLock: boolean; poi: boolean; }

type Caps = MediaTrackCapabilities & Record<string, any>;
type Settings = MediaTrackSettings & Record<string, any>;
const LENS_KEY = 'ply-lens';
const apply = async (track: MediaStreamTrack, c: Record<string, any>) => { try { await track.applyConstraints({ advanced: [c] } as any); return true; } catch { return false; } };
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

export function savedLens() { try { return localStorage.getItem(LENS_KEY); } catch { return null; } }
export function saveLens(id: string) { try { localStorage.setItem(LENS_KEY, id); } catch { /* ignore */ } }

export async function openCamera(deviceId?: string | null): Promise<{ stream: MediaStream; track: MediaStreamTrack; info: CamInfo }> {
  if (!window.isSecureContext) throw Object.assign(new Error('Camera needs a secure (https) page'), { name: 'InsecureContext' });
  if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error('This browser cannot open the camera'), { name: 'NoMediaDevices' });
  const size = { width: { ideal: 3840 }, height: { ideal: 2160 }, frameRate: { ideal: 30, max: 30 } };
  let stream: MediaStream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: deviceId ? { deviceId: { exact: deviceId }, ...size } : { facingMode: { ideal: 'environment' }, ...size } }); }
  catch (e) { if (!deviceId || (e as Error).name === 'NotAllowedError') throw e; return openCamera(null); } // a saved lens that vanished: fall back
  const track = stream.getVideoTracks()[0], s = track.getSettings();
  // logical multi-cameras (one device, several sensors behind a zoom control) start at their widest; 1× is the main sensor
  const caps = (track.getCapabilities?.() ?? {}) as Caps;
  if (caps.zoom && caps.zoom.min < 1 && caps.zoom.max >= 1) await apply(track, { zoom: 1 });
  return { stream, track, info: { width: s.width ?? 0, height: s.height ?? 0, fps: Math.round(s.frameRate ?? 30), label: track.label, deviceId: s.deviceId ?? '' } };
}

/** Rear lenses, main first. Labels only exist after camera permission. */
export async function listLenses(): Promise<Lens[]> {
  const all = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  const back = all.filter(d => /back|rear|environment|world/i.test(d.label));
  const list = (back.length ? back : all).map((d, i) => { const m = d.label.match(/camera2?\s*(\d+)/i); return { deviceId: d.deviceId, label: d.label, index: m ? +m[1] : 100 + i, name: '' }; });
  list.sort((a, b) => a.index - b.index);
  return list.map((l, i) => ({ ...l, name: i === 0 ? 'Main' : list.length === 3 && i === 1 ? 'Ultra-wide' : list.length === 3 && i === 2 ? 'Telephoto' : `Lens ${i + 1}` }));
}

/** The main sensor unless the user picked another: a saved choice, else the lowest-numbered rear camera. */
export async function preferredLens(current: string): Promise<string | null> {
  const lenses = await listLenses().catch(() => [] as Lens[]); if (!lenses.length) return null;
  const saved = savedLens(); if (saved && lenses.some(l => l.deviceId === saved)) return saved === current ? null : saved;
  return lenses[0].deviceId === current ? null : lenses[0].deviceId;
}

export function controls(track: MediaStreamTrack): CamControls {
  const c = (track.getCapabilities?.() ?? {}) as Caps;
  return {
    zoom: c.zoom ? { min: c.zoom.min, max: Math.min(c.zoom.max, 10), step: c.zoom.step || .1 } : null,
    ev: c.exposureCompensation ? { min: Math.max(-2, c.exposureCompensation.min), max: Math.min(2, c.exposureCompensation.max), step: c.exposureCompensation.step || 1 / 3 } : null,
    canLock: !!(c.exposureMode && (c.exposureMode.includes('manual') || c.exposureMode.includes('none') || c.exposureMode.includes('single-shot'))),
    poi: 'pointsOfInterest' in ((navigator.mediaDevices.getSupportedConstraints?.() ?? {}) as any),
  };
}

export const setZoom = (t: MediaStreamTrack, z: number) => apply(t, { zoom: z });
export const setEv = (t: MediaStreamTrack, ev: number) => apply(t, { exposureCompensation: ev });

/** Focus and meter on a point of the preview (0..1). */
export async function meterAt(t: MediaStreamTrack, x: number, y: number) {
  const c = (t.getCapabilities?.() ?? {}) as Caps;
  await apply(t, { pointsOfInterest: [{ x, y }], ...(c.exposureMode?.includes('continuous') ? { exposureMode: 'continuous' } : {}), ...(c.focusMode?.includes('continuous') ? { focusMode: 'continuous' } : c.focusMode?.includes('single-shot') ? { focusMode: 'single-shot' } : {}) });
}

export async function unlock(t: MediaStreamTrack) {
  const c = (t.getCapabilities?.() ?? {}) as Caps;
  await apply(t, { ...(c.exposureMode?.includes('continuous') ? { exposureMode: 'continuous' } : {}), ...(c.focusMode?.includes('continuous') ? { focusMode: 'continuous' } : {}), ...(c.whiteBalanceMode?.includes('continuous') ? { whiteBalanceMode: 'continuous' } : {}) });
}

/** Lock, then correct: `luma()` is the live highlight-ignoring median brightness (0..255) of the preview. */
export async function lockCamera(track: MediaStreamTrack, luma: () => number | null, at?: { x: number; y: number }): Promise<LockReport> {
  const caps = (track.getCapabilities?.() ?? {}) as Caps; const rep: LockReport = { unsupported: [] };
  if (at) await meterAt(track, at.x, at.y);
  await sleep(1100); // let auto exposure settle on the metered point
  const s = track.getSettings() as Settings;
  const TARGET = 118;
  if (caps.exposureMode?.includes('manual') && caps.exposureTime && caps.iso) {
    const want = Math.min(caps.exposureTime.max, Math.max(caps.exposureTime.min, 100)); // 100 × 100 µs = 1/100 s
    let iso = Math.round(s.iso && s.exposureTime ? s.iso * s.exposureTime / want : (caps.iso.min + caps.iso.max) / 4);
    const clampIso = (v: number) => Math.min(caps.iso.max, Math.max(caps.iso.min, Math.round(v)));
    iso = clampIso(iso);
    if (await apply(track, { exposureMode: 'manual', exposureTime: want, iso })) {
      // closed loop: brightness responds ~ exposure^(1/2.2), so correct ISO by the ratio to the power 2.2
      for (let k = 0; k < 8; k++) {
        await sleep(260); const L = luma(); if (L == null) break;
        if (Math.abs(L - TARGET) < 14) break;
        const next = clampIso(iso * Math.min(2.5, Math.max(.4, (TARGET / Math.max(L, 4)) ** 2.2)));
        if (next === iso) break; iso = next; await apply(track, { iso });
      }
      const L = luma();
      if (L != null && (L < 60 || L > 190)) { await unlock(track); return { unsupported: [], note: L > 190 ? 'Too bright to hold at 1/100 s, kept on Auto' : 'Too dark to hold at 1/100 s, kept on Auto' }; }
      rep.exposure = `1/${Math.round(10000 / want)} s · ISO ${iso}`;
    } else rep.unsupported.push('exposure');
  } else if (caps.exposureMode?.includes('none') || caps.exposureMode?.includes('single-shot')) {
    rep.exposure = (await apply(track, { exposureMode: caps.exposureMode.includes('none') ? 'none' : 'single-shot' })) ? 'held' : undefined;
    if (!rep.exposure) rep.unsupported.push('exposure');
  } else rep.unsupported.push('exposure');

  if (caps.focusMode?.includes('manual') && s.focusDistance !== undefined) {
    const fd = Math.min(4, Math.max(s.focusDistance, Math.min(caps.focusDistance?.max ?? 1.5, 1.5)));
    rep.focus = (await apply(track, { focusMode: 'manual', focusDistance: fd })) ? `${fd.toFixed(1)} m` : undefined;
  } else if (caps.focusMode?.includes('single-shot')) rep.focus = (await apply(track, { focusMode: 'single-shot' })) ? 'held' : undefined;
  if (!rep.focus) rep.unsupported.push('focus');
  if (caps.whiteBalanceMode?.includes('manual') && s.colorTemperature) rep.whiteBalance = (await apply(track, { whiteBalanceMode: 'manual', colorTemperature: s.colorTemperature })) ? `${Math.round(s.colorTemperature)} K` : undefined;
  else if (caps.whiteBalanceMode?.includes('single-shot')) rep.whiteBalance = (await apply(track, { whiteBalanceMode: 'single-shot' })) ? 'held' : undefined;
  if (!rep.whiteBalance) rep.unsupported.push('white balance');
  return rep;
}

/** MP4/H.264 where the browser can (what the pipeline's ffmpeg likes best), WebM otherwise. */
export function pickRecorderType() {
  const list = ['video/mp4;codecs=avc1.640033', 'video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  return list.find(t => window.MediaRecorder?.isTypeSupported?.(t)) ?? '';
}
export const extFor = (mime: string) => (mime.startsWith('video/mp4') ? 'mp4' : 'webm');
/** Bits per second: enough that compression doesn't smear the texture SfM matches on. */
export const bitrateFor = (w: number, h: number) => (Math.max(w, h) >= 3000 ? 40e6 : Math.max(w, h) >= 1900 ? 16e6 : 8e6);
