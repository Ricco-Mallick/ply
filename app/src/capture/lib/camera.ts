/* Camera: the best rear stream the phone will give, and locking exposure / focus / white balance once metered.
   Chrome on Android exposes these as MediaTrack constraints (Image Capture spec); Safari and desktop mostly don't, so
   every lock is best-effort and reported back so the UI can say what it could and couldn't hold. */

export interface CamInfo { width: number; height: number; fps: number; label: string; }
export interface LockReport { exposure?: string; focus?: string; whiteBalance?: string; unsupported: string[]; }

type Caps = MediaTrackCapabilities & Record<string, any>;
type Settings = MediaTrackSettings & Record<string, any>;

export async function openCamera(): Promise<{ stream: MediaStream; track: MediaStreamTrack; info: CamInfo }> {
  if (!window.isSecureContext) throw Object.assign(new Error('Camera needs a secure (https) page'), { name: 'InsecureContext' });
  if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error('This browser cannot open the camera'), { name: 'NoMediaDevices' });
  // 4K30 first; phones that can't fall back to their best mode on their own because these are `ideal`, not `exact`
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 3840 }, height: { ideal: 2160 }, frameRate: { ideal: 30, max: 30 } },
  });
  const track = stream.getVideoTracks()[0];
  const s = track.getSettings();
  return { stream, track, info: { width: s.width ?? 0, height: s.height ?? 0, fps: Math.round(s.frameRate ?? 30), label: track.label } };
}

/** Meter for a moment in auto, then hold what the camera chose (with the shutter pinned to 1/100 s against 50 Hz flicker). */
export async function lockCamera(track: MediaStreamTrack): Promise<LockReport> {
  const caps = (track.getCapabilities?.() ?? {}) as Caps; const rep: LockReport = { unsupported: [] };
  const apply = async (c: Record<string, any>) => { try { await track.applyConstraints({ advanced: [c] } as any); return true; } catch { return false; } };
  const s = track.getSettings() as Settings;

  if (caps.exposureMode?.includes('manual') && caps.exposureTime) {
    // exposureTime is in 100 µs units: 100 = 10 ms = 1/100 s. Keep the brightness the auto mode found by trading ISO.
    const want = Math.min(caps.exposureTime.max, Math.max(caps.exposureTime.min, 100));
    const iso = s.iso && s.exposureTime ? Math.round(s.iso * s.exposureTime / want) : undefined;
    const isoC = caps.iso && iso ? Math.min(caps.iso.max, Math.max(caps.iso.min, iso)) : undefined;
    if (await apply({ exposureMode: 'manual', exposureTime: want, ...(isoC ? { iso: isoC } : {}) })) rep.exposure = `1/${Math.round(10000 / want)} s${isoC ? ` · ISO ${isoC}` : ''}`;
    else rep.unsupported.push('exposure');
  } else if (caps.exposureMode?.includes('single-shot') || caps.exposureMode?.includes('none')) {
    rep.exposure = (await apply({ exposureMode: caps.exposureMode.includes('none') ? 'none' : 'single-shot' })) ? 'held' : undefined;
    if (!rep.exposure) rep.unsupported.push('exposure');
  } else rep.unsupported.push('exposure');

  if (caps.focusMode?.includes('manual') && s.focusDistance !== undefined) {
    // a room is mostly 1–4 m away; hold near the hyperfocal-ish 1.5 m if the metered distance is closer
    const fd = Math.min(4, Math.max(s.focusDistance, Math.min(caps.focusDistance?.max ?? 1.5, 1.5)));
    rep.focus = (await apply({ focusMode: 'manual', focusDistance: fd })) ? `${fd.toFixed(1)} m` : undefined;
  } else if (caps.focusMode?.includes('single-shot')) rep.focus = (await apply({ focusMode: 'single-shot' })) ? 'held' : undefined;
  if (!rep.focus) rep.unsupported.push('focus');

  if (caps.whiteBalanceMode?.includes('manual') && s.colorTemperature) rep.whiteBalance = (await apply({ whiteBalanceMode: 'manual', colorTemperature: s.colorTemperature })) ? `${Math.round(s.colorTemperature)} K` : undefined;
  else if (caps.whiteBalanceMode?.includes('single-shot')) rep.whiteBalance = (await apply({ whiteBalanceMode: 'single-shot' })) ? 'held' : undefined;
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
