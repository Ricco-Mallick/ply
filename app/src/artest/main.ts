/* One-minute AR capability check for the live-coverage capture: does this phone's WebXR give depth (for the live
   mesh), the camera image (for keyframes) and at what size, and how steady is tracking? Prints a report to copy. */
import '../shared/tokens.css';

const root = document.getElementById('root')!;
root.innerHTML = `<main style="max-width:560px;margin:0 auto;padding:28px 18px;font-family:var(--font-body)">
  <div class="wide" style="color:var(--accent)">Ply · AR check</div>
  <h1 style="font:400 40px/1.05 var(--font-display);margin:10px 0 12px">What can this phone's AR do?</h1>
  <p style="color:var(--muted);margin:0 0 18px">Tap start, point at the floor and furniture, move slowly for 10 seconds. Then copy the report and send it.</p>
  <button id="go" style="height:48px;padding:0 24px;border:0;border-radius:999px;background:var(--accent);color:#0B1220;font:400 11px/1 var(--font-wide);letter-spacing:.16em;text-transform:uppercase">Start AR check</button>
  <pre id="out" style="white-space:pre-wrap;font:12px/1.6 var(--font-mono);color:var(--body);background:rgba(255,255,255,.05);padding:14px;border-radius:12px;margin-top:18px">checking support…</pre>
  <button id="copy" style="display:none;height:40px;padding:0 18px;border:1px solid var(--line-2);border-radius:999px;background:none;color:var(--fg)">Copy report</button>
</main>`;
const out = document.getElementById('out')!, rep: Record<string, unknown> = { ua: navigator.userAgent };
const show = () => { out.textContent = JSON.stringify(rep, null, 2); };
const xr = (navigator as any).xr;

(async () => {
  rep.webxr = !!xr; rep.immersiveAr = xr ? await xr.isSessionSupported('immersive-ar').catch(() => false) : false; show();
})();

document.getElementById('go')!.onclick = async () => {
  if (!rep.immersiveAr) { rep.error = 'immersive-ar not supported in this browser'; show(); return; }
  const overlay = document.querySelector('main')!;
  let session: any;
  try {
    session = await xr.requestSession('immersive-ar', {
      requiredFeatures: ['local-floor'],
      optionalFeatures: ['depth-sensing', 'camera-access', 'hit-test', 'dom-overlay', 'plane-detection', 'anchors'],
      depthSensing: { usagePreference: ['cpu-optimized', 'gpu-optimized'], dataFormatPreference: ['luminance-alpha', 'float32', 'unsigned-short'] },
      domOverlay: { root: overlay },
    });
  } catch (e) { rep.error = `requestSession failed: ${(e as Error).message}`; show(); return; }
  rep.enabledFeatures = session.enabledFeatures ?? 'not reported'; rep.depthUsage = session.depthUsage; rep.depthDataFormat = session.depthDataFormat;
  const canvas = document.createElement('canvas'); const gl = canvas.getContext('webgl2', { xrCompatible: true }) as WebGL2RenderingContext;
  await session.updateRenderState({ baseLayer: new (window as any).XRWebGLLayer(session, gl) });
  const space = await session.requestReferenceSpace('local-floor');
  const binding = 'XRWebGLBinding' in window ? new (window as any).XRWebGLBinding(session, gl) : null;
  let frames = 0, depthOk = 0, camOk = 0, lost = 0; const t0 = performance.now(); let lastPos: number[] | null = null, path = 0;
  const onFrame = (_t: number, frame: any) => {
    session.requestAnimationFrame(onFrame); frames++;
    const pose = frame.getViewerPose(space); if (!pose) { lost++; return; }
    const p = pose.transform.position; if (lastPos) path += Math.hypot(p.x - lastPos[0], p.y - lastPos[1], p.z - lastPos[2]); lastPos = [p.x, p.y, p.z];
    const view = pose.views[0];
    if (session.depthUsage === 'cpu-optimized') { try { const d = frame.getDepthInformation(view); if (d) { depthOk++; rep.depthSize = `${d.width}x${d.height}`; rep.depthMetersPerUnit = d.rawValueToMeters; } } catch (e) { rep.depthErr = String(e); } }
    else if (binding && session.depthUsage === 'gpu-optimized') { try { const d = binding.getDepthInformation(view); if (d) { depthOk++; rep.depthSize = `${d.width}x${d.height} (gpu)`; } } catch (e) { rep.depthErr = String(e); } }
    if (binding && view.camera) { try { const tex = binding.getCameraImage(view.camera); if (tex) { camOk++; rep.cameraImage = `${view.camera.width}x${view.camera.height}`; } } catch (e) { rep.camErr = String(e); } }
    gl.bindFramebuffer(gl.FRAMEBUFFER, session.renderState.baseLayer.framebuffer); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    const s = (performance.now() - t0) / 1000;
    if (frames % 15 === 0) { Object.assign(rep, { seconds: +s.toFixed(1), fps: +(frames / s).toFixed(1), framesWithDepth: depthOk, framesWithCamera: camOk, trackingLostFrames: lost, walkedMetres: +path.toFixed(2), eyeHeight: +p.y.toFixed(2) }); show(); }
    if (s > 12) session.end();
  };
  session.requestAnimationFrame(onFrame);
  session.addEventListener('end', () => { rep.done = true; show(); const c = document.getElementById('copy')!; c.style.display = 'inline-block'; c.onclick = () => navigator.clipboard?.writeText(out.textContent || ''); });
};
