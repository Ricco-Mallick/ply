/* Capture visualisations: the walked path coloured by quality, the ring of directions filmed, and the quality timeline. */
import { useEffect, useRef } from 'react';
import type { Quality } from '../lib/analyzer';
import type { SensorSample } from '../lib/sensors';

export const GRADE_COL = ['#3DDC97', '#FFB84D', '#FF5A4E'];
const GRADE_NAME = ['good', 'okay', 'recapture'];

/** Dead-reckoned path (top-down) with each stretch coloured by the worst grade recorded while walking it. */
export function PathMap({ path, quality, size = 132, live = false, label = true }: { path: { x: number; z: number; t: number }[]; quality: Quality[]; size?: number; live?: boolean; label?: boolean }) {
  const cv = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = cv.current!, dpr = Math.min(2, devicePixelRatio); c.width = c.height = size * dpr; const x = c.getContext('2d')!; x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, size, size);
    const pts = path.length ? path : [{ x: 0, z: 0, t: 0 }];
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity; for (const p of pts) { minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z); }
    const span = Math.max(4, maxX - minX, maxZ - minZ), sc = (size - 24) / span, cx = size / 2 - (minX + maxX) / 2 * sc, cz = size / 2 - (minZ + maxZ) / 2 * sc;
    // metre grid
    x.strokeStyle = 'rgba(255,255,255,.07)'; x.lineWidth = 1; for (let m = Math.floor(minX - span); m < maxX + span; m++) { const X = cx + m * sc; x.beginPath(); x.moveTo(X, 0); x.lineTo(X, size); x.stroke(); }
    for (let m = Math.floor(minZ - span); m < maxZ + span; m++) { const Z = cz + m * sc; x.beginPath(); x.moveTo(0, Z); x.lineTo(size, Z); x.stroke(); }
    const gradeAt = (t: number) => { let g = 0; for (const q of quality) { if (q.t > t) break; if (q.t > t - 700) g = Math.max(g, q.grade); } return g; };
    x.lineCap = 'round'; x.lineJoin = 'round'; x.lineWidth = 4;
    for (let i = 1; i < pts.length; i++) { x.strokeStyle = GRADE_COL[gradeAt(pts[i].t)]; x.beginPath(); x.moveTo(cx + pts[i - 1].x * sc, cz + pts[i - 1].z * sc); x.lineTo(cx + pts[i].x * sc, cz + pts[i].z * sc); x.stroke(); }
    x.fillStyle = '#fff'; x.beginPath(); x.arc(cx, cz, 4, 0, 7); x.fill(); x.strokeStyle = '#0B1220'; x.lineWidth = 1.5; x.stroke();
    const last = pts[pts.length - 1]; if (live || pts.length > 1) { x.fillStyle = '#FF6B00'; x.beginPath(); x.arc(cx + last.x * sc, cz + last.z * sc, 5, 0, 7); x.fill(); }
  }, [path, quality, size, live, path.length, quality.length]);
  return <canvas ref={cv} style={{ width: size, height: size, display: 'block' }} role="img" aria-label={label ? `Walked path, ${path.length - 1} steps, coloured by capture quality` : undefined} aria-hidden={!label} />;
}

/** Which compass directions have been filmed, each 10° bin coloured by its best grade so far. */
export function HeadingRing({ sensors, quality, heading, size = 132 }: { sensors: SensorSample[]; quality: Quality[]; heading: number | null; size?: number }) {
  const bins = new Array(36).fill(-1);
  for (let i = 0; i < sensors.length; i++) { const h = sensors[i].heading; if (h == null) continue; const b = Math.floor(((h % 360) + 360) % 360 / 10), g = quality[i]?.grade ?? 1; bins[b] = bins[b] < 0 ? g : Math.min(bins[b], g); }
  const covered = bins.filter(b => b >= 0 && b < 2).length, r = size / 2 - 10, c = size / 2;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Directions filmed: ${Math.round(covered / 36 * 100)} percent`}>
      {bins.map((g, i) => { const a0 = (i * 10 - 90) * Math.PI / 180, a1 = ((i + 1) * 10 - 91) * Math.PI / 180;
        return <path key={i} d={`M${c + Math.cos(a0) * r} ${c + Math.sin(a0) * r}A${r} ${r} 0 0 1 ${c + Math.cos(a1) * r} ${c + Math.sin(a1) * r}`} stroke={g < 0 ? 'rgba(255,255,255,.14)' : GRADE_COL[g]} strokeWidth="7" fill="none" />; })}
      {heading != null && <g transform={`rotate(${heading} ${c} ${c})`}><circle cx={c} cy={c - r} r="6.5" fill="#fff" stroke="#0B1220" strokeWidth="2" /></g>}
      <text x={c} y={c + (size < 110 ? 5 : -2)} textAnchor="middle" className="ring-n">{Math.round(covered / 36 * 100)}%</text>
      {size >= 110 && <text x={c} y={c + 14} textAnchor="middle" className="ring-l">FILMED</text>}
    </svg>
  );
}

/** Green/orange/red strip over the take; tap to seek. */
export function Timeline({ quality, duration, current, onSeek, height = 28 }: { quality: Quality[]; duration: number; current?: number; onSeek?: (t: number) => void; height?: number }) {
  const cv = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = cv.current!, w = c.clientWidth || 300, dpr = Math.min(2, devicePixelRatio); c.width = w * dpr; c.height = height * dpr; const x = c.getContext('2d')!; x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.clearRect(0, 0, w, height); const D = Math.max(.1, duration);
    for (let i = 0; i < quality.length; i++) { const q = quality[i], t1 = quality[i + 1]?.t ?? q.t + 125; x.fillStyle = GRADE_COL[q.grade]; x.fillRect(q.t / 1000 / D * w, 0, Math.max(1, (t1 - q.t) / 1000 / D * w + .5), height); }
  }, [quality, duration, height]);
  const pct = current != null ? Math.min(100, current / Math.max(.1, duration) * 100) : null;
  return (
    <div className="timeline" style={{ height }} onPointerDown={e => { if (!onSeek) return; const r = e.currentTarget.getBoundingClientRect(); onSeek((e.clientX - r.left) / r.width * duration); }}
      role={onSeek ? 'slider' : 'img'} aria-label="Capture quality over time" aria-valuemin={0} aria-valuemax={Math.round(duration)} aria-valuenow={current != null ? Math.round(current) : undefined} tabIndex={onSeek ? 0 : -1}
      onKeyDown={e => { if (!onSeek || current == null) return; if (e.key === 'ArrowRight') onSeek(Math.min(duration, current + 2)); if (e.key === 'ArrowLeft') onSeek(Math.max(0, current - 2)); }}>
      <canvas ref={cv} style={{ width: '100%', height }} />
      {pct != null && <i className="playhead" style={{ left: `${pct}%` }} />}
    </div>
  );
}

export function Legend() {
  return <div className="legend mono">{GRADE_COL.map((c, i) => <span key={i}><i style={{ background: c }} />{GRADE_NAME[i]}</span>)}</div>;
}

export const fmtTime = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
export const fmtBytes = (b: number) => (b > 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
