import { useState } from 'react';
import { Glass } from '../../shared/Glass';

const CARDS = [
  { t: 'Lights on, people out', d: 'Switch on every light and prop the doors open. Anyone walking through the shot turns into a ghost in the twin.', art: 'light' },
  { t: 'Hold it like this', d: 'Upright, at chest height, tilted a little toward the floor. Ply shows a level line while you record.', art: 'hold' },
  { t: 'Slow, steady loops', d: 'About one step a second. Walk the edge of each room twice and turn with your whole body, not your wrist.', art: 'loop' },
  { t: 'Doors slowly, end at the start', d: 'Cross every doorway slowly so rooms join up, and finish where you began. Ply tells you when the loop closes.', art: 'door' },
];

function Art({ k }: { k: string }) {
  const s = { fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  if (k === 'light') return <svg viewBox="0 0 120 90" {...s}><path d="M60 14v8M38 22l5 6M82 22l-5 6" stroke="#FFB84D" /><path d="M48 54a12 12 0 1 1 24 0c0 6-6 8-6 14H54c0-6-6-8-6-14Z" /><path d="M54 74h12" /><circle cx="22" cy="58" r="5" opacity=".35" /><path d="M22 64v14M16 70l6-4 6 4" opacity=".35" /><path d="M12 50 32 82" stroke="#FF5A4E" /></svg>;
  if (k === 'hold') return <svg viewBox="0 0 120 90" {...s}><rect x="50" y="14" width="22" height="40" rx="4" transform="rotate(-12 61 34)" /><path d="M20 76h80" opacity=".4" /><path d="M66 50 88 78" stroke="#FF6B00" strokeDasharray="3 5" /><path d="M30 20v52" opacity=".4" /><text x="92" y="46" fill="#FFB84D" stroke="none" fontSize="11" fontFamily="Space Mono">10–15°</text></svg>;
  if (k === 'loop') return <svg viewBox="0 0 120 90" {...s}><rect x="14" y="10" width="92" height="70" rx="3" opacity=".4" /><rect x="26" y="20" width="68" height="50" rx="10" stroke="#FF6B00" /><rect x="38" y="30" width="44" height="30" rx="8" stroke="#FFB84D" strokeDasharray="4 5" /><path d="m88 40 6 6 6-6" stroke="#FF6B00" /></svg>;
  return <svg viewBox="0 0 120 90" {...s}><path d="M10 12h40v24M50 54v24H10V12M70 12h40v66H70V54M70 36V12" opacity=".4" /><path d="M30 45h60" stroke="#fff" /><path d="m84 39 6 6-6 6" stroke="#fff" /><circle cx="30" cy="45" r="4" fill="#3DDC97" stroke="none" /></svg>;
}

export function Tutorial({ onDone }: { onDone: () => void }) {
  const [i, setI] = useState(0); const c = CARDS[i], last = i === CARDS.length - 1;
  return (
    <main className="tut">
      <button className="skip mono" onClick={onDone}>Skip</button>
      <Glass className="tut-card" opts={{ radius: 26 }} key={i}>
        <div className="tut-art" aria-hidden="true"><Art k={c.art} /></div>
        <div className="wide eyebrow">{String(i + 1).padStart(2, '0')} / {String(CARDS.length).padStart(2, '0')}</div>
        <h2>{c.t}</h2>
        <p>{c.d}</p>
      </Glass>
      <div className="tut-dots" role="tablist" aria-label="Tips">{CARDS.map((_, k) => <button key={k} role="tab" aria-selected={k === i} aria-label={`Tip ${k + 1}`} className={k === i ? 'on' : ''} onClick={() => setI(k)} />)}</div>
      <div className="tut-nav">
        {i > 0 && <button className="btn ghost" onClick={() => setI(i - 1)}>Back</button>}
        <button className="btn big" onClick={() => (last ? onDone() : setI(i + 1))}>{last ? 'Start recording' : 'Next'}</button>
      </div>
    </main>
  );
}
