import { useEffect, useMemo, useRef, useState } from 'react';
import { marked } from 'marked';

// The guide is docs/SOP.md, the same file submitted for R1, so the page and the document never drift apart.
const docs = import.meta.glob('../../../docs/SOP.md', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;
const SOP = Object.values(docs)[0] ?? '# Capture guide\n\nThe guide is being written.';
const BASE = import.meta.env.BASE_URL;
const KEY = 'ply-capture-checks';

const slug = (s: string) => s.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');

export function Capture() {
  const { html, toc, title, lead } = useMemo(() => {
    const md = SOP.replace(/\r/g, '');
    const t = /^#\s+(.+)$/m.exec(md)?.[1] ?? 'Capture guide';
    let body = md.replace(/^#\s+.+$/m, '').trim();
    const lead = /^(?!#)(.+)$/m.exec(body)?.[1] ?? '';
    if (lead) body = body.replace(lead, '').trim(); // shown big in the hero, not twice
    const renderer = new marked.Renderer();
    const toc: { id: string; text: string }[] = [];
    renderer.heading = ({ tokens, depth }) => { const text = marked.Parser.parseInline(tokens); const id = slug(text); if (depth === 2) toc.push({ id, text: text.replace(/<[^>]+>/g, '') }); return `<h${depth} id="${id}">${text}</h${depth}>`; };
    const html = marked.parse(body, { renderer, gfm: true, async: false }) as string;
    return { html, toc, title: t, lead };
  }, []);

  const root = useRef<HTMLDivElement>(null);
  const [done, setDone] = useState({ n: 0, of: 0 });
  const [active, setActive] = useState('');

  // GitHub task lists ("- [ ]") become real checkboxes, remembered on this device
  useEffect(() => {
    const el = root.current!; let saved: Record<string, boolean> = {};
    try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { /* private mode */ }
    const boxes = [...el.querySelectorAll<HTMLInputElement>('input[type=checkbox]')];
    const count = () => setDone({ n: boxes.filter(b => b.checked).length, of: boxes.length });
    boxes.forEach((b, i) => {
      const label = b.parentElement?.textContent?.trim() ?? `item ${i}`; b.disabled = false; b.checked = !!saved[label];
      b.setAttribute('aria-label', label); b.parentElement?.classList.add('task');
      b.onchange = () => { saved[label] = b.checked; try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch { /* ignore */ } count(); };
    });
    count();
    const io = new IntersectionObserver(es => { for (const e of es) if (e.isIntersecting) setActive(e.target.id); }, { rootMargin: '-20% 0px -70% 0px' });
    el.querySelectorAll('h2[id]').forEach(h => io.observe(h));
    return () => io.disconnect();
  }, [html]);

  const reset = () => { try { localStorage.removeItem(KEY); } catch { /* ignore */ } root.current?.querySelectorAll<HTMLInputElement>('input[type=checkbox]').forEach(b => { b.checked = false; }); setDone(d => ({ ...d, n: 0 })); };

  return (
    <div className="cap">
      <header className="cap-top">
        <a className="lg brand pill" href={BASE}><span className="mark" aria-hidden="true" />PLY</a>
        <a className="cta-ghost" href={`${BASE}walk.html`}>Explore the demo space →</a>
      </header>

      <section className="cap-hero">
        <div className="wide eyebrow">Capture guide · SOP</div>
        <h1>{title}</h1>
        {lead && <p className="lead" dangerouslySetInnerHTML={{ __html: marked.parseInline(lead) as string }} />}
        <WalkPattern />
      </section>

      <div className="cap-body">
        <nav className="toc" aria-label="Sections">
          <div className="wide toc-h">Sections</div>
          <ol>{toc.map(t => <li key={t.id}><a href={`#${t.id}`} className={active === t.id ? 'on' : ''}>{t.text}</a></li>)}</ol>
          {done.of > 0 && (
            <div className="progress">
              <div className="mono">Pre-flight {done.n}/{done.of}</div>
              <div className="pbar"><i style={{ transform: `scaleX(${done.of ? done.n / done.of : 0})` }} /></div>
              {done.n > 0 && <button className="mono reset" onClick={reset}>Reset ticks</button>}
            </div>
          )}
        </nav>
        <article className="prose" ref={root} dangerouslySetInnerHTML={{ __html: html }} />
      </div>
    </div>
  );
}

/** Top-down sketch of the capture walk: two perimeter loops (eye and chest height) and a slow doorway crossing. */
function WalkPattern() {
  return (
    <figure className="pattern">
      <svg viewBox="0 0 640 260" role="img" aria-label="Plan view: walk the room's perimeter twice, once at eye height and once at chest height, then cross each doorway slowly into the next room.">
        <defs><marker id="ar" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="currentColor" /></marker></defs>
        <g fill="none" stroke="rgba(255,255,255,.55)" strokeWidth="2"><path d="M300 110V20H20V240H300V150" /><path d="M300 20H620V240H300" /></g>
        <rect x="294" y="110" width="12" height="40" fill="#070B14" />
        <g fill="none" strokeWidth="2.2" strokeLinecap="round">
          <path className="loop1" d="M60 200V60H260V200H70" stroke="#FF6B00" markerEnd="url(#ar)" style={{ color: '#FF6B00' }} />
          <path className="loop2" d="M90 175V85H235V175H100" stroke="#FFB84D" strokeDasharray="6 7" markerEnd="url(#ar)" style={{ color: '#FFB84D' }} />
          <path className="door" d="M235 130H360" stroke="#fff" markerEnd="url(#ar)" style={{ color: '#fff' }} />
          <path className="loop1" d="M360 130V60H580V200H360V140" stroke="#FF6B00" markerEnd="url(#ar)" style={{ color: '#FF6B00' }} />
        </g>
        <g className="mono-svg"><text x="64" y="50">LOOP 1 · EYE HEIGHT</text><text x="96" y="160">LOOP 2 · CHEST</text><text x="312" y="122">DOOR · SLOW</text></g>
      </svg>
      <figcaption className="mono">Plan view · two loops per room, then cross each doorway slowly</figcaption>
    </figure>
  );
}
