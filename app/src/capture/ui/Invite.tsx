// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Invite only: there are no accounts to sign into. A signed code unlocks capture, the GPU queue and
   the status page; the worker verifies it on every request, and it expires without anyone revoking it. */
import { useState } from 'react';
import { Glass } from '../../shared/Glass';
import { setTeamKey, whoami } from '../lib/backend';

export function Invite({ onDone }: { onDone: () => void }) {
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!code.trim() || busy) return;
    setBusy(true); setErr(null);
    try { await whoami(code); setTeamKey(code); onDone(); }
    catch (x) { setErr((x as Error).message); }
    finally { setBusy(false); }
  };
  return (
    <main className="start invite-wrap" onSubmit={submit}>
      <section className="start-hero">
        <div className="wide eyebrow">Private preview</div>
        <h1>Invite only.</h1>
        <p className="lead">Ply has no sign-up and no accounts. A member of the team sends you a signed invite code. It unlocks the capture app and the GPU queue, it works without a password, and it expires on its own.</p>
      </section>
      <form className="choices" onSubmit={submit}>
        <Glass className="choice invite-card" opts={{ radius: 22 }}>
          <span className="choice-ico" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="4" y="10" width="16" height="10" rx="2.5" /><path d="M8 10V7a4 4 0 0 1 8 0v3" /><circle cx="12" cy="15" r="1.4" fill="currentColor" stroke="none" /></svg>
          </span>
          <span className="choice-t">Your invite code</span>
          <span className="choice-d">Paste the whole code, it looks like <span className="mono">name.1767225600.3f9a…</span>. Each one carries its own expiry.</span>
          <input className="invite-input" value={code} onChange={e => setCode(e.target.value)} placeholder="name.expiry.signature" autoComplete="off" spellCheck={false} aria-label="Invite code" />
          <button className="btn invite-go" type="submit" disabled={busy || !code.trim()}>{busy ? 'Checking…' : 'Unlock capture'}</button>
          {err && <span className="invite-err" role="alert">{err}</span>}
        </Glass>
      </form>
      <p className="mono dim invite-foot">No invite yet? Ask the team, or explore the demo space — the walkthrough stays open to everyone.</p>
    </main>
  );
}
