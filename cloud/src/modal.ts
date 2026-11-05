/* Wake the Modal GPU worker (pipeline/modal/ply_modal.py `trigger`): it starts unless one is already up, and stops
   itself when the queue is empty.

   The URL lives in wrangler.toml (MODAL_TRIGGER_URL) so it changes with the code when the worker moves account; the
   older hand-set secret MODAL_TRIGGER is tried after it. The call carries WORKER_TOKEN too, which the trigger accepts
   as well as the separate trigger token: the API and the worker already share it for every other call. */

export interface ModalEnv { MODAL_TRIGGER_URL?: string; MODAL_TRIGGER?: string; MODAL_TRIGGER_TOKEN?: string; WORKER_TOKEN?: string }

export type Kick = { ok: boolean; started?: boolean; error?: string; status?: number; url?: string; [k: string]: unknown };

export async function kickModal(env: ModalEnv, why: string): Promise<Kick | null> {
  const urls = [...new Set([env.MODAL_TRIGGER_URL, env.MODAL_TRIGGER].filter((u): u is string => !!u))];
  if (!urls.length || !(env.MODAL_TRIGGER_TOKEN || env.WORKER_TOKEN)) return null;   // no Modal worker configured
  const body = JSON.stringify({ token: env.MODAL_TRIGGER_TOKEN ?? '', workerToken: env.WORKER_TOKEN ?? '', why });
  let last: Kick = { ok: false, error: 'no trigger answered' };
  for (const url of urls) {
    try {
      const r = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body });
      const text = await r.text();
      let res: Kick | null = null;
      try { res = JSON.parse(text); } catch { /* an app that isn't deployed answers 404 with plain text */ }
      if (r.ok && res && res.ok) return res;
      last = { ...(res ?? {}), ok: false, status: r.status, url: url.replace(/^https:\/\/([^.]+).*/, '$1'), error: String(res?.error ?? text.slice(0, 120)) };
    } catch (e) {
      last = { ok: false, url: url.replace(/^https:\/\/([^.]+).*/, '$1'), error: (e as Error).message };
    }
  }
  console.log('modal trigger failed', JSON.stringify(last));
  return last;
}
