/** The two rules the API enforces beyond the gate: what an invite may spend, and when results are public. */

/** What an invite may spend: one job in flight, and a daily ceiling (plan 0.3). */
export const INVITE_ACTIVE_LIMIT = 1;
export const INVITE_DAILY_LIMIT = 10;
export function inviteBudget(active: number, today: number): string | null {
  if (active >= INVITE_ACTIVE_LIMIT) return `an invite can have ${INVITE_ACTIVE_LIMIT} job in flight; this one is still ${active > 1 ? 'waiting' : 'running'}`;
  if (today >= INVITE_DAILY_LIMIT) return `an invite can start ${INVITE_DAILY_LIMIT} jobs a day`;
  return null;
}

/** Results are served only for a job a human has seen (review) or approved (done), plan 0.4. */
export const PUBLISHED = new Set(['review', 'done']);
export function resultsVisible(status: string | null | undefined, file: string, publishedOnly: boolean): boolean {
  if (!publishedOnly) return true;
  if (!status) return true;                       // legacy scenes on the results host with no job row
  if (PUBLISHED.has(status)) return true;
  return file.startsWith('live/') || file === 'live-index.json' || file === 'pipeline.log' || file === 'report.json';
}
