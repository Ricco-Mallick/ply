/* Machines: PCs that join as workers by pairing (pipeline/local/start.bat), each with its own revocable token.

   A borrowed laptop is not trusted like a Modal container. It gets no shared secret: the zip holds none. The PC asks
   for a pairing code and shows it, the team approves the code (scripts/plyctl.py machines approve), and the PC's next
   poll receives a token made for it alone ("pm_<name>.<secret>"; only its SHA-256 is stored).
   With that token the machine can do exactly three things, and only for jobs the team assigned to it by name:
   claim one, report on it, and read its inputs and write scenes/<that job>/. It never sees the shared queue that
   Modal and Kaggle take from, and a revoke ends it on the next request. */

export const PAIR_TTL_MS = 24 * 3600 * 1000;
export const PAIR_PER_HOUR = 20;           // pending pairings the whole API accepts per hour (no auth on /pair)
const CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';  // no 0/O, 1/I/L: read off a screen and typed

const enc = new TextEncoder();

export function randomHex(bytes: number): string {
  const b = new Uint8Array(bytes);
  crypto.getRandomValues(b);
  return Array.from(b, x => x.toString(16).padStart(2, '0')).join('');
}

export function pairCode(): string {
  const b = new Uint8Array(8);
  crypto.getRandomValues(b);
  const s = Array.from(b, x => CODE_ALPHABET[x % CODE_ALPHABET.length]).join('');
  return `${s.slice(0, 4)}-${s.slice(4)}`;
}

export function normCode(code: unknown): string {
  const s = String(code ?? '').toUpperCase().replace(/[^A-Z0-9]/g, '');
  return s.length === 8 ? `${s.slice(0, 4)}-${s.slice(4)}` : '';
}

export async function sha256hex(s: string): Promise<string> {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', enc.encode(s)));
  return Array.from(d, x => x.toString(16).padStart(2, '0')).join('');
}

/** A machine's name: what the PC calls itself, cleaned, plus a short random tail so two "DESKTOP-..." never collide. */
export function machineName(hint: unknown): string {
  const base = String(hint ?? '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 28) || 'pc';
  return `pc-${base}-${randomHex(2)}`;
}

export function parseMachineAuth(header: string | null): { name: string; secret: string } | null {
  const m = /^Bearer pm_([a-z0-9-]{3,48})\.([0-9a-f]{64})$/.exec(header ?? '');
  return m ? { name: m[1], secret: m[2] } : null;
}

/** Paths a machine token may use at all; everything else answers as if it had no key. */
export function machinePathAllowed(path: string): boolean {
  return path === '/v1/worker/claim' || path === '/v1/worker/hello' || path === '/v1/worker/file' || /^\/v1\/worker\/jobs\/[\w-]+$/.test(path)
    || path === '/v1/uploads' || path === '/v1/uploads/sign' || path === '/v1/uploads/complete' || path === '/v1/files';
}

/** Storage keys a machine may write: its current job's results, nothing else. */
export function machineKeyAllowed(key: string, busyJob: string | null): boolean {
  if (!busyJob || !key || key.includes('..') || key.startsWith('/')) return false;
  return key.startsWith(`scenes/${busyJob}/`) || key.startsWith(`restore/${busyJob}/`);
}

/** Inputs a machine may read: the files of its current job, and that job's own checkpoints and results. */
export function machineReadAllowed(key: string, busyJob: string | null, jobFiles: string[]): boolean {
  if (!busyJob || !key || key.includes('..')) return false;
  return jobFiles.includes(key) || key.startsWith(`scenes/${busyJob}/`) || key.startsWith(`restore/${busyJob}/`);
}

/** The specs a machine reports (plypipe.device's plan), cut to what the team list shows. */
export function trimSpecs(specs: any): string | null {
  if (!specs || typeof specs !== 'object') return null;
  const g = specs.machine?.gpu ?? specs.gpu ?? null;
  const out = {
    gpu: g ? { name: String(g.name ?? '').slice(0, 60), totalGB: Number(g.totalGB) || null, sm: Number(g.sm) || null } : null,
    ramGB: Number(specs.machine?.ramGB ?? specs.ramGB) || null,
    cpus: Number(specs.machine?.cpus ?? specs.cpus) || null,
    freeDiskGB: Number(specs.machine?.freeDiskGB ?? specs.freeDiskGB) || null,
    trainable: specs.trainable === undefined ? null : !!specs.trainable,
    sam3: specs.sam3 === undefined ? null : !!specs.sam3,
    maxGaussians: Number(specs.maxGaussians) || null,
    why: String(specs.why ?? '').slice(0, 400),
  };
  return JSON.stringify(out);
}
