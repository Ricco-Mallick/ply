/* Writing tar archives in the browser (the AR scan streams keyframes into one file). Checked against Python's
   tarfile by app/scripts/tar-check.mjs. */
/** Minimal ustar writer: one header block per file, data padded to 512 bytes, two zero blocks at the end. */
export function tarHeader(name: string, size: number, mtime = Math.floor(Date.now() / 1000)) {
  const h = new Uint8Array(512), enc = new TextEncoder();
  const put = (s: string, off: number, len: number) => h.set(enc.encode(s).slice(0, len), off);
  const oct = (n: number, len: number) => n.toString(8).padStart(len - 1, '0') + '\0';
  put(name, 0, 100); put(oct(0o644, 8), 100, 8); put(oct(0, 8), 108, 8); put(oct(0, 8), 116, 8); put(oct(size, 12), 124, 12); put(oct(mtime, 12), 136, 12);
  put('        ', 148, 8); put('0', 156, 1); put('ustar\0', 257, 6); put('00', 263, 2);
  let sum = 0; for (const b of h) sum += b; put(sum.toString(8).padStart(6, '0') + '\0 ', 148, 8);
  return h;
}
export const tarPad = (size: number) => new Uint8Array((512 - (size % 512)) % 512);
