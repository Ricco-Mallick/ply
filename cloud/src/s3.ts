// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* Minimal S3 client for Workers (AWS Signature V4 on WebCrypto): enough for Backblaze B2 / R2 / any S3-compatible store.
   - presign(): query-string signed URLs, so the phone and the GPU worker move bytes directly to and from storage
   - signedFetch(): header-signed calls for the few control requests (create / complete multipart, small puts, reads) */

export interface S3Conf { endpoint: string; region: string; bucket: string; keyId: string; secret: string; }

const enc = new TextEncoder();
const hex = (b: ArrayBuffer) => [...new Uint8Array(b)].map(x => x.toString(16).padStart(2, '0')).join('');
const sha256 = async (d: string | ArrayBuffer) => hex(await crypto.subtle.digest('SHA-256', (typeof d === 'string' ? enc.encode(d) : d) as BufferSource));
async function hmac(key: ArrayBuffer | Uint8Array, msg: string) {
  const k = await crypto.subtle.importKey('raw', key as BufferSource, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return crypto.subtle.sign('HMAC', k, enc.encode(msg));
}
/** RFC 3986 encoding as SigV4 wants it; `/` kept in object keys. */
export const uriEncode = (s: string, keepSlash = false) => { const e = encodeURIComponent(s).replace(/[!'()*]/g, c => `%${c.charCodeAt(0).toString(16).toUpperCase()}`); return keepSlash ? e.replace(/%2F/g, '/') : e; };
const amzDate = (d: Date) => d.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');

async function signingKey(secret: string, date: string, region: string) {
  let k = await hmac(enc.encode(`AWS4${secret}`), date); k = await hmac(k, region); k = await hmac(k, 's3'); return hmac(k, 'aws4_request');
}
function canonicalQuery(q: Record<string, string>) {
  return Object.keys(q).sort().map(k => `${uriEncode(k)}=${uriEncode(q[k])}`).join('&');
}
const host = (c: S3Conf) => new URL(c.endpoint).host;
const objPath = (c: S3Conf, key: string) => `/${c.bucket}/${uriEncode(key, true)}`;

/** Query-signed URL (path-style). Only `host` is signed, so callers may add Content-Type etc. */
export async function presign(c: S3Conf, method: string, key: string, query: Record<string, string> = {}, expires = 3600, now = new Date()) {
  const t = amzDate(now), day = t.slice(0, 8), scope = `${day}/${c.region}/s3/aws4_request`;
  const q: Record<string, string> = { ...query, 'X-Amz-Algorithm': 'AWS4-HMAC-SHA256', 'X-Amz-Credential': `${c.keyId}/${scope}`, 'X-Amz-Date': t, 'X-Amz-Expires': String(expires), 'X-Amz-SignedHeaders': 'host' };
  const path = objPath(c, key), cq = canonicalQuery(q);
  const creq = [method, path, cq, `host:${host(c)}\n`, 'host', 'UNSIGNED-PAYLOAD'].join('\n');
  const sts = ['AWS4-HMAC-SHA256', t, scope, await sha256(creq)].join('\n');
  const sig = hex(await hmac(await signingKey(c.secret, day, c.region), sts));
  return `${c.endpoint.replace(/\/$/, '')}${path}?${cq}&X-Amz-Signature=${sig}`;
}

/** Header-signed request. Body must be buffered (small control payloads only). */
export async function signedFetch(c: S3Conf, method: string, key: string, query: Record<string, string> = {}, body?: ArrayBuffer | string, headers: Record<string, string> = {}, now = new Date()) {
  const t = amzDate(now), day = t.slice(0, 8), scope = `${day}/${c.region}/s3/aws4_request`;
  const payload = body === undefined ? '' : body; const ph = await sha256(typeof payload === 'string' ? payload : payload);
  const h: Record<string, string> = { host: host(c), 'x-amz-content-sha256': ph, 'x-amz-date': t, ...Object.fromEntries(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v])) };
  const names = Object.keys(h).sort(), path = objPath(c, key), cq = canonicalQuery(query);
  const creq = [method, path, cq, names.map(n => `${n}:${String(h[n]).trim()}\n`).join(''), names.join(';'), ph].join('\n');
  const sts = ['AWS4-HMAC-SHA256', t, scope, await sha256(creq)].join('\n');
  const sig = hex(await hmac(await signingKey(c.secret, day, c.region), sts));
  const { host: _h, ...send } = h;
  return fetch(`${c.endpoint.replace(/\/$/, '')}${path}${cq ? `?${cq}` : ''}`, { method, body, headers: { ...send, authorization: `AWS4-HMAC-SHA256 Credential=${c.keyId}/${scope}, SignedHeaders=${names.join(';')}, Signature=${sig}` } });
}

const xmlTag = (xml: string, tag: string) => xml.match(new RegExp(`<${tag}>([^<]*)</${tag}>`))?.[1] ?? null;

export async function createMultipart(c: S3Conf, key: string, contentType: string) {
  const r = await signedFetch(c, 'POST', key, { uploads: '' }, '', { 'content-type': contentType });
  const x = await r.text(); if (!r.ok) throw new Error(`storage refused the upload (${r.status}): ${x.slice(0, 200)}`);
  const id = xmlTag(x, 'UploadId'); if (!id) throw new Error('storage returned no UploadId'); return id;
}
export async function completeMultipart(c: S3Conf, key: string, uploadId: string, parts: { partNumber: number; etag: string }[]) {
  const body = `<CompleteMultipartUpload>${parts.map(p => `<Part><PartNumber>${p.partNumber}</PartNumber><ETag>${p.etag.startsWith('"') ? p.etag : `"${p.etag}"`}</ETag></Part>`).join('')}</CompleteMultipartUpload>`;
  const r = await signedFetch(c, 'POST', key, { uploadId }, body, { 'content-type': 'application/xml' });
  const x = await r.text(); if (!r.ok || x.includes('<Error>')) throw new Error(`storage could not join the parts (${r.status}): ${x.slice(0, 200)}`);
}
