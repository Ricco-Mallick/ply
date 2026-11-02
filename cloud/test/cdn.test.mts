/* The CDN redirect against a stand-in Hugging Face. Run: node --experimental-strip-types test/cdn.test.mts */
import { cdnRedirect, isBulk } from '../src/cdn.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
const CDN = 'https://us.aws.cdn.hf.co/xet-bridge-us/abc?Expires=1793620528&Signature=x';
const calls: { url: string; init: any }[] = [];
globalThis.fetch = (async (url: string, init: any) => {
  calls.push({ url, init });
  if (url.endsWith('/missing.spz')) return new Response('', { status: 404 });
  if (url.endsWith('/small.spz')) return new Response('', { status: 200 });   // HF serves tiny files inline
  return new Response(null, { status: 302, headers: { location: CDN } });
}) as any;
class MemCache { m = new Map<string, Response>(); async match(r: Request) { return this.m.get(r.url)?.clone(); } async put(r: Request, res: Response) { this.m.set(r.url, res); } }

check('splats, meshes and LOD chunks are bulk', ['a/scene.spz', 'a/scene_lite.spz', 'a/mesh.ply', 'a/lod/x.rad', 'a/lod/x-3.radc'].every(isBulk));
check('JSON and PNG are not', !['a/scene.json', 'a/occupancy.png', 'a/report.json'].some(isBulk));

const cache = new MemCache();
const r = await cdnRedirect('Owner/repo', 'hf_secret', 'scenes/j1/scene.spz', 'https://api.example/v1/scenes/j1/scene.spz', cache);
check('stored file -> 307 to the signed CDN URL', r?.status === 307 && r.headers.get('location') === CDN);
check('the token goes to HF only, as a HEAD with manual redirects',
  calls[0].url === 'https://huggingface.co/datasets/Owner/repo/resolve/main/scenes/j1/scene.spz' &&
  calls[0].init.method === 'HEAD' && calls[0].init.redirect === 'manual' && calls[0].init.headers.authorization === 'Bearer hf_secret' &&
  !r!.headers.get('location')!.includes('hf_secret'));
check('browsers keep the redirect briefly (the signature expires)', r!.headers.get('cache-control') === 'private, max-age=300');

const again = await cdnRedirect('Owner/repo', 'hf_secret', 'scenes/j1/scene.spz', 'https://api.example/v1/scenes/j1/scene.spz', cache);
check('second view is served from the edge cache without asking HF', again?.headers.get('location') === CDN && calls.length === 1 && again.headers.get('x-ply-cdn') === 'cached');

check('not on HF -> null (the bucket path takes over)', await cdnRedirect('Owner/repo', 't', 'scenes/old/missing.spz', 'https://api.example/x', new MemCache()) === null);
check('served inline by HF -> null (proxied as before)', await cdnRedirect('Owner/repo', 't', 'scenes/j1/small.spz', 'https://api.example/x', new MemCache()) === null);
process.exit(fail ? 1 : 0);
