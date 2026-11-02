/* Large scene files go to the browser straight from Hugging Face's CDN.

   Results live in a private HF dataset repo. For a stored (LFS/Xet) file, HF's resolve endpoint answers with a
   302 to a signed CloudFront URL (cdn.hf.co, valid ~1 h, CORS *, Range supported) that needs no token. Proxying
   the bytes through the Worker instead ran as slow as ~7 kB/s from India for a 13 MB phone file; the redirect
   leaves only a few hundred bytes on the Worker and the download on HF's nearest edge. */

// files worth a redirect: splats, meshes, streaming-LOD chunks. JSON and PNG stay proxied (small, and scene.json
// must never be cached).
const BULK = /\.(spz|ply|rad|radc|sog|splat|ksplat|glb|bin)$/i;
export const isBulk = (key: string) => BULK.test(key);

// signed URLs last about an hour: hand out a cached one for at most 20 minutes, browsers keep it 5
const EDGE_TTL = 1200;
const BROWSER_TTL = 300;

type CacheLike = { match(req: Request): Promise<Response | undefined>; put(req: Request, res: Response): Promise<void> };

/** A 307 to the CDN copy of `key`, or null when HF does not hold it as a stored file (then proxy as before). */
export async function cdnRedirect(repo: string, token: string, key: string, reqUrl: string,
                                  cache?: CacheLike, waitUntil?: (p: Promise<unknown>) => void): Promise<Response | null> {
  const cacheKey = new Request(`${new URL(reqUrl).origin}/__cdn/${key}`, { method: 'GET' });
  const hit = cache && await cache.match(cacheKey);
  const loc = hit?.headers.get('location') ?? await resolve(repo, token, key);
  if (!loc) return null;
  if (cache && !hit) {
    const stored = new Response(null, { status: 307, headers: { location: loc, 'cache-control': `public, max-age=${EDGE_TTL}` } });
    const p = cache.put(cacheKey, stored).catch(() => undefined);
    waitUntil ? waitUntil(p) : await p;
  }
  return new Response(null, { status: 307, headers: {
    location: loc, 'cache-control': `private, max-age=${BROWSER_TTL}`, 'accept-ranges': 'bytes', 'x-ply-cdn': hit ? 'cached' : 'resolved' } });
}

async function resolve(repo: string, token: string, key: string): Promise<string | null> {
  const r = await fetch(`https://huggingface.co/datasets/${repo}/resolve/main/${key}`,
                        { method: 'HEAD', headers: { authorization: `Bearer ${token}` }, redirect: 'manual' });
  const loc = r.headers.get('location');
  return loc && r.status >= 300 && r.status < 400 && /^https:\/\//.test(loc) ? loc : null;
}
