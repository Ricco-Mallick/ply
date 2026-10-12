/* SigV4 signing checked against botocore (the AWS SDK), same keys and a frozen clock. Expected values were produced by
   botocore 1.x S3SigV4QueryAuth / S3SigV4Auth. Run: node --experimental-strip-types test/sigv4.test.mts */
import { presign, signedFetch } from '../src/s3.ts';
const c = { endpoint: 'https://s3.us-west-004.backblazeb2.com', region: 'us-west-004', bucket: 'ply-data', keyId: '004abc0123456789000000001', secret: 'K004SECRETsecretSECRETsecret/+xyz' };
const now = new Date(Date.UTC(2026, 9, 12, 16, 30, 5));
const want = {
 "get": "https://s3.us-west-004.backblazeb2.com/ply-data/scenes/abc/scene.spz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=004abc0123456789000000001%2F20261012%2Fus-west-004%2Fs3%2Faws4_request&X-Amz-Date=20261012T163005Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=449d3675582adfd6dcb06aadd9bde9e43a4b6ca6384ed899d2102d3653527397",
 "part": "https://s3.us-west-004.backblazeb2.com/ply-data/captures/c%201/clip-01%20%282%29.mp4?partNumber=3&uploadId=u%2Fx%2By%3Dz&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=004abc0123456789000000001%2F20261012%2Fus-west-004%2Fs3%2Faws4_request&X-Amz-Date=20261012T163005Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=ccb297e59fad744d6b4be5ee997bf3c66a19c566fecffbdb3360a6659594b5f9",
 "post": "AWS4-HMAC-SHA256 Credential=004abc0123456789000000001/20261012/us-west-004/s3/aws4_request, SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, Signature=64eb7d7b573409b7ea7f20ca209ce907d07a0d014bd1549e729ebe25f17e55bc"
};
const norm = (u: string) => { const x = new URL(u); return x.origin + x.pathname + '?' + [...x.searchParams.entries()].sort().map(([k, v]) => k + '=' + v).join('&'); };
let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
check('presigned GET', norm(await presign(c, 'GET', 'scenes/abc/scene.spz', {}, 3600, now)) === norm(want.get));
check('presigned part PUT (spaces, brackets, uploadId with /+=)', norm(await presign(c, 'PUT', 'captures/c 1/clip-01 (2).mp4', { partNumber: '3', uploadId: 'u/x+y=z' }, 3600, now)) === norm(want.part));
let auth = ''; globalThis.fetch = (async (_u: string, init: any) => { auth = init.headers.authorization; return new Response(''); }) as any;
await signedFetch(c, 'POST', 'captures/c1/a.mp4', { uploads: '' }, '', { 'content-type': 'video/mp4' }, now);
check('header-signed POST ?uploads (empty query value)', auth === want.post);
process.exit(fail ? 1 : 0);
