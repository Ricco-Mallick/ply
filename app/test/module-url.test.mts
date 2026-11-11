/* Which ?module= links the model page will import. Run: node --experimental-strip-types test/module-url.test.mts */
import { sameOriginModuleUrl } from '../src/model/module-url.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
const PROD = 'https://ricco.codes/ply/model.html?scene=x&module=y', DEV = 'http://localhost:5199/ply/model.html';
const OK = 'https://ricco.codes/ply/models/classroom2.js';

check('a path under the base (the link the library builds)', sameOriginModuleUrl('/ply/models/classroom2.js', PROD) === OK);
check('the same path on the dev server', sameOriginModuleUrl('/ply/models/classroom2.js', DEV) === 'http://localhost:5199/ply/models/classroom2.js');
check('an absolute URL on this origin', sameOriginModuleUrl(OK, PROD) === OK);
check('a relative path resolves against the page, not the bundle', sameOriginModuleUrl('models/classroom.js', PROD) === 'https://ricco.codes/ply/models/classroom.js' && sameOriginModuleUrl('./models/classroom.js', PROD) === 'https://ricco.codes/ply/models/classroom.js');
check('query and hash survive', sameOriginModuleUrl('/ply/models/a.js?v=2#x', PROD) === 'https://ricco.codes/ply/models/a.js?v=2#x');
check('another origin is refused', sameOriginModuleUrl('https://evil.example/x.js', PROD) === null);
check('another port is refused', sameOriginModuleUrl('http://localhost:5200/ply/models/a.js', DEV) === null);
check('http for an https site is refused', sameOriginModuleUrl('http://ricco.codes/ply/models/a.js', PROD) === null);
check('a look-alike host is refused', ['https://ricco.codes.evil.example/x.js', 'https://ricco.codes@evil.example/x.js', 'https://evil.example/https://ricco.codes/x.js'].every(u => sameOriginModuleUrl(u, PROD) === null));
check('protocol-relative //host/x.js is refused', sameOriginModuleUrl('//evil.example/x.js', PROD) === null);
check('backslash tricks do not change the host', sameOriginModuleUrl('/\\evil.example/x.js', PROD) === null && sameOriginModuleUrl('\\\\evil.example\\x.js', PROD) === null);
check('data: is refused', sameOriginModuleUrl('data:text/javascript,alert(1)', PROD) === null);
check('blob: is refused even though it names this origin', sameOriginModuleUrl('blob:https://ricco.codes/3f2c1d0e-0000-4000-8000-000000000000', PROD) === null);
check('javascript: is refused', sameOriginModuleUrl('javascript:alert(1)', PROD) === null && sameOriginModuleUrl(' JavaScript:alert(1)', PROD) === null);
check('file: and ftp: are refused', sameOriginModuleUrl('file:///etc/passwd', PROD) === null && sameOriginModuleUrl('ftp://ricco.codes/x.js', PROD) === null);
check('garbage is refused', ['', '   ', 'http://', 'https://[::1', 'http://exa mple.com/x.js'].every(u => sameOriginModuleUrl(u, PROD) === null));
check('a page that is not an http(s) origin never matches', sameOriginModuleUrl('/ply/models/a.js', 'about:blank') === null && sameOriginModuleUrl('data:text/javascript,1', 'data:text/html,x') === null);
check('a bad base is refused, not thrown', sameOriginModuleUrl('/ply/models/a.js', 'not a url') === null);
process.exit(fail ? 1 : 0);
