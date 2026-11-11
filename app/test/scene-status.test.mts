/* Which badge a scene URL gets. Run: node --experimental-strip-types test/scene-status.test.mts */
import { sceneStatus } from '../src/shared/scene-status.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
const PAGE = 'https://ricco.codes/ply/walk.html?scene=x';
const API = 'https://ply-api.example/v1/scenes';
const UUID = '6f1c2b9e-3d4a-4e8b-9c1d-0a7b5e2f4d31';

check('a listed id gets its badge', sceneStatus(`${API}/chapel-high-2/scene.json`, PAGE)?.label === 'Latest pipeline');
check('an old run keeps its honest label', sceneStatus(`${API}/hq-chapel-1/scene.json`, PAGE)?.tone === 'old');
check('the first capture and the contract-test scene keep theirs', sceneStatus(`${API}/0ffc07fb-4d8d-49eb-9a40-22b9c6fddd76/`, PAGE)?.tone === 'old' && sceneStatus('/ply/scenes/synth-test/scene.json', PAGE)?.tone === 'old');
check('a fresh job id (UUID) gets no badge', sceneStatus(`${API}/${UUID}/scene.json`, PAGE) === null);
check('an unlisted name gets no badge', sceneStatus(`${API}/drjohnson-high/scene.json`, PAGE) === null);
check('Object.prototype names are not ids', ['constructor', '__proto__', 'toString', 'hasOwnProperty'].every(n => sceneStatus(`${API}/${n}/scene.json`, PAGE) === null));
check('a folder URL works too', sceneStatus(`${API}/chapel-high-2/`, PAGE)?.tone === 'new');
check('a relative scene URL resolves against the page', sceneStatus('/ply/scenes/c09a4c82-ca08-47b3-9850-e8e30a54d064/scene.json', PAGE)?.label === 'Latest pipeline');
check('a relative URL with no base is not a scene', sceneStatus('/ply/scenes/c09a4c82-ca08-47b3-9850-e8e30a54d064/scene.json', undefined) === null);
check('an absolute URL needs no base', sceneStatus(`${API}/c09a4c82-ca08-47b3-9850-e8e30a54d064/scene.json`, undefined)?.tone === 'new');
check('a URL with no /scenes/<id>/ segment gets nothing', sceneStatus('https://example.com/other/c09a4c82-ca08-47b3-9850-e8e30a54d064/scene.json', PAGE) === null);
check('a malformed URL gets nothing', sceneStatus('http://', PAGE) === null && sceneStatus('https://[::1', PAGE) === null);
check('null, undefined and empty get nothing', sceneStatus(null, PAGE) === null && sceneStatus(undefined, PAGE) === null && sceneStatus('', PAGE) === null);
check('a bad base does not throw', sceneStatus('scenes/c09a4c82-ca08-47b3-9850-e8e30a54d064/scene.json', 'not a url') === null);
process.exit(fail ? 1 : 0);
