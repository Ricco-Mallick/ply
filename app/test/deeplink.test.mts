/* The model page's hand-off: walk.html?scene=<encoded>&place=<id>. Run:
   node --experimental-strip-types test/deeplink.test.mts */
import { buildWalkHref, formatAt, isPlaceId, parseAt, parseDeepLink, PLACE_ID, shareHref } from '../src/walk/deeplink.ts';
import { isPlaceId as editorPlaceId, PLACE_ID as EDITOR_PLACE_ID } from '../src/lab/place-edit.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
const SCENE = 'https://ply-api.example/v1/scenes/6f1c2b9e-3d4a-4e8b-9c1d-0a7b5e2f4d31/scene.json';

// ---------- reading a link
check('scene and place are both read', (() => { const d = parseDeepLink('?scene=x&place=door'); return d.scene === 'x' && d.place === 'door'; })());
check('a link with no query asks for nothing', (() => { const d = parseDeepLink(''); return d.scene === null && d.place === null; })());
check('scene alone leaves the place unset', (() => { const d = parseDeepLink('scene=x'); return d.scene === 'x' && d.place === null; })());
check('the encoded scene URL comes back whole', parseDeepLink(`?scene=${encodeURIComponent(SCENE)}`).scene === SCENE);
check('a scene URL with its own query survives the trip', (() => { const s = `${SCENE}?v=2&at=1,2`; return parseDeepLink(buildWalkHref(s, 'door')).scene === s; })());
check('a whole walk.html URL is read like a query', (() => { const d = parseDeepLink(`https://ricco.codes/ply/walk.html?scene=${encodeURIComponent(SCENE)}&place=desk`); return d.scene === SCENE && d.place === 'desk'; })());
check('a fragment is not part of the values', parseDeepLink('walk.html?scene=x&place=door#top').place === 'door');
check('spaces around the id are ignored', parseDeepLink('?place=%20door%20').place === 'door');
check('the first of two place parameters wins', parseDeepLink('?place=door&place=desk').place === 'door');

// ---------- a place id the link may not invent
check('an uppercase id is refused', parseDeepLink('?place=Door').place === null);
check('an id with a space is refused', parseDeepLink('?place=front%20door').place === null);
check('an id over 41 characters is refused', parseDeepLink(`?place=${'x'.repeat(42)}`).place === null);
check('markup in the place parameter is refused', parseDeepLink('?place=%3Cscript%3E').place === null);
check('an empty place is refused', parseDeepLink('?place=').place === null);
check('a contract-valid id with dashes and digits is accepted', parseDeepLink('?place=2nd-floor').place === '2nd-floor');
check('a refused place does not lose the scene', (() => { const d = parseDeepLink('?scene=x&place=Bad'); return d.scene === 'x' && d.place === null; })());

// ---------- writing the link
const href = buildWalkHref(SCENE, 'door');
check('the navigate link is walk.html with an encoded scene and the place', href.startsWith('walk.html?scene=') && href.endsWith('&place=door') && !href.includes(SCENE));
check('the link round-trips through the parser', (() => { const d = parseDeepLink(href); return d.scene === SCENE && d.place === 'door'; })());
check('walk.html can live under a base path', buildWalkHref(SCENE, 'door', '/ply/walk.html').startsWith('/ply/walk.html?scene='));
check('no place means no place parameter', !buildWalkHref(SCENE, null).includes('place=') && !buildWalkHref(SCENE, 'Bad Id').includes('place='));
check('a scene with a query still round-trips', (() => { const s = `${SCENE}?cam=1`; return parseDeepLink(buildWalkHref(s, 'desk')).scene === s; })());
check('an encoded ampersand in the scene is decoded, not split', parseDeepLink(buildWalkHref('a&b=c', 'door')).scene === 'a&b=c');

// ---------- one id rule, two copies
const ids = ['desk', '2nd-floor', 'x'.repeat(41), 'x'.repeat(42), 'Desk', '-door', 'door-', 'a_b', '', ' crème'];
check('both modules use the same id rule', ids.every(v => isPlaceId(v) === editorPlaceId(v)));
check('both modules carry the same id pattern', PLACE_ID.source === EDITOR_PLACE_ID.source && PLACE_ID.flags === EDITOR_PLACE_ID.flags);

// ---------- a shared view (?at=)
check('a shared view is read', (() => { const a = parseAt('1.5,0,-2.25,0.785,-0.1,f'); return !!a && a.x === 1.5 && a.z === -2.25 && a.yaw === 0.785 && a.pitch === -0.1 && a.mode === 'fly'; })());
check('walk is the default mode', parseAt('1,0,2,0,0')?.mode === 'walk');
check('too few or too many numbers are refused', parseAt('1,2,3,4') === null && parseAt('1,2,3,4,0,w,9') === null);
check('a non-number is refused', parseAt('1,2,x,4,0') === null && parseAt('1,2,Infinity,4,0') === null);
check('a camera 6 km out is refused', parseAt('6000,0,0,0,0') === null);
check('a pitch past straight up is refused', parseAt('0,0,0,0,2') === null);
check('an unknown mode is refused', parseAt('0,0,0,0,0,x') === null);
check('a view survives format and parse', (() => { const p = { x: 1.234, y: 0, z: -9.87, yaw: 3.14159, pitch: -0.25, mode: 'walk' as const }; const q = parseAt(formatAt(p))!; return Math.abs(q.x - 1.23) < 1e-9 && Math.abs(q.yaw - 3.142) < 1e-9 && q.mode === 'walk'; })());
check('the share link keeps the scene, drops the place and adds the view', (() => {
  const h = shareHref(`https://ricco.codes/ply/walk.html?scene=${encodeURIComponent(SCENE)}&place=door#x`, { x: 1, y: 0, z: 2, yaw: 0.5, pitch: 0, mode: 'walk' });
  const d = parseDeepLink(h); return d.scene === SCENE && d.place === null && d.at?.x === 1 && !h.includes('#');
})());
check('embed=1 is read, anything else is not embed', parseDeepLink('?embed=1').embed && !parseDeepLink('?embed=0').embed && !parseDeepLink('').embed);

process.exit(fail ? 1 : 0);
