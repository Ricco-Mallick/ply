/* Places as data: the id / name / category / position rules, the slug for a new tag, the tolerant read of a
   places.json and the file a failed save offers as a download. Run: node --experimental-strip-types test/place-edit.test.mts */
import { asPoint, CATEGORIES, cleanName, isPlaceId, MAX_PLACES, NAME_MAX, normalisePlace, parsePlacesPayload, placesFile, slugId, validatePlaces } from '../src/lab/place-edit.ts';
import type { Place } from '../src/lab/places.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };
const place = (id: string, name = id, cat = 'Things', p: [number, number, number] = [0, 0, 0]): Place => ({ id, name, cat: cat as Place['cat'], p });
const fields = (list: unknown, index: number) => validatePlaces(list).issues.filter(i => i.index === index).map(i => i.field);

// ---------- names
check('a name is trimmed and its runs of spaces collapse', cleanName('  Study   desk \n') === 'Study desk');
check('a name stops at 80 characters', cleanName('x'.repeat(120)).length === NAME_MAX);
check('an empty name stays empty', cleanName(undefined) === '' && cleanName('   ') === '');

// ---------- ids
check('the id rule from the contract accepts a normal tag id', ['desk', 'fire-exit-2', 'a', '2nd-floor', 'x'.repeat(41)].every(isPlaceId));
check('the id rule refuses uppercase, spaces, a leading dash, an underscore and 42 characters', ['Desk', 'fire exit', '-door', 'x'.repeat(42), 'a_b', ''].every(v => !isPlaceId(v)));
check('the contract’s rule lets a trailing dash through, so the editor never makes one', isPlaceId('door-') && slugId('door-') === 'door');
check('a slug from a name is lowercase words joined by dashes', slugId('Fire Exit 2') === 'fire-exit-2');
check('a slug folds accents', slugId('Crème brûlée') === 'creme-brulee');
check('a slug of punctuation only is "place"', slugId('!!!') === 'place' && slugId('') === 'place');
check('a slug of digits is allowed by the contract', slugId('2nd floor') === '2nd-floor');
check('a long name gives an id of at most 41 characters and no trailing dash', (() => { const id = slugId('A very long place name that keeps going past the limit'); return id.length <= 41 && id === id.replace(/-+$/, ''); })());
check('a slug avoids the ids already in the scene', slugId('Door', ['door']) === 'door-2' && slugId('Door', ['door', 'door-2']) === 'door-3');
check('a slug shortened to make room for the suffix still fits', slugId('x'.repeat(39) + ' a', ['x'.repeat(39) + '-a']).length <= 41);
check('every slug matches the id rule', ['Front door', '  ', 'A/B', 'Ω omega', 'x'.repeat(90)].every(n => isPlaceId(slugId(n, ['place']))));

// ---------- a whole list
const good = [place('desk', 'Desk'), place('door', 'Main door', 'Exits')];
check('a good list passes', validatePlaces(good).ok && validatePlaces(good).issues.length === 0);
check('unknown fields are not a problem', validatePlaces([{ ...good[0], note: 'kept', conf: 0.4 }]).ok);
check('a list that is not a list names the list', !validatePlaces({ places: good }).ok && validatePlaces(null).issues[0].field === 'places');
check('a bad id names the entry and the field', fields([good[0], place('Bad Id', 'Bad')], 1).join() === 'id');
check('a duplicate id names the second entry', fields([place('desk'), place('desk', 'Desk two')], 1).join() === 'id');
check('a missing name names the name field', fields([{ ...place('x'), name: '   ' }], 0).join() === 'name');
check('a name over 80 characters is refused', fields([{ ...place('x'), name: 'x'.repeat(81) }], 0).join() === 'name');
check('an unknown category is refused', fields([{ ...place('x'), cat: 'Rooms' }], 0).join() === 'cat');
check('every category the dialog offers is accepted', CATEGORIES.every(c => validatePlaces([place('x', 'X', c)]).ok));
check('a position that is not three numbers is refused', fields([{ ...place('x'), p: [0, 0] }], 0).join() === 'p');
check('a position with a string in it is refused', fields([{ ...place('x'), p: [0, 'up', 0] }], 0).join() === 'p');
check('an entry that is not an object names its index', validatePlaces([good[0], 7]).issues.some(i => i.index === 1 && i.field === 'places'));
const many = Array.from({ length: MAX_PLACES + 1 }, (_, i) => place(`p${i}`));
check('a list over 500 names the list, not an entry', validatePlaces(many).issues.some(i => i.index === -1 && i.field === 'places'));
check('exactly 500 places is allowed', validatePlaces(many.slice(0, MAX_PLACES)).ok);

// ---------- reading a places.json however it was written
const file = { version: 1, source: 'sam3', places: [place('desk', 'Desk'), { ...place('lamp', 'Lamp'), ai: { cls: 'lamp', conf: 0.8, half: [0.1, 0.2, 0.1] } }] };
check('the pipeline shape reads its places', parsePlacesPayload(file).map(p => p.id).join() === 'desk,lamp');
check('a bare array reads too', parsePlacesPayload([place('door', 'Door')]).length === 1);
check('an API payload reads the same way', parsePlacesPayload({ schema: 1, updatedAt: null, places: [place('bin', 'Bin')] })[0].id === 'bin');
check('fields the reader does not know are carried through', (parsePlacesPayload([{ ...place('x'), note: 'hi' }])[0] as any).note === 'hi');
check('junk entries are dropped, not fatal', parsePlacesPayload({ places: [place('ok'), null, 7, { id: 'Bad' }, { id: 'no-p', name: 'No p', cat: 'Things' }, { id: 'no-cat', name: 'No cat', p: [0, 0, 0] }] }).map(p => p.id).join() === 'ok');
check('an empty or unknown payload is an empty list', parsePlacesPayload(null).length === 0 && parsePlacesPayload({ foo: [] }).length === 0 && parsePlacesPayload('nope').length === 0);
check('a payload is capped at 500 places', parsePlacesPayload({ places: Array.from({ length: MAX_PLACES + 20 }, (_, i) => place(`p${i}`)) }).length === MAX_PLACES);

// ---------- building the place the dialog saves
check('a new place gets a slug id, a clean name and the category', (() => {
  const p = normalisePlace({ name: '  Front   Door ', cat: 'Exits', p: [1, 2, 3] });
  return p.id === 'front-door' && p.name === 'Front Door' && p.cat === 'Exits' && p.p.join() === '1,2,3';
})());
check('a new place keeps its id unique against the scene', normalisePlace({ name: 'Door', cat: 'Exits', p: [0, 0, 0] }, ['door']).id === 'door-2');
check('an edited place keeps the id it had', normalisePlace({ id: 'door-2', name: 'Side door', cat: 'Exits', p: [0, 0, 0] }, ['door']).id === 'door-2');
check('an edited place keeps its AI box', (() => {
  const ai = { cls: 'door', conf: 0.9, half: [0.4, 1, 0.1] as [number, number, number] };
  return normalisePlace({ id: 'door', name: 'Door', cat: 'Exits', p: [0, 0, 0], ai }).ai?.cls === 'door';
})());
check('a human tag is pinned', normalisePlace({ name: 'Bench', cat: 'Furniture', p: [0, 0, 0], pinned: true }).pinned === true);
check('a bad id in the input is replaced by a slug', normalisePlace({ id: 'Not An Id', name: 'Good Name', cat: 'Things', p: [0, 0, 0] }).id === 'good-name');
check('asPoint reads numbers and refuses anything else', asPoint([1, '2', 3])?.join() === '1,2,3' && asPoint([1, 2]) === null && asPoint('nope') === null);

// ---------- the download
const text = placesFile(good), back = parsePlacesPayload(JSON.parse(text));
check('the export is a places.json the pipeline can read', back.length === 2 && back[0].id === 'desk' && back[0].p.join() === '0,0,0');
check('the export says what wrote it', (() => { const j = JSON.parse(text); return j.version === 1 && /walkthrough/.test(j.source) && Array.isArray(j.places); })());
check('the export ends with a newline', text.endsWith('\n'));
check('the export passes the same validation as a save', validatePlaces(JSON.parse(text).places).ok);

process.exit(fail ? 1 : 0);
