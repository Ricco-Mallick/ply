/* The search index: ranking tiers, prefix and plural matching, the synonym table, category filters and the empty
   query. Run: node --experimental-strip-types test/index.test.mts */
import { buildIndex, matchPlace, normalise, search, synonymsOf, tokenise, TIERS } from '../src/lab/index.ts';
import type { PlaceLike } from '../src/lab/index.ts';

let fail = 0; const check = (name: string, ok: boolean) => { console.log(ok ? 'ok  ' : 'FAIL', name); if (!ok) fail++; };

const at = (id: string, name: string, cat = 'Things', cls?: string): PlaceLike => ({ id, name, cat, ai: cls ? { cls } : null });
const rank = (places: PlaceLike[], q: string, cat?: string) => search(buildIndex(places), q, { cat }).map(h => h.id);
const score = (places: PlaceLike[], id: string, q: string) => search(buildIndex(places), q).find(h => h.id === id)?.score ?? -1;

// ---------- an empty query is the whole list
const room = [at('desk', 'Desk', 'Furniture', 'desk'), at('door', 'Main door', 'Exits', 'door'), at('bin', 'Dustbin', 'Storage', 'bin'), at('sofa', 'Sofa', 'Furniture', 'couch')];
const all = search(buildIndex(room), '');
check('an empty query returns every place, in the list’s own order', all.map(h => h.id).join() === 'desk,door,bin,sofa');
check('an empty query scores everything zero', all.every(h => h.score === 0));
check('a whitespace query is the empty query', search(buildIndex(room), '   ').length === room.length);
check('an empty index has nothing to return', search(buildIndex([]), 'desk').length === 0 && buildIndex([]).entries.length === 0);

// ---------- the ranking tiers, best first
const tiers = [at('exact', 'Desk'), at('prefix', 'Desk lamp'), at('token', 'Reading desk'), at('tokenPrefix', 'Oak desktop'), at('substring', 'Workdesk')];
check('exact name beats a name prefix', score(tiers, 'exact', 'desk') > score(tiers, 'prefix', 'desk'));
check('a name prefix beats a whole token', score(tiers, 'prefix', 'desk') > score(tiers, 'token', 'desk'));
check('a whole token beats a token prefix', score(tiers, 'token', 'desk') > score(tiers, 'tokenPrefix', 'desk'));
check('a token prefix beats a substring', score(tiers, 'tokenPrefix', 'desk') > score(tiers, 'substring', 'desk'));
check('the four kinds come back in that order', rank(tiers, 'desk').join() === 'exact,prefix,token,tokenPrefix,substring');
check('the tiers are the numbers the ranking uses', score(tiers, 'exact', 'desk') === TIERS.exact * 2 && score(tiers, 'substring', 'desk') === TIERS.substring);
check('scores never increase down the list', search(buildIndex(tiers), 'desk').every((h, i, a) => i === 0 || a[i - 1].score >= h.score));

// ---------- prefix, plurals, classes
const room2 = [at('door', 'Doorway', 'Exits', 'door'), at('window', 'Window', 'Things', 'window'), at('laptop', 'Laptop', 'Things', 'laptop'), at('shelf', 'Flower shelf', 'Storage', 'potted plant'), at('bins', 'Bins', 'Storage', 'dustbin')];
check('a prefix finds the word it starts', rank(room2, 'doo').includes('door'));
check('a prefix does not find an unrelated word', !rank(room2, 'win').includes('laptop'));
check('a plural query finds the singular name', rank(room2, 'laptops').includes('laptop'));
check('a singular query finds the plural name', rank(room2, 'bin').includes('bins'));
check('the AI class is searchable', rank(room2, 'potted').includes('shelf'));
check('the category is searchable', rank(room2, 'storage').join() === 'shelf,bins');
check('a token inside a longer word is found', rank([at('work', 'Woodwork bench')], 'work').includes('work'));

// ---------- synonyms, both ways
const syn = [at('toilet', 'Toilet', 'Storage'), at('wc', 'WC', 'Storage'), at('door', 'Main door', 'Exits'), at('exit', 'Fire exit', 'Exits'), at('sofa', 'Sofa', 'Furniture'), at('couch', 'Couch', 'Furniture'), at('dustbin', 'Dustbin', 'Storage'), at('bin', 'Bin', 'Storage'), at('ac', 'AC unit', 'Things', 'air conditioner')];
for (const [q, id] of [['wc', 'toilet'], ['toilet', 'wc'], ['exit', 'door'], ['door', 'exit'], ['sofa', 'couch'], ['couch', 'sofa'], ['dustbin', 'bin'], ['bin', 'dustbin'], ['ac', 'ac'], ['air conditioner', 'ac'], ['air', 'ac']] as const) {
  check(`“${q}” finds ${id} through the synonym table`, rank(syn, q).includes(id));
}
check('the direct word beats the synonym', rank(syn, 'wc').indexOf('wc') < rank(syn, 'wc').indexOf('toilet'));
check('a synonym match scores under the same word written plainly', score(syn, 'toilet', 'wc') < score(syn, 'wc', 'wc'));
check('an unknown word has no synonyms', synonymsOf('hydrant').length === 0);
check('the synonym table normalises its key', synonymsOf(' WC ').includes('toilet'));

// ---------- several words, categories, accents, stability
const room3 = [at('main', 'Main door', 'Exits'), at('side', 'Side door', 'Exits'), at('lamp', 'Door lamp', 'Furniture'), at('cafe', 'Café shelf', 'Furniture'), at('store', 'Storage box', 'Storage')];
check('two words must both be there', rank(room3, 'main door')[0] === 'main');
check('a word that is missing excludes the place', rank(room3, 'door hydrant').length === 0);
check('the category filter keeps only that category', rank(room3, 'door', 'Furniture').join() === 'lamp');
check('a category filter that matches nothing returns nothing', rank(room3, 'door', 'Things').length === 0);
check('accents and case do not matter', rank(room3, 'CAFE')[0] === 'cafe');
check('the query is trimmed', rank(room3, '  cafe  ')[0] === 'cafe');
check('equal scores keep the input order', rank([at('b', 'Same'), at('a', 'Same')], 'same').join() === 'b,a');
check('a limit stops after that many hits', search(buildIndex(room), '', { limit: 2 }).length === 2 && search(buildIndex(room), 'o', { limit: 2 }).length === 2);
check('no limit returns the whole ranked list', search(buildIndex(room), 'o').length > 2);

// ---------- the lab's filter wrapper
check('matchPlace accepts everything on an empty query', matchPlace(room[0], ''));
check('matchPlace filters like the palette', matchPlace(room[0], 'desk') && !matchPlace(room[0], 'hydrant'));

// ---------- tokenising
check('normalise folds accents, case and punctuation', normalise('  Crème  Brûlée! ') === 'creme brulee');
check('tokenise keeps words apart', tokenise('Potted plant / door').join('|') === 'potted|plant|door');
check('a duplicate id is indexed once', buildIndex([at('desk', 'Desk'), at('desk', 'Other desk')]).entries.length === 1);

process.exit(fail ? 1 : 0);
