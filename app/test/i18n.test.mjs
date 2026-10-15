// Korean coverage of the viewer: every translatable string has an entry in js/i18n.js, so no label stays English in
// the Korean interface. Checked: the static page (data-i18n text exactly as written, data-i18n-html keys,
// data-i18n-attr attribute values), every literal t('...') in js/app.js, the word tables app.js passes to t()
// (statuses, titles, legs, map sides, building classes, events) and the class names of the exported scenes.
//
//   node --test app/test/i18n.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const APP = join(dirname(fileURLToPath(import.meta.url)), '..');
globalThis.location = { hash: '#lang=ko' };                   // i18n.js reads the language from the address
const { t, LANG } = await import('../js/i18n.js');
const html = readFileSync(join(APP, 'index.html'), 'utf8');
const js = readFileSync(join(APP, 'js', 'app.js'), 'utf8');
const quoted = (s) => [...s.matchAll(/'((?:[^'\\]|\\.)*)'/g)].map(m => m[1].replace(/\\'/g, "'"));

function missing(keys) {
  return [...new Set(keys)].filter(k => t(k) === k);
}

test('the address picks Korean', () => assert.equal(LANG, 'ko'));

test('static page: data-i18n text, data-i18n-html keys, data-i18n-attr values', () => {
  const text = [...html.matchAll(/<\w+[^>]*\sdata-i18n(?=[\s>])[^>]*>([^<]*)</g)].map(m => m[1]);
  const keys = [...html.matchAll(/data-i18n-html="([^"]+)"/g)].map(m => m[1]);
  const attrs = [];
  for (const m of html.matchAll(/<\w+([^>]*\sdata-i18n-attr="([^"]+)"[^>]*)>/g)) {
    for (const pair of m[2].split('|')) {
      const name = pair.split(':')[0], v = m[1].match(new RegExp(`\\s${name}="([^"]*)"`));
      assert.ok(v, `attribute ${name} listed in data-i18n-attr but missing: ${m[0].slice(0, 80)}`);
      attrs.push(v[1]);
    }
  }
  assert.ok(text.length > 40 && keys.length >= 4 && attrs.length > 10, 'the page parse found the translatable strings');
  assert.deepEqual(missing([...text, ...keys, ...attrs]), []);
});

test("app.js: literal t('...') calls and the word tables passed to t()", () => {
  const literal = [...js.matchAll(/\bt\('((?:[^'\\]|\\.)*)'\)/g)].map(m => m[1].replace(/\\'/g, "'"));
  const table = (name) => quoted(js.match(new RegExp(`const ${name} = ([\\s\\S]*?);\\n`))[1]);
  const words = [...table('WORD'), ...table('TITLE'),
    ...[...js.matchAll(/name: '([^']+)', icon/g)].map(m => m[1]),                  // LEG
    ...quoted(js.match(/const SIDE = ({[^}]*})/)[1]),
    ...[...js.match(/const CLASS_LABEL = {([\s\S]*?)\n};/)[1].matchAll(/\d: '([^']+)'/g)].map(m => m[1]),
    ...[...js.matchAll(/event: '([^']+)'/g)].map(m => m[1])];
  assert.ok(literal.length > 20 && words.length > 15, 'the parse found the strings');
  assert.deepEqual(missing([...literal, ...words]), []);
});

test('class names of the exported scenes', () => {
  const index = JSON.parse(readFileSync(join(APP, 'scenes', 'scenes.json'), 'utf8'));
  const names = [];
  for (const ds of ['floodnet', 'rescuenet']) {
    const sc = index.scenes.find(s => s.dataset === ds);
    names.push(...JSON.parse(readFileSync(join(APP, 'scenes', sc.key, 'meta.json'), 'utf8')).class_names);
  }
  assert.deepEqual(missing(names), []);
});
