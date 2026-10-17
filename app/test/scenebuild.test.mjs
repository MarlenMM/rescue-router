// Parity test of the in-browser upload (app/js/scenebuild.js, PLAN §16.10): from the same uint8 probabilities the JS
// builder must give exactly the scene Python builds for an uploaded photo (src/upload_fixtures.py ->
// app/test/fixtures/: the probabilities, Python's scene.rrb and its statuses): every cell array (classes, risk to the
// last float32 bit, both maps' flags), the entry cells, every house (id, class, centroid, its three goal rings), the
// grid parameters and the statuses the JS router finds. The outlines are drawn another way (display only): same
// count, similar area.
//
//   node --test app/test/scenebuild.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseScene, sceneGrids, STATUS } from '../js/router.js';
import { buildUploadScene, quantize } from '../js/scenebuild.js';

const FIX = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');
const index = JSON.parse(readFileSync(join(FIX, 'uploads.json'), 'utf8'));

const area = (poly) => Math.abs(poly.reduce((s, [x, y], i) => { const [x2, y2] = poly[(i + 1) % poly.length]; return s + x * y2 - x2 * y; }, 0)) / 2;
const firstDiff = (a, b) => { for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return i; return -1; };

for (const fx of index) {
  test(`upload ${fx.id}: the JS scene equals Python's`, async () => {
    const [C, H, W] = fx.shape;
    const probs = new Uint8Array(gunzipSync(readFileSync(join(FIX, `upload_${fx.id}.probs.gz`))));
    assert.equal(probs.length, C * H * W);
    const py = await parseScene(readFileSync(join(FIX, `upload_${fx.id}.rrb`)));
    const js = buildUploadScene(probs, H, W, 'floodnet', fx.decode.tau);
    const g = py.grid, j = js.grid;
    assert.equal(j.h, g.h); assert.equal(j.w, g.w);
    for (const [k, v] of Object.entries(g.params)) assert.deepEqual(j.params[k], v, `param ${k}`);
    assert.deepEqual(j.profile, g.profile);
    assert.deepEqual(j.eta, g.eta);
    assert.deepEqual(j.base, g.base, 'entry cells');
    for (const [layer, key] of [['pred', 'cls'], ['pred', 'risk'], ['pred', 'flags'], ['gt', 'cls'], ['gt', 'flags'],
                                ['pred_cautious', 'flags']]) {
      const a = j[layer][key], b = g[layer][key], i = firstDiff(a, b);
      assert.equal(a.length, b.length, `${layer}.${key} length`);
      assert.equal(i, -1, `${layer}.${key} differs first at cell ${i}: JS ${a[i]} Python ${b[i]}`);
    }
    assert.equal(js.buildings.length, py.buildings.length, 'houses');
    for (let i = 0; i < py.buildings.length; i++) {
      const a = js.buildings[i], b = py.buildings[i];
      assert.equal(a.id, b.id); assert.equal(a.cls, b.cls, `house ${i} class`);
      assert.deepEqual(a.centroid, b.centroid, `house ${i} centroid`);
      for (let m = 0; m < 3; m++) assert.deepEqual(a.goals[m], b.goals[m], `house ${i} goal ring ${m}`);
      const ra = area(a.polygon), rb = area(b.polygon);
      assert.ok(Math.abs(ra - rb) <= 0.05 * rb + 20, `house ${i} outline area ${ra} vs ${rb}`);
    }
    const sol = sceneGrids(j, 5).pred.solve(j.base);
    assert.deepEqual(js.buildings.map(b => STATUS[sol.status(b.goals)]), fx.status, 'statuses');
  });
}

test('quantize rounds half to even, as np.round', () => {
  assert.deepEqual([...quantize(new Float32Array([0, 1, 0.5 / 255, 1.5 / 255, 2.5 / 255, 0.2]))],
                   [0, 255, 0, 2, 2, 51]);
});
