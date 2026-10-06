// Parity test: the JS router (app/js/router.js) against the Python reference (src/router.py) on every
// exported scene, both maps, risk-aware (lam 5) and plain (lam 0) truck costs.
//
//   node --test app/test/
//
// Statuses must match exactly; route costs to 1e-9 relative (equal-cost ties may pick another path,
// whose float sum can differ in the last bits). Every JS route must also be a valid path whose
// re-summed cost equals its reported cost.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { sceneGrids, STATUS, TRUCK, BOAT } from '../js/router.js';

const SCENES = join(dirname(fileURLToPath(import.meta.url)), '..', 'scenes');
const index = JSON.parse(readFileSync(join(SCENES, 'scenes.json'), 'utf8'));

function recost(grid, r) {
  let c = 0;
  for (let k = 1; k < r.cells.length; k++) {
    const [y0, x0] = r.cells[k - 1], [y, x] = r.cells[k], b = y * grid.w + x;
    assert.ok(Math.max(Math.abs(y - y0), Math.abs(x - x0)) === 1, 'steps are 8-neighbours');
    assert.ok(r.modes[k] >= r.modes[k - 1], 'no switch back to truck');
    const L = y !== y0 && x !== x0 ? Math.SQRT2 : 1;
    if (r.modes[k] === TRUCK) { assert.ok(grid.truckOk[b]); c += grid.truckStep(L, b); }
    else { assert.ok(grid.boatOk[b]); c += L * grid.par.boat_cost + (r.modes[k - 1] === TRUCK ? grid.par.launch : 0); }
  }
  return c;
}

for (const s of index.scenes) {
  test(`${s.key}: statuses and costs match Python`, () => {
    const gj = JSON.parse(readFileSync(join(SCENES, s.key, 'grid.json'), 'utf8'));
    const blds = JSON.parse(readFileSync(join(SCENES, s.key, 'buildings.json'), 'utf8'));
    let checked = 0;
    for (const lam of [5, 0]) {
      const grids = sceneGrids(gj, lam);
      for (const name of ['pred', 'gt']) {
        const sol = grids[name].solve(gj.base);
        for (const b of blds) {
          const ref = b.py[`lam${lam}`]?.[name];
          if (!ref) continue;                                   // no base: nothing to compare
          assert.equal(STATUS[sol.status(b.ring)], ref.status, `${name} lam${lam} building ${b.id} status`);
          const r = sol.route(b.ring);
          if (ref.cost === null) { assert.equal(r, null); continue; }
          assert.ok(Math.abs(r.cost - ref.cost) <= 1e-9 * Math.max(1, ref.cost),
                    `${name} lam${lam} building ${b.id}: JS ${r.cost} vs Python ${ref.cost}`);
          assert.ok(Math.abs(recost(grids[name], r) - r.cost) <= 1e-9 * Math.max(1, r.cost));
          if (STATUS[sol.status(b.ring)] === 'BOAT') assert.equal(r.modes.at(-1), BOAT);
          checked++;
        }
      }
    }
    assert.ok(checked > 0 || blds.length === 0);
  });
}
