// The in-browser upload (PLAN §16.10): a routing scene built from a model's class probabilities in the browser, as
// Python builds it for an uploaded photo (src/gradio_viewer.export_upload): the threshold decode (rr.metrics.decode),
// 4x4-px cells (router.cell_classes / cell_probs), each map's passability flags (router.Grid: off-road gate, closed
// car gaps, vehicle widths, canopy strips), the entry cells (router.base_cells), the houses (router.buildings_from_gt,
// on the prediction: an upload has no labels), their goal rings per mode (router.Grid.goal) and outlines
// (export_web_assets.polygon). It returns what router.js parseScene returns, so the viewer routes it like any scene.
// app/test/scenebuild.test.mjs checks every array, house and goal against Python's scene on the same probabilities.
//
// NumPy semantics kept on purpose: float32 arithmetic (Math.fround; a float32 operation done in float64 and rounded
// once is exact for + - * /), Python floats compared with float32 arrays in float32 (NumPy 2), SciPy's binary
// erosion with the outside as 0, labels in raster order of each component's first pixel, exact Euclidean distances.

const f32 = Math.fround;
const CELL = 4, N_CH = 5, OTHER = 5;
export const TRUCK = 0, BOAT = 1, FOOT = 2;
const STATUS = ['TRUCK', 'BOAT', 'FOOT', 'ISOLATED'];

// = router.PROFILES / DEFAULTS / ETA and route_eval's gates (the parity test compares them with a Python header)
export const PROFILES = {
  floodnet: { truck: [0], boat: [1, 4], wall: [2, 3], risk: [1, 4], bldg: [2, 3], hazard: [1, 4], walk: [], cell_m: 0.66 },
  rescuenet: { truck: [0], boat: [4], wall: [1, 2, 3], risk: [1, 4], bldg: [2, 3], hazard: [1, 4], walk: [1], cell_m: 0.31 },
};
export const PARAMS = { lam: 5.0, boat_cost: 3.0, launch: 40.0, boat_p: 0.5, offroad: 4.0, foot_cost: 8.0, alight: 10.0,
  canopy_cost: 2.0, corner: true, truck_halfwidth: 1.25, boat_halfwidth: 0.8, car_gap: 2.0, canopy_gap: 3.0 };
export const GATES = { floodnet: { pred: 0.3, cautious: 0.05 }, rescuenet: { pred: 0.2, cautious: 0.15 } };
export const ETA = { truck: 20.0, truck_offroad: 5.0, boat: 6.0, boat_canopy: 3.0, foot: 3.0, launch_min: 5.0, alight_min: 1.0 };
const TIE = [0, 3, 5, 4, 2, 1].map(x => x * 1e-3);   // on equal counts the less drivable class wins

/** rr.metrics.decode: channel c is a candidate if p_c >= tau_c; the largest p_c / tau_c wins; else OTHER.
 *  probs: Uint8Array (C * H * W) as the caches store them (p * 255, rounded). */
export function decode(probs, H, W, tau) {
  const n = H * W, out = new Uint8Array(n), t = tau.map(f32);
  for (let i = 0; i < n; i++) {
    let best = 0, bs = -Infinity;
    for (let c = 0; c < N_CH; c++) {
      const s = f32(f32(probs[c * n + i] / 255) / t[c]);
      if (s > bs) { bs = s; best = c; }                       // first maximum, as np.argmax
    }
    out[i] = bs >= 1 ? best : OTHER;
  }
  return out;
}

/** Softmax probabilities (Float32Array C * H * W) -> uint8 as np.round(p * 255) (round half to even). */
export function quantize(p) {
  const out = new Uint8Array(p.length);
  for (let i = 0; i < p.length; i++) {
    const x = f32(p[i] * 255), r = Math.round(x);
    out[i] = (r - x === 0.5 && r % 2) ? r - 1 : r;              // Math.round rounds .5 up; NumPy to even
  }
  return out;
}

/** router.cell_classes: majority vote per 4x4 cell (ties: the less drivable class). */
export function cellClasses(cls, H, W) {
  const h = (H / CELL) | 0, w = (W / CELL) | 0, out = new Uint8Array(h * w), cnt = new Float64Array(N_CH + 1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      cnt.fill(0);
      for (let dy = 0; dy < CELL; dy++) for (let dx = 0; dx < CELL; dx++) {
        const k = cls[(y * CELL + dy) * W + x * CELL + dx];
        if (k <= OTHER) cnt[k] += 1;
      }
      let best = 0, bv = -Infinity, tot = 0;
      for (let k = 0; k <= OTHER; k++) {
        tot += cnt[k];
        const v = cnt[k] + TIE[k];
        if (v > bv) { bv = v; best = k; }
      }
      out[y * w + x] = tot ? best : OTHER;
    }
  }
  return out;
}

/** router.cell_probs: per-cell means of the uint8 probabilities / 255, float32 (C * h * w). */
export function cellProbs(probs, H, W) {
  const h = (H / CELL) | 0, w = (W / CELL) | 0, n = H * W, out = new Float32Array(N_CH * h * w);
  for (let c = 0; c < N_CH; c++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let dy = 0; dy < CELL; dy++) for (let dx = 0; dx < CELL; dx++) s += probs[c * n + (y * CELL + dy) * W + x * CELL + dx];
        out[c * h * w + y * w + x] = f32(f32(s / 16) / 255);  // a sum of 16 integers is exact in float32
      }
    }
  }
  return out;
}

// ---------- binary morphology on (h, w) Uint8Array masks, SciPy semantics ----------
function disk(r) {                                             // router._disk
  const k = Math.max(Math.ceil(r), 1), rr = Math.max(r, 1) ** 2, off = [];
  for (let dy = -k; dy <= k; dy++) for (let dx = -k; dx <= k; dx++) if (dx * dx + dy * dy <= rr) off.push([dy, dx]);
  return off;
}
const SQUARE = [-1, 0, 1].flatMap(dy => [-1, 0, 1].map(dx => [dy, dx]));
const CROSS = [[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]];

function dilate(m, h, w, off) {                                 // outside counts as 0 (it cannot add anything)
  const out = new Uint8Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!m[y * w + x]) continue;
    for (const [dy, dx] of off) {
      const yy = y + dy, xx = x + dx;
      if (yy >= 0 && yy < h && xx >= 0 && xx < w) out[yy * w + xx] = 1;
    }
  }
  return out;
}

function erode(m, h, w, off) {                                  // border_value 0: the outside erodes the edge
  const out = new Uint8Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let ok = 1;
    for (const [dy, dx] of off) {
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || yy >= h || xx < 0 || xx >= w || !m[yy * w + xx]) { ok = 0; break; }
    }
    out[y * w + x] = ok;
  }
  return out;
}

const closing = (m, h, w, off) => erode(dilate(m, h, w, off), h, w, off);
const opening = (m, h, w, off) => dilate(erode(m, h, w, off), h, w, off);

function dilateIter(m, h, w, k) {                               // k iterations of the 3x3 square
  let cur = m;
  for (let i = 0; i < k; i++) cur = dilate(cur, h, w, SQUARE);
  return cur;
}

/** Connected components (8-connected with SQUARE, 4 with CROSS), labelled 1.. in raster order of each component's
 *  first pixel, as scipy.ndimage.label; returns {lab: Int32Array, k}. */
function label(m, h, w, off) {
  const lab = new Int32Array(h * w), stack = [];
  const nb = off.filter(([dy, dx]) => dy || dx);
  let k = 0;
  for (let s = 0; s < h * w; s++) {
    if (!m[s] || lab[s]) continue;
    lab[s] = ++k;
    stack.push(s);
    while (stack.length) {
      const p = stack.pop(), py = (p / w) | 0, px = p - py * w;
      for (const [dy, dx] of nb) {
        const y = py + dy, x = px + dx;
        if (y < 0 || y >= h || x < 0 || x >= w) continue;
        const q = y * w + x;
        if (m[q] && !lab[q]) { lab[q] = k; stack.push(q); }
      }
    }
  }
  return { lab, k };
}

/** Exact Euclidean distance of every cell to the nearest cell with wall[i] = 1 (scipy distance_transform_edt of
 *  ~wall): squared distances by the two-pass lower-envelope method (Felzenszwalb and Huttenlocher), integers. */
function edt(wall, h, w) {
  const INF = 1e20, d2 = new Float64Array(h * w), n = Math.max(h, w);
  const f = new Float64Array(n), z = new Float64Array(n + 1), v = new Int32Array(n), out = new Float64Array(n);
  const pass = (len, get, set) => {
    for (let q = 0; q < len; q++) f[q] = get(q);
    const cross = (q, p) => ((f[q] + q * q) - (f[p] + p * p)) / (2 * q - 2 * p);   // where parabolas q and p meet
    let k = 0;
    v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
    for (let q = 1; q < len; q++) {
      let s = cross(q, v[k]);
      while (s <= z[k]) { k--; s = cross(q, v[k]); }                 // z[0] = -Infinity stops it at k = 0
      k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < len; q++) {
      while (z[k + 1] < q) k++;
      out[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
    }
    for (let q = 0; q < len; q++) set(q, out[q]);
  };
  for (let x = 0; x < w; x++) pass(h, y => (wall[y * w + x] ? 0 : INF), (y, val) => { d2[y * w + x] = val; });
  for (let y = 0; y < h; y++) pass(w, x => d2[y * w + x], (x, val) => { d2[y * w + x] = val; });
  return d2;
}

/** router.Grid flags of one map (bits: 1 truck, 2 boat, 4 foot, 8 off-road, 16 canopy, 32 wall, 64 building) and
 *  its risk p1 + p4 (float32), for the off-road gate `maxwet`. */
export function gridFlags(cls, probs, h, w, dataset, maxwet) {
  const pr = PROFILES[dataset], p = PARAMS, n = h * w, cellM = pr.cell_m;
  const inSet = (set) => { const m = new Uint8Array(n); for (let i = 0; i < n; i++) m[i] = set.includes(cls[i]) ? 1 : 0; return m; };
  const chanSum = (chs) => {                                    // probs[list(chs)].sum(0), float32
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) { let s = 0; for (const c of chs) s = f32(s + probs[c * n + i]); out[i] = s; }
    return out;
  };
  const wall = inSet(pr.wall), bldg = inSet(pr.bldg), risk = chanSum(pr.risk), g = f32(maxwet);
  const land = new Uint8Array(n);
  for (let i = 0; i < n; i++) land[i] = cls[i] === OTHER && risk[i] < g ? 1 : 0;
  const road = inSet(pr.truck);
  if (p.car_gap) {                                             // parked cars: close small gaps in the road
    const c = closing(road, h, w, disk(p.car_gap / 2 / cellM));
    for (let i = 0; i < n; i++) road[i] |= c[i] & land[i];
  }
  const offroad = new Uint8Array(n);
  if (p.offroad) for (let i = 0; i < n; i++) offroad[i] = land[i] & (road[i] ^ 1);
  const anyWall = wall.some(Boolean), d2 = anyWall ? edt(wall, h, w) : null;
  const room = (half) => {                                      // centre >= half-width + half a cell from every wall
    const m = new Uint8Array(n).fill(1);
    if (half == null || !anyWall) return m;
    const c = half / cellM + 0.5;
    for (let i = 0; i < n; i++) m[i] = Math.sqrt(d2[i]) >= c ? 1 : 0;
    return m;
  };
  const roomTruck = room(p.truck_halfwidth), roomBoat = room(p.boat_halfwidth);
  const pBoat = chanSum(pr.boat), boat = inSet(pr.boat);
  for (let i = 0; i < n; i++) if (pBoat[i] >= p.boat_p) boat[i] = 1;
  const canopy = new Uint8Array(n);
  if (p.canopy_gap) {                                          // narrow land strips between water
    const c = closing(boat, h, w, disk(p.canopy_gap / 2 / cellM));
    for (let i = 0; i < n; i++) canopy[i] = c[i] & (boat[i] ^ 1) & (cls[i] === OTHER ? 1 : 0);
  }
  const walkable = inSet(pr.truck.concat(pr.walk)), flags = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const truckOk = (road[i] | offroad[i]) & (wall[i] ^ 1) & roomTruck[i];
    const boatOk = (boat[i] | canopy[i]) & (wall[i] ^ 1) & roomBoat[i];
    const footOk = (walkable[i] | land[i]) & (bldg[i] ^ 1);
    flags[i] = truckOk | (boatOk << 1) | (footOk << 2) | (offroad[i] << 3) | (canopy[i] << 4) | (wall[i] << 5) | (bldg[i] << 6);
  }
  return { risk, flags };
}

/** router.base_cells on the cell classes: the border cells of the largest component touching the border, of clear
 *  road, else dry land, else water (boat). {cells, mode} or null. */
export function baseCells(cls, h, w, dataset) {
  const pr = PROFILES[dataset], n = h * w;
  const border = (i) => { const y = (i / w) | 0, x = i - y * w; return y === 0 || y === h - 1 || x === 0 || x === w - 1; };
  for (const [classes, mode] of [[pr.truck, 'TRUCK'], [pr.truck.concat([OTHER]), 'TRUCK'], [pr.boat, 'BOAT']]) {
    const m = new Uint8Array(n);
    for (let i = 0; i < n; i++) m[i] = classes.includes(cls[i]) ? 1 : 0;
    const { lab, k } = label(m, h, w, SQUARE);
    if (!k) continue;
    const size = new Float64Array(k + 1), touch = new Uint8Array(k + 1);
    for (let i = 0; i < n; i++) if (lab[i]) { size[lab[i]]++; if (border(i)) touch[lab[i]] = 1; }
    let best = 0;
    for (let l = 1; l <= k; l++) if (touch[l] && (!best || size[l] > size[best])) best = l;   // first maximum
    if (!best) continue;
    const cells = [];
    for (let i = 0; i < n; i++) if (lab[i] === best && border(i)) cells.push(i);
    return { cells, mode };
  }
  return null;
}

const roundHalfEven1 = (v) => {                                 // Python round(v, 1)
  const s = v * 10, f = Math.floor(s), d = s - f;
  return (d > 0.5 || (d === 0.5 && f % 2) ? f + 1 : f) / 10;
};

/** router.buildings_from_gt on a class map (H, W): opened building pixels, 4-connected components >= min_area px, their
 *  cells (>= half covered, else every touched cell), the 2-cell ring, sub-class by pixel majority, centroid, and the
 *  pixel mask for the outline. */
export function buildings(map, H, W, dataset, minArea = 150, ring = 2) {
  const pr = PROFILES[dataset], h = (H / CELL) | 0, w = (W / CELL) | 0;
  const bm = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) bm[i] = pr.bldg.includes(map[i]) ? 1 : 0;
  const { lab, k } = label(opening(bm, H, W, SQUARE), H, W, CROSS);
  const area = new Float64Array(k + 1);
  for (let i = 0; i < H * W; i++) if (lab[i]) area[lab[i]]++;
  const out = [];
  for (let l = 1; l <= k; l++) {
    if (area[l] < minArea) continue;
    const cnt = new Uint8Array(h * w), sub = new Float64Array(N_CH + 1);
    let sy = 0, sx = 0;
    for (let y = 0; y < h * CELL; y++) for (let x = 0; x < w * CELL; x++) if (lab[y * W + x] === l) cnt[((y / CELL) | 0) * w + ((x / CELL) | 0)]++;
    for (let i = 0; i < H * W; i++) if (lab[i] === l) { sub[map[i]]++; sy += (i / W) | 0; sx += i % W; }
    let cells = new Uint8Array(h * w), any = false;
    for (let c = 0; c < h * w; c++) if (cnt[c] >= 8) { cells[c] = 1; any = true; }
    if (!any) for (let c = 0; c < h * w; c++) cells[c] = cnt[c] > 0 ? 1 : 0;
    const dil = dilateIter(cells, h, w, ring), ringCells = [], cellList = [];
    for (let c = 0; c < h * w; c++) { if (dil[c] && !cells[c]) ringCells.push(c); if (cells[c]) cellList.push(c); }
    const cls = sub[pr.bldg[1]] > sub[pr.bldg[0]] ? pr.bldg[1] : pr.bldg[0];
    const cy = sy / area[l], cx = sx / area[l];
    out.push({ id: out.length, cls, area_px: area[l], cells: cellList, cellMask: cells, ring: ringCells, ringK: ring,
               centroid: [roundHalfEven1(cx), roundHalfEven1(cy)], label: l, lab });
  }
  return out;
}

/** router.Grid.goal: a vehicle's goal is the building dilated by ceil(clearance) + ring cells; on foot the ring. */
export function goals(b, h, w, cellM) {
  const out = [];
  for (const half of [PARAMS.truck_halfwidth, PARAMS.boat_halfwidth]) {
    const k = Math.ceil(half / cellM + 0.5) + b.ringK, dil = dilateIter(b.cellMask, h, w, k), g = [];
    for (let c = 0; c < h * w; c++) if (dil[c] && !b.cellMask[c]) g.push(c);
    out.push(g);
  }
  out.push(b.ring);
  return out;
}

/** Outer outline of one building as [[x, y], ...] (pixel centres): boundary tracing of the mask, then the
 *  Douglas-Peucker simplification with tolerance 1.5 px (export_web_assets.polygon draws the same outline with
 *  skimage; display only, the routing never reads it). */
export function outline(lab, l, H, W, tol = 1.5) {
  // marching squares on the 0/1 mask at level 0.5: the contour runs through the edge midpoints between a pixel
  // of the building and one outside; walk the longest closed loop
  const inside = (y, x) => y >= 0 && y < H && x >= 0 && x < W && lab[y * W + x] === l;
  const key = (y, x) => `${y},${x}`;
  const next = new Map();                                       // directed boundary edges, building on the left
  let minY = H, minX = W, maxY = 0, maxX = 0;
  for (let i = 0; i < H * W; i++) if (lab[i] === l) {
    const y = (i / W) | 0, x = i - y * W;
    minY = Math.min(minY, y); maxY = Math.max(maxY, y); minX = Math.min(minX, x); maxX = Math.max(maxX, x);
  }
  for (let y = minY; y <= maxY; y++) for (let x = minX; x <= maxX; x++) {
    if (!inside(y, x)) continue;
    // corners of pixel (y, x) at (y - .5, x - .5) etc.; edges where the neighbour is outside, counter-clockwise
    if (!inside(y - 1, x)) next.set(key(y - 0.5, x + 0.5), [y - 0.5, x - 0.5]);
    if (!inside(y, x - 1)) next.set(key(y - 0.5, x - 0.5), [y + 0.5, x - 0.5]);
    if (!inside(y + 1, x)) next.set(key(y + 0.5, x - 0.5), [y + 0.5, x + 0.5]);
    if (!inside(y, x + 1)) next.set(key(y + 0.5, x + 0.5), [y - 0.5, x + 0.5]);
  }
  const seen = new Set();
  let best = [];
  for (const start of next.keys()) {
    if (seen.has(start)) continue;
    const loop = [];
    let k = start;
    while (!seen.has(k) && next.has(k)) {
      seen.add(k);
      const [y, x] = k.split(',').map(Number);
      loop.push([y, x]);
      const [ny, nx] = next.get(k);
      k = key(ny, nx);
    }
    if (loop.length > best.length) best = loop;
  }
  // corner points -> edge midpoints (the 0.5 level of marching squares lies half a pixel inside the corner path)
  const mid = best.map(([y, x], i) => { const [y2, x2] = best[(i + 1) % best.length]; return [(y + y2) / 2, (x + x2) / 2]; });
  return simplify(mid.concat([mid[0]]), tol).map(([y, x]) => [roundHalfEven1(x), roundHalfEven1(y)]);
}

function simplify(pts, tol) {                                   // Douglas-Peucker on [[y, x], ...]
  if (pts.length < 3) return pts;
  const keep = new Uint8Array(pts.length);
  keep[0] = keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const [ay, ax] = pts[a], [by, bx] = pts[b], dy = by - ay, dx = bx - ax, L = Math.hypot(dy, dx);
    let worst = -1, wd = 0;
    for (let i = a + 1; i < b; i++) {
      const [py, px] = pts[i];
      const d = L ? Math.abs(dy * (px - ax) - dx * (py - ay)) / L : Math.hypot(py - ay, px - ax);
      if (d > wd) { wd = d; worst = i; }
    }
    if (worst >= 0 && wd > tol) { keep[worst] = 1; stack.push([a, worst], [worst, b]); }
  }
  return pts.filter((_, i) => keep[i]);
}

/** The whole upload scene from uint8 probabilities (C * H * W): {grid, buildings, cls} with grid as parseScene returns
 *  it (pred = the tuned gate, gt = the same map: no labels, pred_cautious = the cautious gate's flags). */
export function buildUploadScene(probs, H, W, dataset, tau) {
  const cls = decode(probs, H, W, tau);
  const h = (H / CELL) | 0, w = (W / CELL) | 0;
  const ccls = cellClasses(cls, H, W), cprobs = cellProbs(probs, H, W);
  const pred = gridFlags(ccls, cprobs, h, w, dataset, GATES[dataset].pred);
  const cautious = gridFlags(ccls, cprobs, h, w, dataset, GATES[dataset].cautious);
  const base = baseCells(ccls, h, w, dataset);
  const pr = PROFILES[dataset];
  const blds = buildings(cls, H, W, dataset).map(b => ({ id: b.id, cls: b.cls, centroid: b.centroid,
    polygon: outline(b.lab, b.label, H, W), goals: goals(b, h, w, pr.cell_m) }));
  const layer = { cls: ccls, risk: pred.risk, flags: pred.flags };
  const grid = { h, w, cell: CELL, dataset, base,
    params: { ...PARAMS, cell_m: pr.cell_m, offroad_maxwet_pred: GATES[dataset].pred,
              offroad_maxwet_cautious: GATES[dataset].cautious, offroad_maxwet_gt: null },
    profile: pr, eta: ETA, pred: layer, gt: layer, pred_cautious: { flags: cautious.flags } };
  return { grid, buildings: blds, cls };
}

export { STATUS };
