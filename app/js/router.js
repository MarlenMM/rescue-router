// JavaScript port of src/router.py (PLAN §6.1): truck/boat Dijkstra on the 4x4-px cell grid.
// It must reproduce the Python statuses and route costs on the exported grids (app/test/router.test.mjs).
//
// States (cell, mode), mode TRUCK (layer 0) or BOAT (layer 1), 8-connected, step length 1 or sqrt(2).
//   TRUCK enters a road_clear cell at len * (1 + lam * risk), or dry 'other' land at `offroad` x that
//         (on predicted maps only where risk < offroad_maxwet, the off-road gate); risk = p1 + p4
//   BOAT  enters a boat cell (class, or p_boat >= boat_p) at len * boat_cost
//   launch TRUCK -> BOAT into a neighbouring boat cell: len * boat_cost + launch; no switch back
//   walls  buildings (RescueNet also road_blocked) are never entered
// The truck cost is computed in float32 like NumPy does ((1 + lam * risk) on a float32 array), then
// multiplied by the float64 step length, so the costs match Python to the last bit (up to tie order).

export const TRUCK = 0, BOAT = 1, ISOLATED = 2;
export const STATUS = ['TRUCK', 'BOAT', 'ISOLATED'];
const OTHER = 5;
const DIRS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
const f32 = Math.fround;

export function b64ToBytes(s) {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function b64ToFloat32(s) {
  const b = b64ToBytes(s);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}

/** One map's routing graph. layer = {cls, risk, pboat} (base64 strings from grid.json, or typed arrays). */
export class Grid {
  constructor(gridJson, layer, opts = {}) {
    const g = gridJson;
    this.h = g.h; this.w = g.w; this.n = g.h * g.w;
    this.prof = g.profile;
    this.par = { lam: g.params.lam, boat_cost: g.params.boat_cost, launch: g.params.launch,
                 boat_p: g.params.boat_p, offroad: g.params.offroad, offroad_maxwet: null, ...opts };
    this.cls = typeof layer.cls === 'string' ? b64ToBytes(layer.cls) : layer.cls;
    this.risk = typeof layer.risk === 'string' ? b64ToFloat32(layer.risk) : layer.risk;
    this.pboat = typeof layer.pboat === 'string' ? b64ToFloat32(layer.pboat) : layer.pboat;
    const n = this.n, p = this.par, pr = this.prof;
    const gate = p.offroad_maxwet == null ? null : f32(p.offroad_maxwet);
    this.wall = new Uint8Array(n); this.truckOk = new Uint8Array(n); this.boatOk = new Uint8Array(n);
    this.slow = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const c = this.cls[i];
      const wall = pr.wall.includes(c);
      let off = p.offroad ? c === OTHER : false;
      if (off && gate !== null) off = this.risk[i] < gate;
      this.wall[i] = wall;
      this.truckOk[i] = (pr.truck.includes(c) || off) && !wall;
      this.boatOk[i] = (pr.boat.includes(c) || this.pboat[i] >= p.boat_p) && !wall;
      this.slow[i] = off ? (p.offroad || 1) : 1;
    }
  }

  truckStep(L, b) {
    return L * f32(1 + f32(this.par.lam * this.risk[b])) * this.slow[b];
  }

  /** Multi-source Dijkstra from base = {cells: [flat ids], mode: 'TRUCK'|'BOAT'} (entry cost 0). */
  solve(base) {
    const n = this.n, w = this.w, h = this.h, p = this.par;
    const dist = new Float64Array(2 * n).fill(Infinity);
    const pred = new Int32Array(2 * n).fill(-1);
    const heap = new MinHeap();
    if (base) {
      const off = base.mode === 'BOAT' ? n : 0;
      for (const c of base.cells) { dist[c + off] = 0; heap.push(0, c + off); }
    }
    const boatL = [p.boat_cost, Math.SQRT2 * p.boat_cost];
    while (heap.size) {
      const [d, u] = heap.pop();
      if (d > dist[u]) continue;
      const boat = u >= n, a = boat ? u - n : u, ay = (a / w) | 0, ax = a - ay * w;
      for (const [dy, dx] of DIRS) {
        const y = ay + dy, x = ax + dx;
        if (y < 0 || y >= h || x < 0 || x >= w) continue;
        const b = y * w + x, diag = dy !== 0 && dx !== 0, L = diag ? Math.SQRT2 : 1;
        if (!boat && this.truckOk[b]) relax(b, d + this.truckStep(L, b), u);
        if (this.boatOk[b]) relax(n + b, d + (boat ? boatL[+diag] : boatL[+diag] + p.launch), u);
      }
    }
    function relax(v, nd, u) {
      if (nd < dist[v]) { dist[v] = nd; pred[v] = u; heap.push(nd, v); }
    }
    return new Solution(this, dist, pred);
  }
}

export class Solution {
  constructor(grid, dist, pred) { this.grid = grid; this.dist = dist; this.pred = pred; }

  status(ring) {
    const n = this.grid.n;
    if (ring.some(c => Number.isFinite(this.dist[c]))) return TRUCK;
    if (ring.some(c => Number.isFinite(this.dist[n + c]))) return BOAT;
    return ISOLATED;
  }

  /** Cheapest route to the ring: {cells: [[y, x]], modes: [0|1], cost} or null (first minimum, like np.argmin).
   *  truckOnly: only routes that end in the truck layer (the viewer shows the road route for TRUCK houses;
   *  the default, like router.py, may prefer a cheaper truck + boat route to a truck-reachable house). */
  route(ring, truckOnly = false) {
    const n = this.grid.n, w = this.grid.w;
    let best = -1;
    for (const j of truckOnly ? ring : [...ring, ...ring.map(c => n + c)]) if (best < 0 || this.dist[j] < this.dist[best]) best = j;
    if (best < 0 || !Number.isFinite(this.dist[best])) return null;
    const nodes = [best];
    while (this.pred[nodes[nodes.length - 1]] >= 0) nodes.push(this.pred[nodes[nodes.length - 1]]);
    nodes.reverse();
    return { cells: nodes.map(v => { const c = v % n; return [(c / w) | 0, c % w]; }),
             modes: nodes.map(v => (v >= n ? BOAT : TRUCK)), cost: this.dist[best] };
  }
}

class MinHeap {
  constructor() { this.k = []; this.v = []; }
  get size() { return this.k.length; }
  push(key, val) {
    const k = this.k, v = this.v;
    let i = k.length; k.push(key); v.push(val);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= key) break;
      k[i] = k[p]; v[i] = v[p]; i = p;
    }
    k[i] = key; v[i] = val;
  }
  pop() {
    const k = this.k, v = this.v, top = [k[0], v[0]];
    const lk = k.pop(), lv = v.pop(), m = k.length;
    if (m) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= m) break;
        if (c + 1 < m && k[c + 1] < k[c]) c++;
        if (k[c] >= lk) break;
        k[i] = k[c]; v[i] = v[c]; i = c;
      }
      k[i] = lk; v[i] = lv;
    }
    return top;
  }
}

/** Both maps of a scene: pred (gated) and GT (no gate), as route_eval.py builds them. */
export function sceneGrids(gridJson, lam) {
  const o = lam == null ? {} : { lam };
  return {
    pred: new Grid(gridJson, gridJson.pred, { ...o, offroad_maxwet: gridJson.params.offroad_maxwet_pred }),
    gt: new Grid(gridJson, gridJson.gt, { ...o, offroad_maxwet: gridJson.params.offroad_maxwet_gt }),
  };
}
