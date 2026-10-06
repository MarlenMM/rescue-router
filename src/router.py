"""Truck/boat rescue router on a coarse cell grid (PLAN §6.1) and the routing metrics (§5.4).

Pure NumPy/SciPy, CPU only. Works on any decoded class map (prediction or ground truth) with the
class ids of `rr.metrics` (0-4 = task channels, 5 = other, 255 = ignore).

    cls = cell_classes(class_map)                       # 4x4-px cells, majority vote
    grid = Grid(cls, cell_probs(probs), 'floodnet')     # probs (C, H, W) or None for a hard map
    base = base_cells(cell_classes(gt_map), 'floodnet') # same base for prediction and GT
    blds = buildings_from_gt(gt_map, 'floodnet')
    sol = grid.solve(base)
    sol.status(b), sol.route(b)                         # TRUCK / BOAT / ISOLATED, cell path + modes

Graph: states (cell, mode), mode in {TRUCK, BOAT}, 8-connected, step length 1 or sqrt(2).
  TRUCK  enters a road_clear cell, cost len * (1 + lam * (p1 + p4))   (risk-aware; lam = 0 = plain),
         or dry 'other' land (lawn, trees, background) at `offroad` x that cost (None = roads only,
         the strict PLAN §6.1 rule; on GT maps it leaves ~75 % of FloodNet houses ISOLATED behind
         their own yards, so the default lets the truck cross dry land slowly)
  BOAT   enters a boat cell (FloodNet road_flooded or water; RescueNet water) or a cell with
         p_boat >= 0.5, cost len * boat_cost
  launch TRUCK -> BOAT into a neighbouring boat cell, extra cost `launch`; no switch back
  walls  buildings (and RescueNet road_blocked) are never entered
One multi-source Dijkstra (scipy.sparse.csgraph) from the base gives every building's status.
"""

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

CELL = 4
N_CH, OTHER, IGNORE = 5, 5, 255
TRUCK, BOAT, ISOLATED = 0, 1, 2
STATUS = ('TRUCK', 'BOAT', 'ISOLATED')

# truck: drivable classes; boat: navigable classes; wall: never entered; risk: channels in the
# risk-aware truck cost; hazard: GT classes a truck must never drive on (route safety, §5.4)
PROFILES = {
    'floodnet': dict(truck=(0,), boat=(1, 4), wall=(2, 3), risk=(1, 4), bldg=(2, 3), hazard=(1, 4)),
    'rescuenet': dict(truck=(0,), boat=(4,), wall=(1, 2, 3), risk=(1, 4), bldg=(2, 3), hazard=(1, 4)),
}
DEFAULTS = dict(lam=5.0, boat_cost=3.0, launch=40.0, boat_p=0.5, offroad=4.0)

# majority-vote tie-break: on equal pixel counts the less drivable class wins
# (road_clear < other < water < road_flooded < bldg_dry < bldg_flooded)
_TIE = np.array([0, 3, 5, 4, 2, 1], np.float64) * 1e-3
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _blocks(a, cell):
    """(H, W, ...) -> (h, w, cell*cell, ...) non-overlapping blocks (edge remainder dropped)."""
    h, w = a.shape[0] // cell, a.shape[1] // cell
    a = a[:h * cell, :w * cell]
    return a.reshape(h, cell, w, cell, *a.shape[2:]).swapaxes(1, 2).reshape(h, w, cell * cell, *a.shape[2:])


def cell_counts(cls_map, cell=CELL):
    """class map (H, W) -> per-cell pixel counts (h, w, 6) of classes 0-5 (ignore not counted)."""
    b = _blocks(np.asarray(cls_map), cell)
    return np.stack([(b == k).sum(-1) for k in range(N_CH + 1)], -1)


def cell_classes(cls_map, cell=CELL):
    """class map (H, W) -> cell class grid (h, w) by majority vote; all-ignore cells become OTHER."""
    n = cell_counts(cls_map, cell)
    out = (n + _TIE).argmax(-1).astype(np.uint8)
    out[n.sum(-1) == 0] = OTHER
    return out


def cell_probs(probs, cell=CELL):
    """probabilities (C, H, W), float in [0, 1] or uint8 in [0, 255] -> per-cell means (C, h, w)."""
    p = np.asarray(probs)
    scale = 255.0 if p.dtype == np.uint8 else 1.0
    return _blocks(np.moveaxis(p, 0, -1), cell).astype(np.float32).mean(2).transpose(2, 0, 1) / scale


def onehot_probs(cls_map, cell=CELL):
    """class map -> per-cell class fractions (C, h, w): the 'probabilities' of a hard (e.g. GT) map."""
    n = cell_counts(cls_map, cell).astype(np.float32)
    return (n[..., :N_CH] / np.maximum(n.sum(-1, keepdims=True), 1)).transpose(2, 0, 1)


class Grid:
    """Routing graph of one map. cls: cell classes (h, w); probs: cell probabilities (C, h, w)
    (None = one-hot of cls, i.e. a hard map)."""

    def __init__(self, cls, probs=None, dataset='floodnet', **kw):
        self.cls = np.asarray(cls, np.uint8)
        self.h, self.w = self.cls.shape
        self.n = self.h * self.w
        self.prof = PROFILES[dataset]
        self.par = {**DEFAULTS, **kw}
        if probs is None:
            probs = np.stack([(self.cls == c) for c in range(N_CH)]).astype(np.float32)
        self.probs = np.asarray(probs, np.float32)
        pr = self.prof
        self.wall = np.isin(self.cls, pr['wall'])
        self.offroad = (self.cls == OTHER) if self.par['offroad'] else np.zeros_like(self.wall)
        self.truck_ok = (np.isin(self.cls, pr['truck']) | self.offroad) & ~self.wall
        self.boat_ok = (np.isin(self.cls, pr['boat']) | (self.probs[list(pr['boat'])].sum(0) >= self.par['boat_p'])) \
            & ~self.wall
        self.risk = self.probs[list(pr['risk'])].sum(0)
        self.graph = self._build()

    def _build(self):
        h, w, n, p = self.h, self.w, self.n, self.par
        idx = np.arange(n).reshape(h, w)
        rows, cols, vals = [], [], []
        for dy, dx in _DIRS:
            L = np.hypot(dy, dx)
            ys, yd = slice(max(-dy, 0), h - max(dy, 0)), slice(max(dy, 0), h - max(-dy, 0))
            xs, xd = slice(max(-dx, 0), w - max(dx, 0)), slice(max(dx, 0), w - max(-dx, 0))
            a, b = idx[ys, xs].ravel(), idx[yd, xd].ravel()
            t, bo = self.truck_ok[yd, xd].ravel(), self.boat_ok[yd, xd].ravel()
            risk = self.risk[yd, xd].ravel()
            slow = np.where(self.offroad[yd, xd].ravel(), self.par['offroad'] or 1.0, 1.0)
            rows += [a[t], n + a[bo], a[bo]]                       # truck->truck, boat->boat, launch
            cols += [b[t], n + b[bo], n + b[bo]]
            vals += [L * (1 + p['lam'] * risk[t]) * slow[t],
                     np.full(bo.sum(), L * p['boat_cost']),
                     np.full(bo.sum(), L * p['boat_cost'] + p['launch'])]
        return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n))

    def solve(self, base):
        """Multi-source Dijkstra from base = (flat cell indices, mode). The vehicle enters the image at
        the base cells (cost 0) whatever the map says there; every later step obeys the map."""
        cells, mode = base
        src = np.asarray(cells, np.int64) + (self.n if mode == BOAT else 0)
        dist, pred, _ = dijkstra(self.graph, directed=True, indices=src, min_only=True, return_predecessors=True)
        return Solution(self, dist, pred)


class Solution:
    def __init__(self, grid, dist, pred):
        self.grid, self.dist, self.pred = grid, dist, pred

    def status(self, bld):
        n, r = self.grid.n, bld['ring']
        if len(r) and np.isfinite(self.dist[r]).any():
            return TRUCK
        if len(r) and np.isfinite(self.dist[n + r]).any():
            return BOAT
        return ISOLATED

    def route(self, bld):
        """Cheapest route to the building's ring: dict(cells=[(y, x)], modes=[TRUCK|BOAT], cost) or None."""
        n, r = self.grid.n, bld['ring']
        if not len(r):
            return None
        cand = np.concatenate([r, n + r])
        j = cand[np.argmin(self.dist[cand])]
        if not np.isfinite(self.dist[j]):
            return None
        nodes = [j]
        while self.pred[nodes[-1]] >= 0:
            nodes.append(self.pred[nodes[-1]])
        nodes = np.array(nodes[::-1])
        cells = nodes % n
        return dict(cells=list(zip(*np.divmod(cells, self.grid.w))), modes=(nodes >= n).astype(int).tolist(),
                    cost=float(self.dist[j]))


def base_cells(gt_cls, dataset='floodnet', offroad=True):
    """Entry cells (PLAN §5.4): the border cells of the largest GT clear-road component touching the
    image border (TRUCK). Fallbacks, in order: the largest dry-land (road_clear or other) component
    touching the border (TRUCK, only if offroad), then the largest boat-navigable one (BOAT).
    gt_cls = GT cell classes (h, w). Returns (flat indices, mode) or None."""
    pr = PROFILES[dataset]
    border = np.zeros(gt_cls.shape, bool)
    border[0], border[-1], border[:, 0], border[:, -1] = True, True, True, True
    order = [(pr['truck'], TRUCK)] + ([(pr['truck'] + (OTHER,), TRUCK)] if offroad else []) + [(pr['boat'], BOAT)]
    for classes, mode in order:
        lab, k = ndimage.label(np.isin(gt_cls, classes), structure=np.ones((3, 3)))
        if not k:
            continue
        touch = np.unique(lab[border & (lab > 0)])
        if len(touch):
            sizes = ndimage.sum(np.ones_like(lab), lab, touch)
            best = touch[np.argmax(sizes)]
            return np.flatnonzero((lab == best) & border), mode
    return None


def buildings_from_gt(gt_map, dataset='floodnet', min_area=150, ring=2, cell=CELL):
    """GT building components (PLAN §5.4): building pixels (both building channels) opened with a 3x3
    kernel, connected components with area >= min_area px. Each gets its cells (>= half of the cell
    covered; else every touched cell), a `ring`-cell dilation ring as the goal, and its GT sub-class
    (FloodNet: flooded/dry; RescueNet: intact/damaged) by pixel majority."""
    pr = PROFILES[dataset]
    gt_map = np.asarray(gt_map)
    b = ndimage.binary_opening(np.isin(gt_map, pr['bldg']), structure=np.ones((3, 3)))
    lab, k = ndimage.label(b)
    h, w = gt_map.shape[0] // cell, gt_map.shape[1] // cell
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab == i
        area = int(m.sum())
        if area < min_area:
            continue
        frac = _blocks(m, cell).mean(-1)
        cells = frac >= 0.5
        if not cells.any():
            cells = frac > 0
        rg = ndimage.binary_dilation(cells, structure=np.ones((3, 3)), iterations=ring) & ~cells
        sub = np.bincount(gt_map[m], minlength=N_CH + 1)
        ys, xs = np.nonzero(m)
        out.append(dict(id=len(out), area_px=area, cells=np.flatnonzero(cells), ring=np.flatnonzero(rg),
                        cls=int(pr['bldg'][int(np.argmax(sub[list(pr['bldg'])]))]),
                        centroid=(float(ys.mean()), float(xs.mean())), shape=(h, w)))
    return out


def eval_route_on_gt(route, gt_grid):
    """Score a route found on one map against the GT grid: cost under GT costs, feasibility, RSV.
    Infeasible = it drives a truck onto GT water/flooded/blocked road or enters a GT wall
    (a boat over dry ground is charged boat cost but not flagged; a truck on GT dry 'other' land
    pays the off-road cost, or is infeasible if the GT grid is roads-only). RSV = share of truck-leg cells
    that are GT hazard or building. unsafe = a truck leg crosses >= 3 consecutive GT hazard cells."""
    g, p = gt_grid, gt_grid.par
    haz = np.isin(g.cls, g.prof['hazard'])
    bad_truck = haz | np.isin(g.cls, g.prof['bldg'])
    cells, modes = route['cells'], route['modes']
    cost, feasible, run, unsafe, truck_cells, truck_bad = 0.0, True, 0, False, 0, 0
    for k, ((y, x), m) in enumerate(zip(cells, modes)):
        if m == TRUCK:
            truck_cells += 1
            truck_bad += bool(bad_truck[y, x])
            run = run + 1 if haz[y, x] else 0
            unsafe |= run >= 3
        if k == 0:
            continue                                   # entry cell: no step cost
        if g.wall[y, x] or (m == TRUCK and (haz[y, x] or not (g.truck_ok[y, x] or bad_truck[y, x]))):
            feasible = False
        (y0, x0), m0 = cells[k - 1], modes[k - 1]
        L = np.hypot(y - y0, x - x0)
        slow = p['offroad'] if (p['offroad'] and g.cls[y, x] == OTHER) else 1.0
        cost += L * (1 + p['lam'] * g.risk[y, x]) * slow if m == TRUCK else L * p['boat_cost'] + p['launch'] * (m0 == TRUCK)
    return dict(cost_gt=cost, feasible=feasible, unsafe=unsafe,
                rsv=truck_bad / truck_cells if truck_cells else None)


def route_metrics(pred_grid, gt_grid, base, blds):
    """Per-image routing comparison (PLAN §5.4) of a predicted map against the GT map, same base and
    GT buildings. Returns the 3x3 status confusion (rows GT, cols pred) and per-building records."""
    cm = np.zeros((3, 3), np.int64)
    recs = []
    if base is None:
        return cm, recs
    sp, sg = pred_grid.solve(base), gt_grid.solve(base)
    for b in blds:
        st_p, st_g = sp.status(b), sg.status(b)
        cm[st_g, st_p] += 1
        rec = dict(id=b['id'], cls=b['cls'], status_pred=st_p, status_gt=st_g)
        rp, rg = sp.route(b), sg.route(b)
        if rg is not None:
            rec['cost_gt_opt'] = rg['cost']
        if rp is not None:
            rec.update(eval_route_on_gt(rp, gt_grid))
            if rg is not None and rec['feasible'] and rg['cost'] > 0:   # cost 0 = house touches the entry
                rec['cost_ratio'] = rec['cost_gt'] / rg['cost']
        recs.append(rec)
    return cm, recs


def summarize_routing(cm, recs):
    """Dataset-level routing metrics from accumulated status confusion and building records."""
    pos_g, pos_p = cm[1:, :].sum(), cm[:, 1:].sum()          # "not TRUCK" = stranded
    tp = cm[1:, 1:].sum()
    prec, rec = tp / max(pos_p, 1), tp / max(pos_g, 1)
    with_route = [r for r in recs if 'feasible' in r]
    truck_routes = [r for r in with_route if r['rsv'] is not None]
    gt_reach = [r for r in recs if 'cost_gt_opt' in r]
    ratios = [r['cost_ratio'] for r in gt_reach if 'cost_ratio' in r]
    infeasible = [not r.get('feasible', False) for r in gt_reach]          # no predicted route, or unsafe
    return dict(
        n_buildings=int(cm.sum()), gt_status_counts=cm.sum(1).tolist(), pred_status_counts=cm.sum(0).tolist(),
        stranded_precision=float(prec), stranded_recall=float(rec),
        stranded_f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
        reach_agreement=float(np.trace(cm) / max(cm.sum(), 1)), status_confusion=cm.tolist(),
        rsv_mean=float(np.mean([r['rsv'] for r in truck_routes])) if truck_routes else None,
        unsafe_route_rate=float(np.mean([r['unsafe'] for r in truck_routes])) if truck_routes else None,
        cost_ratio_median=float(np.median(ratios)) if ratios else None,
        infeasible_rate=float(np.mean(infeasible)) if gt_reach else None,
        n_routes_gt=len(gt_reach), n_routes_pred=len(with_route), n_cost_ratio=len(ratios),
        n_trivial_gt_routes=sum(r['cost_gt_opt'] == 0 for r in gt_reach))
