"""Truck/boat rescue router on a coarse cell grid (PLAN §6.1) and the routing metrics (§5.4).

Pure NumPy/SciPy, CPU only. Works on any decoded class map (prediction or ground truth) with the
class ids of `rr.metrics` (0-4 = task channels, 5 = other, 255 = ignore).

    cls = cell_classes(class_map)                       # 4x4-px cells, majority vote
    grid = Grid(cls, cell_probs(probs), 'floodnet')     # probs (C, H, W) or None for a hard map
    base = base_cells(cell_classes(gt_map), 'floodnet') # same base for prediction and GT
    blds = buildings_from_gt(gt_map, 'floodnet')
    sol = grid.solve(base)
    sol.status(b), sol.route(b)                         # TRUCK / BOAT / FOOT / ISOLATED, cell path + modes

Graph: states (cell, mode), mode in {TRUCK, BOAT, FOOT}, 8-connected, step length 1 or sqrt(2).
  TRUCK  enters a road_clear cell, cost len * (1 + lam * (p1 + p4))   (risk-aware; lam = 0 = plain),
         or dry 'other' land (lawn, trees, background) at `offroad` x that cost (None = roads only,
         the strict PLAN §6.1 rule; on GT maps it leaves ~75 % of FloodNet houses ISOLATED behind
         their own yards, so the default lets the truck cross dry land slowly). offroad_maxwet = g:
         off-road only where p1 + p4 < g (92 % of the GT-water cells on predicted truck routes were
         predicted 'other' land, median p_wet .21; tuned on val30, see PLAN §15)
  BOAT   enters a boat cell (FloodNet road_flooded or water; RescueNet water) or a cell with
         p_boat >= 0.5, cost len * boat_cost
  FOOT   enters dry land (road_clear or 'other', with the same off-road gate) or, RescueNet only,
         blocked road (debris: walkable, not drivable), cost len * foot_cost (default 8: walking is the
         slowest mode); foot_cost=None = no foot layer (the 6 Oct truck/boat router)
  launch TRUCK -> BOAT into a neighbouring boat cell, extra cost `launch`
  alight TRUCK or BOAT -> FOOT into a neighbouring foot cell, extra cost `alight`
         modes only go forward (truck -> boat -> foot): no switch back to a vehicle
  walls  buildings are never entered (RescueNet road_blocked: by truck and boat only)
Physical rules (6 Oct, Batch 1; sizes in metres, converted with the dataset's cell size `cell_m`):
  width   a truck (half-width 1.25 m) and a boat (0.8 m) keep their body clear of walls: a cell is open to
          them only if its centre is >= half-width + half a cell from every wall cell (distance transform)
  cars    gaps of up to `car_gap` m in the road are closed (PLAN §3.1: parked cars and their shadows split
          decoded roads); only dry land becomes road, so the off-road gate still holds
  canopy  a boat may cross strips of non-building, non-road land up to `canopy_gap` m wide between water
          (tree crowns over floodwater, which the 5-class task decodes as 'other'), at canopy_cost x
  corners no diagonal step squeezes between two obstacle cells that touch at a corner
  goal    a vehicle reaches a house when its body edge is within the house's ring: its ring is the
          building dilated by ceil(clearance) + ring cells (people on foot keep the plain ring)
POINT_VEHICLES switches all of them off (the router before Batch 1).
Status of a building: TRUCK if a truck reaches its ring, else BOAT if a boat does, else FOOT if people
on foot do (after a truck or a boat), else ISOLATED (no route: air team).
One multi-source Dijkstra (scipy.sparse.csgraph) from the base gives every building's status.
"""

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

CELL = 4
N_CH, OTHER, IGNORE = 5, 5, 255
TRUCK, BOAT, FOOT, ISOLATED = 0, 1, 2, 3
STATUS = ('TRUCK', 'BOAT', 'FOOT', 'ISOLATED')

# truck: drivable classes; boat: navigable classes; wall: never entered; risk: channels in the
# risk-aware truck cost; hazard: GT classes a truck must never drive on (route safety, §5.4);
# walk: classes people can cross on foot besides dry land (RescueNet debris); cell_m: ground size of a
# 4-px cell at 768x576, from the labelled cars (median car 1.8 m wide): RescueNet 7.7 cm/px -> 0.31 m;
# FloodNet flies at two heights (ids 7000-7999: 16.6 cm/px, most others ~8.5) -> the larger cell, 0.66 m,
# so clearances in cells are never too large (scale errors can only under-block, never close a real gap)
PROFILES = {
    'floodnet': dict(truck=(0,), boat=(1, 4), wall=(2, 3), risk=(1, 4), bldg=(2, 3), hazard=(1, 4), walk=(),
                     cell_m=0.66),
    'rescuenet': dict(truck=(0,), boat=(4,), wall=(1, 2, 3), risk=(1, 4), bldg=(2, 3), hazard=(1, 4), walk=(1,),
                      cell_m=0.31),
}
DEFAULTS = dict(lam=5.0, boat_cost=3.0, launch=40.0, boat_p=0.5, offroad=4.0, offroad_maxwet=None,
                foot_cost=8.0, alight=10.0, cell_m=None,
                truck_halfwidth=1.25, boat_halfwidth=0.8, car_gap=2.0, canopy_gap=3.0, canopy_cost=2.0, corner=True)
POINT_VEHICLES = dict(truck_halfwidth=None, boat_halfwidth=None, car_gap=None, canopy_gap=None, corner=False)

# confidence tiers of a house on a predicted map (Batch 2, src/tiers.py): truck-reachable even with the cautious
# off-road gate -> CONFIDENT_TRUCK; only with the tuned gate -> PROBABLE_TRUCK (check first); else its status
TIERS = ('CONFIDENT_TRUCK', 'PROBABLE_TRUCK', 'BOAT', 'FOOT', 'ISOLATED')

# travel-time estimate for a route (display only; routes are chosen by cost): assumed flood-time speeds in
# km/h (truck on road 20, off-road 5 = the 4x cost; boat 6, under canopy 3; on foot 3) and handover minutes
ETA = dict(truck=20.0, truck_offroad=5.0, boat=6.0, boat_canopy=3.0, foot=3.0, launch_min=5.0, alight_min=1.0)

# majority-vote tie-break: on equal pixel counts the less drivable class wins
# (road_clear < other < water < road_flooded < bldg_dry < bldg_flooded)
_TIE = np.array([0, 3, 5, 4, 2, 1], np.float64) * 1e-3
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _disk(r):
    """disk structuring element of radius r cells (at least the 3x3 cross)"""
    k = max(int(np.ceil(r)), 1)
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    return x * x + y * y <= max(r, 1) ** 2


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
        pr, p = self.prof, self.par
        self.cell_m = p['cell_m'] or pr['cell_m']
        self.wall = np.isin(self.cls, pr['wall'])
        self.bldg = np.isin(self.cls, pr['bldg'])
        self.risk = self.probs[list(pr['risk'])].sum(0)
        land = self.cls == OTHER
        if p['offroad_maxwet'] is not None:
            land &= self.risk < p['offroad_maxwet']
        road = np.isin(self.cls, pr['truck'])
        if p['car_gap']:                                           # parked cars: close small gaps in the road
            road |= ndimage.binary_closing(road, _disk(p['car_gap'] / 2 / self.cell_m)) & land
        self.road = road
        self.offroad = land & ~road if p['offroad'] else np.zeros_like(self.wall)
        self.room_truck = self._room(p['truck_halfwidth'])
        self.room_boat = self._room(p['boat_halfwidth'])
        self.truck_ok = (road | self.offroad) & ~self.wall & self.room_truck
        boat = np.isin(self.cls, pr['boat']) | (self.probs[list(pr['boat'])].sum(0) >= p['boat_p'])
        self.canopy = np.zeros_like(self.wall)
        if p['canopy_gap']:                                        # narrow land strips between water
            self.canopy = ndimage.binary_closing(boat, _disk(p['canopy_gap'] / 2 / self.cell_m)) & ~boat \
                & (self.cls == OTHER)
        self.boat_ok = (boat | self.canopy) & ~self.wall & self.room_boat
        self.foot = p['foot_cost'] is not None
        self.foot_ok = (np.isin(self.cls, pr['truck'] + pr['walk']) | land) & ~self.bldg \
            if self.foot else np.zeros_like(self.wall)
        self.layers = 3 if self.foot else 2
        self.slow = np.where(self.offroad, p['offroad'] or 1.0, 1.0)            # truck step factor
        self.slow_boat = np.where(self.canopy, p['canopy_cost'], 1.0)           # boat step factor
        self._goals = {}
        self.graph = self._build()

    def _room(self, halfwidth):
        """cells whose centre is >= halfwidth + half a cell from every wall cell (None = a point vehicle)"""
        if halfwidth is None or not self.wall.any():
            return np.ones_like(self.wall)
        return ndimage.distance_transform_edt(~self.wall) >= self.clearance(halfwidth)

    def clearance(self, halfwidth):
        return halfwidth / self.cell_m + 0.5

    def goal(self, bld, layer):
        """Goal cells of a building for one mode: its ring; for a vehicle with a width, the building dilated
        by ceil(clearance) + ring cells (the vehicle's edge within the ring)."""
        half = {TRUCK: self.par['truck_halfwidth'], BOAT: self.par['boat_halfwidth']}.get(layer)
        if half is None or 'cells' not in bld:
            return bld['ring']
        key = (id(bld), layer)
        if key not in self._goals:
            m = np.zeros(self.n, bool)
            m[bld['cells']] = True
            m = m.reshape(self.h, self.w)
            k = int(np.ceil(self.clearance(half))) + bld.get('ring_k', 2)
            self._goals[key] = np.flatnonzero(ndimage.binary_dilation(m, np.ones((3, 3)), iterations=k) & ~m)
        return self._goals[key]

    def flags(self):
        """Per-cell bit flags for the web viewer: 1 truck, 2 boat, 4 foot, 8 off-road (truck slow), 16 canopy
        (boat slow), 32 wall, 64 building. The JS router reads these instead of re-deriving the rules."""
        f = np.zeros((self.h, self.w), np.uint8)
        for bit, m in ((1, self.truck_ok), (2, self.boat_ok), (4, self.foot_ok), (8, self.offroad),
                       (16, self.canopy), (32, self.wall), (64, self.bldg)):
            f |= np.where(m, bit, 0).astype(np.uint8)
        return f

    def _build(self):
        h, w, n, p = self.h, self.w, self.n, self.par
        idx = np.arange(n).reshape(h, w)
        rows, cols, vals = [], [], []
        for dy, dx in _DIRS:
            L = np.hypot(dy, dx)
            ys, yd = slice(max(-dy, 0), h - max(dy, 0)), slice(max(dy, 0), h - max(-dy, 0))
            xs, xd = slice(max(-dx, 0), w - max(dx, 0)), slice(max(dx, 0), w - max(-dx, 0))
            a, b = idx[ys, xs].ravel(), idx[yd, xd].ravel()
            ok_v = ok_f = np.ones(a.size, bool)
            if dy and dx and p['corner']:                          # no squeeze between two touching corners
                ok_v = ~(self.wall[yd, xs] & self.wall[ys, xd]).ravel()
                ok_f = ~(self.bldg[yd, xs] & self.bldg[ys, xd]).ravel()
            t, bo = self.truck_ok[yd, xd].ravel() & ok_v, self.boat_ok[yd, xd].ravel() & ok_v
            risk = self.risk[yd, xd].ravel()
            slow, slow_b = self.slow[yd, xd].ravel(), self.slow_boat[yd, xd].ravel()
            rows += [a[t], n + a[bo], a[bo]]                       # truck->truck, boat->boat, launch
            cols += [b[t], n + b[bo], n + b[bo]]
            vals += [L * (1 + p['lam'] * risk[t]) * slow[t],
                     L * p['boat_cost'] * slow_b[bo],
                     L * p['boat_cost'] * slow_b[bo] + p['launch']]
            if self.foot:                                          # foot->foot, truck->foot, boat->foot
                f = self.foot_ok[yd, xd].ravel() & ok_f
                rows += [2 * n + a[f], a[f], n + a[f]]
                cols += [2 * n + b[f]] * 3
                vals += [np.full(f.sum(), L * p['foot_cost'])] + [np.full(f.sum(), L * p['foot_cost'] + p['alight'])] * 2
        m = self.layers * n
        return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, m))

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
        n = self.grid.n
        for layer in range(self.grid.layers):                   # TRUCK, BOAT, FOOT
            r = self.grid.goal(bld, layer)
            if len(r) and np.isfinite(self.dist[layer * n + r]).any():
                return layer
        return ISOLATED

    def route(self, bld, layer=None):
        """Cheapest route to the building's ring that ends in `layer` (default: the layer of its status, so a
        TRUCK house gets its road route, a BOAT house its boat route): dict(cells=[(y, x)], modes, cost) or None."""
        n = self.grid.n
        layer = self.status(bld) if layer is None else layer
        if layer == ISOLATED:
            return None
        r = self.grid.goal(bld, layer)
        if not len(r):
            return None
        cand = layer * n + r
        j = cand[np.argmin(self.dist[cand])]
        if not np.isfinite(self.dist[j]):
            return None
        nodes = [j]
        while self.pred[nodes[-1]] >= 0:
            nodes.append(self.pred[nodes[-1]])
        nodes = np.array(nodes[::-1])
        cells = nodes % n
        return dict(cells=list(zip(*np.divmod(cells, self.grid.w))), modes=(nodes // n).astype(int).tolist(),
                    cost=float(self.dist[j]))


def tier(st_tuned, st_cautious):
    """tier name of a house from its predicted status with the tuned gate and with the cautious gate"""
    if st_cautious == TRUCK:
        return 'CONFIDENT_TRUCK'
    if st_tuned == TRUCK:
        return 'PROBABLE_TRUCK'
    return STATUS[st_tuned]


def eta(route, grid, cell_m=None):
    """metres per mode and minutes of a route (ETA speeds; cell_m defaults to the grid's)"""
    cell_m = cell_m or grid.cell_m
    metres, minutes = [0.0, 0.0, 0.0], 0.0
    cells, modes = route['cells'], route['modes']
    for k in range(1, len(cells)):
        (y0, x0), (y, x), m = cells[k - 1], cells[k], modes[k]
        d = np.hypot(y - y0, x - x0) * cell_m
        metres[m] += d
        if m == TRUCK:
            v = ETA['truck_offroad'] if grid.offroad[y, x] else ETA['truck']
        elif m == BOAT:
            v = ETA['boat_canopy'] if grid.canopy[y, x] else ETA['boat']
        else:
            v = ETA['foot']
        minutes += d / (v * 1000 / 60)
        if m != modes[k - 1]:
            minutes += ETA['launch_min'] if m == BOAT else ETA['alight_min']
    return dict(metres=metres, minutes=minutes)


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
        out.append(dict(id=len(out), area_px=area, cells=np.flatnonzero(cells), ring=np.flatnonzero(rg), ring_k=ring,
                        cls=int(pr['bldg'][int(np.argmax(sub[list(pr['bldg'])]))]),
                        centroid=(float(ys.mean()), float(xs.mean())), shape=(h, w)))
    return out


def eval_route_on_gt(route, gt_grid):
    """Score a route found on one map against the GT grid: cost under GT costs, feasibility, RSV.
    Infeasible = it drives a truck onto GT water/flooded/blocked road or enters a GT wall
    (a boat over dry ground is charged boat cost but not flagged; a truck on GT dry 'other' land
    pays the off-road cost, or is infeasible if the GT grid is roads-only; a walk is infeasible on GT
    water, flooded road or a building, gate or not). RSV = share of truck-leg cells
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
        if m == FOOT:
            feasible &= bool(np.isin(g.cls[y, x], g.prof['truck'] + g.prof['walk'] + (OTHER,))
                             and not np.isin(g.cls[y, x], g.prof['bldg']))
        elif g.wall[y, x] or (m == TRUCK and (haz[y, x] or not (g.truck_ok[y, x] or bad_truck[y, x]))) \
                or (m == BOAT and not g.room_boat[y, x]):
            feasible = False
        (y0, x0), m0 = cells[k - 1], modes[k - 1]
        L = np.hypot(y - y0, x - x0)
        slow = g.slow[y, x] if g.truck_ok[y, x] else (p['offroad'] if (p['offroad'] and g.cls[y, x] == OTHER) else 1.0)
        if m == TRUCK:
            cost += L * (1 + p['lam'] * g.risk[y, x]) * slow
        elif m == BOAT:
            cost += L * p['boat_cost'] * g.slow_boat[y, x] + p['launch'] * (m0 == TRUCK)
        else:
            cost += L * (p['foot_cost'] or 0.0) + p['alight'] * (m0 != FOOT)
    return dict(cost_gt=cost, feasible=feasible, unsafe=unsafe,
                rsv=truck_bad / truck_cells if truck_cells else None)


def route_metrics(pred_grid, gt_grid, base, blds):
    """Per-image routing comparison (PLAN §5.4) of a predicted map against the GT map, same base and
    GT buildings. Returns the 4x4 status confusion (rows GT, cols pred) and per-building records."""
    cm = np.zeros((len(STATUS), len(STATUS)), np.int64)
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


def false_safe_from(cm):
    """share of predicted-TRUCK houses whose labelled-map status is not TRUCK (rows GT, cols pred)"""
    cm = np.asarray(cm)
    return float(cm[1:, 0].sum() / max(cm[:, 0].sum(), 1))


def summarize_routing(cm, recs):
    """Dataset-level routing metrics from accumulated status confusion and building records.
    false_safe_rate = share of the houses shown as truck-reachable that are not truck-reachable on the labelled map
    (the dangerous error: a team is sent by road and cannot get through)."""
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
        false_safe_rate=false_safe_from(cm), n_false_safe=int(cm[1:, 0].sum()),
        reach_agreement=float(np.trace(cm) / max(cm.sum(), 1)), status_confusion=cm.tolist(),
        rsv_mean=float(np.mean([r['rsv'] for r in truck_routes])) if truck_routes else None,
        unsafe_route_rate=float(np.mean([r['unsafe'] for r in truck_routes])) if truck_routes else None,
        cost_ratio_median=float(np.median(ratios)) if ratios else None,
        infeasible_rate=float(np.mean(infeasible)) if gt_reach else None,
        n_routes_gt=len(gt_reach), n_routes_pred=len(with_route), n_cost_ratio=len(ratios),
        n_trivial_gt_routes=sum(r['cost_gt_opt'] == 0 for r in gt_reach))
