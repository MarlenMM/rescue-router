"""Unit tests for src/router.py on synthetic grids with known answers (PLAN §6.1).

    ~/.venvs/rr/bin/python -m pytest src/tests -q

Maps are drawn at cell level (cell=1, so pixels = cells): '.' other, R road_clear, F road_flooded
(RescueNet: road_blocked), B flooded building, D dry building, W water.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import router as R  # noqa: E402

CODE = {'.': 5, 'R': 0, 'F': 1, 'B': 2, 'D': 3, 'W': 4}


def grid_map(s):
    return np.array([[CODE[c] for c in row] for row in s.split()], np.uint8)


def setup(s, dataset='floodnet', **kw):
    m = grid_map(s)
    g = R.Grid(m, None, dataset, **kw)
    base = R.base_cells(m, dataset, offroad=bool(kw.get('offroad', R.DEFAULTS['offroad'])))
    return m, g, base, R.buildings_from_gt(m, dataset, min_area=1, ring=1, cell=1)


def test_truck_straight_road():
    m, g, base, blds = setup("""
        ..........
        RRRRRRRRRR
        ....DDD...
        ....DDD...
        ....DDD...""")
    assert base[1] == R.TRUCK and set(base[0]) == {10, 19}        # both border cells of the road
    assert len(blds) == 1 and blds[0]['cls'] == 3
    sol = g.solve(base)
    assert sol.status(blds[0]) == R.TRUCK
    r = sol.route(blds[0])
    assert r['cost'] == pytest.approx(2.0)                         # (1,9) -> (1,7)
    assert set(r['modes']) == {R.TRUCK}


def test_boat_across_water():
    m, g, base, blds = setup("""
        RRRWWWWWWW
        RRRWWWWWWW
        ...WWDDDWW
        ...WWDDDWW
        ...WWDDDWW""")
    assert base[1] == R.TRUCK                                      # road preferred over water
    sol = g.solve(base)
    assert sol.status(blds[0]) == R.BOAT
    r = sol.route(blds[0])
    # truck 1 step to (1,2), launch into (1,3) = 3 + 40, one boat step to the ring (1,4) = 3
    assert r['cost'] == pytest.approx(47.0)
    assert r['modes'][0] == R.TRUCK and r['modes'][-1] == R.BOAT
    assert r['modes'] == sorted(r['modes'])                        # never switches back to truck


GRASS = """
    RRRRRRRRRR
    ..........
    ..........
    ....DDD...
    ....DDD...
    ....DDD..."""


def test_isolated_in_grass_roads_only():
    m, g, base, blds = setup(GRASS, offroad=None)                   # strict PLAN §6.1: roads only
    sol = g.solve(base)
    assert sol.status(blds[0]) == R.ISOLATED and sol.route(blds[0]) is None


def test_offroad_across_the_yard():
    m, g, base, blds = setup(GRASS)                                 # default: dry land at 4x cost
    r = g.solve(base).route(blds[0])
    assert g.solve(base).status(blds[0]) == R.TRUCK
    assert r['cost'] == pytest.approx(8.0)                          # two lawn cells, rows 1-2, to the ring


def test_water_still_isolates_with_offroad():
    # an island house: water all round, roads only on the far side -> boat, never truck
    m, g, base, blds = setup("""
        RRRRRRRRRR
        WWWWWWWWWW
        WWW...WWWW
        WWW.DDD.WW
        WWW.DDD.WW
        WWW.DDD.WW
        WWW.....WW""")
    assert g.solve(base).status(blds[0]) == R.BOAT


def test_no_switch_back_to_truck():
    # the second road (row 1, cols 6-10) is only reachable by boat, and boats cannot drive on roads
    m, g, base, blds = setup("""
        RRRRWW......
        ....WWRRRRR.
        ........DDD.
        ........DDD.
        ........DDD.""", offroad=None)
    assert base[1] == R.TRUCK and set(base[0]) == {0, 1, 2, 3}
    assert g.solve(base).status(blds[0]) == R.ISOLATED


def test_buildings_are_walls():
    # the direct way along row 0 is blocked by the building column; the truck goes round its end
    m = grid_map("""
        R.D.R
        R.D.R
        R.D.R
        RRRRR""")
    sol = R.Grid(m).solve((np.array([0]), R.TRUCK))
    r = sol.route(dict(ring=np.array([4])))                         # cell (0, 4)
    assert all(m[y, x] != 3 for y, x in r['cells'])
    assert r["cost"] == pytest.approx(6 + 2 * np.sqrt(2))


def test_risk_aware_avoids_wet_road():
    m = grid_map("""
        RRRRRRR
        R.....R
        RRRRRRR""")
    p = np.stack([(m == c) for c in range(5)]).astype(np.float32)
    p[1, 0, 2:5] = 0.3                                              # row 0 still decoded clear, but 30 % flooded
    target, base = dict(ring=np.array([6])), (np.array([0]), R.TRUCK)
    short = R.Grid(m, p, lam=0).solve(base).route(target)
    safe = R.Grid(m, p, lam=5).solve(base).route(target)
    assert short['cost'] == pytest.approx(6.0) and all(y == 0 for y, _ in short['cells'])
    assert safe['cost'] == pytest.approx(6 + 2 * np.sqrt(2)) and any(y == 2 for y, _ in safe['cells'])


def test_soft_wet_prob_opens_boat_cell():
    m = grid_map("RR.W")
    p = np.stack([(m == c) for c in range(5)]).astype(np.float32)
    base, target = (np.array([0]), R.TRUCK), dict(ring=np.array([3]))
    assert R.Grid(m, p, offroad=None).solve(base).route(target) is None   # the '.' cell blocks
    p[4, 0, 2] = 0.6                                                # p_water 0.6 >= 0.5 -> boat may cross
    r = R.Grid(m, p, offroad=None).solve(base).route(target)
    assert r['cost'] == pytest.approx(1 + 43 + 3) and r['modes'] == [0, 0, 1, 1]


def test_rescuenet_blocked_road_is_a_wall():
    m = grid_map("RRRFRRR")
    base, target = (np.array([0]), R.TRUCK), dict(ring=np.array([3]))
    assert R.Grid(m, None, 'floodnet').solve(base).status(target) == R.BOAT      # flooded road: boat
    assert R.Grid(m, None, 'rescuenet').solve(base).status(target) == R.ISOLATED  # blocked: nobody


def test_base_fallbacks():
    assert R.base_cells(grid_map("WW..\n....\n.DD."), offroad=False)[1] == R.BOAT
    assert R.base_cells(grid_map("WWWW\nW..W\nWWWW"))[1] == R.BOAT                  # no dry land on the border
    land = R.base_cells(grid_map("....\n.RR.\n...."))                             # road inside, lawn on the border
    assert land[1] == R.TRUCK and len(land[0]) == 10
    assert R.base_cells(grid_map("....\n.RR.\n...."), offroad=False) is None


def test_cell_classes_majority_and_ties():
    px = np.zeros((4, 12), np.uint8)
    px[:, :4] = 0
    px[:2, 4:8], px[2:, 4:8] = 0, 4                                  # 8 road vs 8 water -> water (safer)
    px[:, 8:] = 255                                                  # all ignore -> other
    assert R.cell_classes(px).tolist() == [[0, 4, 5]]
    probs = np.zeros((5, 4, 8), np.uint8)
    probs[1, :, :4] = 255
    cp = R.cell_probs(probs)
    assert cp.shape == (5, 1, 2) and cp[1, 0, 0] == pytest.approx(1.0) and cp[1, 0, 1] == 0


def test_metrics_identical_maps_are_perfect():
    m, g, base, blds = setup("""
        ..........
        RRRFFFFRR.
        ......DDD.
        ......DDD.
        ......DDD.""")
    cm, recs = R.route_metrics(g, g, base, blds)
    s = R.summarize_routing(cm, recs)
    assert s['reach_agreement'] == 1 and s['cost_ratio_median'] == pytest.approx(1)
    assert s['rsv_mean'] == 0 and s['infeasible_rate'] == 0 and s['unsafe_route_rate'] == 0


def test_metrics_false_clear_road_is_unsafe():
    gt = """
        ..........
        RRRFFFFRR.
        ......DDD.
        ......DDD.
        ......DDD."""
    m, g, base, blds = setup(gt, offroad=None)
    assert g.solve(base).status(blds[0]) == R.BOAT
    assert g.solve(base).route(blds[0])['cost'] == pytest.approx(2 + 43 + 6)
    pred = R.Grid(grid_map(gt.replace('F', 'R')), offroad=None)                   # flooded stretch predicted clear
    cm, recs = R.route_metrics(pred, g, base, blds)
    assert cm[R.BOAT, R.TRUCK] == 1                                 # stranded house missed
    r = recs[0]
    assert r['status_pred'] == R.TRUCK and not r['feasible'] and r['unsafe']
    assert r['rsv'] == pytest.approx(3 / 6)                         # (1,3),(1,4),(1,5) of 6 truck cells
    s = R.summarize_routing(cm, recs)
    assert s['stranded_recall'] == 0 and s['infeasible_rate'] == 1 and s['cost_ratio_median'] is None


def test_cost_ratio_skips_houses_at_the_entry():
    # the building's ring touches the border road (the base): GT cost 0, so no cost ratio, but feasible
    m, g, base, blds = setup("""
        RRRRRRRRRR
        ....DDD...
        ....DDD...
        ....DDD...""")
    assert g.solve(base).route(blds[0])['cost'] == 0
    cm, recs = R.route_metrics(g, g, base, blds)
    assert 'cost_ratio' not in recs[0] and recs[0]['feasible']
    s = R.summarize_routing(cm, recs)
    assert s['infeasible_rate'] == 0 and s['cost_ratio_median'] is None and s['n_trivial_gt_routes'] == 1
