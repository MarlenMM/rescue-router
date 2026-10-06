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


def Grid(*a, **kw):
    """the router with point vehicles (no widths, car closing, canopy or corner rule) unless a test asks"""
    return R.Grid(*a, **{**R.POINT_VEHICLES, **kw})


def setup(s, dataset='floodnet', **kw):
    m = grid_map(s)
    g = Grid(m, None, dataset, **kw)
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
    m, g, base, blds = setup(GRASS, offroad=None, foot_cost=None)   # strict PLAN §6.1: roads only, no walking
    sol = g.solve(base)
    assert sol.status(blds[0]) == R.ISOLATED and sol.route(blds[0]) is None


def test_roads_only_truck_then_walk_across_the_yard():
    m, g, base, blds = setup(GRASS, offroad=None)                   # truck stays on the road, people walk
    r = g.solve(base).route(blds[0])
    assert g.solve(base).status(blds[0]) == R.FOOT
    assert r['modes'] == [R.TRUCK, R.FOOT, R.FOOT] and r['cost'] == pytest.approx(2 * 8 + 10)


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
        ........DDD.""", offroad=None, foot_cost=None)
    assert base[1] == R.TRUCK and set(base[0]) == {0, 1, 2, 3}
    assert g.solve(base).status(blds[0]) == R.ISOLATED


def test_boat_then_foot_never_back_to_a_vehicle():
    # a full-height water channel: truck to the bank, boat across, walk to the house; nobody drives after the boat
    m, g, base, blds = setup("""
        RRRRRWW.RRR..
        .....WW......
        .....WW..DDD.
        .....WW..DDD.
        .....WW..DDD.""", offroad=None)
    sol = g.solve(base)
    r = sol.route(blds[0])
    assert sol.status(blds[0]) == R.FOOT
    assert r['modes'] == sorted(r['modes']) and r['modes'][-1] == R.FOOT and R.BOAT in r['modes']


def test_walls_stop_walkers_and_water_stops_walkers():
    m, g, base, blds = setup("""
        RRRR......
        WWWWWWWWWW
        ..........
        ....DDD...
        ....DDD...
        ....DDD...""")
    assert g.solve(base).status(blds[0]) == R.FOOT                 # truck -> boat across row 1 -> walk
    g2 = Grid(m, None, 'floodnet', foot_cost=8.0)
    g2.boat_ok[:] = False                                            # no boats: walkers cannot cross water
    g2.graph = g2._build()
    assert g2.solve(base).status(blds[0]) == R.ISOLATED


def test_buildings_are_walls():
    # the direct way along row 0 is blocked by the building column; the truck goes round its end
    m = grid_map("""
        R.D.R
        R.D.R
        R.D.R
        RRRRR""")
    sol = Grid(m).solve((np.array([0]), R.TRUCK))
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
    short = Grid(m, p, lam=0).solve(base).route(target)
    safe = Grid(m, p, lam=5).solve(base).route(target)
    assert short['cost'] == pytest.approx(6.0) and all(y == 0 for y, _ in short['cells'])
    assert safe['cost'] == pytest.approx(6 + 2 * np.sqrt(2)) and any(y == 2 for y, _ in safe['cells'])


def test_soft_wet_prob_opens_boat_cell():
    m = grid_map("RR.W")
    p = np.stack([(m == c) for c in range(5)]).astype(np.float32)
    base, target = (np.array([0]), R.TRUCK), dict(ring=np.array([3]))
    assert Grid(m, p, offroad=None).solve(base).route(target) is None   # the '.' cell blocks
    p[4, 0, 2] = 0.6                                                # p_water 0.6 >= 0.5 -> boat may cross
    r = Grid(m, p, offroad=None).solve(base).route(target)
    assert r['cost'] == pytest.approx(1 + 43 + 3) and r['modes'] == [0, 0, 1, 1]


def test_rescuenet_blocked_road_is_a_wall():
    m = grid_map("RRRFRRR")
    base, target = (np.array([0]), R.TRUCK), dict(ring=np.array([3]))
    assert Grid(m, None, 'floodnet').solve(base).status(target) == R.BOAT      # flooded road: boat
    assert Grid(m, None, 'rescuenet', foot_cost=None).solve(base).status(target) == R.ISOLATED  # no vehicle
    sol = Grid(m, None, 'rescuenet').solve(base)                                # debris: walkable
    assert sol.status(target) == R.FOOT and sol.route(target)['modes'][-1] == R.FOOT


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
    pred = Grid(grid_map(gt.replace('F', 'R')), offroad=None)                   # flooded stretch predicted clear
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


def test_offroad_gate_blocks_wet_looking_land():
    m = grid_map("RR..D")
    p = np.stack([(m == c) for c in range(5)]).astype(np.float32)
    p[4, 0, 3] = 0.3                                                # lawn cell (0,3) looks 30 % water
    base, target = (np.array([0]), R.TRUCK), dict(ring=np.array([3]))
    assert Grid(m, p).solve(base).status(target) == R.TRUCK
    assert Grid(m, p, offroad_maxwet=0.2).solve(base).status(target) == R.ISOLATED
    assert Grid(m, p, offroad_maxwet=0.5).solve(base).status(target) == R.TRUCK


# ---------------------------------------------------------------- physical rules (Batch 1), cell_m = 1 m
WIDE = dict(cell_m=1.0, truck_halfwidth=1.0, boat_halfwidth=None, car_gap=None, canopy_gap=None, corner=True)


def test_truck_too_wide_for_the_gap_people_walk_through():
    m, g, base, blds = setup("""
        RRRRRRR
        DDD.DDD
        DDD.DDD
        .......
        ...BBB.
        ...BBB.
        ...BBB.""")
    house = [b for b in blds if b['cls'] == 2][0]
    assert g.solve(base).status(house) == R.TRUCK                  # a point truck slips through the 1 m gap
    wide = R.Grid(m, None, 'floodnet', **WIDE)
    assert wide.solve(base).status(house) == R.FOOT                # 2 m truck stops, people walk through


def test_wide_truck_still_reaches_a_house_by_the_road():
    m, g, base, blds = setup("""
        RRRRRRR
        .......
        ..DDD..
        ..DDD..
        ..DDD..""")
    wide = R.Grid(m, None, 'floodnet', **WIDE)
    assert not wide.truck_ok[1, 3] and wide.truck_ok[0, 3]          # body would touch the wall in row 1
    assert wide.solve(base).status(blds[0]) == R.TRUCK              # its edge is still within the ring


def test_no_squeeze_between_touching_corners():
    m = grid_map("""
        RD.
        D..""")
    base, target = (np.array([0]), R.TRUCK), dict(ring=np.array([4]))   # cell (1, 1)
    assert Grid(m).solve(base).status(target) == R.TRUCK                  # diagonal squeeze
    assert Grid(m, corner=True).solve(base).status(target) == R.ISOLATED  # blocked for everyone


def test_parked_car_gap_is_closed_on_dry_land_only():
    m = grid_map("""
        .....
        RR.RR
        .....""")
    base, target = (np.array([5]), R.TRUCK), dict(ring=np.array([9]))    # row 1, left end -> right end
    assert Grid(m).solve(base).route(target)['cost'] == pytest.approx(1 + 4 + 1 + 1)   # gap crossed off-road
    g = Grid(m, cell_m=1.0, car_gap=3.0)
    assert g.road[1, 2] and g.solve(base).route(target)['cost'] == pytest.approx(4.0)  # gap closed: all road
    p = np.stack([(m == c) for c in range(5)]).astype(np.float32)
    p[4, 1, 2] = 0.5                                                     # the gap looks wet
    assert not Grid(m, p, cell_m=1.0, car_gap=3.0, offroad_maxwet=0.3).road[1, 2]   # the gate still holds


def test_boat_crosses_a_narrow_canopy_strip():
    m, g, base, blds = setup("""
        RWW.WW.DDD
        RWW.WW.DDD
        RWW.WW.DDD
        RWW.WW.DDD""")
    assert g.solve(base).status(blds[0]) == R.ISOLATED             # people stranded on the strip
    c = R.Grid(m, None, 'floodnet', **{**R.POINT_VEHICLES, 'cell_m': 1.0, 'canopy_gap': 2.0})
    assert c.canopy[1, 3]
    sol = c.solve(base)
    r = sol.route(blds[0])
    assert sol.status(blds[0]) == R.FOOT and (1, 3) in [cell for cell, md in zip(r['cells'], r['modes']) if md == R.BOAT]


# ---------------------------------------------------------------- confidence tiers and travel times (Batch 2)
def test_tiers_nest_and_name():
    m = grid_map("""
        RRR....DDD
        .......DDD
        .......DDD""")
    p = np.stack([(m == c) for c in range(5)]).astype(np.float32)
    p[4, :, 3:7] = 0.2                                               # the lawn before the house may be wet
    base, blds = R.base_cells(m), R.buildings_from_gt(m, min_area=1, ring=1, cell=1)
    st = {g: Grid(m, p, offroad_maxwet=g).solve(base).status(blds[0]) for g in (0.3, 0.1)}
    assert st == {0.3: R.TRUCK, 0.1: R.ISOLATED}                     # only the tuned gate opens the lawn (walkers too)
    assert R.tier(st[0.3], st[0.1]) == 'PROBABLE_TRUCK'
    assert R.tier(R.TRUCK, R.TRUCK) == 'CONFIDENT_TRUCK' and R.tier(R.BOAT, R.BOAT) == 'BOAT'


def test_eta_counts_metres_speeds_and_handovers():
    m = grid_map("RRWW.DDD")
    g = Grid(m, cell_m=1.0)
    r = dict(cells=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], modes=[R.TRUCK, R.TRUCK, R.BOAT, R.BOAT, R.FOOT])
    e = R.eta(r, g)
    assert e['metres'] == [1.0, 2.0, 1.0]
    want = 1 / (20 / 0.06) + 2 / (6 / 0.06) + 1 / (3 / 0.06) + 5 + 1   # km/h -> m/min = v * 1000 / 60
    assert e['minutes'] == pytest.approx(want)
