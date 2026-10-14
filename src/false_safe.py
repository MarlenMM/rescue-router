"""Why a house is falsely shown as truck-reachable (PLAN §16, A2). CPU only, from the caches.

A false-safe house: router v3 on the predicted map says TRUCK, on the labelled map it is not truck-reachable. For each
one, the predicted truck route is followed on the labelled map to the first cell a real truck could not use, and that
cell's labelled vs predicted class names the error: e.g. a flooded road read as clear road, water read as land, a
building or wall missed (the gap is narrower than it looks), a truck route through dry land the labels mark as
blocked. Counts per cause (eval100 and full test) and the worst photos, for the model work to target.

    python src/false_safe.py [--results results/kaggle_v2/rr] [--out results/routing/false_safe]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import router as R  # noqa: E402
from route_eval import PRED_OFFROAD_MAXWET  # noqa: E402
from rr.final_decode import decode_photo, load_chosen, photo  # noqa: E402
from rr.metrics import gt_classes  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

DS = 'floodnet'
RUN = 'floodnet_50shot_s0_stratified'
CLS = {0: 'clear road', 1: 'flooded road', 2: 'flooded building', 3: 'dry building', 4: 'water', 5: 'other'}


def cause(g_cls, p_cls, g_grid, y, x):
    """name of the error at the first cell of the predicted truck leg that a truck cannot use on the labelled map"""
    gc, pc = int(g_cls[y, x]), int(p_cls[y, x])
    if gc in (1, 4):
        return f'{CLS[gc]} read as {CLS[pc]}'
    if gc in (2, 3):
        return f'building read as {CLS[pc]}'
    if g_grid.wall[y, x] or not g_grid.room_truck[y, x]:
        return 'gap narrower than predicted (building/wall missed nearby)'
    return f'labelled {CLS[gc]}, not truck-passable ({CLS[pc]} predicted)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--out', default='results/routing/false_safe')
    ap.add_argument('--splits', default='floodnet_test_eval100,floodnet_test_full')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dec = load_chosen(a.results, RUN)
    out, md = {}, ['# False-safe houses: first point where the predicted truck route fails on the labelled map', '',
                   f'Run {RUN}, router v3 (risk-aware, off-road gate {PRED_OFFROAD_MAXWET[DS]}), the published decode.', '']
    for sf in a.splits.split(','):
        z = np.load(os.path.join(a.results, 'preds', f'{RUN}_{sf}_fp16.npz'))
        P = dict(zip([str(i) for i in z['ids']], z['probs']))
        ids = open(os.path.join(os.path.dirname(HERE), 'splits', f'{sf}.txt')).read().split()
        causes, per_img, n_fs, n_truck, along = collections.Counter(), collections.Counter(), 0, 0, []
        for i in ids:
            gt = gt_classes(np.asarray(Image.open(os.path.join(RR_DATA, f'{DS}_r768', 'test', 'labels', f'{i}.png'))))
            gcls = R.cell_classes(gt)
            base = R.base_cells(gcls, DS, offroad=True)
            if base is None:
                continue
            blds = R.buildings_from_gt(gt, DS)
            pred = decode_photo(P[i].astype(np.float32) / 255, dec, img=photo(RR_DATA, DS, 'test', i), split_file=sf,
                                photo_id=i)
            pcls = R.cell_classes(pred)
            pg = R.Grid(pcls, R.cell_probs(P[i]), DS, offroad_maxwet=PRED_OFFROAD_MAXWET[DS])
            gg = R.Grid(gcls, R.onehot_probs(gt), DS)
            sp, sg = pg.solve(base), gg.solve(base)
            for b in blds:
                if sp.status(b) != R.TRUCK:
                    continue
                n_truck += 1
                if sg.status(b) == R.TRUCK:
                    continue
                n_fs += 1
                per_img[i] += 1
                r = sp.route(b)
                hit = None
                for k, ((y, x), m) in enumerate(zip(r['cells'], r['modes'])):
                    if k and m == R.TRUCK and not gg.truck_ok[y, x]:
                        hit = (y, x, k / max(len(r['cells']) - 1, 1))
                        break
                if hit is None:
                    causes['route passes, but the labelled-map goal ring differs'] += 1
                    continue
                causes[cause(gcls, pcls, gg, hit[0], hit[1])] += 1
                along.append(hit[2])
        out[sf] = dict(n_truck_pred=n_truck, n_false_safe=n_fs, causes=dict(causes.most_common()),
                       worst_photos=per_img.most_common(10),
                       fail_point_along_route_median=float(np.median(along)) if along else None)
        md += [f'## {sf}: {n_fs} false-safe of {n_truck} houses shown truck-reachable ({n_fs / max(n_truck, 1):.1%})', '',
               '| first failure on the labelled map | houses | share |', '|---|---|---|']
        md += [f'| {c} | {k} | {k / max(n_fs, 1):.0%} |' for c, k in causes.most_common()]
        md += ['', f'Median position of the failure along the route: {out[sf]["fail_point_along_route_median"]:.2f} '
               '(0 = entry, 1 = house). Photos with the most false-safe houses: '
               + ', '.join(f'{i} ({k})' for i, k in per_img.most_common(8)), '']
        print('\n'.join(md[-(len(causes) + 6):]), flush=True)
    json.dump(out, open(os.path.join(a.out, 'false_safe.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'false_safe.md'), 'w').write('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
