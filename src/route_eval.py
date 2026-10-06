"""Routing evaluation (PLAN §5.4) on ground-truth maps and, given a prediction cache, on predictions.
CPU only.

    python src/route_eval.py --dataset floodnet --split_file floodnet_test_eval100 \
        [--preds results/kaggle_v2/rr/preds/<run>_floodnet_test_eval100_fp16.npz] [--tau 0.5 | --decode <json>]

Without --preds: GT routing statistics (base found?, buildings, GT status counts) and a GT-vs-GT
sanity check (must be perfect). With --preds: the predicted map is routed from the same GT base to
the same GT buildings, for the risk-aware (lam = 5) and the plain (lam = 0) truck cost.
Writes a JSON summary (--out).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import router as R  # noqa: E402
from rr.metrics import decode, gt_classes  # noqa: E402

SPLIT = {'test': 'test', 'val': 'val'}
# off-road gate for predicted maps (the GT grid has none): tuned on FloodNet 50-shot val30 (stranded F1
# .548 -> .971 at 0.3), see PLAN §15 "Off-road gate"; the web viewer's JS router uses the same value
PRED_OFFROAD_MAXWET = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='floodnet', choices=list(R.PROFILES))
    ap.add_argument('--split_file', required=True)
    ap.add_argument('--split', default=None, help='image folder (default: from the split file name)')
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--repo', default=os.path.dirname(HERE))
    ap.add_argument('--preds', default=None)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--decode', default=None, help='JSON with per-channel "tau" (and "bias"), see decode_tune.py')
    ap.add_argument('--lams', default='5,0')
    ap.add_argument('--offroad', type=float, default=R.DEFAULTS['offroad'], help='0 = roads only (strict §6.1)')
    ap.add_argument('--offroad_maxwet', type=float, default=PRED_OFFROAD_MAXWET,
                    help='prediction grid only: off-road where p_wet < this (<0 = no gate)')
    ap.add_argument('--foot_cost', type=float, default=R.DEFAULTS['foot_cost'],
                    help='walking cost per cell after a truck or boat (<0 = no walking: the truck/boat router)')
    ap.add_argument('--point_vehicles', action='store_true',
                    help='no vehicle widths, car closing, canopy or corner rule (the router before 6 Oct Batch 1)')
    ap.add_argument('--router', default='{}', help='JSON overrides of router parameters (ablations), both maps')
    ap.add_argument('--ring', type=int, default=2)
    ap.add_argument('--min_area', type=int, default=150)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    split = a.split or ('test' if '_test_' in a.split_file else 'val')
    ids = [l.strip() for l in open(os.path.join(a.repo, 'splits', f'{a.split_file}.txt')) if l.strip()]
    root = os.path.join(a.data, f'{a.dataset}_r768', split)
    P = None
    if a.preds:
        z = np.load(a.preds)
        P = dict(zip([str(i) for i in z['ids']], z['probs']))
        assert set(ids) <= set(P), 'prediction cache does not cover the split'
    dec = dict(tau=a.tau)
    if a.decode:
        dec = json.load(open(a.decode))
    lams = [float(x) for x in a.lams.split(',')]

    NS = len(R.STATUS)
    cm_gt = np.zeros((NS, NS), np.int64)
    acc = {lam: (np.zeros((NS, NS), np.int64), []) for lam in lams}
    n_base, gt_counts, per_img, t0 = 0, np.zeros(NS, np.int64), [], time.time()
    for i in ids:
        gt = gt_classes(np.asarray(Image.open(os.path.join(root, 'labels', f'{i}.png'))))
        gcls = R.cell_classes(gt)
        base = R.base_cells(gcls, a.dataset, offroad=bool(a.offroad))
        blds = R.buildings_from_gt(gt, a.dataset, min_area=a.min_area, ring=a.ring)
        rec = dict(id=i, base=None if base is None else R.STATUS[base[1]], n_buildings=len(blds))
        if base is None:
            per_img.append(rec)
            continue
        n_base += 1
        foot = None if a.foot_cost < 0 else a.foot_cost
        phys = {**(R.POINT_VEHICLES if a.point_vehicles else {}), **json.loads(a.router)}
        ggrid = {lam: R.Grid(gcls, R.onehot_probs(gt), a.dataset, lam=lam, offroad=a.offroad or None, foot_cost=foot,
                             **phys) for lam in lams}
        g0 = ggrid[lams[0]]
        st = [g0.solve(base).status(b) for b in blds]
        gt_counts += np.bincount(st, minlength=NS)
        rec['gt_status'] = np.bincount(st, minlength=NS).tolist()
        cm, _ = R.route_metrics(g0, g0, base, blds)                 # sanity: GT vs GT
        cm_gt += cm
        if P is not None:
            probs = P[i]
            pred = decode(probs.astype(np.float32) / 255, **dec)
            pc, pp = R.cell_classes(pred), R.cell_probs(probs)
            for lam in lams:
                cm, recs = R.route_metrics(R.Grid(pc, pp, a.dataset, lam=lam, offroad=a.offroad or None, foot_cost=foot, **phys,
                                           offroad_maxwet=None if a.offroad_maxwet < 0 else a.offroad_maxwet), ggrid[lam], base, blds)
                acc[lam][0][...] += cm
                acc[lam][1].extend(dict(r, image=i) for r in recs)
                if lam == lams[0]:                      # this image only (rows GT, cols pred)
                    rec['status_confusion'] = cm.tolist()
                    rec['pred_status'] = cm.sum(0).tolist()
                    rec['n_unsafe_routes'] = sum(bool(r.get('unsafe')) for r in recs)
                    rec['n_infeasible'] = sum(not r.get('feasible', False) for r in recs if 'cost_gt_opt' in r)
        per_img.append(rec)

    out = dict(dataset=a.dataset, split_file=a.split_file, n_images=len(ids), n_with_base=n_base,
               ring=a.ring, min_area=a.min_area, params=dict(R.DEFAULTS, offroad=a.offroad or None, offroad_maxwet=None if a.offroad_maxwet < 0 else a.offroad_maxwet,
                           foot_cost=None if a.foot_cost < 0 else a.foot_cost,
                           **(R.POINT_VEHICLES if a.point_vehicles else {}), **json.loads(a.router)),
               gt_status_counts=dict(zip(R.STATUS, gt_counts.tolist())),
               base_mode_counts={m: sum(r['base'] == m for r in per_img) for m in ('TRUCK', 'BOAT', None)},
               gt_vs_gt_agreement=float(np.trace(cm_gt) / max(cm_gt.sum(), 1)),
               seconds=round(time.time() - t0, 1))
    if P is not None:
        out.update(preds=os.path.basename(a.preds), decode=dec,
                   routing={f'lam{lam:g}': R.summarize_routing(*acc[lam]) for lam in lams})
    out['per_image'] = per_img
    print(json.dumps({k: v for k, v in out.items() if k != 'per_image'}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
