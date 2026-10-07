"""Router sensitivity and building detection (Batch 3). CPU only, FloodNet 50-shot (tuned decode) on eval100
and RescueNet 20-shot on eval100.

    python src/sensitivity.py [--out results/routing/sensitivity]

1. Cost independence: the status of a house (truck / boat / foot / none) depends only on which cells each mode may
   enter, never on the cost values. Checked by re-routing with every cost changed (lam, boat cost, launch, foot
   cost, alight, off-road factor, canopy factor): the statuses on both maps must be identical.
2. Sensitivity of the statuses and the stranded metrics to the settings that do matter: off-road gate, goal ring,
   boat probability threshold, cell size (4 vs 8 px, ring kept in metres), vehicle half-widths x0.5 / x1.5.
3. Building detection on the predicted map (what an uploaded photo relies on): predicted building components (same
   opening and area rule as the labelled ones) matched to labelled buildings by IoU (>= 0.5, and >= 0.25).
Writes sensitivity.json and sensitivity.md.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import router as R  # noqa: E402
from export_web_assets import building_masks  # noqa: E402
from route_eval import PRED_OFFROAD_MAXWET  # noqa: E402
from rr.metrics import decode, gt_classes  # noqa: E402
from rr.final_decode import decode_photo, photo  # noqa: E402

RUNS = {'floodnet': 'floodnet_50shot_s0_stratified', 'rescuenet': 'rescuenet_20shot_s0_stratified'}
COSTS = [dict(lam=0.0), dict(lam=20.0), dict(boat_cost=1.5), dict(boat_cost=6.0), dict(launch=0.0), dict(launch=200.0),
         dict(foot_cost=4.0), dict(foot_cost=16.0), dict(alight=0.0), dict(alight=50.0), dict(offroad=2.0),
         dict(offroad=8.0), dict(canopy_cost=1.0), dict(canopy_cost=4.0)]


def load(ds, a):
    run = RUNS[ds]
    z = np.load(os.path.join(a.results, 'preds', f'{run}_{ds}_test_eval100_fp16.npz'))
    dec = {k: v for k, v in json.load(open(os.path.join(a.results, 'decode', f'{run}.json')))['chosen'].items()
           if k != 'variant'}
    out = []
    for i, p in zip([str(x) for x in z['ids']], z['probs']):
        gt = gt_classes(np.asarray(Image.open(os.path.join(a.data, f'{ds}_r768', 'test', 'labels', f'{i}.png'))))
        out.append((i, gt, decode_photo(p.astype(np.float32) / 255, dec, img=photo(a.data, ds, 'test', i),
                                        split_file=f'{ds}_test_eval100', photo_id=i), p))
    return out


def route_all(ds, imgs, cell=R.CELL, ring=2, gate=None, **kw):
    """statuses on both maps for every labelled building, and the summed status confusion (rows GT, cols pred)"""
    gate = PRED_OFFROAD_MAXWET[ds] if gate is None else gate
    if cell != R.CELL:
        kw = {'cell_m': R.PROFILES[ds]['cell_m'] * cell / R.CELL, **kw}
    cm, sts = np.zeros((4, 4), np.int64), []
    for i, gt, pred, p in imgs:
        gc = R.cell_classes(gt, cell)
        base = R.base_cells(gc, ds)
        if base is None:
            continue
        blds = R.buildings_from_gt(gt, ds, ring=ring, cell=cell)
        sg = R.Grid(gc, R.onehot_probs(gt, cell), ds, **kw).solve(base)
        sp = R.Grid(R.cell_classes(pred, cell), R.cell_probs(p, cell), ds, offroad_maxwet=gate, **kw).solve(base)
        for b in blds:
            g, q = sg.status(b), sp.status(b)
            cm[g, q] += 1
            sts.append((g, q))
    return cm, sts


def summary(cm):
    s = R.summarize_routing(cm, [])
    return dict(n=int(cm.sum()), labelled=cm.sum(1).tolist(), predicted=cm.sum(0).tolist(),
                P=s['stranded_precision'], R=s['stranded_recall'], F1=s['stranded_f1'], agreement=s['reach_agreement'])


def detect(imgs, ds, thr):
    tp = fp = fn = same = 0
    for i, gt, pred, p in imgs:
        G, P = building_masks(gt, ds), building_masks(pred, ds)
        used = set()
        for pm in P:
            best, bj = 0.0, None
            for j, gm in enumerate(G):
                if j in used:
                    continue
                inter = (pm & gm).sum()
                if inter:
                    iou = inter / (pm | gm).sum()
                    if iou > best:
                        best, bj = iou, j
            if best >= thr:
                tp += 1
                used.add(bj)
                gcls = np.bincount(gt[G[bj]], minlength=6)[2:4].argmax()
                pcls = np.bincount(pred[pm], minlength=6)[2:4].argmax()
                same += int(gcls == pcls)
            else:
                fp += 1
        fn += len(G) - len(used)
    P, Rr = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(iou=thr, tp=tp, fp=fp, fn=fn, precision=P, recall=Rr, f1=2 * P * Rr / max(P + Rr, 1e-9),
                subclass_agreement=same / max(tp, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--out', default='results/routing/sensitivity')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = dict(cost_independence={}, sensitivity={}, detection={})
    md = []
    for ds in RUNS:
        imgs = load(ds, a)
        cm0, st0 = route_all(ds, imgs)
        ref = summary(cm0)
        # 1. cost independence
        bad = {}
        for c in COSTS:
            _, st = route_all(ds, imgs, **c)
            bad[json.dumps(c)] = sum(x != y for x, y in zip(st, st0))
        out['cost_independence'][ds] = bad
        md += [f'**{ds}: cost independence** — houses whose status changed when one cost value changed '
               f'(of {len(st0)} houses x 2 maps): ' + ', '.join(f'{k}: {v}' for k, v in bad.items()), '']
        # 2. sensitivity
        hw = (R.DEFAULTS['truck_halfwidth'], R.DEFAULTS['boat_halfwidth'])
        variants = [('default', {}), ('gate 0.2', dict(gate=0.2)), ('gate 0.5', dict(gate=0.5)),
                    ('ring 1 cell', dict(ring=1)), ('ring 3 cells', dict(ring=3)),
                    ('boat threshold 0.3', dict(boat_p=0.3)), ('boat threshold 0.7', dict(boat_p=0.7)),
                    ('cell 8 px (ring 1 = same metres)', dict(cell=8, ring=1)),
                    ('half-widths x0.5', dict(truck_halfwidth=hw[0] / 2, boat_halfwidth=hw[1] / 2)),
                    ('half-widths x1.5', dict(truck_halfwidth=hw[0] * 1.5, boat_halfwidth=hw[1] * 1.5))]
        md += [f'| {ds} ({RUNS[ds].split("_")[1]}, eval100) | houses | labelled: truck/boat/foot/none | predicted | '
               'stranded P | R | F1 | agreement |', '|---|---|---|---|---|---|---|---|']
        out['sensitivity'][ds] = {}
        for name, kw in variants:
            s = ref if not kw else summary(route_all(ds, imgs, **kw)[0])
            out['sensitivity'][ds][name] = s
            md.append(f"| {name} | {s['n']} | {'/'.join(map(str, s['labelled']))} | {'/'.join(map(str, s['predicted']))} | "
                      f"{s['P']:.3f} | {s['R']:.3f} | {s['F1']:.3f} | {s['agreement']:.3f} |")
            print(md[-1], flush=True)
        # 3. building detection
        out['detection'][ds] = [detect(imgs, ds, t) for t in (0.5, 0.25)]
        md += ['', f'| {ds}: predicted buildings vs labelled | IoU | precision | recall | F1 | flooded/dry (or intact/damaged) agreement |',
               '|---|---|---|---|---|---|']
        for d in out['detection'][ds]:
            md.append(f"| tp {d['tp']}, fp {d['fp']}, fn {d['fn']} | ≥ {d['iou']} | {d['precision']:.3f} | {d['recall']:.3f} | "
                      f"{d['f1']:.3f} | {d['subclass_agreement']:.3f} |")
        md.append('')
    json.dump(out, open(os.path.join(a.out, 'sensitivity.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'sensitivity.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
