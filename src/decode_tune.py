"""Decode tuning on val30 from the prediction caches (PLAN §15 step 3). CPU only.

    python src/decode_tune.py --results results/kaggle_v2/rr [--runs floodnet_50shot_s0_stratified,...]

For every run with a val30 cache (preds/<run>_<ds>_val_tune30.npz) it tunes, on val30 only:
  1. per-channel tau (rr.metrics.decode: candidate if p_c >= tau_c, winner = largest p_c / tau_c),
     by coordinate ascent on val30 mIoU over a grid, 3 passes;
     Constraint: val30 drivable precision may not drop below the default decode's (a false
     "clear" sends a truck into water, PLAN §5.1).
  2. dry-building fixes, each tuned on val30 on top of the tuned tau:
     - building vote: one label per predicted building, flooded iff sum(p2)/(sum(p2)+sum(p3)) >= bldg_t;
     - wet context (FloodNet): flooded iff >= f of the ring within d px is decoded wet
       (the model's own p_dry carries no flooded/dry signal at >= 5 shots: per-building AUC ~0.5).
Then it applies default (tau 0.5), tau-only, tau + vote and tau + wet context unchanged to eval100
(preds/<run>_<ds>_test_eval100_fp16.npz) and writes decode/<run>.json (params + all metrics).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402

NAMES = {'floodnet': ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water'],
         'rescuenet': ['road_clear', 'road_blocked', 'bldg_intact', 'bldg_damaged', 'water']}
TAU_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
BLDG_GRID = [None, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CTX_D = [16, 32, 48, 64, 96, 128]
CTX_F = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]


def load(npz, data_root, split):
    z = np.load(npz)
    ids = [str(i) for i in z['ids']]
    gts = [gt_classes(np.asarray(Image.open(os.path.join(data_root, split, 'labels', f'{i}.png')))) for i in ids]
    return z['probs'], gts


def evaluate(probs, gts, ds, **dec):
    cm = np.zeros((6, 6), np.int64)
    for P, g in zip(probs, gts):
        cm += confusion(decode(P if P.dtype == np.float32 else P.astype(np.float32) / 255, **dec), g)
    return summarize(cm, NAMES[ds], WET[ds]), cm


def tune(probs, gts, ds):
    probs = [P.astype(np.float32) / 255 for P in probs]      # convert once: ~200 evaluations follow
    tau = [0.5] * 5
    r0 = evaluate(probs, gts, ds, tau=tau)[0]
    best, p_min = r0['miou'], r0['drivable_precision']
    for _ in range(3):
        changed = False
        for c in range(5):
            for t in TAU_GRID:
                cand = tau[:c] + [t] + tau[c + 1:]
                r = evaluate(probs, gts, ds, tau=cand)[0]
                if r['miou'] > best + 1e-6 and r['drivable_precision'] >= p_min - 1e-9:
                    best, tau, changed = r['miou'], cand, True
        if not changed:
            break
    bt, best_b = None, best
    for t in BLDG_GRID[1:]:
        m = evaluate(probs, gts, ds, tau=tau, bldg_t=t)[0]['miou']
        if m > best_b + 1e-6:
            best_b, bt = m, t
    ctx, best_c = None, best
    if ds == 'floodnet':
        for d in CTX_D:
            for f in CTX_F:
                m = evaluate(probs, gts, ds, tau=tau, ctx=(d, f))[0]['miou']
                if m > best_c + 1e-6:
                    best_c, ctx = m, (d, f)
    return dict(tau=tau, bldg_t=bt, ctx=ctx, val_drivable_precision_min=p_min, val_miou_tau=best,
                val_miou_tau_bldg=best_b, val_miou_tau_ctx=best_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--runs', default=None)
    a = ap.parse_args()
    out_dir = os.path.join(a.results, 'decode')
    os.makedirs(out_dir, exist_ok=True)
    vals = sorted(glob.glob(os.path.join(a.results, 'preds', '*_val_tune30.npz')))
    for v in vals:
        run = os.path.basename(v).split('_floodnet_val')[0].split('_rescuenet_val')[0]
        if a.runs and run not in a.runs.split(','):
            continue
        ds = run.split('_')[0]
        root = os.path.join(a.data, f'{ds}_r768')
        vp, vg = load(v, root, 'val')
        params = tune(vp, vg, ds)
        variants = {'default': dict(tau=0.5), 'tau': dict(tau=params['tau']),
                    'tau_bldg': dict(tau=params['tau'], bldg_t=params['bldg_t']),
                    'tau_ctx': dict(tau=params['tau'], ctx=params['ctx'])}
        # the decode used from here on: tuned tau + whichever building rule helped most on val30
        fix = max([('tau', params['val_miou_tau']), ('tau_bldg', params['val_miou_tau_bldg']),
                   ('tau_ctx', params['val_miou_tau_ctx'])], key=lambda kv: kv[1] + 1e-6 * (kv[0] == 'tau'))[0]
        params['chosen'] = fix
        res = dict(run=run, dataset=ds, params=params, chosen=dict(variants[fix], variant=fix), val30={}, eval100={})
        for k, dec in variants.items():
            res['val30'][k], cm = evaluate(vp, vg, ds, **dec)
        del vp, vg
        e = os.path.join(a.results, 'preds', f'{run}_{ds}_test_eval100_fp16.npz')
        if os.path.exists(e):
            ep, eg = load(e, root, 'test')
            for k, dec in variants.items():
                r, cm = evaluate(ep, eg, ds, **dec)
                res['eval100'][k] = dict(r, confusion=cm.tolist())
            del ep, eg
        json.dump(res, open(os.path.join(out_dir, f'{run}.json'), 'w'), indent=1)
        print(run, json.dumps(params), {s: {k: round(v['miou'], 3) for k, v in res[s].items()} for s in ('val30', 'eval100')},
              flush=True)


if __name__ == '__main__':
    main()
