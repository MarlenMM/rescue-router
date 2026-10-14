"""The final decode chain for any FloodNet run, chosen on val30 exactly as for the submitted model (PLAN §16, Phase 2).
CPU only, from the prediction caches.

    python src/run_decode.py --results results/phase2/rr --run floodnet_50shot_s1_stratified_sup2 [--sam]

Chain (the submitted model's, 8 Oct): guided filter (r 2, eps 1e-3, guided by the grey photo) -> per-class tau ->
SAM "grow" of the building outlines. tau: decode_tune's coordinate ascent on val30 mIoU, first on the raw
probabilities from 0.5, then on the filtered ones from there, under the same fixed drivable-precision floor as the
submitted model (postproc_sweep.P_MIN = the first model's tau-0.5 val30 value, .567). Also the label-free thresholds
of src/prior_decode.py (50 training labels only). Writes <results>/decode/<run>.json (decode_tune format: 'chosen'
is used by every consumer through rr.final_decode) with the val30 numbers and, for every split cache found
(eval100, confirm100, full test), mIoU with tau 0.5 / prior-matched / chosen (no SAM) / chosen + SAM (--sam).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import postproc_sweep as PS  # noqa: E402
from prior_decode import match, support_prior  # noqa: E402
from rr.final_decode import decode_photo  # noqa: E402
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

DS = 'floodnet'
NAMES = PS.NAMES[DS]
GF = (2, 1e-3)
SAM = dict(mode='grow', min_area=50)
SPLITS = (('floodnet_test_eval100', 'test'), ('floodnet_val_confirm100', 'val'), ('floodnet_test_full', 'test'))


def cache(results, run, split_file):
    for suffix in ('', '_fp16'):
        p = os.path.join(results, 'preds', f'{run}_{split_file}{suffix}.npz')
        if os.path.exists(p):
            return p
    return None


def load(path, split):
    z = np.load(path)
    ids = [str(i) for i in z['ids']]
    root = os.path.join(RR_DATA, f'{DS}_r768', split)
    gts = [gt_classes(np.asarray(Image.open(os.path.join(root, 'labels', f'{i}.png')))) for i in ids]
    grey = [np.asarray(Image.open(os.path.join(root, 'images', f'{i}.jpg')).convert('L'), np.float32) / 255 for i in ids]
    return ids, [P.astype(np.float32) / 255 for P in z['probs']], gts, grey


def scores(cms):
    return {k: {kk: float(v) for kk, v in summarize(cm, NAMES, WET[DS]).items()} for k, cm in cms.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--sam', action='store_true', help='also score the chain with SAM (CPU, ~1 s per photo)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--support_seed', type=int, default=None, help='for the prior (default: parsed from the run id)')
    a = ap.parse_args()
    val = cache(a.results, a.run, 'floodnet_val_tune30')
    assert val, f'no val30 cache for {a.run}'
    _, probs, gts, grey = load(val, 'val')
    S = PS.Scorer(probs, gts, a.workers)
    r05 = S('tau', tau=0.5)
    tau_raw, r_raw = PS.tune_tau(S)
    S.close()
    filt = [PS.guided_filter(I, P, *GF) for I, P in zip(grey, probs)]
    S = PS.Scorer(filt, gts, a.workers)
    tau, r_gf = PS.tune_tau(S, start=tau_raw)
    S.close()
    shot = int(a.run.split('_')[1][:-4])
    ss = a.support_seed if a.support_seed is not None else \
        int(a.run.split('_sup')[1].split('_')[0]) if '_sup' in a.run else int(a.run.split('_')[2][1:])
    tau_prior, _ = match([P[:, ::4, ::4] for P in filt], support_prior(shot, ss))   # all of the run's training labels
    chosen = dict(tau=[float(t) for t in tau], gf=list(GF), sam=SAM, variant='final')
    out = dict(run=a.run, dataset=DS, rule='run_decode.py (submitted chain; drivable-precision floor P_MIN = %.4f)' % PS.P_MIN,
               chosen=chosen, prior_tau=[float(t) for t in tau_prior], tau_raw=tau_raw,
               val30=dict(tau05=r05, tau_raw=r_raw, chosen_no_sam=r_gf), splits={})
    for sf, split in SPLITS:
        path = cache(a.results, a.run, sf)
        if not path:
            continue
        ids, probs, gts, grey = load(path, split)
        keys = ['tau0.5', 'prior', 'chosen'] + (['chosen_sam'] if a.sam else [])
        cms = {k: np.zeros((6, 6), np.int64) for k in keys}
        maps = {}
        for i, P, g, I in zip(ids, probs, gts, grey):
            Pf = PS.guided_filter(I, P, *GF)
            cms['tau0.5'] += confusion(decode(P, tau=0.5), g)
            cms['prior'] += confusion(decode(Pf, tau=tau_prior), g)
            cms['chosen'] += confusion(decode(Pf, tau=chosen['tau']), g)
            if a.sam:
                img = np.asarray(Image.open(os.path.join(RR_DATA, f'{DS}_r768', split, 'images', f'{i}.jpg')).convert('RGB'))
                maps[i] = decode_photo(P, chosen, img=img)
                cms['chosen_sam'] += confusion(maps[i], g)
        if maps:                                  # consumers (route_eval, viewer) reuse these instead of re-running SAM
            os.makedirs(os.path.join(a.results, 'classmaps'), exist_ok=True)
            f = os.path.join(a.results, 'classmaps', f'classmaps_{a.run}_{sf}.npz')
            np.savez_compressed(f, ids=np.array(list(maps)), maps=np.stack(list(maps.values())))
            chosen.setdefault('classmaps', {})[sf] = f
        out['splits'][sf] = dict(n=len(ids), **scores(cms))
        print(a.run, sf, {k: round(v['miou'], 4) for k, v in out['splits'][sf].items() if isinstance(v, dict)}, flush=True)
    os.makedirs(os.path.join(a.results, 'decode'), exist_ok=True)
    json.dump(out, open(os.path.join(a.results, 'decode', f'{a.run}.json'), 'w'), indent=1)
    print(f'{a.run}: tau {chosen["tau"]} (raw {tau_raw}), prior tau {[round(t, 3) for t in tau_prior]}; '
          f'val30 tau .5 {r05["miou"]:.4f}, chosen {r_gf["miou"]:.4f}')


if __name__ == '__main__':
    main()
