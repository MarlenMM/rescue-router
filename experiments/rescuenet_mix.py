"""Exploratory (PLAN §16.7, no choice depends on it): Chameleon RescueNet 20 shots + the 20-photo DeepLabV3 baseline,
probability average P = w C + (1 - w) B, per-class thresholds tuned on RescueNet val30 for each w (decode_tune's
coordinate ascent, drivable-precision floor min(P_MIN, the mix's own tau .5 value)); eval100 mIoU, paired bootstrap
(2,000) against Chameleon's published decode (tau + building vote, .421). CPU, ~25 min.

    python experiments/rescuenet_mix.py [--out results/baselines/report/rescuenet_mix.json]

16 Oct: w 0 .359, .25 .380, .5 .402, .75 .430 (+.009 [-.003, +.021]; best val30), 1 .407: on RescueNet the baseline
adds at most a non-significant +.009 (on FloodNet every mix with w < 1 beat Chameleon, src/ensemble.py).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
import postproc_sweep as PS  # noqa: E402
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

NAMES = PS.NAMES['rescuenet']
CHAM = 'results/kaggle_v2/rr/preds/rescuenet_20shot_s0_stratified_rescuenet_{}'
BASE = 'results/baselines/preds/baseline_deeplabv3_r50_rescuenet_20shot_s0_rescuenet_{}.npz'
SPLITS = (('val_tune30', 'val'), ('test_eval100', 'test'))


def load(f):
    z = np.load(os.path.join(REPO, f))
    return {str(i): p for i, p in zip(z['ids'], z['probs'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'results/baselines/report/rescuenet_mix.json'))
    a = ap.parse_args()
    C = {s: load(CHAM.format(s) + ('_fp16.npz' if 'test' in s else '.npz')) for s, _ in SPLITS}
    B = {s: load(BASE.format(s)) for s, _ in SPLITS}
    G = {s: {i: gt_classes(np.asarray(Image.open(f'{RR_DATA}/rescuenet_r768/{sp}/labels/{i}.png'))) for i in C[s]}
         for s, sp in SPLITS}

    def cms(s, w, **dec):
        return {i: confusion(decode(w * C[s][i].astype(np.float32) / 255 + (1 - w) * B[s][i].astype(np.float32) / 255,
                                    **dec), G[s][i]) for i in C[s]}

    def miou(cm):
        return summarize(cm, NAMES, WET['rescuenet'])['miou']

    pub = json.load(open(os.path.join(REPO, 'results/kaggle_v2/rr/decode/rescuenet_20shot_s0_stratified.json')))['chosen']
    ref = cms('test_eval100', 1.0, tau=pub['tau'], bldg_t=pub.get('bldg_t'))
    ids = sorted(ref)
    rng = np.random.default_rng(0)
    resamples = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
    out = dict(chameleon_published=miou(sum(ref.values())), mixes={})
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        S = lambda kind, tau, **_: summarize(sum(cms('val_tune30', w, tau=tau).values()), NAMES, WET['rescuenet'])  # noqa: E731
        tau, best = PS.tune_tau(S, floor=min(PS.P_MIN, S('tau', [0.5] * 5)['drivable_precision']))
        ev = cms('test_eval100', w, tau=tau)
        d = [miou(sum(ev[ids[j]] for j in k)) - miou(sum(ref[ids[j]] for j in k)) for k in resamples]
        m = miou(sum(ev.values()))
        out['mixes'][str(w)] = dict(tau=tau, val30=best['miou'], eval100=m, diff=m - out['chameleon_published'],
                                    lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
        print(f"w {w}: val30 {best['miou']:.4f} eval100 {m:.4f} vs Chameleon {m - out['chameleon_published']:+.4f} "
              f"[{out['mixes'][str(w)]['lo']:+.3f}, {out['mixes'][str(w)]['hi']:+.3f}]", flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
