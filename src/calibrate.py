"""Per-class calibration of the final FloodNet 50-shot model (PLAN §16, B3). CPU only, from the caches.

At a plain threshold 0.5 the final model scores only .278 on the full test; its val30-tuned per-class thresholds
(tau .25/.25/.45/.10/.15) are part of the model. This script asks how much of that tuning is plain calibration:
  1. Platt scaling per channel, fitted on val30 pixels: p' = sigmoid(a * logit(p) + b), p = the probabilities after
     the decode's guided filter (the same input the tuned thresholds see);
  2. expected calibration error (ECE, 15 bins) per channel before and after, on val30 (fit set) and the full test;
  3. full-test mIoU of the plain 0.5 rule on the calibrated probabilities (no SAM) vs the tuned thresholds (no SAM);
  4. the raw threshold at which each calibrated channel reaches 0.5, next to the tuned threshold.

    python src/calibrate.py [--results results/kaggle_v2/rr] [--out results/kaggle_v2/report/calibration]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decode_tune import NAMES  # noqa: E402
from postproc_sweep import guided_filter  # noqa: E402
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

RUN, DS = 'floodnet_50shot_s0_stratified', 'floodnet'
BINS = 15
EPS = 1e-4


def photos(results, split_file, split):
    """(id, filtered probs float32 (C, H, W), gt classes) per photo, in cache order"""
    name = f'{RUN}_{split_file}' + ('' if split_file.endswith('tune30') else '_fp16')
    z = np.load(os.path.join(results, 'preds', f'{name}.npz'))
    dec = json.load(open(os.path.join(results, 'decode', f'{RUN}.json')))['chosen']
    root = os.path.join(RR_DATA, f'{DS}_r768', split)
    for i, P in zip(z['ids'], z['probs']):
        i = str(i)
        img = np.asarray(Image.open(os.path.join(root, 'images', f'{i}.jpg')).convert('RGB'))
        P = guided_filter(img.mean(2).astype(np.float32) / 255, P.astype(np.float32) / 255, *dec['gf'])
        g = gt_classes(np.asarray(Image.open(os.path.join(root, 'labels', f'{i}.png'))))
        yield i, np.clip(P, 0, 1), g


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_platt(x, y):
    """a, b minimising the log loss of sigmoid(a x + b) (x = logit of the raw probability)"""
    def nll(w):
        z = w[0] * x + w[1]
        return float(np.mean(np.logaddexp(0, z) - y * z))

    def grad(w):
        r = sigmoid(w[0] * x + w[1]) - y
        return np.array([np.mean(r * x), np.mean(r)])
    return minimize(nll, np.array([1.0, 0.0]), jac=grad, method='L-BFGS-B').x


class Reliability:
    """streaming per-channel reliability bins: count, sum of predicted p, sum of outcomes"""
    def __init__(self, C):
        self.n, self.p, self.y = (np.zeros((C, BINS)) for _ in range(3))

    def add(self, P, g, valid):
        for c in range(P.shape[0]):
            p, y = P[c][valid], (g[valid] == c)
            b = np.minimum((p * BINS).astype(int), BINS - 1)
            self.n[c] += np.bincount(b, minlength=BINS)
            self.p[c] += np.bincount(b, p, minlength=BINS)
            self.y[c] += np.bincount(b, y, minlength=BINS)

    def ece(self):
        tot = self.n.sum(1, keepdims=True)
        return (np.abs(self.p - self.y).sum(1) / tot[:, 0]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--out', default='results/kaggle_v2/report/calibration')
    ap.add_argument('--pixels', type=int, default=400_000, help='pixels per channel sampled for the fit')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dec = json.load(open(os.path.join(a.results, 'decode', f'{RUN}.json')))['chosen']
    tau = np.array(dec['tau'], np.float32)
    names = NAMES[DS]
    C = len(names)

    # 1. fit on val30 (all valid pixels, then a seeded subsample per channel)
    val = list(photos(a.results, 'floodnet_val_tune30', 'val'))
    X = np.concatenate([P.reshape(C, -1)[:, (g != 255).ravel()] for _, P, g in val], 1)
    Y = np.concatenate([g[g != 255] for _, _, g in val])
    rng = np.random.default_rng(0)
    pick = rng.choice(X.shape[1], min(a.pixels, X.shape[1]), replace=False)
    platt = np.array([fit_platt(logit(X[c, pick]), (Y[pick] == c).astype(np.float64)) for c in range(C)])
    equiv_tau = sigmoid(-platt[:, 1] / platt[:, 0])          # raw p where the calibrated p reaches 0.5

    def cal(P):
        return sigmoid(platt[:, 0, None, None] * logit(P) + platt[:, 1, None, None]).astype(np.float32)

    # 2./3. reliability and decodes on val30 and the full test
    out = dict(run=RUN, decode=dec, platt={n: dict(a=float(p[0]), b=float(p[1])) for n, p in zip(names, platt)},
               tau_tuned=dict(zip(names, tau.tolist())), tau_equivalent=dict(zip(names, equiv_tau.tolist())), splits={})
    for split_file, split, it in (('floodnet_val_tune30', 'val', val),
                                  ('floodnet_test_full', 'test', photos(a.results, 'floodnet_test_full', 'test'))):
        raw, calr = Reliability(C), Reliability(C)
        cms = {k: np.zeros((6, 6), np.int64) for k in ('tau0.5_raw', 'tuned_raw', 'tau0.5_calibrated')}
        n = 0
        for _, P, g in it:
            valid = g != 255
            Pc = cal(P)
            raw.add(P, g, valid)
            calr.add(Pc, g, valid)
            cms['tau0.5_raw'] += confusion(decode(P, tau=0.5), g)
            cms['tuned_raw'] += confusion(decode(P, tau=tau), g)
            cms['tau0.5_calibrated'] += confusion(decode(Pc, tau=0.5), g)
            n += 1
        out['splits'][split_file] = dict(
            n=n, ece_raw=dict(zip(names, raw.ece())), ece_calibrated=dict(zip(names, calr.ece())),
            miou={k: summarize(cm, names, WET[DS])['miou'] for k, cm in cms.items()},
            iou={k: {c: summarize(cm, names, WET[DS])[f'iou_{c}'] for c in names} for k, cm in cms.items()})
        print(split_file, out['splits'][split_file]['miou'], flush=True)

    json.dump(out, open(os.path.join(a.out, 'calibration.json'), 'w'), indent=1)
    md = ['# Calibration of the final FloodNet 50-shot model (guided-filtered probabilities, no SAM)', '',
          'Platt scaling per channel fitted on val30 pixels; the full test is never used for the fit.', '',
          '| channel | Platt a | Platt b | tuned τ | τ where calibrated p = 0.5 |', '|---|---|---|---|---|']
    md += [f'| {c} | {platt[k, 0]:.2f} | {platt[k, 1]:+.2f} | {tau[k]:.2f} | {equiv_tau[k]:.3f} |'
           for k, c in enumerate(names)]
    for sf, s in out['splits'].items():
        md += ['', f'**{sf}** ({s["n"]} photos)', '', '| | ' + ' | '.join(names) + ' | mean |', '|---|' + '---|' * (C + 1)]
        for k in ('ece_raw', 'ece_calibrated'):
            v = list(s[k].values())
            md.append(f'| ECE {k[4:]} | ' + ' | '.join(f'{x:.3f}' for x in v) + f' | {np.mean(v):.3f} |')
        for k, m in s['miou'].items():
            md.append(f'| IoU, {k} | ' + ' | '.join(f'{s["iou"][k][c]:.3f}' for c in names) + f' | mIoU {m:.3f} |')
    open(os.path.join(a.out, 'calibration.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
