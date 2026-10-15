"""A decode that needs no labels beyond the 50 training photos (PLAN §16, fix F1). CPU only, from the caches.

The published decode thresholds (and the guided filter, SAM mode, router gates) were chosen on 30 labelled
validation photos, so the system has used 80 labels. Here the per-class thresholds come from the 50 training labels
only: choose tau so that, on unlabelled photos (val30 photos with their labels ignored), each class takes the same
share of the decoded pixels as it does in the 50 training labels ("prior matching"; a coordinate search on log tau
with the real decode rule, which couples the classes).

What still used val30: the 1,000-step schedule and the step-750/1000 weight average (Batch 13) and the guided-filter
radius. Reported: full-test mIoU of prior-matched tau vs tau .5 and the val30-tuned tau, with the same filter, no SAM.

    python src/prior_decode.py [--results results/kaggle_v2/rr] [--out results/kaggle_v2/report/prior_decode]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from calibrate import RUN, photos  # noqa: E402
from decode_tune import NAMES  # noqa: E402
from rr.metrics import OTHER_IDX, WET, confusion, decode, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

DS = 'floodnet'
REPO = os.path.dirname(HERE)


def support_prior(n_shot=50, seed=0, ds=DS):
    """share of each class (and 'other') among the labelled pixels of the training photos (RescueNet: its supports
    come from its val split)"""
    ids = open(os.path.join(REPO, 'supports', f'{ds}_stratified_s{seed}.txt')).read().split()[:n_shot]
    cnt = np.zeros(OTHER_IDX + 1)
    for i in ids:
        split = 'train' if ds == 'floodnet' else 'val'
        g = gt_classes(np.asarray(Image.open(os.path.join(RR_DATA, f'{ds}_r768', split, 'labels', f'{i}.png'))))
        cnt += np.bincount(g[g != 255].ravel(), minlength=OTHER_IDX + 1)[:OTHER_IDX + 1]
    return cnt / cnt.sum()


def shares(probs, tau):
    cnt = np.zeros(OTHER_IDX + 1)
    for P in probs:
        cnt += np.bincount(decode(P, tau=tau).ravel(), minlength=OTHER_IDX + 1)
    return cnt / cnt.sum()


def match(probs, prior, iters=150, eta=0.5):
    """per-class tau whose decoded shares match the prior (5 classes + 'other'): simultaneous multiplicative updates
    tau_c <- tau_c * (share_c / prior_c)^(eta / sqrt(t)) (dual ascent on the argmax 'prices'; a class with too many pixels raises its threshold; 'other' follows), then
    the best iterate by sum |log(share / prior)| over all 6 shares. (A coordinate search stalls: classes trade pixels.)"""
    tau = np.full(len(prior) - 1, 0.5, np.float64)
    err = lambda s: float(np.abs(np.log((s + 1e-6) / (prior + 1e-6))).sum())   # noqa: E731
    best = (np.inf, tau)
    for t in range(1, iters + 1):
        s = shares(probs, tau.astype(np.float32))
        e = err(s)
        if e < best[0]:
            best = (e, tau.copy())
        tau = np.clip(tau * ((s[:-1] + 1e-6) / (prior[:-1] + 1e-6)) ** (eta / np.sqrt(t)), 0.02, 0.98)
    return best[1].astype(np.float32), best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--out', default='results/kaggle_v2/report/prior_decode')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = NAMES[DS]
    prior = support_prior()
    val = [P[:, ::4, ::4] for _, P, _ in photos(a.results, 'floodnet_val_tune30', 'val')]  # labels unused; shares only
    tau_p, loss = match(val, prior)
    tuned = np.array(json.load(open(os.path.join(a.results, 'decode', f'{RUN}.json')))['chosen']['tau'], np.float32)
    cms = {k: np.zeros((6, 6), np.int64) for k in ('tau0.5', 'prior_matched', 'val30_tuned')}
    pred_share = np.zeros(OTHER_IDX + 1)
    true_share = np.zeros(OTHER_IDX + 1)
    for _, P, g in photos(a.results, 'floodnet_test_full', 'test'):
        for k, t in (('tau0.5', 0.5), ('prior_matched', tau_p), ('val30_tuned', tuned)):
            cms[k] += confusion(decode(P, tau=t), g)
        pred_share += np.bincount(decode(P, tau=tau_p).ravel(), minlength=OTHER_IDX + 1)
        true_share += np.bincount(g[g != 255].ravel(), minlength=OTHER_IDX + 1)[:OTHER_IDX + 1]
    res = {k: summarize(cm, names, WET[DS]) for k, cm in cms.items()}
    out = dict(run=RUN, prior=dict(zip(names + ['other'], prior.tolist())), tau_prior_matched=dict(zip(names, tau_p.tolist())),
               tau_val30_tuned=dict(zip(names, tuned.tolist())), match_loss=loss,
               test_share_true=dict(zip(names + ['other'], (true_share / true_share.sum()).tolist())),
               test_share_pred=dict(zip(names + ['other'], (pred_share / pred_share.sum()).tolist())),
               full_test={k: dict(miou=r['miou'], **{c: r[f'iou_{c}'] for c in names},
                                  drivable_precision=r['drivable_precision']) for k, r in res.items()})
    json.dump(out, open(os.path.join(a.out, 'prior_decode.json'), 'w'), indent=1)
    md = ['# Decode thresholds from the 50 training labels only (prior matching)', '',
          'Per-class τ chosen so the decoded class shares on the 30 val photos (labels ignored) match the class shares '
          'of the 50 training labels. Same model and guided filter as the published decode; no SAM. Full test, 448 photos.', '',
          '| | ' + ' | '.join(names) + ' | other |', '|---|' + '---|' * (len(names) + 1),
          '| share in the 50 training labels | ' + ' | '.join(f'{x:.3f}' for x in prior) + ' |',
          '| share in the test labels | ' + ' | '.join(f'{x:.3f}' for x in out['test_share_true'].values()) + ' |',
          '| decoded share, prior-matched τ | ' + ' | '.join(f'{x:.3f}' for x in out['test_share_pred'].values()) + ' |',
          '| prior-matched τ | ' + ' | '.join(f'{x:.2f}' for x in tau_p) + ' | |',
          '| val30-tuned τ | ' + ' | '.join(f'{x:.2f}' for x in tuned) + ' | |', '',
          '| decode (full test) | ' + ' | '.join(names) + ' | mIoU | drivable precision |', '|---|' + '---|' * (len(names) + 2)]
    for k, r in out['full_test'].items():
        md.append(f'| {k} | ' + ' | '.join(f'{r[c]:.3f}' for c in names) + f' | **{r["miou"]:.3f}** | {r["drivable_precision"]:.3f} |')
    open(os.path.join(a.out, 'prior_decode.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
