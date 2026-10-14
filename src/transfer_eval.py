"""Cross-disaster evaluation on shared super-classes (PLAN §3.4). CPU only, from the caches.

    python src/transfer_eval.py [--out results/kaggle_v2/report/transfer.md]

Rows on RescueNet eval100, super-classes road (channels 0+1), building (2+3), water (4):
  * zero-label transfer: FloodNet 50-shot model with its own FloodNet supports
    (preds/floodnet_50shot_s0_stratified_rescuenet_test_eval100_fp16.npz), FloodNet decodes;
  * re-adaptation: RescueNet 20- and 50-shot models (their own caches and decodes);
  * the same-budget baselines (results/baselines, argmax decode), where their caches exist.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr.metrics import gt_classes  # noqa: E402
from rr.final_decode import decode_photo, photo  # noqa: E402
from viz import load_label  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

SUPER = {'road': (0, 1), 'building': (2, 3), 'water': (4,)}


def super_map(cls):
    out = np.full(cls.shape, 3, np.uint8)                   # 3 = other
    for k, chans in enumerate(SUPER.values()):
        out[np.isin(cls, chans)] = k
    out[cls == 255] = 255
    return out


def score(npz, dec, data):
    z = np.load(npz)
    cm = np.zeros((4, 4), np.int64)
    for i, P in zip(z['ids'], z['probs']):
        g = super_map(gt_classes(load_label(data, 'rescuenet', 'test', str(i))))
        p = super_map(decode_photo(P.astype(np.float32) / 255, dec, img=photo(data, 'rescuenet', 'test', str(i)),
                                   split_file='rescuenet_test_eval100', photo_id=str(i)))
        v = g != 255
        cm += np.bincount(g[v].astype(np.int64) * 4 + p[v], minlength=16).reshape(4, 4)
    tp = np.diag(cm).astype(float)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    return {n: float(iou[k]) for k, n in enumerate(SUPER)} | {'mean': float(iou[:3].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--data', default=RR_DATA)
    ap.add_argument('--out', default='results/kaggle_v2/report/transfer.md')
    a = ap.parse_args()
    dec_of = lambda run: {k: v for k, v in json.load(open(os.path.join(a.results, 'decode', f'{run}.json')))['chosen'].items()
                          if k != 'variant'}
    rows = []
    fn = os.path.join(a.results, 'preds', 'floodnet_50shot_s0_stratified_rescuenet_test_eval100_fp16.npz')
    if os.path.exists(fn):
        rows.append(('zero-label transfer: FloodNet 50-shot (0 RescueNet labels), τ 0.5', score(fn, dict(tau=0.5), a.data)))
        rows.append(('zero-label transfer: FloodNet 50-shot, FloodNet-tuned decode',
                     score(fn, dec_of('floodnet_50shot_s0_stratified'), a.data)))
    for k in (20, 50):
        run = f'rescuenet_{k}shot_s0_stratified'
        npz = os.path.join(a.results, 'preds', f'{run}_rescuenet_test_eval100_fp16.npz')
        rows.append((f're-adapted: RescueNet {k}-shot, τ 0.5', score(npz, dict(tau=0.5), a.data)))
        rows.append((f're-adapted: RescueNet {k}-shot, tuned decode', score(npz, dec_of(run), a.data)))
    # same-budget baselines (src/baseline_seg.py, argmax decode; 14 Oct): zero-label transfer of the FloodNet 50-shot
    # baseline (--query_dataset rescuenet) and the RescueNet-trained ones (COCO start, or started from the FloodNet model)
    B = os.path.join(os.path.dirname(os.path.dirname(a.results)), 'baselines', 'preds')
    for label, run in (('zero-label transfer: FloodNet 50-shot **baseline** (DeepLabV3), argmax', 'baseline_deeplabv3_r50_50shot_s0'),
                       ('RescueNet 20-shot **baseline**, argmax', 'baseline_deeplabv3_r50_rescuenet_20shot_s0'),
                       ('RescueNet 50-shot **baseline**, argmax', 'baseline_deeplabv3_r50_rescuenet_50shot_s0'),
                       ('RescueNet 20-shot **baseline** started from the FloodNet 50-shot baseline, argmax',
                        'baseline_deeplabv3_r50_rescuenet_20shot_s0_ft50')):
        npz = os.path.join(B, f'{run}_rescuenet_test_eval100.npz')
        if os.path.exists(npz):
            rows.append((label, score(npz, {'argmax': True}, a.data)))
    lines = ['| model (RescueNet eval100) | road IoU | building IoU | water IoU | mean |', '|---|---|---|---|---|']
    lines += [f'| {n} | ' + ' | '.join(f'{r[c]:.3f}' for c in ('road', 'building', 'water', 'mean')) + ' |' for n, r in rows]
    open(a.out, 'w').write('\n'.join(lines) + '\n')
    json.dump(dict(rows), open(a.out.replace('.md', '.json'), 'w'), indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
