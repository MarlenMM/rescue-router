"""I20 (Batch 13): pseudo-labels for unlabelled FloodNet train photos (confident pixels only). CPU.

    python src/pseudo_labels.py pick [--n 100]                     # -> supports/floodnet_pseudo100_s0.txt
    python src/pseudo_labels.py make --cache <preds .npz of those photos> --decode <json with tau> --out <dir>

pick: n train photos (seed 0) that are not among the 50 stratified supports and whose frame number is more than 3
away from every val/test photo (FloodNet frames are numbered along the flight, so near-duplicates of evaluation
photos are left out). Their labels are never read: only the photos are used.
make: the model's probabilities -> label PNG (prep_data format: channel id 0-4, 254 other, 255 ignore): a pixel gets
the decoded class (decode params from --decode: tau [+ t_w two-step]) only if that channel's probability is >= --hi
(other: every channel < --lo); everything else is ignored (255) in training. Writes <out>/labels/<id>.png and
<out>/stats.json (kept share per class).
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr.metrics import OTHER_IDX, decode  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402


def pick(a):
    ids = lambda f: [l.strip() for l in open(f) if l.strip()]
    lab_dir = os.path.join(a.data, 'train', 'labels')
    train = sorted((f[:-4] for f in os.listdir(lab_dir) if f.endswith('.png')), key=int)
    sup = set(ids('supports/floodnet_stratified_s0.txt'))
    evals = np.array([int(i) for f in ('splits/floodnet_val_tune30.txt', 'splits/floodnet_test_full.txt') for i in ids(f)]
                     + [int(f[:-4]) for f in os.listdir(os.path.join(a.data, 'val', 'labels')) if f.endswith('.png')])
    pool = [i for i in train if i not in sup and np.abs(evals - int(i)).min() > 3]
    rng = np.random.default_rng(0)
    out = sorted(rng.choice(pool, a.n, replace=False).tolist(), key=int)
    f = f'supports/floodnet_pseudo{a.n}_s0.txt'
    open(f, 'w').write('\n'.join(out) + '\n')
    print(f'{len(pool)} eligible of {len(train)} train photos; wrote {f}')


def make(a):
    z = np.load(a.cache)
    d = json.load(open(a.decode))
    tau = d.get('tau', [0.5] * 5)
    os.makedirs(os.path.join(a.out, 'labels'), exist_ok=True)
    kept = np.zeros(6)
    total = 0
    for i, P in zip(z['ids'], z['probs']):
        P = P.astype(np.float32) / 255
        if d.get('t_w'):
            from postproc_sweep import decode_two_step
            cls = decode_two_step(P, tau, d['t_w'])
        else:
            cls = decode(P, tau=tau)
        conf = np.take_along_axis(P, np.minimum(cls, 4)[None].astype(np.int64), 0)[0]
        lab = np.full(cls.shape, 255, np.uint8)
        ok = (cls < OTHER_IDX) & (conf >= a.hi)
        lab[ok] = cls[ok]
        other = (cls == OTHER_IDX) & (P.max(0) < a.lo)
        lab[other] = 254
        Image.fromarray(lab).save(os.path.join(a.out, 'labels', f'{i}.png'))
        kept += np.array([(lab == c).sum() for c in range(5)] + [(lab == 254).sum()])
        total += lab.size
    stats = dict(n=int(len(z['ids'])), hi=a.hi, lo=a.lo, decode=d, share_labelled=float(kept.sum() / total),
                 share_per_class=dict(zip(['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water', 'other'],
                                          (kept / total).round(4).tolist())))
    json.dump(stats, open(os.path.join(a.out, 'stats.json'), 'w'), indent=1)
    print(json.dumps(stats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['pick', 'make'])
    ap.add_argument('--data', default=os.path.join(RR_DATA, 'floodnet_r768'))
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--cache')
    ap.add_argument('--decode')
    ap.add_argument('--out')
    ap.add_argument('--hi', type=float, default=0.7)
    ap.add_argument('--lo', type=float, default=0.2)
    a = ap.parse_args()
    (pick if a.cmd == 'pick' else make)(a)


if __name__ == '__main__':
    main()
