"""Near-duplicate check (Batch 3): do test photos show the same ground as the labelled support photos?
CPU only. FloodNet's official split puts neighbouring frames of one flight on both sides (e.g. test 7085 /
support 7081), so a few test scenes may have been "seen" through the supports.

    python src/dupes.py [--out results/kaggle_v2/report/dupes]

For each test photo: the 3 supports with the highest coarse correlation (64x48 grey, normalised) are candidates;
each candidate is verified with ORB features + RANSAC homography on 768x576 grey images. A pair overlaps if the
fit has >= MIN_INLIERS inliers and the support, warped into the test photo, covers >= MIN_COVER of it.
Pairs with 10-24 inliers (and cover >= 0.15) were checked by eye on 6 Oct (MANUAL): the FloodNet ones show the
same field, pond and canal; the RescueNet ones are different streets. `--recount` re-applies the rules to an
existing dupes.json without recomputing features.
Writes dupes.json (all candidates with inliers and cover; the overlapping test ids per run).
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

RUNS = {'floodnet': ('floodnet_test_full', 'train', 50), 'rescuenet': ('rescuenet_test_eval100', 'val', 20)}
MIN_INLIERS, MIN_COVER, TOPK = 25, 0.2, 3
MANUAL = {('floodnet', '7926', '7925'): True, ('floodnet', '8945', '8206'): True, ('floodnet', '8972', '8967'): True,
          ('rescuenet', '13795', '12003'): False, ('rescuenet', '14882', '14852'): False}


def overlaps(ds, c):
    m = MANUAL.get((ds, c['test'], c['support']))
    return m if m is not None else (c['inliers'] >= MIN_INLIERS and c['cover'] >= MIN_COVER)


def count(ds, cands, shot):
    hit = [c for c in cands if overlaps(ds, c)]
    return {f'{k}shot': sorted({c['test'] for c in hit if c['support_rank'] < k}) for k in sorted({shot, 20, 50})}


def grey(path, size=None):
    im = Image.open(path).convert('L')
    return np.asarray(im.resize(size, Image.BOX) if size else im, np.float32) / 255


def coarse(path):
    a = grey(path, (64, 48))
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a.ravel() / a.size ** 0.5


def orb(path):
    o = ORB(n_keypoints=800, fast_threshold=0.05)
    o.detect_and_extract(grey(path))
    return o.keypoints, o.descriptors


def verify(fa, fb, shape=(576, 768)):
    """inliers of a homography from support b into test a, and the share of a covered by the warped b"""
    (ka, da), (kb, db) = fa, fb
    m = match_descriptors(da, db, cross_check=True, max_ratio=0.8)
    if len(m) < 8:
        return 0, 0.0
    src, dst = kb[m[:, 1]][:, ::-1], ka[m[:, 0]][:, ::-1]            # (x, y)
    model, inl = ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=3, max_trials=1000,
                        rng=0)
    if model is None or inl is None or inl.sum() < 4:
        return 0, 0.0
    h, w = shape
    ys, xs = np.mgrid[0:h:16, 0:w:16]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    back = model.inverse(pts)                                        # test pixel -> support pixel
    cover = float(((back[:, 0] >= 0) & (back[:, 0] < w) & (back[:, 1] >= 0) & (back[:, 1] < h)).mean())
    return int(inl.sum()), cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--repo', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--out', default='results/kaggle_v2/report/dupes')
    ap.add_argument('--recount', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.recount:
        out = json.load(open(os.path.join(a.out, 'dupes.json')))
        for ds, r in out['runs'].items():
            r['overlapping'] = count(ds, r['candidates'], RUNS[ds][2])
            print(ds, {k: len(v) for k, v in r['overlapping'].items()}, 'of', r['n_test'], r['overlapping'])
        out['manual'] = [dict(dataset=d, test=t, support=s, overlap=v) for (d, t, s), v in MANUAL.items()]
        json.dump(out, open(os.path.join(a.out, 'dupes.json'), 'w'), indent=1)
        return
    out = dict(min_inliers=MIN_INLIERS, min_cover=MIN_COVER, topk=TOPK, runs={})
    for ds, (split_file, sup_split, shot) in RUNS.items():
        root = os.path.join(a.data, f'{ds}_r768')
        sup = [l.strip() for l in open(os.path.join(a.repo, 'supports', f'{ds}_stratified_s0.txt'))][:50]
        test = [l.strip() for l in open(os.path.join(a.repo, 'splits', f'{split_file}.txt')) if l.strip()]
        sp = {i: os.path.join(root, sup_split, 'images', f'{i}.jpg') for i in sup}
        tp = {i: os.path.join(root, 'test', 'images', f'{i}.jpg') for i in test}
        S = np.stack([coarse(sp[i]) for i in sup])
        fs = {i: orb(sp[i]) for i in sup}
        cands = []
        for t in test:
            c = S @ coarse(tp[t])
            ft = None
            for k in np.argsort(-c)[:TOPK]:
                ft = ft or orb(tp[t])
                n, cover = verify(ft, fs[sup[k]])
                cands.append(dict(test=t, support=sup[k], support_rank=sup.index(sup[k]), corr=round(float(c[k]), 3),
                                  inliers=n, cover=round(cover, 3)))
        hit = [c for c in cands if overlaps(ds, c)]
        res = count(ds, cands, shot)                                 # per run: supports actually used
        out['runs'][ds] = dict(split=split_file, n_test=len(test), overlapping=res, candidates=cands)
        print(ds, {k: len(v) for k, v in res.items()}, 'of', len(test), flush=True)
        for c in sorted(hit, key=lambda c: -c['inliers'])[:12]:
            print('   ', c)
    out['manual'] = [dict(dataset=d, test=t, support=s, overlap=v) for (d, t, s), v in MANUAL.items()]
    json.dump(out, open(os.path.join(a.out, 'dupes.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
