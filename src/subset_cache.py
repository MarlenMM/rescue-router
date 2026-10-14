"""Cut a prediction cache down to the photos of a smaller split (Phase 2 amendment 1). CPU only.

Phase 2 predicts the FloodNet curve points (1-20 shots) on the whole test set; eval100 is a subset of it, and every
photo is predicted on its own (same model, same supports, tiles of that photo only), so its eval100 cache is the
full-test cache restricted to the eval100 ids, in the split file's order. The confusion matrix (tau .5) and a
predict.py-style metrics JSON are recomputed for the subset.

    python src/subset_cache.py --src results/phase2/rr/preds/<run>_floodnet_test_full_fp16.npz \
        --split_file floodnet_test_eval100 [--split test] [--dataset floodnet]
    -> results/phase2/rr/preds/<run>_floodnet_test_eval100_fp16.npz (+ metrics/<same>.json)
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

NAMES = {'floodnet': ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water'],
         'rescuenet': ['road_clear', 'road_blocked', 'bldg_intact', 'bldg_damaged', 'water']}


def subset(src, split_file, out=None, split='test', dataset='floodnet', data=RR_DATA):
    z = np.load(src)
    pos = {str(i): k for k, i in enumerate(z['ids'])}
    ids = open(os.path.join(os.path.dirname(HERE), 'splits', f'{split_file}.txt')).read().split()
    missing = [i for i in ids if i not in pos]
    assert not missing, f'{len(missing)} photos of {split_file} are not in {src}'
    probs = z['probs'][[pos[i] for i in ids]]
    cm = np.zeros((6, 6), np.int64)
    for i, P in zip(ids, probs):
        lab = np.asarray(Image.open(os.path.join(data, f'{dataset}_r768', split, 'labels', f'{i}.png')))
        cm += confusion(decode(P.astype(np.float32) / 255, 0.5), gt_classes(lab))
    m = re.match(r'(.+?)_(floodnet|rescuenet)_(\w+?)(_es\d+)?(_fp16)?\.npz$', os.path.basename(src))
    assert m, src
    name = os.path.basename(src).replace(f'{m.group(2)}_{m.group(3)}', split_file, 1)
    out = out or os.path.join(os.path.dirname(src), name)
    tmp = out[:-4] + '.tmp.npz'                    # atomic: a stopped run never leaves a truncated cache behind
    np.savez_compressed(tmp, probs=probs, ids=np.array(ids), cm=cm)
    os.replace(tmp, out)
    res = summarize(cm, NAMES[dataset], WET[dataset])
    res.update(split=split_file, n_images=len(ids), tau=0.5, subset_of=os.path.basename(src))
    mdir = os.path.join(os.path.dirname(os.path.dirname(out)), 'metrics')
    if os.path.isdir(mdir):
        json.dump(res, open(os.path.join(mdir, os.path.basename(out)[:-4] + '.json'), 'w'), indent=1)
    return out, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, nargs='+')
    ap.add_argument('--split_file', default='floodnet_test_eval100')
    ap.add_argument('--split', default='test')
    ap.add_argument('--dataset', default='floodnet')
    a = ap.parse_args()
    for src in a.src:
        out, res = subset(src, a.split_file, split=a.split, dataset=a.dataset)
        print(f'{out}: {res["n_images"]} photos, mIoU (tau .5) {res["miou"]:.4f}')


if __name__ == '__main__':
    main()
