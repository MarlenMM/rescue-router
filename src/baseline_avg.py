"""Average the probability caches of several baseline runs into one run (PROTOCOL amendment 3: the seed ensemble).

    python src/baseline_avg.py --runs baseline_deeplabv3_r50_50shot_s0,baseline_deeplabv3_r50_50shot_s1_sup0,\
baseline_deeplabv3_r50_50shot_s2_sup0 --out_run baseline_deeplabv3_r50_50shot_s0_ens3 [--suffix _tta]

For every split file that all runs have predicted (results/baselines/preds/<run><suffix>_<split>.npz, same photos),
the 5 class channels are averaged (the softmax's "other" = 1 - sum is averaged with them, as it is linear) and
written as <out_run><suffix>_<split>.npz in the same uint8 format, with the argmax confusion and a metrics JSON, so
baseline_report.py, tiers.py, router_sweep.py and ensemble.py read it like any other run. No labels are used.
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
from rr.metrics import WET, confusion, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

REPO = os.path.dirname(HERE)
NAMES = {'floodnet': ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water'],
         'rescuenet': ['road_clear', 'road_blocked', 'bldg_intact', 'bldg_damaged', 'water']}


def average(paths):
    """mean of uint8 probability caches (same photo ids, same order) -> (uint8 probs, ids)"""
    zs = [np.load(p) for p in paths]
    ids = zs[0]['ids']
    for z, p in zip(zs[1:], paths[1:]):
        assert list(z['ids']) == list(ids), f'{p}: other photos or order than {paths[0]}'
    mean = np.mean([z['probs'].astype(np.float32) for z in zs], axis=0)
    return np.round(mean).astype(np.uint8), ids


def splits_of(files, prefix, repo=REPO):
    """the split files of a run's caches: the part after '<run><suffix>_' that is a split in splits/ (17 Oct: the glob of
    run ..._20shot_s0 also matches ..._s0_tta_<split>, ..._s0_ft50_<split>, ...: 'tta_rescuenet_test_full' is no split)"""
    names = [os.path.basename(f)[len(prefix):-4] for f in files]
    return [sf for sf in names if os.path.exists(os.path.join(repo, 'splits', f'{sf}.txt'))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='comma list of baseline runs (results/baselines/preds/<run>_...)')
    ap.add_argument('--out_run', required=True)
    ap.add_argument('--suffix', default='', help="cache suffix of the members and the output, e.g. '_tta'")
    ap.add_argument('--baselines', default='results/baselines')
    a = ap.parse_args()
    runs = a.runs.split(',')
    pre = [f'{r}{a.suffix}_' for r in runs]
    first = sorted(glob.glob(os.path.join(a.baselines, 'preds', f'{pre[0]}*.npz')))
    splits = splits_of(first, pre[0])
    done = []
    for sf in splits:
        paths = [os.path.join(a.baselines, 'preds', f'{p}{sf}.npz') for p in pre]
        if not all(os.path.exists(p) for p in paths):
            print(f'skip {sf}: not predicted by every run')
            continue
        P, ids = average(paths)
        ds = 'rescuenet' if sf.startswith('rescuenet') else 'floodnet'
        root = os.path.join(RR_DATA, f'{ds}_r768', 'val' if '_val_' in sf else 'test', 'labels')
        cm = np.zeros((6, 6), np.int64)
        for i, p in zip(ids, P):
            full = np.concatenate([p, 255 - p.astype(np.int32).sum(0, keepdims=True)])     # + other, uint8 scale
            cm += confusion(full.argmax(0).astype(np.uint8), gt_classes(np.asarray(Image.open(os.path.join(root, f'{i}.png')))))
        name = f'{a.out_run}{a.suffix}_{sf}'
        np.savez_compressed(os.path.join(a.baselines, 'preds', f'{name}.npz'), probs=P, ids=ids, cm=cm)
        res = summarize(cm, NAMES[ds], WET[ds])
        res.update(n_images=len(ids), members=[f'{r}{a.suffix}' for r in runs])
        json.dump(res, open(os.path.join(a.baselines, 'metrics', f'{name}.json'), 'w'), indent=1)
        print(f'[{name}] mIoU (argmax) {res["miou"]:.3f} from {len(runs)} runs', flush=True)
        done.append(sf)
    log = os.path.join(a.baselines, 'logs', f'{a.out_run}.json')
    old = json.load(open(log)) if os.path.exists(log) else {}
    old.update(run=a.out_run, members=runs, kind='mean of the members\' softmax (PROTOCOL amendment 3)',
               **{f'splits{a.suffix or "_plain"}': done})
    json.dump(old, open(log, 'w'), indent=1)
    if not done:
        sys.exit('no split predicted by every run')


if __name__ == '__main__':
    main()
