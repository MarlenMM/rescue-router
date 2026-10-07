"""Per-class comparison with the FloodNet paper baselines (PLAN §5.2). CPU only.

    python src/baseline_table.py --results results/kaggle_v2/rr --out results/kaggle_v2/report/baselines.md

Our rows: FloodNet 50-shot on eval100 (τ 0.5 and the tuned decode) and, if the cache exists, on the
full 448-image test (preds/floodnet_50shot_s0_stratified_floodnet_test_full_part{1,2}_fp16.npz).
Baselines: FloodNet paper Table 4 (test, 713x713, 1,445 training images, 10-class training).
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402
from rr.final_decode import decode_photo, photo  # noqa: E402
from viz import load_label  # noqa: E402

CLS = ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water']
PAPER = {  # bldg flooded, bldg non-flooded, road flooded, road non-flooded, water, mIoU (9 classes)
    'ENet': dict(bldg_flooded=6.94, bldg_dry=47.35, road_flooded=12.49, road_clear=48.43, water=48.95, miou9=42.61),
    'DeepLabV3+': dict(bldg_flooded=32.7, bldg_dry=72.8, road_flooded=52.00, road_clear=70.2, water=75.2, miou9=61.53),
    'PSPNet': dict(bldg_flooded=68.93, bldg_dry=89.75, road_flooded=82.16, road_clear=91.18, water=92.00, miou9=79.69),
}


def full_test(res_dir, data, dec):
    parts = [os.path.join(res_dir, 'preds', f'floodnet_50shot_s0_stratified_floodnet_test_full_part{k}_fp16.npz')
             for k in (1, 2)]
    if not all(os.path.exists(p) for p in parts):
        return None
    out = {}
    for name, d in dec.items():
        cm = np.zeros((6, 6), np.int64)
        for p in parts:
            z = np.load(p)
            for i, P in zip(z['ids'], z['probs']):
                cm += confusion(decode_photo(P.astype(np.float32) / 255, d, img=photo(data, 'floodnet', 'test', str(i)),
                                             split_file='floodnet_test_full', photo_id=str(i)),
                                gt_classes(load_label(data, 'floodnet', 'test', str(i))))
        out[name] = summarize(cm, CLS, WET['floodnet'])
        out[name]['n_images'] = sum(len(np.load(p)['ids']) for p in parts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--out', default='results/kaggle_v2/report/baselines.md')
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.results, 'decode', 'floodnet_50shot_s0_stratified.json')))
    dec = {'τ 0.5': dict(tau=0.5), 'tuned': {k: v for k, v in d['chosen'].items() if k != 'variant'}}
    rows = [(f'Chameleon 50-shot, eval100, {k}', {c: 100 * d['eval100'][v][f'iou_{c}'] for c in CLS},
             100 * d['eval100'][v]['miou'], '50')
            for k, v in (('τ 0.5', 'default'), ('tuned', d['chosen']['variant']))]
    ft = full_test(a.results, a.data, dec)
    if ft:
        rows += [(f'Chameleon 50-shot, full test ({ft[k]["n_images"]} img), {k}', {c: 100 * ft[k][f'iou_{c}'] for c in CLS},
                  100 * ft[k]['miou'], '50') for k in dec]
    order = ['bldg_flooded', 'bldg_dry', 'road_flooded', 'road_clear', 'water']
    lines = ['| method | labels | bldg flooded | bldg dry (non-flooded) | road flooded | road clear (non-flooded) | water '
             '| mean of these 5 | paper mIoU (9 cl.) |', '|' + '---|' * 9]
    for name, v in PAPER.items():
        m5 = np.mean([v[c] for c in order])
        lines.append(f'| {name} (paper) | 1,445 | ' + ' | '.join(f'{v[c]:.1f}' for c in order) + f' | {m5:.1f} | {v["miou9"]:.1f} |')
    for name, v, m5, lab in rows:
        lines.append(f'| {name} | {lab} | ' + ' | '.join(f'{v[c]:.1f}' for c in order) + f' | {m5:.1f} | – |')
    note = ('\nCaveats: the paper trains on 10 classes at 713×713 and reports the full 448-image test; we train a '
            '5-channel task (other classes decoded as "other") at 768×576 with 50 labels (3.5 % of 1,445). The 5-class '
            'decode removes confusions with tree/grass/vehicle/pool, which can inflate our IoU a little. eval100 is a '
            'stratified subset (50 % with flooded road/building), not the paper split.\n')
    open(a.out, 'w').write('\n'.join(lines) + '\n' + note)
    if ft:
        json.dump(ft, open(os.path.join(os.path.dirname(a.out), 'full_test_floodnet50.json'), 'w'), indent=1)
    print('\n'.join(lines) + '\n' + note)


if __name__ == '__main__':
    main()
