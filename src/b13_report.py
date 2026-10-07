"""Batch 13 report from the caches (CPU, ~3 min): the headline switch, recomputed. Used by the notebook (§5.4).

    python src/b13_report.py [--out results/kaggle_v2/report/b13]

1. val30 (where everything was chosen): the new model with tuned τ, + guided filter, + SAM (the stored class maps),
   with building-detection F1 (the guard that removed the clean-up step) and drivable precision.
2. Full test and eval100, paired bootstrap over photos (B 2000, seed 0): published model + published decode vs the new
   model with Chameleon only (guided filter → τ) and with SAM (stored class maps of the same chain).
Old = run floodnet_50shot_s0_stratified_pub (the published caches, kept by src/b13_switch.py); new = the headline run.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b13_final_check import boot_diff  # noqa: E402
from postproc_sweep import guided_filter  # noqa: E402
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402
from sensitivity import detect  # noqa: E402

R = 'results/kaggle_v2/rr'
FN50 = 'floodnet_50shot_s0_stratified'
NAMES = ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water']


def load(name):
    z = np.load(f'{R}/preds/{name}.npz')
    return [str(i) for i in z['ids']], z['probs']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--out', default='results/kaggle_v2/report/b13')
    ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dec = json.load(open(f'{R}/decode/{FN50}.json'))['chosen']
    old_dec = {k: v for k, v in json.load(open(f'{R}/decode/{FN50}_pub.json'))['chosen'].items() if k != 'variant'}
    maps = {sf: dict(zip([str(i) for i in np.load(p)['ids']], np.load(p)['maps'])) for sf, p in dec['classmaps'].items()}
    img = lambda split, i: np.asarray(Image.open(os.path.join(a.data, 'floodnet_r768', split, 'images', f'{i}.jpg')).convert('RGB'))
    lab = lambda split, i: gt_classes(np.asarray(Image.open(os.path.join(a.data, 'floodnet_r768', split, 'labels', f'{i}.png'))))
    res, lines = {}, ['# Batch 13: the new FloodNet 50-shot model, recomputed from the caches', '']

    # 1. val30
    ids, probs = load(f'{FN50}_floodnet_val_tune30')
    rows = {'tuned τ': [], '+ guided filter': [], '+ SAM (final)': []}
    for i, P8 in zip(ids, probs):
        P, g = P8.astype(np.float32) / 255, lab('val', i)
        rows['tuned τ'].append((i, g, decode(P, tau=dec['tau']), None))
        rows['+ guided filter'].append((i, g, decode(guided_filter(img('val', i).mean(2).astype(np.float32) / 255, P, *dec['gf']),
                                                      tau=dec['tau']), None))
        rows['+ SAM (final)'].append((i, g, maps['floodnet_val_tune30'][i], None))
    lines += ['## val30 (the decode chain was chosen here)', '',
              '| decode | mIoU | drivable P | building detection F1 (IoU .5) |', '|---|---|---|---|']
    res['val30'] = {}
    for k, v in rows.items():
        s = summarize(sum(confusion(m, g) for _, g, m, _ in v), NAMES, WET['floodnet'])
        d = detect(v, 'floodnet', 0.5)
        res['val30'][k] = dict(miou=s['miou'], drivable_precision=s['drivable_precision'], detection_f1=d['f1'])
        lines.append(f"| {k} | {s['miou']:.4f} | {s['drivable_precision']:.3f} | {d['f1']:.3f} |")

    # 2. full test / eval100, paired
    ids, probs = load(f'{FN50}_floodnet_test_full_fp16')
    oid, oprobs = load(f'{FN50}_pub_floodnet_test_full_fp16')
    opos = {i: k for k, i in enumerate(oid)}
    e100 = {l.strip() for l in open('splits/floodnet_test_eval100.txt') if l.strip()}
    cms = {'old': [], 'new': [], 'new_sam': []}
    for i, P8 in zip(ids, probs):
        g = lab('test', i)
        P = P8.astype(np.float32) / 255
        cms['old'].append(confusion(decode(oprobs[opos[i]].astype(np.float32) / 255, **old_dec), g))
        cms['new'].append(confusion(decode(guided_filter(img('test', i).mean(2).astype(np.float32) / 255, P, *dec['gf']),
                                           tau=dec['tau']), g))
        cms['new_sam'].append(confusion(maps['floodnet_test_full'][i], g))
    lines += ['', '## Test (paired bootstrap over photos, B %d; old = published model + published decode)' % a.B, '',
              '| split | new decode | mIoU new | mIoU old | Δ [95 % CI] | drivable P new / old |', '|---|---|---|---|---|---|']
    for split, keep in (('full test (448)', ids), ('eval100', [i for i in ids if i in e100])):
        k = [ids.index(i) for i in keep]
        for v, nm in (('new', 'Chameleon only (filter → τ)'), ('new_sam', '+ SAM')):
            m = boot_diff([cms['old'][j] for j in k], [cms[v][j] for j in k], a.B, 0, 'miou')
            p = boot_diff([cms['old'][j] for j in k], [cms[v][j] for j in k], 200, 0, 'drivable_precision')
            res[f'{split}/{v}'] = dict(miou=m, drivable_precision=p)
            lines.append(f"| {split} | {nm} | {m['new']:.3f} [{m['new_ci'][0]:.3f}, {m['new_ci'][1]:.3f}] | {m['old']:.3f} | "
                         f"{m['diff']:+.3f} [{m['diff_ci'][0]:+.3f}, {m['diff_ci'][1]:+.3f}] | {p['new']:.3f} / {p['old']:.3f} |")
    pc = {v: summarize(np.sum(cms[v], 0), NAMES, WET['floodnet']) for v in cms}
    lines += ['', '| full test, per class | ' + ' | '.join(NAMES) + ' | mIoU |', '|---|' + '---|' * 6]
    for v, nm in (('old', 'published'), ('new', 'new, Chameleon only'), ('new_sam', 'new + SAM')):
        lines.append(f'| {nm} | ' + ' | '.join(f"{pc[v][f'iou_{n}']:.3f}" for n in NAMES) + f" | {pc[v]['miou']:.3f} |")
    res['per_class'] = pc
    json.dump(res, open(os.path.join(a.out, 'b13.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'b13.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
