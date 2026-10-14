"""Batch 13.5: the headline rule. Paired full-test comparison of the Batch 13 setup with the published model. CPU.

    python src/b13_final_check.py --new_full <preds part1>,<preds part2> --decode results/b13/r2/decode_wavg.json \
        [--sam ~/rr_data/weights/sam_vit_b_01ec64.pth] [--out results/b13/final]

Old = published FloodNet 50-shot full-test cache + its published decode (tuned tau). New = the Batch 13 checkpoint's
full-test caches + its own val30-chosen decode (guided filter -> tau -> clean-up), with and without SAM grow (I31).
Images are resampled with replacement (B 2000, seed 0, the same resamples for every variant: paired); mIoU from the
summed confusion. Rule (PLAN §15, fixed before any result): switch only if the full-test difference new - old has a
95 % CI lower bound > 0. eval100 (a subset of the full test) is reported as well. Also writes the decoded class maps
of the new setup (classmaps_<split>.npz) for the downstream rebuild. Writes final.json / final.md.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from postproc_sweep import cleanup, guided_filter  # noqa: E402
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA, SAM_WEIGHTS  # noqa: E402

NAMES = ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water']
_PUB = '_pub' if os.path.exists('results/kaggle_v2/rr/decode/floodnet_50shot_s0_stratified_pub.json') else ''  # after b13_switch
OLD_CACHE = f'results/kaggle_v2/rr/preds/floodnet_50shot_s0_stratified{_PUB}_floodnet_test_full_fp16.npz'
OLD_DECODE = f'results/kaggle_v2/rr/decode/floodnet_50shot_s0_stratified{_PUB}.json'


def load_parts(paths):
    ps, ids = [], []
    for p in paths:
        z = np.load(p)
        ps.append(z['probs'])
        ids += [str(i) for i in z['ids']]
    return np.concatenate(ps), ids


def new_decode(P, img, d):
    Q = guided_filter(img.mean(2).astype(np.float32) / 255, P, *d['gf']) if d['gf'] else P
    return cleanup(decode(Q, tau=d['tau']), d['k'], d['r'])


def boot_diff(cms_a, cms_b, B=2000, seed=0, metric='miou'):
    rng = np.random.default_rng(seed)
    n = len(cms_a)
    A, Bm = np.stack(cms_a), np.stack(cms_b)
    f = lambda cm: summarize(cm, NAMES, WET['floodnet'])[metric]
    pa, pb = f(A.sum(0)), f(Bm.sum(0))
    da, db, dd = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        a, b = f(A[idx].sum(0)), f(Bm[idx].sum(0))
        da.append(a); db.append(b); dd.append(b - a)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return dict(old=pa, old_ci=ci(da), new=pb, new_ci=ci(db), diff=pb - pa, diff_ci=ci(dd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new_full', required=True)
    ap.add_argument('--decode', required=True)
    ap.add_argument('--data', default=os.path.join(RR_DATA, 'floodnet_r768'))
    ap.add_argument('--sam', default=SAM_WEIGHTS)
    ap.add_argument('--out', default='results/b13/final')
    ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = max(json.load(open(a.decode)), key=lambda r: r['miou'])
    old_dec = {k: v for k, v in json.load(open(OLD_DECODE))['chosen'].items() if k != 'variant'}
    Pn, ids = load_parts(a.new_full.split(','))
    zo = np.load(OLD_CACHE)
    old_ids = [str(i) for i in zo['ids']]
    pos = {i: k for k, i in enumerate(old_ids)}
    assert set(ids) == set(old_ids), 'new and old full-test caches hold different photos'
    eval100 = set(l.strip() for l in open('splits/floodnet_test_eval100.txt') if l.strip())

    import torch
    from segment_anything import SamPredictor, sam_model_registry
    from sam_refine import refine
    dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    pred = SamPredictor(sam_model_registry['vit_b'](checkpoint=a.sam).to(dev).eval())

    cms = {'old': [], 'new': [], 'new_sam': []}
    maps_new, maps_sam = {}, {}
    t0 = time.time()
    for n, (i, P8) in enumerate(zip(ids, Pn)):
        P = P8.astype(np.float32) / 255
        img = np.asarray(Image.open(os.path.join(a.data, 'test', 'images', f'{i}.jpg')).convert('RGB'))
        g = gt_classes(np.asarray(Image.open(os.path.join(a.data, 'test', 'labels', f'{i}.png'))))
        cms['old'].append(confusion(decode(zo['probs'][pos[i]].astype(np.float32) / 255, **old_dec), g))
        cn = new_decode(P, img, d)
        with torch.no_grad():
            pred.set_image(img)

            def masks_for(boxes):
                tb = pred.transform.apply_boxes_torch(torch.as_tensor(boxes, device=dev), img.shape[:2])
                m, _, _ = pred.predict_torch(point_coords=None, point_labels=None, boxes=tb, multimask_output=False)
                return list(m[:, 0].cpu().numpy())
            cs = refine(cn, P, masks_for, 'grow', 50, d['tau'])
        cms['new'].append(confusion(cn, g))
        cms['new_sam'].append(confusion(cs, g))
        maps_new[i], maps_sam[i] = cn, cs
        if n % 50 == 0:
            print(f'{n}/{len(ids)} {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(os.path.join(a.out, 'classmaps_floodnet_test_full.npz'), ids=np.array(ids),
                        new=np.stack([maps_new[i] for i in ids]), new_sam=np.stack([maps_sam[i] for i in ids]))
    res = dict(decode=d, old_decode=old_dec, n_full=len(ids), B=a.B)
    for split, keep in (('full', ids), ('eval100', [i for i in ids if i in eval100])):
        k = [ids.index(i) for i in keep]
        for v in ('new', 'new_sam'):
            for metric in ('miou', 'drivable_precision'):
                res[f'{split}_{v}_{metric}'] = boot_diff([cms['old'][j] for j in k], [cms[v][j] for j in k], a.B, 0, metric)
        res[f'{split}_per_class'] = {v: summarize(np.sum([cms[v][j] for j in k], 0), NAMES, WET['floodnet'])
                                     for v in ('old', 'new', 'new_sam')}
    full = res['full_new_sam_miou']
    res['switch'] = bool(full['diff_ci'][0] > 0)
    res['switch_chameleon_only'] = bool(res['full_new_miou']['diff_ci'][0] > 0)
    json.dump(res, open(os.path.join(a.out, 'final.json'), 'w'), indent=1)
    f = lambda r: f"{r['new']:.3f} [{r['new_ci'][0]:.3f}, {r['new_ci'][1]:.3f}] vs {r['old']:.3f}: Δ {r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]"
    lines = [f'Paired bootstrap over photos (B {a.B}). Old = published model + published decode; new = Batch 13 checkpoint '
             '+ its val30-chosen decode (guided filter -> tau [-> clean-up if chosen]), "+SAM" = with SAM grow.', '']
    for split in ('full', 'eval100'):
        for v, lab in (('new', 'Chameleon only'), ('new_sam', '+SAM')):
            lines.append(f"- {split} test, {lab}: mIoU {f(res[f'{split}_{v}_miou'])}; drivable precision "
                         f"{res[f'{split}_{v}_drivable_precision']['new']:.3f} vs {res[f'{split}_{v}_drivable_precision']['old']:.3f}")
    lines.append('')
    lines.append(f"**Rule: switch = {res['switch']}** (with SAM); Chameleon only: {res['switch_chameleon_only']}")
    pc = res['full_per_class']
    lines.append('')
    lines.append('| full test | ' + ' | '.join(NAMES) + ' | mIoU | drv P |')
    lines.append('|---|' + '---|' * 7)
    for v in ('old', 'new', 'new_sam'):
        lines.append(f'| {v} | ' + ' | '.join(f"{pc[v][f'iou_{n}']:.3f}" for n in NAMES) + f" | {pc[v]['miou']:.3f} | {pc[v]['drivable_precision']:.3f} |")
    open(os.path.join(a.out, 'final.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
