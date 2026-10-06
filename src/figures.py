"""Report figures from the caches (PLAN §5.3, §7.5). CPU only.

    python src/figures.py [--out results/kaggle_v2/report/figures]

failure_gallery.png  2x3: GT | prediction (FloodNet 50-shot, tuned decode) for six tagged failures,
                     captioned with the tag and its routing consequence (results/routing, gated router).
qualitative.png      eval100 scenes x (photo, GT, 1 / 5 / 20 / 50-shot predictions, tuned decodes).
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr.metrics import decode, gt_classes  # noqa: E402
from viz import blend, legend_handles, load_image, load_label  # noqa: E402

RUN = 'floodnet_50shot_s0_stratified'
GALLERY = [  # id, tag (PLAN §5.3 categories, + 8 label ambiguity, 9 thin structures)
    ('7370', '2 · shallow brown floodwater over yards decoded as dry land'),
    ('7486', '2 · sand bar in floodwater decoded as clear road'),
    ('7308', '2 · shallow flooded street decoded as clear road'),
    ('7291', '8 · bridge over the flood labelled "flooded road" in GT'),
    ('6775', '5+7 · river split into water / flooded road along a tile row'),
    ('6420', '4 · pale concrete street missed (decoded as other)'),
]


def load_run(results, run, split='test_eval100_fp16', ds='floodnet'):
    z = np.load(os.path.join(results, 'preds', f'{run}_{ds}_{split}.npz'))
    d = json.load(open(os.path.join(results, 'decode', f'{run}.json')))
    dec = {k: v for k, v in d['chosen'].items() if k != 'variant'}
    return dict(zip([str(i) for i in z['ids']], z['probs'])), dec


def predicted(P, dec):
    return decode(P.astype(np.float32) / 255, **dec)


def gallery(a, plt):
    probs, dec = load_run(a.results, RUN)
    rt = {r['id']: r for r in json.load(open(os.path.join(a.routing, f'pred_{RUN}_tuned.json')))['per_image']}
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.2))
    for ax, (i, tag) in zip(axes.ravel(), GALLERY):
        img, gt = load_image(a.data, 'floodnet', 'test', i), gt_classes(load_label(a.data, 'floodnet', 'test', i))
        pair = np.concatenate([blend(img, gt), np.full((img.shape[0], 12, 3), 255, np.uint8),
                               blend(img, predicted(probs[i], dec))], 1)
        ax.imshow(pair)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        r = rt[i]
        cm = np.array(r.get('status_confusion', np.zeros((4, 4))))
        cons = (f'routing: {r["n_buildings"]} houses, {int(cm[1:, 0].sum())} stranded shown as truck-reachable, '
                f'{r.get("n_unsafe_routes", 0)} unsafe truck routes') if r['n_buildings'] else 'routing: no houses, no effect'
        ax.set_title(f'{i}  {tag}\n{cons}', fontsize=8.5, loc='left')
    fig.legend(handles=legend_handles(), loc='lower center', ncol=5, frameon=False, fontsize=9)
    fig.suptitle('Failure cases, FloodNet 50-shot (left: ground truth, right: prediction)', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=1.0)
    fig.savefig(os.path.join(a.out, 'failure_gallery.png'), dpi=130, bbox_inches='tight')


def qualitative(a, plt, ids):
    runs = [(f'floodnet_{k}shot_s0_stratified', f'{k}-shot') for k in (1, 5, 20, 50)]
    loaded = [load_run(a.results, r) for r, _ in runs]
    cols = ['photo', 'ground truth'] + [n for _, n in runs]
    fig, axes = plt.subplots(len(ids), len(cols), figsize=(2.6 * len(cols), 2.05 * len(ids)))
    for r, i in enumerate(ids):
        img, gt = load_image(a.data, 'floodnet', 'test', i), gt_classes(load_label(a.data, 'floodnet', 'test', i))
        tiles = [img, blend(img, gt)] + [blend(img, predicted(p[i], dec)) for p, dec in loaded]
        for c, t in enumerate(tiles):
            ax = axes[r, c]
            ax.imshow(t)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
        axes[r, 0].set_ylabel(i, fontsize=8)
    fig.legend(handles=legend_handles(), loc='lower center', ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(a.out, 'qualitative.png'), dpi=130)


def pick_qualitative(a, n=4):
    """Two flooded scenes and two dry ones where the 50-shot model does well (by road score)."""
    rows = {r['id']: r for r in json.load(open(os.path.join(a.failures, 'per_image.json')))}
    ok = [r for r in rows.values() if r['road_score'] is not None and r['n_buildings'] >= 3]
    wet = sorted([r for r in ok if r['gt_frac']['road_flooded'] + r['gt_frac']['water'] > 0.1], key=lambda r: -r['road_score'])
    dry = sorted([r for r in ok if r['gt_frac']['road_flooded'] + r['gt_frac']['water'] < 0.02], key=lambda r: -r['road_score'])
    return [r['id'] for r in wet[:n // 2] + dry[:n - n // 2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--routing', default='results/routing')
    ap.add_argument('--failures', default='results/kaggle_v2/report/failures')
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--out', default='results/kaggle_v2/report/figures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gallery(a, plt)
    ids = pick_qualitative(a)
    qualitative(a, plt, ids)
    print('qualitative scenes:', ids)


if __name__ == '__main__':
    main()
