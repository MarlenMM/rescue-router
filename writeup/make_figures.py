"""Figures for the write-up (Batch 10). CPU, from the reports and caches; run from the repository root.

    ~/.venvs/rr/bin/python writeup/make_figures.py
writes writeup/figs/curve.pdf (learning curve with bootstrap CIs) and writeup/figs/routes.jpg (router on a predicted
and a labelled map, drawn with the Gradio app's renderer).
"""

import json
import os
import sys
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
OUT = os.path.join(ROOT, 'writeup', 'figs')
CI = json.load(open(os.path.join(ROOT, 'results/kaggle_v2/report/ci/ci.json')))['curve']


def curve():
    shots = [1, 5, 10, 20, 50]
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    # τ 0.5 only up to 20 shots: the final 50-shot model (Batch 13: shorter schedule) is calibrated differently and is
    # always used with its val30-tuned decode; the first 50-shot model is shown as open markers
    for dec, col, lab, dx, sh in (('tau0.5', '#7a8699', 'τ = 0.5', 0.96, shots[:4]), ('tuned', '#1f5fbf', 'decode tuned on val30', 1.04, shots)):
        m = [CI[f'floodnet/{s}/floodnet_test_eval100/{dec}']['miou'] for s in sh]
        y = np.array([v['value'] for v in m])
        err = np.array([[v['value'] - v['lo'] for v in m], [v['hi'] - v['value'] for v in m]])
        ax.errorbar(np.array(sh) * dx, y, yerr=err, marker='o', ms=3.5, lw=1.4, capsize=2, color=col, label=f'FloodNet eval100, {lab}')
    first = json.load(open(os.path.join(ROOT, 'results/kaggle_v2/rr/decode/floodnet_50shot_s0_stratified_pub.json')))
    first_full = json.load(open(os.path.join(ROOT, 'results/kaggle_v2/report/b13/b13.json')))['full test (448)/new_sam']['miou']['old']
    ax.plot([50 * 1.04], [first['eval100'][first['chosen']['variant']]['miou']], marker='o', ms=4, mfc='white', color='#1f5fbf',
            ls='none', label='first 50-shot model (eval100, full test)')
    ax.plot([50 * 1.25], [first_full], marker='D', ms=4, mfc='white', color='#1f5fbf', ls='none')
    v = CI['floodnet/50/floodnet_test_full/tuned']['miou']
    ax.errorbar([50 * 1.25], [v['value']], yerr=[[v['value'] - v['lo']], [v['hi'] - v['value']]], marker='D', ms=4,
                color='#1f5fbf', capsize=2, lw=1.2)
    ax.annotate('full test\n(448 photos)', (70, .40), fontsize=7, color='#333', ha='center', va='top')
    for s, mk in ((20, 's'), (50, 's')):
        v = CI[f'rescuenet/{s}/rescuenet_test_eval100/tuned']['miou']
        ax.errorbar([s * 0.9], [v['value']], yerr=[[v['value'] - v['lo']], [v['hi'] - v['value']]], marker=mk, ms=3.5,
                    color='#c0392b', capsize=2, lw=1, ls='none', label='RescueNet eval100, tuned' if s == 20 else None)
    ax.axhline(.848, ls='--', lw=1, color='#e67e22')
    ax.text(0.85, .862, 'PSPNet, 1,445 labels (same 5 classes): .848', fontsize=7, color='#e67e22', va='bottom')
    ax.set(xscale='log', xticks=shots, xticklabels=[str(s) for s in shots], xlim=(0.8, 105), ylim=(0.1, 0.93),
           xlabel='labelled support photos', ylabel='mIoU (5 classes)')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, 0.88), frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, 'curve.pdf'))
    print('curve.pdf')


def routes():
    import gradio_app as G
    a = types.SimpleNamespace(results=os.path.join(ROOT, 'results/kaggle_v2/rr'), data=os.path.expanduser('~/rr_data'))
    panels = []
    st = G.cached_state('floodnet:7301', a, src='pred')
    gt = G.cached_state('floodnet:7301', a, src='gt')
    j = next(k for k in range(len(st['scene']['bld'])) if st['status']['pred'][k] == 1 and gt['status']['gt'][k] == 1)
    for s, side, title in ((st, 'pred', 'FloodNet 7301: prediction (50 labels)'), (gt, 'gt', 'FloodNet 7301: labelled map')):
        s['sel'] = (j, *G.route_for(s, j))
        panels.append((G.render(s), title))
    rn = G.cached_state('rescuenet:12195', a, src='gt')
    rn['sel'] = (8, *G.route_for(rn, 8))
    panels.append((G.render(rn), 'RescueNet 12195: truck, boat, then on foot'))
    W = 900
    ims = []
    font = ImageFont.load_default(size=30)
    for arr, title in panels:
        im = Image.fromarray(arr).resize((W, round(W * arr.shape[0] / arr.shape[1])), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, 44], fill=(255, 255, 255))
        d.text((10, 6), title, fill=(0, 0, 0), font=font)
        ims.append(np.asarray(im))
    gap = np.full((ims[0].shape[0], 14, 3), 255, np.uint8)
    row = np.concatenate([ims[0], gap, ims[1], gap, ims[2]], 1)
    Image.fromarray(row).save(os.path.join(OUT, 'routes.jpg'), quality=88)
    print('routes.jpg, house', j)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    curve()
    routes()
