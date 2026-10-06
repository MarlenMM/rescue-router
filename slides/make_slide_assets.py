"""Images for the slide deck and the GIF backup of the demo (Batch 11). CPU; run from the repository root.

    ~/.venvs/rr/bin/python slides/make_slide_assets.py     -> slides/assets/*.jpg|png, slides/demo_route.gif
"""

import json
import os
import sys
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import gradio_app as G  # noqa: E402
import router as R  # noqa: E402
from viz import blend, load_image, load_label  # noqa: E402
from rr.metrics import gt_classes  # noqa: E402

OUT = os.path.join(ROOT, 'slides', 'assets')
DATA = os.path.expanduser('~/rr_data')
A = types.SimpleNamespace(results=os.path.join(ROOT, 'results/kaggle_v2/rr'), data=DATA)
CI = json.load(open(os.path.join(ROOT, 'results/kaggle_v2/report/ci/ci.json')))['curve']


def save(arr, name, width=None, q=88):
    im = Image.fromarray(arr) if isinstance(arr, np.ndarray) else arr
    if width:
        im = im.resize((width, round(width * im.height / im.width)), Image.LANCZOS)
    im.convert('RGB').save(os.path.join(OUT, name), quality=q)


def photos():
    Image.open(os.path.join(ROOT, 'app/scenes/floodnet_7301/photo.jpg')).convert('RGB').save(os.path.join(OUT, 'cover.jpg'), quality=86)
    img, lab = load_image(DATA, 'floodnet', 'test', '7301'), gt_classes(load_label(DATA, 'floodnet', 'test', '7301'))
    save(img, 'photo_7301.jpg')
    save(blend(img, lab, 'floodnet', 0.7), 'label_7301.jpg')


def curve():
    shots = [1, 5, 10, 20, 50]
    plt.rcParams.update({'font.size': 20, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(10, 6.6))
    for dec, col, lab, dx in (('tau0.5', '#8a94a6', 'τ = 0.5', 0.97), ('tuned', '#1F6FB2', 'decode tuned on 30 val photos', 1.03)):
        m = [CI[f'floodnet/{s}/floodnet_test_eval100/{dec}']['miou'] for s in shots]
        y = np.array([v['value'] for v in m])
        err = np.array([[v['value'] - v['lo'] for v in m], [v['hi'] - v['value'] for v in m]])
        ax.errorbar(np.array(shots) * dx, y, yerr=err, marker='o', ms=9, lw=3, capsize=5, color=col, label=lab)
    v = CI['floodnet/50/floodnet_test_full/tuned']['miou']
    ax.errorbar([64], [v['value']], yerr=[[v['value'] - v['lo']], [v['hi'] - v['value']]], marker='D', ms=11, color='#1F6FB2',
                mfc='#F6F4EF', mew=3, capsize=5, lw=2.5)
    ax.annotate(f"full test\n{v['value']:.3f}", (64, v['lo'] - 0.015), ha='center', va='top', fontsize=17, color='#1F2A3A')
    ax.axhline(.848, ls='--', lw=2.5, color='#D9731F')
    ax.text(1.0, .865, 'PSPNet with 1,445 labels: .848', fontsize=18, color='#B85A12', va='bottom')
    ax.set(xscale='log', xticks=shots, xticklabels=[str(s) for s in shots], xlim=(0.8, 100), ylim=(0.1, 0.95),
           xlabel='labelled photos', ylabel='mIoU, 5 classes')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 0.9), fontsize=18)
    fig.patch.set_facecolor('#F6F4EF'); ax.set_facecolor('#F6F4EF')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'curve.png'), dpi=110, facecolor='#F6F4EF')
    plt.close(fig)


def scenes():
    st = G.cached_state('floodnet:7301', A, src='pred')
    gt = G.cached_state('floodnet:7301', A, src='gt')
    j = next(k for k in range(len(st['scene']['bld'])) if st['status']['pred'][k] == 1 and gt['status']['gt'][k] == 1)
    for s, name in ((st, 'route_7301_pred.jpg'), (gt, 'route_7301_gt.jpg')):
        s['sel'] = (j, *G.route_for(s, j))
        save(G.render(s), name, 1200)
    rn = G.cached_state('rescuenet:12195', A, src='gt')
    rn['sel'] = (8, *G.route_for(rn, 8))
    save(G.render(rn), 'route_12195.jpg', 1200)
    f = G.cached_state('floodnet:7241', A, src='pred')
    save(G.render(f), 'fail_7241_pred.jpg', 1200)
    fg = G.cached_state('floodnet:7241', A, src='gt')
    save(G.render(fg), 'fail_7241_gt.jpg', 1200)
    return rn


def gif(rn, path):
    """the demo moment as a GIF: RescueNet 12195, labelled map, house 8: truck -> boat -> on foot, drawn step by step"""
    sel = rn['sel']
    rn['sel'] = None
    base = Image.fromarray(G.render(rn)).convert('RGB')
    rn['sel'] = sel
    route = sel[1]
    K, C = G.K, R.CELL
    pts = [(x * C * K + C * K / 2, y * C * K + C * K / 2) for y, x in route['cells']]
    modes = route['modes']
    col = {R.TRUCK: (255, 255, 255), R.BOAT: (34, 211, 238), R.FOOT: (167, 139, 250)}
    frames, n = [], len(pts)
    steps = list(range(2, n + 1, max(1, n // 40))) + [n] * 12
    poly = [(x * K, y * K) for x, y in rn['scene']['buildings'][8]['polygon']]
    for k in steps:
        im = base.copy()
        d = ImageDraw.Draw(im)
        d.polygon(poly, outline=(34, 211, 238), width=8)
        for a in range(1, k):
            d.line([pts[a - 1], pts[a]], fill=(0, 0, 0), width=12)
        for a in range(1, k):
            if modes[a] == R.FOOT:
                if a % 2 == 0:
                    x, y = pts[a]
                    d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col[R.FOOT])
            else:
                d.line([pts[a - 1], pts[a]], fill=col[modes[a]], width=6)
        x, y = pts[k - 1]
        d.ellipse([x - 14, y - 14, x + 14, y + 14], fill=col[modes[k - 1]], outline=(0, 0, 0), width=4)
        frames.append(im.resize((768, 576), Image.LANCZOS).convert('P', palette=Image.ADAPTIVE, colors=128))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[80] * (len(frames) - 1) + [1500], loop=0, optimize=True)
    print('gif', path, len(frames), 'frames', f'{os.path.getsize(path) / 1e6:.1f} MB')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    photos(); curve()
    rn = scenes()
    gif(rn, os.path.join(ROOT, 'slides', 'demo_route.gif'))
    for f in sorted(os.listdir(OUT)):
        print(f, Image.open(os.path.join(OUT, f)).size, f'{os.path.getsize(os.path.join(OUT, f)) / 1e3:.0f} KB')
