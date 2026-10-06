"""Failure mining (PLAN §5.3) for the headline model on eval100. CPU only, from the caches.

    python src/failures.py --run floodnet_50shot_s0_stratified [--n 12]

Per image (tuned decode of the run): road_clear / road_flooded IoU, a road score (mean of the road IoUs
of the road channels present in GT or prediction), false-"clear" pixels (decoded road_clear where the GT
is road_flooded or water) and the largest connected false-clear cluster, plus the routing outcome from
results/routing/pred_<run>_tuned.json (stranded houses missed, unsafe truck routes).
Writes <out>/per_image.json, <out>/worst.json (the n worst by road score, then by false-clear cluster)
and contact sheets <out>/sheet_<k>.png: photo | GT | prediction | errors (magenta = false clear,
yellow = other errors).
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr.metrics import decode, gt_classes  # noqa: E402
from viz import blend, load_image, load_label  # noqa: E402

WET = (1, 4)


def iou(p, g):
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else None


def per_image(i, P, gt, dec):
    pred = decode(P.astype(np.float32) / 255, **dec)
    r = dict(id=i)
    present = []
    for c, n in ((0, 'road_clear'), (1, 'road_flooded')):
        p, g = pred == c, gt == c
        r[f'iou_{n}'] = iou(p, g)
        if (g.mean() >= 0.01 or p.mean() >= 0.01) and r[f"iou_{n}"] is not None:   # >= 1 % of the image
            present.append(r[f'iou_{n}'])
    r['road_score'] = float(np.mean(present)) if present else None
    fc = (pred == 0) & np.isin(gt, WET)
    lab, n = ndimage.label(fc)
    r['false_clear_px'] = int(fc.sum())
    r['false_clear_max_cluster'] = int(np.bincount(lab.ravel())[1:].max()) if n else 0
    r['gt_frac'] = {k: float((gt == c).mean()) for c, k in enumerate(['road_clear', 'road_flooded', 'bldg_flooded',
                                                                    'bldg_dry', 'water'])}
    return r, pred


def error_map(img, pred, gt):
    out = (img.astype(np.float32) * 0.35).astype(np.uint8)
    wrong = (pred != gt) & (gt != 255)
    out[wrong] = (230, 200, 0)
    out[(pred == 0) & np.isin(gt, WET)] = (255, 0, 200)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='floodnet_50shot_s0_stratified')
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--routing', default='results/routing')
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--out', default='results/kaggle_v2/report/failures')
    ap.add_argument('--n', type=int, default=12)
    a = ap.parse_args()
    ds = a.run.split('_')[0]
    os.makedirs(a.out, exist_ok=True)
    d = json.load(open(os.path.join(a.results, 'decode', f'{a.run}.json')))
    dec = {k: v for k, v in d['chosen'].items() if k != 'variant'}
    z = np.load(os.path.join(a.results, 'preds', f'{a.run}_{ds}_test_eval100_fp16.npz'))
    rt = {r['id']: r for r in json.load(open(os.path.join(a.routing, f'pred_{a.run}_tuned.json')))['per_image']}
    rows, preds = [], {}
    for i, P in zip([str(x) for x in z['ids']], z['probs']):
        gt = gt_classes(load_label(a.data, ds, 'test', i))
        r, preds[i] = per_image(i, P, gt, dec)
        cm = np.array(rt[i].get('status_confusion', np.zeros((4, 4))))
        r.update(n_buildings=rt[i]['n_buildings'], stranded_missed=int(cm[1:, 0].sum()),
                 n_unsafe_routes=rt[i].get('n_unsafe_routes', 0), base=rt[i]['base'])
        rows.append(r)
    json.dump(rows, open(os.path.join(a.out, 'per_image.json'), 'w'), indent=1)
    scored = sorted((r for r in rows if r['road_score'] is not None), key=lambda r: r['road_score'])
    worst = scored[:a.n]
    fc_top = sorted((r for r in rows if r not in worst), key=lambda r: -r['false_clear_max_cluster'])[:4]
    worst += [dict(r, picked='false-clear cluster') for r in fc_top if r['false_clear_max_cluster'] > 0]
    json.dump(worst, open(os.path.join(a.out, 'worst.json'), 'w'), indent=1)
    tw, th = 384, 288
    for k in range(0, len(worst), 4):
        chunk = worst[k:k + 4]
        sheet = Image.new('RGB', (4 * tw, len(chunk) * (th + 22)), 'white')
        dr = ImageDraw.Draw(sheet)
        for row, r in enumerate(chunk):
            i = r['id']
            img, gt = load_image(a.data, ds, 'test', i), gt_classes(load_label(a.data, ds, 'test', i))
            tiles = [img, blend(img, gt, ds), blend(img, preds[i], ds), error_map(img, preds[i], gt)]
            y = row * (th + 22)
            for col, t in enumerate(tiles):
                sheet.paste(Image.fromarray(t).resize((tw, th)), (col * tw, y + 22))
            fmt = lambda v: '–' if v is None else f'{v:.2f}'
            dr.text((4, y + 4), f'#{k + row + 1} id {i} | road score {fmt(r["road_score"])} | clear {fmt(r["iou_road_clear"])} '
                    f'flooded {fmt(r["iou_road_flooded"])} | false-clear {r["false_clear_px"]} px (max cluster '
                    f'{r["false_clear_max_cluster"]}) | houses {r["n_buildings"]}, stranded missed {r["stranded_missed"]}, '
                    f'unsafe routes {r["n_unsafe_routes"]} {r.get("picked", "")}', fill='black')
        sheet.save(os.path.join(a.out, f'sheet_{k // 4 + 1}.png'))
    print(f'{len(rows)} images; worst {len(worst)} -> {a.out}')
    for r in worst:
        print(r['id'], r['road_score'] and round(r['road_score'], 3), r['false_clear_px'], r['stranded_missed'],
              r['n_unsafe_routes'], r.get('picked', ''))


if __name__ == '__main__':
    main()
