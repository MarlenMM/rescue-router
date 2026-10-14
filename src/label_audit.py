"""Label audit of the FloodNet evaluation photos (PLAN §16.5). CPU only; uses the photos and the labels, never a
model's prediction, so whatever it flags affects every model alike.

The two full-test photos with the most false-safe houses (9344, 9348: 42 of the submitted model's 105) carry a
labelling error in FloodNet itself: dry, green lawns and river banks are labelled water (raw class 5, 58 % of the
photo; no grass class at all). Rule: a pixel is vegetation-green when its excess-green index on chromaticity
coordinates, ExG = 2g - r - b (g = G / (R + G + B), ...), is above EXG_T; a photo is flagged when at least FLAG of its
labelled water pixels (and at least MIN_PX of them) are vegetation-green. Floodwater is brown, grey or blue (ExG near or
below 0); lawns and crowns are clearly positive.
Flagged photos are reported (contact sheet for a visual check); every result can then be given with and without them
(`--exclude` lists in report/label_audit/). Thresholds are set from the distribution over all evaluation photos.

    python src/label_audit.py [--splits floodnet_test_full,floodnet_val_tune30,floodnet_val_confirm100]
    python src/label_audit.py --impact [out]     # both models with and without the confirmed errors; val30 re-tune
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr.paths import RR_DATA  # noqa: E402

EXG_T = 0.08
FLAG = 0.5
MIN_PX = 5000


def exg(img):
    x = img.astype(np.float32)
    s = x.sum(2) + 1e-6
    r, g, b = x[..., 0] / s, x[..., 1] / s, x[..., 2] / s
    return 2 * g - r - b


def audit(split_file):
    split = 'val' if '_val_' in split_file else 'test'
    root = os.path.join(RR_DATA, f'{split_file.split("_")[0]}_r768', split)
    rows = []
    for i in open(os.path.join(os.path.dirname(HERE), 'splits', f'{split_file}.txt')).read().split():
        img = np.asarray(Image.open(os.path.join(root, 'images', f'{i}.jpg')).convert('RGB'))
        lab = np.asarray(Image.open(os.path.join(root, 'labels', f'{i}.png')))
        e = exg(img)
        row = dict(id=i, split=split_file)
        for name, c in (('water', 4), ('road_flooded', 1)):
            m = lab == c
            row[f'{name}_px'] = int(m.sum())
            row[f'{name}_green'] = float((e[m] > EXG_T).mean()) if m.any() else 0.0
        row['flag'] = row['water_px'] >= MIN_PX and row['water_green'] >= FLAG
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='floodnet_test_full,floodnet_val_tune30,floodnet_val_confirm100')
    ap.add_argument('--out', default='results/kaggle_v2/report/label_audit')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [r for sf in a.splits.split(',') for r in audit(sf)]
    flagged = [r for r in rows if r['flag']]
    json.dump(dict(rule=dict(exg_t=EXG_T, flag=FLAG, min_px=MIN_PX), rows=rows), open(os.path.join(a.out, 'audit.json'), 'w'),
              indent=1)
    for sf in a.splits.split(','):
        open(os.path.join(a.out, f'flagged_{sf}.txt'), 'w').write(
            ''.join(f"{r['id']}\n" for r in flagged if r['split'] == sf))
    with_water = sorted((r for r in rows if r['water_px'] >= MIN_PX), key=lambda r: -r['water_green'])
    md = ['# Label audit: labelled water that looks like vegetation (FloodNet evaluation photos)', '',
          f'Rule (photo and label only, no prediction): pixel vegetation-green if ExG = 2g − r − b > {EXG_T}; photo flagged if '
          f'≥ {FLAG:.0%} of its ≥ {MIN_PX} labelled water pixels are green. {len(flagged)} of {len(rows)} photos flagged.', '',
          '| photo | split | water px | green share of water | green share of flooded road | flagged |', '|---|---|---|---|---|---|']
    md += [f"| {r['id']} | {r['split']} | {r['water_px']} | {r['water_green']:.2f} | {r['road_flooded_green']:.2f} | "
           f"{'**yes**' if r['flag'] else ''} |" for r in with_water[:30]]
    shares = np.array([r['water_green'] for r in with_water])
    md += ['', f'Distribution over the {len(with_water)} photos with ≥ {MIN_PX} water pixels: green share quantiles '
           + ', '.join(f'{q}% {np.percentile(shares, q):.2f}' for q in (50, 90, 95, 99)) + '.']
    open(os.path.join(a.out, 'audit.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))
    if with_water:                                  # contact sheet: top 12 by green share, photo | label
        from viz import colorize
        tiles = []
        for r in with_water[:12]:
            split = 'val' if '_val_' in r['split'] else 'test'
            root = os.path.join(RR_DATA, f'{r["split"].split("_")[0]}_r768', split)
            img = Image.open(os.path.join(root, 'images', f"{r['id']}.jpg")).convert('RGB').resize((256, 192))
            lab = np.asarray(Image.open(os.path.join(root, 'labels', f"{r['id']}.png")))
            from rr.metrics import gt_classes
            col = Image.fromarray(colorize(gt_classes(lab), r['split'].split('_')[0])[..., :3]).resize((256, 192), Image.NEAREST)
            tiles.append(np.concatenate([np.asarray(img), np.asarray(col)], 1))
        sheet = np.concatenate([np.concatenate(tiles[k:k + 2], 1) if k + 1 < len(tiles) else
                                np.concatenate([tiles[k], np.zeros_like(tiles[k])], 1) for k in range(0, len(tiles), 2)], 0)
        Image.fromarray(sheet).save(os.path.join(a.out, 'top_green_water.jpg'), quality=85)


def errors(split=None):
    """ids of the confirmed label errors (splits/floodnet_label_errors.tsv), optionally of one split ('test' / 'val')"""
    rows = [l.split('\t') for l in open(os.path.join(os.path.dirname(HERE), 'splits', 'floodnet_label_errors.tsv'))
            if l.strip() and not l.startswith('#')][1:]
    return [r[0] for r in rows if split is None or r[1] == split]


def impact(out, B=2000):
    """the submitted Chameleon model and the 50-shot baseline on the full test, with and without the confirmed label
    errors: mIoU, water IoU, stranded F1, false safe, and the paired differences (same resamples)"""
    import baseline_report as BR
    from bootstrap_ci import f1_from
    from router import false_safe_from
    from rr.metrics import WET, confusion, summarize
    from rr.final_decode import decode_photo
    names = BR.NAMES['floodnet']
    gts = BR.Labels()('floodnet', 'floodnet_test_full')
    bad = set(errors('test'))
    cc = BR.cham_cms(os.path.abspath('results/kaggle_v2/rr'), 'floodnet_50shot_s0_stratified', 'floodnet', 'floodnet_test_full', gts)
    Pb = BR.probs('results/baselines/preds/baseline_deeplabv3_r50_50shot_s0_floodnet_test_full.npz')
    cb = {i: confusion(decode_photo(Pb[i], {'argmax': True}), g) for i, g in gts.items()}
    sc = BR.per_photo_status('results/routing/full/pred_floodnet_50shot_full_tuned.json')
    sb = BR.per_photo_status('results/baselines/report/routing_full_baseline_deeplabv3_r50_50shot_s0.json')
    water = lambda c: summarize(c, names, WET['floodnet'])['iou_water']                      # noqa: E731
    res = {}
    for label, keep in (('all', lambda i: True), ('without label errors', lambda i: i not in bad)):
        ids = sorted(i for i in gts if keep(i))
        rids = sorted(i for i in set(sc) & set(sb) if keep(i))
        r = {}
        for m, fn, xb, xc, ii in (('miou', BR.miou_fn('floodnet'), cb, cc, ids), ('water_iou', water, cb, cc, ids),
                                  ('stranded_f1', f1_from, sb, sc, rids), ('false_safe', false_safe_from, sb, sc, rids)):
            p = BR.paired(xb, xc, ii, fn, B)
            r[m] = dict(baseline=p['baseline'], chameleon=p['chameleon'], baseline_minus_chameleon=p['diff'])
        r['n_photos'], r['n_false_safe_chameleon'] = len(ids), int(sum(sc[i][1:, 0].sum() for i in rids))
        r['n_false_safe_baseline'] = int(sum(sb[i][1:, 0].sum() for i in rids))
        res[label] = r
    json.dump(res, open(os.path.join(out, 'impact.json'), 'w'), indent=1, default=float)
    md = ['# Impact of the confirmed label errors (FloodNet full test; 50-shot baseline vs the submitted Chameleon model)', '',
          f'{len(bad)} test photos with large dry areas labelled water (splits/floodnet_label_errors.tsv).', '',
          '| photos | metric | baseline | Chameleon | baseline − Chameleon [95 % CI] |', '|---|---|---|---|---|']
    for label, r in res.items():
        for m in ('miou', 'water_iou', 'stranded_f1', 'false_safe'):
            x = r[m]
            md.append(f"| {label} ({r['n_photos']}) | {m.replace('_', ' ')} | {x['baseline']['value']:.3f} | "
                      f"{x['chameleon']['value']:.3f} | {x['baseline_minus_chameleon']['value']:+.3f} "
                      f"[{x['baseline_minus_chameleon']['lo']:+.3f}, {x['baseline_minus_chameleon']['hi']:+.3f}] |")
        md.append(f"| {label} | false-safe houses | {r['n_false_safe_baseline']} | {r['n_false_safe_chameleon']} | |")
    open(os.path.join(out, 'impact.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


def retune_without(out):
    """Chameleon's decode thresholds re-tuned on val30 without its label error (run_decode.py procedure)"""
    import postproc_sweep as PS
    from run_decode import GF, load
    _, probs, gts, grey = load('results/kaggle_v2/rr/preds/floodnet_50shot_s0_stratified_floodnet_val_tune30.npz', 'val')
    ids = open(os.path.join(os.path.dirname(HERE), 'splits', 'floodnet_val_tune30.txt')).read().split()
    res = {}
    for label, keep in (('all 30', lambda i: True), ('without 7769', lambda i: i not in errors('val'))):
        k = [n for n, i in enumerate(ids) if keep(i)]
        S = PS.Scorer([probs[n] for n in k], [gts[n] for n in k], 4)
        tau_raw, _ = PS.tune_tau(S)
        S.close()
        S = PS.Scorer([PS.guided_filter(grey[n], probs[n], *GF) for n in k], [gts[n] for n in k], 4)
        tau, r = PS.tune_tau(S, start=tau_raw)
        S.close()
        res[label] = dict(tau=[float(t) for t in tau], val_miou=r['miou'], n=len(k))
        print(label, res[label], flush=True)
    json.dump(res, open(os.path.join(out, 'retune_val30.json'), 'w'), indent=1)
    return res


if __name__ == '__main__' and '--impact' in sys.argv:
    o = sys.argv[sys.argv.index('--impact') + 1] if len(sys.argv) > sys.argv.index('--impact') + 1 else \
        'results/kaggle_v2/report/label_audit'
    os.makedirs(o, exist_ok=True)
    retune_without(o)
    impact(o)
elif __name__ == '__main__':
    main()
