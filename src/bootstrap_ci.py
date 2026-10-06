"""Bootstrap confidence intervals over images (Batch 3). CPU only, from the prediction caches.

    python src/bootstrap_ci.py [--B 2000] [--out results/kaggle_v2/report/ci]

Every metric is a dataset-level number (IoU from the summed confusion matrix; stranded F1 from the summed status
confusion), so images are resampled with replacement and the metric recomputed; 95 % CI = 2.5 / 97.5 percentiles.
- curve: mIoU (and drivable precision) per run on eval100, default (tau 0.5) and tuned decode; FloodNet 50-shot
  also on the full 448-image test (per class too, for the PSPNet table)
- paired differences between consecutive FloodNet shot counts on the same resampled images (eval100)
- the same without the test photos that overlap the run's support photos (src/dupes.py)
- stranded-house F1 of the routing (router v3, risk-aware) on eval100 and the full test
Writes ci.json and ci.md.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402

NAMES = {'floodnet': ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water'],
         'rescuenet': ['road_clear', 'road_blocked', 'bldg_intact', 'bldg_damaged', 'water']}
RUNS = [('floodnet', s) for s in (1, 5, 10, 20, 50)] + [('rescuenet', s) for s in (20, 50)]


def run_id(ds, shot):
    return f'{ds}_{shot}shot_s0_stratified'


def per_image_cms(ds, shot, split_file, a, dec):
    """{image id: 6x6 confusion} for one run, split and decode"""
    npz = os.path.join(a.results, 'preds', f'{run_id(ds, shot)}_{split_file}_fp16.npz')
    z = np.load(npz)
    out = {}
    for i, p in zip([str(x) for x in z['ids']], z['probs']):
        gt = gt_classes(np.asarray(Image.open(os.path.join(a.data, f'{ds}_r768', 'test', 'labels', f'{i}.png'))))
        out[i] = confusion(decode(p.astype(np.float32) / 255, **dec), gt)
    return out


def boot(cms, ids, fn, B, rng):
    """point estimate and 95 % CI of fn(summed confusion) over images"""
    C = np.stack([cms[i] for i in ids])
    point = fn(C.sum(0))
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    vals = np.array([fn(C[k].sum(0)) for k in idx])
    return dict(value=point, lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)), n=len(ids)), vals


def f1_from(cm):
    cm = np.asarray(cm)
    pos_g, pos_p, tp = cm[1:, :].sum(), cm[:, 1:].sum(), cm[1:, 1:].sum()
    p, r = tp / max(pos_p, 1), tp / max(pos_g, 1)
    return float(2 * p * r / max(p + r, 1e-9))


def fmt(d):
    return f"{d['value']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--routing', default='results/routing')
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--dupes', default='results/kaggle_v2/report/dupes/dupes.json')
    ap.add_argument('--out', default='results/kaggle_v2/report/ci')
    ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dupes = json.load(open(a.dupes))['runs']
    out = dict(B=a.B, curve={}, paired={}, without_overlap={}, routing={}, full_test_per_class={})
    md = [f'Bootstrap over images, B = {a.B}, 95 % CI in brackets.', '',
          '| run | split | decode | images | mIoU | drivable precision |', '|---|---|---|---|---|---|']
    cms, draws = {}, {}
    for ds, shot in RUNS:
        dec_t = {k: v for k, v in json.load(open(os.path.join(a.results, 'decode', f'{run_id(ds, shot)}.json')))['chosen'].items()
                 if k != 'variant'}
        splits = [f'{ds}_test_eval100'] + ([f'{ds}_test_full'] if (ds, shot) == ('floodnet', 50) else [])
        for sf in splits:
            for dname, dec in (('tau0.5', dict(tau=0.5)), ('tuned', dec_t)):
                key = (ds, shot, sf, dname)
                cms[key] = per_image_cms(ds, shot, sf, a, dec)
                ids = sorted(cms[key])
                rng = np.random.default_rng(0)                       # same resamples for every run (paired)
                m, draws[key] = boot(cms[key], ids, lambda c: summarize(c, NAMES[ds], WET[ds])['miou'], a.B, rng)
                rng = np.random.default_rng(0)
                d, _ = boot(cms[key], ids, lambda c: summarize(c, NAMES[ds], WET[ds])['drivable_precision'], a.B, rng)
                out['curve']['/'.join(map(str, key))] = dict(miou=m, drivable_precision=d)
                md.append(f'| {ds} {shot}-shot | {sf.split("_test_")[1]} | {dname} | {m["n"]} | {fmt(m)} | {fmt(d)} |')
                print(md[-1], flush=True)
    # per class on the full test (for the PSPNet table)
    for dname in ('tau0.5', 'tuned'):
        key = ('floodnet', 50, 'floodnet_test_full', dname)
        ids = sorted(cms[key])
        out['full_test_per_class'][dname] = {}
        for n in NAMES['floodnet']:
            rng = np.random.default_rng(0)
            r, _ = boot(cms[key], ids, lambda c, n=n: summarize(c, NAMES['floodnet'], WET['floodnet'])[f'iou_{n}'], a.B, rng)
            out['full_test_per_class'][dname][n] = r
    md += ['', '| FloodNet 50-shot, full test (448) | ' + ' | '.join(NAMES['floodnet']) + ' |',
           '|---|' + '---|' * 5]
    for dname, v in out['full_test_per_class'].items():
        md.append(f'| {dname} | ' + ' | '.join(fmt(v[n]) for n in NAMES['floodnet']) + ' |')
    # paired differences along the FloodNet curve (same resampled image sets: rng seeded identically)
    md += ['', '| eval100, paired | decode | mIoU difference | share of resamples > 0 |', '|---|---|---|---|']
    for s0, s1 in ((1, 5), (5, 10), (10, 20), (20, 50), (5, 20), (10, 50)):
        for dname in ('tau0.5', 'tuned'):
            k0, k1 = (('floodnet', s, 'floodnet_test_eval100', dname) for s in (s0, s1))
            ids0, ids1 = sorted(cms[k0]), sorted(cms[k1])
            assert ids0 == ids1
            diff = draws[k1] - draws[k0]
            point = out['curve']['/'.join(map(str, k1))]['miou']['value'] - out['curve']['/'.join(map(str, k0))]['miou']['value']
            r = dict(value=point, lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)),
                     p_positive=float((diff > 0).mean()))
            out['paired'][f'{s0}->{s1}/{dname}'] = r
            md.append(f'| {s0} → {s1} shots | {dname} | {r["value"]:+.3f} [{r["lo"]:+.3f}, {r["hi"]:+.3f}] | {r["p_positive"]:.1%} |')
    # without the test photos overlapping the run's supports
    md += ['', '| run | split | decode | overlapping photos removed | mIoU all | mIoU without overlap |', '|---|---|---|---|---|---|']
    for key, cm in cms.items():
        ds, shot, sf, dname = key
        over = set(dupes[ds]['overlapping'].get(f'{shot}shot', [])) if (ds, shot) in (('floodnet', 20), ('floodnet', 50),
                                                                                          ('rescuenet', 20), ('rescuenet', 50)) else set()
        ids = sorted(cm)
        keep = [i for i in ids if i not in over]
        if len(keep) == len(ids):
            continue
        rng = np.random.default_rng(0)
        r, _ = boot(cm, keep, lambda c: summarize(c, NAMES[ds], WET[ds])['miou'], a.B, rng)
        allv = out['curve']['/'.join(map(str, key))]['miou']
        out['without_overlap']['/'.join(map(str, key))] = dict(removed=len(ids) - len(keep), miou=r)
        md.append(f'| {ds} {shot}-shot | {sf.split("_test_")[1]} | {dname} | {len(ids) - len(keep)} | {fmt(allv)} | {fmt(r)} |')
    # routing: stranded F1 (risk-aware, router v3, tuned decode)
    md += ['', '| routing (router v3, tuned decode, λ 5) | images | stranded F1 | without overlapping photos |', '|---|---|---|---|']
    for name, path, ds, shot in (('FloodNet 50-shot eval100', 'pred_floodnet_50shot_s0_stratified_tuned.json', 'floodnet', 50),
                                 ('FloodNet 50-shot full test', 'full/pred_floodnet_50shot_full_tuned.json', 'floodnet', 50),
                                 ('RescueNet 20-shot eval100', 'pred_rescuenet_20shot_s0_stratified_tuned.json', 'rescuenet', 20)):
        per = {r['id']: np.array(r['status_confusion']) for r in json.load(open(os.path.join(a.routing, path)))['per_image']
               if 'status_confusion' in r}
        over = set(dupes[ds]['overlapping'].get(f'{shot}shot', []))
        rng = np.random.default_rng(0)
        r, _ = boot(per, sorted(per), f1_from, a.B, rng)
        rng = np.random.default_rng(0)
        r2, _ = boot(per, sorted(i for i in per if i not in over), f1_from, a.B, rng)
        out['routing'][name] = dict(all=r, without_overlap=r2)
        md.append(f'| {name} | {r["n"]} | {fmt(r)} | {fmt(r2)} ({r["n"] - r2["n"]} removed) |')
    json.dump(out, open(os.path.join(a.out, 'ci.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'ci.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
