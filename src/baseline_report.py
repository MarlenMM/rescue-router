"""Same-budget baselines vs Chameleon (PLAN §16, A5). CPU only, from the caches of src/baseline_seg.py.

For every baseline run (results/baselines/preds/baseline_<arch>_<k>shot_s<seed>_<split>.npz):
  - full-test mIoU with its own decode (argmax over the 5 classes + other: no tuning labels) and with per-class
    thresholds tuned on val30 like Chameleon's (decode_tune rule; the same 80-label budget);
  - 95 % bootstrap CI over photos, and the paired difference to Chameleon's submitted maps on the same photos
    (same resamples), so "Chameleon beats a standard model trained on the same 50 photos" has an interval;
  - routing on eval100 (router v3, the published gate): stranded F1 and false-safe rate.

    python src/baseline_report.py [--baselines results/baselines] [--out results/baselines/report]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import postproc_sweep as PS  # noqa: E402
from bootstrap_ci import boot, f1_from  # noqa: E402
from router import false_safe_from  # noqa: E402
from rr.final_decode import decode_photo, load_chosen  # noqa: E402
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

NAMES = PS.NAMES['floodnet']
CHAM = 'floodnet_50shot_s0_stratified'
miou = lambda c: summarize(c, NAMES, WET['floodnet'])['miou']        # noqa: E731


def gts(split_file, split):
    ids = open(os.path.join(os.path.dirname(HERE), 'splits', f'{split_file}.txt')).read().split()
    root = os.path.join(RR_DATA, 'floodnet_r768', split, 'labels')
    return ids, {i: gt_classes(np.asarray(Image.open(os.path.join(root, f'{i}.png')))) for i in ids}


def per_image(probs_by_id, g, dec):
    return {i: confusion(decode_photo(probs_by_id[i], dec) if dec.get('argmax') else decode(probs_by_id[i], **dec), g[i])
            for i in g}


def load(path):
    z = np.load(path)
    return {str(i): P.astype(np.float32) / 255 for i, P in zip(z['ids'], z['probs'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baselines', default='results/baselines')
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--out', default='results/baselines/report')
    ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--cham_routing', default='results/routing/full/pred_floodnet_50shot_full_tuned.json')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids, g_full = gts('floodnet_test_full', 'test')
    _, g_val = gts('floodnet_val_tune30', 'val')
    # Chameleon's submitted maps (guided filter -> tau -> SAM, precomputed class maps) per photo
    dec_c = load_chosen(a.results, CHAM)
    z = np.load(os.path.join(a.results, 'preds', f'{CHAM}_floodnet_test_full_fp16.npz'))
    cm_c = {str(i): confusion(decode_photo(P.astype(np.float32) / 255, dec_c, split_file='floodnet_test_full', photo_id=str(i)),
                              g_full[str(i)]) for i, P in zip(z['ids'], z['probs'])}
    rows = []
    runs = sorted({re.sub(r'_floodnet_(test_full|test_eval100|val_tune30)$', '', os.path.basename(f)[:-4])
                   for f in glob.glob(os.path.join(a.baselines, 'preds', 'baseline_*_floodnet_test_full.npz'))})
    for run in runs:
        val = load(os.path.join(a.baselines, 'preds', f'{run}_floodnet_val_tune30.npz'))
        S = PS.Scorer([val[i] for i in g_val], [g_val[i] for i in g_val], 6)
        tau, _ = PS.tune_tau(S)
        S.close()
        test = load(os.path.join(a.baselines, 'preds', f'{run}_floodnet_test_full.npz'))
        row = dict(run=run, tau=tau)
        for name, dec in (('argmax', {'argmax': True}), ('tuned', {'tau': tau})):
            cms = per_image(test, g_full, dec)
            r, draws = boot(cms, ids, miou, a.B, np.random.default_rng(0))
            _, draws_c = boot(cm_c, ids, miou, a.B, np.random.default_rng(0))
            d = draws_c - draws
            row[name] = dict(miou=r, per_class={n: summarize(sum(cms.values()), NAMES, WET['floodnet'])[f'iou_{n}'] for n in NAMES},
                             chameleon_minus_baseline=dict(value=float(miou(sum(cm_c.values())) - r['value']),
                                                          lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5))))
        ev = os.path.join(a.baselines, 'preds', f'{run}_floodnet_test_eval100.npz')
        if os.path.exists(ev):
            f = os.path.join(a.out, f'routing_{run}.json')
            if not os.path.exists(f):
                dj = os.path.join(a.out, 'decode_argmax.json')
                json.dump({'argmax': True}, open(dj, 'w'))
                subprocess.run([sys.executable, os.path.join(HERE, 'route_eval.py'), '--dataset', 'floodnet', '--split_file',
                                'floodnet_test_eval100', '--preds', ev, '--decode', dj, '--lams', '5', '--out', f],
                               check=True, stdout=subprocess.DEVNULL)
            rr_ = json.load(open(f))['routing']['lam5']
            row['routing_eval100'] = dict(stranded_f1=rr_['stranded_f1'], false_safe=rr_['false_safe_rate'])
        # full-test routing, paired with Chameleon's published full-test routing (same photos, same resamples)
        f = os.path.join(a.out, f'routing_full_{run}.json')
        if not os.path.exists(f):
            dj = os.path.join(a.out, 'decode_argmax.json')
            json.dump({'argmax': True}, open(dj, 'w'))
            subprocess.run([sys.executable, os.path.join(HERE, 'route_eval.py'), '--dataset', 'floodnet', '--split_file',
                            'floodnet_test_full', '--preds', os.path.join(a.baselines, 'preds', f'{run}_floodnet_test_full.npz'),
                            '--decode', dj, '--lams', '5', '--out', f], check=True, stdout=subprocess.DEVNULL)
        pb = {r['id']: np.array(r['status_confusion']) for r in json.load(open(f))['per_image'] if 'status_confusion' in r}
        pc = {r['id']: np.array(r['status_confusion']) for r in json.load(open(a.cham_routing))['per_image'] if 'status_confusion' in r}
        common = sorted(set(pb) & set(pc))
        row['routing_full'] = {}
        for name, fn in (('stranded_f1', f1_from), ('false_safe', false_safe_from)):
            rb, db = boot(pb, common, fn, a.B, np.random.default_rng(0))
            rc, dc = boot(pc, common, fn, a.B, np.random.default_rng(0))
            row['routing_full'][name] = dict(baseline=rb, chameleon=rc['value'], diff=float(rb['value'] - rc['value']),
                                             lo=float(np.percentile(db - dc, 2.5)), hi=float(np.percentile(db - dc, 97.5)))
        rows.append(row)
        print(run, {k: round(row[k]['miou']['value'], 4) for k in ('argmax', 'tuned')}, row.get('routing_eval100'), flush=True)
    c_all = sum(cm_c.values())
    md = ['# Same-label-budget baselines vs Chameleon (FloodNet full test, 448 photos)', '',
          f'Chameleon (submitted, 50 training + 30 tuning labels): mIoU {miou(c_all):.3f}; per class '
          + ', '.join(f'{n} {summarize(c_all, NAMES, WET["floodnet"])[f"iou_{n}"]:.3f}' for n in NAMES), '',
          '| baseline | decode | mIoU [95 % CI] | Chameleon − baseline [95 % CI] | ' + ' | '.join(NAMES)
          + ' | eval100 stranded F1 | false safe |', '|' + '---|' * (6 + len(NAMES))]
    for r in rows:
        for name in ('argmax', 'tuned'):
            x = r[name]
            rt = r.get('routing_eval100', {}) if name == 'argmax' else {}
            md.append(f"| {r['run']} | {'own argmax (50 labels)' if name == 'argmax' else 'val30-tuned τ (80 labels)'} | "
                      f"{x['miou']['value']:.3f} [{x['miou']['lo']:.3f}, {x['miou']['hi']:.3f}] | "
                      f"{x['chameleon_minus_baseline']['value']:+.3f} [{x['chameleon_minus_baseline']['lo']:+.3f}, "
                      f"{x['chameleon_minus_baseline']['hi']:+.3f}] | " + ' | '.join(f"{x['per_class'][n]:.3f}" for n in NAMES)
                      + f" | {rt.get('stranded_f1', float('nan')):.3f} | {rt.get('false_safe', float('nan')):.3f} |")
    md += ['', '| baseline | full test, router v3 | baseline [95 % CI] | Chameleon | baseline − Chameleon [95 % CI] |', '|---|---|---|---|---|']
    for r in rows:
        for name, x in r['routing_full'].items():
            md.append(f"| {r['run']} | {name.replace('_', ' ')} | {x['baseline']['value']:.3f} [{x['baseline']['lo']:.3f}, "
                      f"{x['baseline']['hi']:.3f}] | {x['chameleon']:.3f} | {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] |")
    json.dump(rows, open(os.path.join(a.out, 'baselines.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'baselines.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
