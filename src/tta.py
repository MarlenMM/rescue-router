"""Test-time flips for FloodNet 50-shot (Batch 7). CPU only, from the prediction caches.

    python src/tta.py build  --flips results/kaggle_v2/flips/rr/preds      # write the averaged (tta4) caches
    python src/decode_tune.py --runs floodnet_50shot_s0_stratified_tta4    # re-tune the decode on val30 (TTA)
    python src/tta.py report                                                # compare with the plain model

build: averages the unflipped cache with the h / v / hv flipped caches (src/predict.py --flip; probabilities already
flipped back) into run 'floodnet_50shot_s0_stratified_tta4': preds/<run>_floodnet_val_tune30.npz, ..._test_full_fp16
and ..._test_eval100_fp16 (subset of the full test, same images). The unflipped val30 cache is the selected snapshot
(step 1000, preds/<run>_floodnet_val_tune30.npz); the unflipped test caches are the existing eval100 and full-test caches.
report: mIoU (tau 0.5 and each run's own val30-tuned decode) on val30, eval100 and the full test, with bootstrap 95 %
CIs and the paired difference tta4 - plain over images; per-class IoU; agreement of the flipped views with the
unflipped one (a mis-registered flip would show up here); post-hoc: TTA probabilities with the plain decode; routing on
the full test if results/routing/tta/ holds it (src/rebuild_all.py runs route_eval there first); the decision.
Writes results/kaggle_v2/report/tta/tta.{md,json}.
The decision whether TTA replaces the headline is taken on val30 (tuned mIoU), then reported on the test splits.
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

RUN = 'floodnet_50shot_s0_stratified'
TTA = RUN + '_tta4'
FLIPS = ('h', 'v', 'hv')
NAMES = ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water']
REPO = os.path.dirname(HERE)


def ids_of(split_file):
    return [l.strip() for l in open(os.path.join(REPO, 'splits', f'{split_file}.txt')) if l.strip()]


def load(path):
    z = np.load(path)
    return dict(zip([str(i) for i in z['ids']], z['probs']))


def write(path, d, ids):
    np.savez_compressed(path, probs=np.stack([d[i] for i in ids]), ids=np.array(ids))
    print('wrote', path, len(ids))


def average(views):
    """mean of uint8 probability maps, rounded back to uint8 (the caches' format)"""
    return np.round(np.mean([v.astype(np.float32) for v in views], 0)).astype(np.uint8)


def build(a):
    P = os.path.join(a.results, 'preds')
    flip = lambda split_files, f: {k: v for s in split_files  # noqa: E731
                                   for k, v in load(os.path.join(a.flips, f'{RUN}_{s}_fp16_flip{f}.npz')).items()}
    v0 = load(os.path.join(P, f'{RUN}_floodnet_val_tune30.npz'))  # = the selected snapshot's (step 1000) cache
    vf = [flip(['floodnet_val_tune30'], f) for f in FLIPS]
    ids = ids_of('floodnet_val_tune30')
    write(os.path.join(P, f'{TTA}_floodnet_val_tune30.npz'), {i: average([v0[i]] + [d[i] for d in vf]) for i in ids}, ids)
    # test: unflipped = the full-test cache (eval100 is a subset; its own cache is compared in the report)
    t0 = load(os.path.join(P, f'{RUN}_floodnet_test_full_fp16.npz'))
    parts = ['floodnet_test_eval100', 'floodnet_test_rest348_a', 'floodnet_test_rest348_b']
    tf = [flip(parts, f) for f in FLIPS]
    full = ids_of('floodnet_test_full')
    avg = {i: average([t0[i]] + [d[i] for d in tf]) for i in full}
    write(os.path.join(P, f'{TTA}_floodnet_test_full_fp16.npz'), avg, full)
    write(os.path.join(P, f'{TTA}_floodnet_test_eval100_fp16.npz'), avg, ids_of('floodnet_test_eval100'))


def per_image(probs, split, data, dec):
    out = {}
    for i, p in probs.items():
        gt = gt_classes(np.asarray(Image.open(os.path.join(data, 'floodnet_r768', split, 'labels', f'{i}.png'))))
        out[i] = confusion(decode(p.astype(np.float32) / 255, **dec), gt)
    return out


def miou(cm):
    return summarize(cm, NAMES, WET['floodnet'])['miou']


def boot(cmsA, cmsB, ids, B, rng):
    """point + 95 % CI of mIoU(A), mIoU(B) and the paired difference B - A over images"""
    A, Bm = np.stack([cmsA[i] for i in ids]), np.stack([cmsB[i] for i in ids])
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    va = np.array([miou(A[k].sum(0)) for k in idx])
    vb = np.array([miou(Bm[k].sum(0)) for k in idx])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return dict(plain=dict(value=miou(A.sum(0)), ci=ci(va)), tta=dict(value=miou(Bm.sum(0)), ci=ci(vb)),
                diff=dict(value=miou(Bm.sum(0)) - miou(A.sum(0)), ci=ci(vb - va)), n=len(ids))


def report(a):
    P = os.path.join(a.results, 'preds')
    dec = {k: json.load(open(os.path.join(a.results, 'decode', f'{k}.json')))['chosen'] for k in (RUN, TTA)}
    for k in dec:
        dec[k] = {kk: vv for kk, vv in dec[k].items() if kk != 'variant'}
    rng = np.random.default_rng(0)
    out = dict(decode={k: v for k, v in dec.items()}, splits={})
    sources = {
        'val30': ('val', load(os.path.join(P, f'{RUN}_floodnet_val_tune30.npz')),
                  load(os.path.join(P, f'{TTA}_floodnet_val_tune30.npz'))),
        'full448': ('test', load(os.path.join(P, f'{RUN}_floodnet_test_full_fp16.npz')),
                    load(os.path.join(P, f'{TTA}_floodnet_test_full_fp16.npz'))),
    }
    ev = ids_of('floodnet_test_eval100')
    for name, (split, plain, tta) in sources.items():
        for dname in ('tau0.5', 'tuned'):
            dA = dict(tau=0.5) if dname == 'tau0.5' else dec[RUN]
            dB = dict(tau=0.5) if dname == 'tau0.5' else dec[TTA]
            cA, cB = per_image(plain, split, a.data, dA), per_image(tta, split, a.data, dB)
            subsets = {name: list(plain)} if name == 'val30' else {'full448': list(plain), 'eval100': ev}
            for sname, ids in subsets.items():
                r = boot(cA, cB, ids, a.B, rng)
                sA, sB = (summarize(sum(c[i] for i in ids), NAMES, WET['floodnet']) for c in (cA, cB))
                r['per_class'] = {k: dict(plain=sA[f'iou_{k}'], tta=sB[f'iou_{k}']) for k in NAMES}
                r['drivable_precision'] = dict(plain=sA['drivable_precision'], tta=sB['drivable_precision'])
                out['splits'].setdefault(sname, {})[dname] = r
    # the eval100 cache vs the full-test cache on the same 100 images (fp16 determinism of the two runs)
    e0 = load(os.path.join(P, f'{RUN}_floodnet_test_eval100_fp16.npz'))
    f0 = sources['full448'][1]
    out['eval100_cache_vs_full_cache_max_abs_diff'] = int(max(np.abs(e0[i].astype(int) - f0[i].astype(int)).max() for i in ev))
    # flipped views vs the unflipped view (decode agreement): a mis-registered flip would be far below the others
    if a.flips and os.path.isdir(a.flips):
        agree = {}
        for f in FLIPS:
            d = load(os.path.join(a.flips, f'{RUN}_floodnet_test_eval100_fp16_flip{f}.npz'))
            agree[f] = float(np.mean([(decode(d[i] / 255., tau=0.5) == decode(e0[i] / 255., tau=0.5)).mean() for i in ev]))
        out['flip_vs_plain_pixel_agreement_eval100'] = agree
    # post-hoc only (not used for the decision): TTA probabilities with the plain model's decode
    plain, tta = sources['full448'][1], sources['full448'][2]
    cA, cB = per_image(plain, 'test', a.data, dec[RUN]), per_image(tta, 'test', a.data, dec[RUN])
    out['posthoc_tta_with_plain_decode'] = {s: boot(cA, cB, ids, a.B, np.random.default_rng(0))
                                            for s, ids in (('full448', list(plain)), ('eval100', ev))}
    cv = per_image(sources['val30'][2], 'val', a.data, dec[RUN])
    out['posthoc_tta_with_plain_decode']['val30_tta'] = miou(sum(cv.values()))
    rt = {}
    for name, f in (('plain, tuned decode (headline)', 'full/pred_floodnet_50shot_full_tuned.json'),
                    ('plain, tau 0.5', 'full/pred_floodnet_50shot_full_default.json'),
                    ('TTA, own tuned decode', 'tta/pred_floodnet_50shot_tta4_full_tuned.json'),
                    ('TTA, tau 0.5', 'tta/pred_floodnet_50shot_tta4_full_default.json')):
        if os.path.exists(os.path.join(a.routing, f)):
            rt[name] = json.load(open(os.path.join(a.routing, f)))['routing']['lam5']
    out['routing_full'] = rt
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, 'tta.json'), 'w'), indent=1)
    f = lambda d: f"{d['value']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"  # noqa: E731
    lines = ['# Test-time flips (FloodNet 50-shot): plain vs mean of 4 views (identity, h, v, hv)', '',
             f'Bootstrap 95 % CIs over images (B = {a.B}); the difference is paired (same resamples). Tuned = each '
             "run's own decode, tuned on its own val30 (src/decode_tune.py).", '',
             '| split | decode | plain | TTA | TTA - plain |', '|---|---|---|---|---|']
    for s in ('val30', 'eval100', 'full448'):
        for dname, r in out['splits'][s].items():
            lines.append(f"| {s} | {dname} | {f(r['plain'])} | {f(r['tta'])} | {r['diff']['value']:+.3f} "
                         f"[{r['diff']['ci'][0]:+.3f}, {r['diff']['ci'][1]:+.3f}] |")
    lines += ['', 'Per class (full test, tuned): ' + ', '.join(
        f"{k} {v['plain']:.3f} → {v['tta']:.3f}" for k, v in out['splits']['full448']['tuned'].get('per_class', {}).items()), '',
        f"Decodes: plain {dec[RUN]}, TTA {dec[TTA]}.", '',
        f"eval100 cache vs full-test cache on the same images: max |diff| {out['eval100_cache_vs_full_cache_max_abs_diff']} "
        '(uint8 probability units).']
    if 'flip_vs_plain_pixel_agreement_eval100' in out:
        lines.append('Pixel agreement of each flipped view with the unflipped one (eval100, tau 0.5): ' + ', '.join(
            f'{k} {v:.3f}' for k, v in out['flip_vs_plain_pixel_agreement_eval100'].items()) + '.')
    ph = out['posthoc_tta_with_plain_decode']
    d = lambda r: f"{r['diff']['value']:+.3f} [{r['diff']['ci'][0]:+.3f}, {r['diff']['ci'][1]:+.3f}]"  # noqa: E731
    lines += ['', '## Post-hoc (not used for any decision)',
              f"TTA probabilities decoded with the **plain** model's val30-tuned decode {dec[RUN]}: full test "
              f"{ph['full448']['tta']['value']:.3f} vs plain {ph['full448']['plain']['value']:.3f}, paired diff {d(ph['full448'])}; "
              f"eval100 {ph['eval100']['tta']['value']:.3f} vs {ph['eval100']['plain']['value']:.3f}, diff {d(ph['eval100'])}; "
              f"val30 {ph['val30_tta']:.3f} (TTA's own decode: {out['splits']['val30']['tuned']['tta']['value']:.3f}, so val30 "
              'prefers the own decode). Picking this combination because of its test score would be test-set selection, so it '
              'is reported only as an observation: averaging the views helps a little when the decode stays conservative; the '
              'TTA-tuned decode lowered the road thresholds (full-test drivable precision '
              f"{out['splits']['full448']['tuned']['drivable_precision']['tta']:.3f} vs "
              f"{out['splits']['full448']['tuned']['drivable_precision']['plain']:.3f}) and lost more than the averaging gained."]
    if rt:
        lines += ['', '## Routing on the full test (router v3, lam 5, gate 0.3; results/routing/tta/)',
                  '| maps | stranded P | R | F1 | unsafe routes | agreement |', '|---|---|---|---|---|---|']
        lines += [f"| {k} | {v['stranded_precision']:.3f} | {v['stranded_recall']:.3f} | {v['stranded_f1']:.3f} | "
                  f"{v['unsafe_route_rate']:.3f} | {v['reach_agreement']:.3f} |" for k, v in rt.items()]
    sv, sf = out['splits']['val30']['tuned'], out['splits']['full448']['tuned']
    adopt = sv['diff']['ci'][0] > 0 and sf['diff']['ci'][0] > 0
    lines += ['', '## Decision',
              'Rule fixed before looking at test: adopt TTA if its val30-tuned mIoU beats the plain one on val30 and the paired '
              f"full-test difference with each run's own decode is clearly positive. val30 {d(sv)}, full test {d(sf)}. "
              + ('**Adopted.**' if adopt else '**Not adopted: the headline stays the plain model.**')
              + ' Cost: 4x inference (about 70 s instead of 17 s per photo on a T4).']
    open(os.path.join(a.out, 'tta.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'report'])
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--flips', default='results/kaggle_v2/flips/rr/preds')
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--out', default='results/kaggle_v2/report/tta')
    ap.add_argument('--routing', default='results/routing')
    ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args()
    build(a) if a.cmd == 'build' else report(a)


if __name__ == '__main__':
    main()
