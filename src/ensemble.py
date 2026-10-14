"""Chameleon + same-budget baseline, probabilities averaged (PLAN §16, A5 follow-up). CPU only, from the caches.

P = w * Chameleon (guided-filtered, as in its decode) + (1 - w) * DeepLabV3 baseline (softmax), both trained on the
same 50 photos. w and the per-class thresholds are chosen on val30 (decode_tune rule, drivable-precision floor as for
the submitted model); eval100 and the full test are reported for every w, the val30 choice marked. No SAM.

    python src/ensemble.py [--baseline baseline_deeplabv3_r50_50shot_s0]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import postproc_sweep as PS  # noqa: E402
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

CHAM = 'floodnet_50shot_s0_stratified'
NAMES = PS.NAMES['floodnet']
WS = (0.0, 0.25, 0.5, 0.75, 1.0)
SPLITS = (('floodnet_val_tune30', 'val', '', ''), ('floodnet_test_eval100', 'test', '_fp16', ''),
          ('floodnet_test_full', 'test', '_fp16', ''))


def load(a, sf, split, cs):
    zc = np.load(os.path.join(a.results, 'preds', f'{CHAM}_{sf}{cs}.npz'))
    zb = np.load(os.path.join(a.baselines, 'preds', f'{a.baseline}_{sf}.npz'))
    pb = {str(i): P for i, P in zip(zb['ids'], zb['probs'])}
    root = os.path.join(RR_DATA, 'floodnet_r768', split)
    out = []
    for i, P in zip(zc['ids'], zc['probs']):
        i = str(i)
        grey = np.asarray(Image.open(os.path.join(root, 'images', f'{i}.jpg')).convert('L'), np.float32) / 255
        Pc = np.clip(PS.guided_filter(grey, P.astype(np.float32) / 255, 2, 1e-3), 0, 1)
        g = gt_classes(np.asarray(Image.open(os.path.join(root, 'labels', f'{i}.png'))))
        out.append((Pc, pb[i].astype(np.float32) / 255, g))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--baselines', default='results/baselines')
    ap.add_argument('--baseline', default='baseline_deeplabv3_r50_50shot_s0')
    ap.add_argument('--out', default='results/baselines/report')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {sf: load(a, sf, split, cs) for sf, split, cs, _ in SPLITS}
    res = {}
    for w in WS:
        val = data['floodnet_val_tune30']
        S = PS.Scorer([w * c + (1 - w) * b for c, b, _ in val], [g for _, _, g in val], 6)
        tau, rv = PS.tune_tau(S)
        S.close()
        res[w] = dict(tau=tau, val30=rv['miou'])
        for sf in ('floodnet_test_eval100', 'floodnet_test_full'):
            cm = sum(confusion(decode(w * c + (1 - w) * b, tau=tau), g) for c, b, g in data[sf])
            r = summarize(cm, NAMES, WET['floodnet'])
            res[w][sf] = dict(miou=r['miou'], drivable_precision=r['drivable_precision'], **{n: r[f'iou_{n}'] for n in NAMES})
        print(w, round(res[w]['val30'], 4), round(res[w]['floodnet_test_eval100']['miou'], 4),
              round(res[w]['floodnet_test_full']['miou'], 4), flush=True)
    best = max(WS, key=lambda w: res[w]['val30'])
    json.dump(dict(baseline=a.baseline, chosen_w=best, results={str(k): v for k, v in res.items()}),
              open(os.path.join(a.out, f'ensemble_{a.baseline}.json'), 'w'), indent=1)
    md = [f'# Chameleon (weight w) + {a.baseline} (1 − w), probabilities averaged; τ tuned on val30, no SAM', '',
          '| w | val30 (tuned) | eval100 | full test | ' + ' | '.join(NAMES) + ' (full test) | drv P |', '|' + '---|' * 10]
    for w in WS:
        r = res[w]['floodnet_test_full']
        md.append(f"| {w}{' **(val30 choice)**' if w == best else ''} | {res[w]['val30']:.3f} | "
                  f"{res[w]['floodnet_test_eval100']['miou']:.3f} | **{r['miou']:.3f}** | " + ' | '.join(f'{r[n]:.3f}' for n in NAMES)
                  + f" | {r['drivable_precision']:.3f} |")
    md += ['', 'w = 1 is Chameleon alone (its submitted chain without SAM: full test .503); w = 0 the baseline with tuned τ.']
    open(os.path.join(a.out, f'ensemble_{a.baseline}.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
