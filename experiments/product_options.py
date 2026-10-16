"""The two product-model options after Phase 2, paired on the FloodNet full test (PLAN §16.8; information for Marlen's
decision, not a choice: the registered rule chose the mix on val30 + confirm100).

  mix       product_choice: the registered choice, 0.5 Chameleon + 0.5 self-trained baseline with flips (GPU + CPU)
  CPU only  product_choice_baselines: the registered baseline-only choice, seed ensemble (3) with flips (CPU)
Both from `src/ensemble.py --paired [--baselines_only]` (caches, val30-tuned thresholds, router v3 routing files).
mIoU and routing (stranded F1, false-safe rate), 2,000 resamples over photos, the same resamples for both.

    python experiments/product_options.py     # -> results/baselines/report/product_options.{json,md}
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
from baseline_report import Labels, miou_fn, per_photo_status  # noqa: E402
from bootstrap_ci import boot, f1_from  # noqa: E402
from router import false_safe_from  # noqa: E402
from rr.metrics import confusion, decode  # noqa: E402

REP = os.path.join(REPO, 'results', 'baselines', 'report')
OPTS = {'mix (registered choice)': 'product_choice', 'CPU only (registered baseline-only choice)': 'product_choice_baselines'}


def main():
    g = Labels()('floodnet', 'floodnet_test_full')
    cms, st = {}, {}
    for name, run in OPTS.items():
        tau = json.load(open(os.path.join(REP, f'decode_{run}.json')))['tau']
        z = np.load(os.path.join(REPO, 'results', 'baselines', 'preds', f'{run}_floodnet_test_full.npz'))
        cms[name] = {str(i): confusion(decode(P.astype(np.float32) / 255, tau=tau), g[str(i)]) for i, P in zip(z['ids'], z['probs'])}
        st[name] = per_photo_status(os.path.join(REP, f'routing_full_{run}.json'))
    (a, b) = OPTS
    out, md = {}, ['# Product-model options on the FloodNet full test (paired, 2,000 resamples)', '',
                   f'| metric | {a} | {b} | mix − CPU only [95 % CI] |', '|---|---|---|---|']
    for metric, data, fn in (('mIoU', cms, miou_fn('floodnet')), ('stranded F1', st, f1_from),
                             ('false-safe rate', st, false_safe_from)):
        ids = sorted(set(data[a]) & set(data[b]))
        (ra, da), (rb, db) = (boot(data[k], ids, fn, 2000, np.random.default_rng(0)) for k in (a, b))
        d = da - db
        out[metric] = dict(n=len(ids), mix=ra, cpu=rb, diff=dict(value=ra['value'] - rb['value'],
                                                                 lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5))))
        x = out[metric]['diff']
        md.append(f"| {metric} ({len(ids)} photos) | {ra['value']:.3f} [{ra['lo']:.3f}, {ra['hi']:.3f}] | "
                  f"{rb['value']:.3f} [{rb['lo']:.3f}, {rb['hi']:.3f}] | **{x['value']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]** |")
    json.dump(out, open(os.path.join(REP, 'product_options.json'), 'w'), indent=1)
    open(os.path.join(REP, 'product_options.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
