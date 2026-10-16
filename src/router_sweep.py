"""Router options against the false-safe error (PLAN §16, A2/C2). CPU only, from the caches.

false_safe.py: 83-87 % of the falsely truck-reachable houses fail on water or flooded road that the model left
unlabelled ('other'), which the router lets a truck cross as dry off-road land. Options swept (FloodNet 50-shot,
published decode, lam 5):
  H1 calibrated wetness: the router reads Platt-scaled probabilities (calibrate.py; raw water p is too low)
  gate g: off-road only where p_wet < g (published 0.3)
  H2 reach D: off-road only within D m of a predicted road (yards and driveways, not open country)
All three act on the predicted map only (route_eval --router_pred); the labelled-map reference is the published one.
Choice rule (pre-registered, PLAN §16.5): on val30 + confirm100 (val30 alone has 52 stranded houses and its F1
saturates at .98), max stranded F1, adopted only if F1 does not drop and the false-safe rate does not rise vs the
published router; reported on eval100 / full test. Until confirm100 predictions exist this table is information only.

    python src/router_sweep.py [--splits val30,eval100] [--workers 8]
    python src/router_sweep.py --results results/baselines --run baseline_deeplabv3_r50_50shot_s0 --argmax --raw_only \
        --no_reach --splits val30,confirm100,eval100 --out results/baselines/report/router_sweep    # the baseline's gate
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = 'floodnet_50shot_s0_stratified'
SPLITS = {'val30': 'floodnet_val_tune30', 'confirm100': 'floodnet_val_confirm100', 'eval100': 'floodnet_test_eval100',
          'full': 'floodnet_test_full'}


def cache(results, run, sf):
    for suffix in ('', '_fp16'):
        p = os.path.join(results, 'preds', f'{run}_{sf}{suffix}.npz')
        if os.path.exists(p):
            return p
    return None
GATES = ['none', 0.5, 0.3, 0.2, 0.15, 0.1]
REACH = [None, 40, 20, 10]


def one(args):
    a, split, cal, g, reach = args
    sf = SPLITS[split]
    name = f'{split}_{"cal" if cal else "raw"}_g{g}_r{reach}'
    f = os.path.join(a.out, f'{name}.json')
    if not os.path.exists(f):
        cmd = [sys.executable, os.path.join(HERE, 'route_eval.py'), '--dataset', 'floodnet', '--split_file', sf,
               '--preds', cache(a.results, a.run, sf), '--decode', a.dec, '--lams', '5',
               '--offroad_maxwet', str(-1 if g == 'none' else g), '--out', f,
               '--router_pred', json.dumps({} if reach is None else {'offroad_reach': reach})]
        if cal:
            cmd += ['--calibration', a.calibration]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    r = json.load(open(f))['routing']['lam5']
    return dict(split=split, cal=cal, gate=g, reach=reach, f1=r['stranded_f1'], false_safe=r['false_safe_rate'],
                precision=r['stranded_precision'], recall=r['stranded_recall'], infeasible=r['infeasible_rate'],
                n_false_safe=r['n_false_safe'], cm=r['status_confusion'])


def pooled(rows, splits, cal, g, reach):
    """stranded F1 and false-safe rate of the summed status confusion over several splits"""
    import numpy as np
    sys.path.insert(0, HERE)
    import router as R
    cm = sum(np.array(next(x for x in rows if (x['split'], x['cal'], x['gate'], x['reach']) == (s, cal, g, reach))['cm'])
             for s in splits)
    return R.summarize_routing(cm, [])['stranded_f1'], R.false_safe_from(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--out', default='results/routing/router_sweep')
    ap.add_argument('--calibration', default='results/kaggle_v2/report/calibration/calibration.json')
    ap.add_argument('--splits', default='val30,eval100')
    ap.add_argument('--workers', type=int, default=8, help='at most; fewer when they do not fit in free memory (rr.mem)')
    ap.add_argument('--run', default=RUN, help='any run with caches in --results (e.g. a baseline in results/baselines)')
    ap.add_argument('--argmax', action='store_true', help='decode by argmax (softmax baselines) instead of decode/<run>.json')
    ap.add_argument('--raw_only', action='store_true', help='raw probabilities only (the calibration is fitted to Chameleon)')
    ap.add_argument('--no_reach', action='store_true', help='unlimited off-road reach only')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a.dec = os.path.join(a.out, f'decode_{a.run}.json')
    json.dump({'argmax': True} if a.argmax else json.load(open(os.path.join(a.results, 'decode', f'{a.run}.json')))['chosen'],
              open(a.dec, 'w'))
    splits = [s for s in a.splits.split(',') if cache(a.results, a.run, SPLITS[s])]
    cals, reaches = (False,) if a.raw_only else (False, True), [None] if a.no_reach else REACH
    jobs = [(a, s, c, g, r) for s, c, g, r in itertools.product(splits, cals, GATES, reaches)]
    sys.path.insert(0, HERE)
    from rr import mem
    # each route_eval holds ~4.6 GB on confirm100 / eval100: 8 of them next to a baseline training froze the 24 GB Mac
    with ThreadPoolExecutor(mem.workers(a.workers, 5.0, 'route_eval workers')) as ex:
        rows = list(ex.map(one, jobs))
    json.dump(rows, open(os.path.join(a.out, 'router_sweep.json'), 'w'), indent=1)
    who = 'FloodNet 50-shot, published decode' if (a.run, a.argmax) == (RUN, False) else \
        f'{a.run}, {"argmax" if a.argmax else "its chosen"} decode'
    md = [f'# Router options vs false-safe ({who}, lam 5)', '',
          'published router = raw probabilities, gate 0.3, unlimited reach', '',
          '| wetness | gate | reach (m) | ' + ' | '.join(f'{s} F1 | {s} false safe' for s in splits) + ' |',
          '|---|---|---|' + '---|---|' * len(splits)]
    for c, g, r in itertools.product(cals, GATES, reaches):
        cells = []
        for s in splits:
            x = next(x for x in rows if (x['split'], x['cal'], x['gate'], x['reach']) == (s, c, g, r))
            cells.append(f"{x['f1']:.3f} | {x['false_safe']:.3f} ({x['n_false_safe']})")
        mark = ' **(published)**' if (c, g, r) == (False, 0.3, None) else ''
        md.append(f"| {'calibrated' if c else 'raw'} | {g} | {r or '∞'}{mark} | " + ' | '.join(cells) + ' |')
    if {'val30', 'confirm100'} <= set(splits):                   # the pre-registered choice (PLAN §16, PROTOCOL.md)
        md += ['', '## Gate choice on val30 + confirm100 (raw wetness, unlimited reach; pre-registered rule)', '',
               '| gate | pooled F1 | pooled false safe |', '|---|---|---|']
        pool = {g: pooled(rows, ('val30', 'confirm100'), False, g, None) for g in GATES}
        md += [f'| {g} | {f:.3f} | {fs:.3f} |' for g, (f, fs) in pool.items()]
        top = max(round(f, 6) for f, _ in pool.values())
        tied = [g for g in GATES if round(pool[g][0], 6) == top]
        best = 0.3 if 0.3 in tied else min(tied, key=lambda g: pool[g][1])   # a tie keeps the published gate
        adopt = best != 0.3 and pool[best][1] <= pool[0.3][1]
        md += ['', f'Max pooled F1 {top:.3f}: gate(s) {tied}. ' + (f'**Adopt {best}** (higher F1, false safe not higher than 0.3).'
                                                               if adopt else '**Keep 0.3**.')]
    open(os.path.join(a.out, 'router_sweep.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
