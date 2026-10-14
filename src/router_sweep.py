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
SPLITS = {'val30': ('floodnet_val_tune30', f'{RUN}_floodnet_val_tune30.npz'),
          'confirm100': ('floodnet_val_confirm100', f'{RUN}_floodnet_val_confirm100_fp16.npz'),
          'eval100': ('floodnet_test_eval100', f'{RUN}_floodnet_test_eval100_fp16.npz')}
GATES = ['none', 0.5, 0.3, 0.2, 0.15, 0.1]
REACH = [None, 40, 20, 10]


def one(args):
    a, split, cal, g, reach = args
    sf, cache = SPLITS[split]
    name = f'{split}_{"cal" if cal else "raw"}_g{g}_r{reach}'
    f = os.path.join(a.out, f'{name}.json')
    if not os.path.exists(f):
        cmd = [sys.executable, os.path.join(HERE, 'route_eval.py'), '--dataset', 'floodnet', '--split_file', sf,
               '--preds', os.path.join(a.results, 'preds', cache), '--decode', a.dec, '--lams', '5',
               '--offroad_maxwet', str(-1 if g == 'none' else g), '--out', f,
               '--router_pred', json.dumps({} if reach is None else {'offroad_reach': reach})]
        if cal:
            cmd += ['--calibration', a.calibration]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    r = json.load(open(f))['routing']['lam5']
    return dict(split=split, cal=cal, gate=g, reach=reach, f1=r['stranded_f1'], false_safe=r['false_safe_rate'],
                precision=r['stranded_precision'], recall=r['stranded_recall'], infeasible=r['infeasible_rate'],
                n_false_safe=r['n_false_safe'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--out', default='results/routing/router_sweep')
    ap.add_argument('--calibration', default='results/kaggle_v2/report/calibration/calibration.json')
    ap.add_argument('--splits', default='val30,eval100')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a.dec = os.path.join(a.out, f'decode_{RUN}.json')
    json.dump(json.load(open(os.path.join(a.results, 'decode', f'{RUN}.json')))['chosen'], open(a.dec, 'w'))
    splits = [s for s in a.splits.split(',') if os.path.exists(os.path.join(a.results, 'preds', SPLITS[s][1]))]
    jobs = [(a, s, c, g, r) for s, c, g, r in itertools.product(splits, (False, True), GATES, REACH)]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, jobs))
    json.dump(rows, open(os.path.join(a.out, 'router_sweep.json'), 'w'), indent=1)
    md = ['# Router options vs false-safe (FloodNet 50-shot, published decode, lam 5)', '',
          'published router = raw probabilities, gate 0.3, unlimited reach', '',
          '| wetness | gate | reach (m) | ' + ' | '.join(f'{s} F1 | {s} false safe' for s in splits) + ' |',
          '|---|---|---|' + '---|---|' * len(splits)]
    for c, g, r in itertools.product((False, True), GATES, REACH):
        cells = []
        for s in splits:
            x = next(x for x in rows if (x['split'], x['cal'], x['gate'], x['reach']) == (s, c, g, r))
            cells.append(f"{x['f1']:.3f} | {x['false_safe']:.3f} ({x['n_false_safe']})")
        mark = ' **(published)**' if (c, g, r) == (False, 0.3, None) else ''
        md.append(f"| {'calibrated' if c else 'raw'} | {g} | {r or '∞'}{mark} | " + ' | '.join(cells) + ' |')
    open(os.path.join(a.out, 'router_sweep.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
