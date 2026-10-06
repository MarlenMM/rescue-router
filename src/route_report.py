"""Routing evaluation of every run's eval100 prediction cache (PLAN §5.4), default and tuned decode,
risk-aware and plain truck costs. CPU only; wraps route_eval.py.

    python src/route_report.py --results results/kaggle_v2/rr --out results/routing

Writes results/routing/pred_<run>_{default,tuned}.json and routing.md (one table).
"""

import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ['stranded_precision', 'stranded_recall', 'stranded_f1', 'reach_agreement', 'rsv_mean', 'unsafe_route_rate',
        'cost_ratio_median', 'infeasible_rate']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--out', default='results/routing')
    ap.add_argument('--table', default='routing.md')
    ap.add_argument('--extra', default='', help='extra route_eval.py arguments, e.g. "--offroad_maxwet -1"')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = sorted((json.load(open(f)) for f in glob.glob(os.path.join(a.results, 'decode', '*.json'))
                   if '_tta' not in os.path.basename(f)),                 # test-time flips: own report (src/tta.py)
                  key=lambda d: (d['dataset'], int(d['run'].split('_')[1][:-4])))
    lines = ['| run | decode | truck cost | stranded P | stranded R | stranded F1 | status agreement | RSV | unsafe routes '
             '| cost ratio (median) | infeasible |', '|' + '---|' * 11]
    for d in runs:
        run, ds = d['run'], d['dataset']
        dec = os.path.join(a.out, f'decode_{run}.json')
        json.dump(d['chosen'], open(dec, 'w'))
        preds = os.path.join(a.results, 'preds', f'{run}_{ds}_test_eval100_fp16.npz')
        for name, extra in (('default', []), ('tuned', ['--decode', dec])):
            out = os.path.join(a.out, f'pred_{run}_{name}.json')
            subprocess.run([sys.executable, os.path.join(HERE, 'route_eval.py'), '--dataset', ds, '--split_file',
                            f'{ds}_test_eval100', '--preds', preds, '--out', out, *extra, *a.extra.split()], check=True,
                           stdout=subprocess.DEVNULL)
            r = json.load(open(out))['routing']
            for lam, lab in (('lam5', 'risk-aware λ=5'), ('lam0', 'plain λ=0')):
                v = [r[lam][c] for c in COLS]
                lines.append(f'| {run.replace("_s0_stratified", "")} | {"τ=0.5" if name == "default" else d["chosen"]["variant"]} '
                             f'| {lab} | ' + ' | '.join('–' if x is None else f'{x:.3f}' for x in v) + ' |')
        print('done', run, flush=True)
    open(os.path.join(a.out, a.table), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
