"""Ablation of the router's physical rules (Batch 1, PLAN §6.1 v3): point vehicles, each rule alone, all four.
CPU only; eval100 of FloodNet 50-shot and RescueNet 20-shot, tuned decode, the dataset's off-road gate, lam 5.

    python src/physical_ablation.py [--out results/routing/physical]

The rules change passability on both maps (the labelled map too), so its status counts are listed per row.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import router as R  # noqa: E402

RUNS = {'floodnet': 'floodnet_50shot_s0_stratified', 'rescuenet': 'rescuenet_20shot_s0_stratified'}
ONE = [('truck width', {'truck_halfwidth': R.DEFAULTS['truck_halfwidth']}),
       ('boat width', {'boat_halfwidth': R.DEFAULTS['boat_halfwidth']}),
       ('car closing', {'car_gap': R.DEFAULTS['car_gap']}),
       ('canopy', {'canopy_gap': R.DEFAULTS['canopy_gap']}),
       ('corner rule', {'corner': True})]
VARIANTS = ([('point vehicles (before)', ['--point_vehicles'])]
            + [(f'+ {n} only', ['--point_vehicles', '--router', json.dumps(o)]) for n, o in ONE]
            + [('all four (new default)', [])])
COLS = ['stranded_precision', 'stranded_recall', 'stranded_f1', 'rsv_mean', 'unsafe_route_rate', 'infeasible_rate']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--out', default='results/routing/physical')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = ['| dataset | router | labelled map: truck / boat / foot / none | stranded P | R | F1 | RSV | unsafe | infeasible |',
             '|' + '---|' * 9]
    for ds, run in RUNS.items():
        dec = os.path.join(a.out, f'decode_{run}.json')
        json.dump(json.load(open(os.path.join(a.results, 'decode', f'{run}.json')))['chosen'], open(dec, 'w'))
        for k, (label, extra) in enumerate(VARIANTS):
            f = os.path.join(a.out, f'{ds}_{k}.json')
            subprocess.run([sys.executable, os.path.join(HERE, 'route_eval.py'), '--dataset', ds, '--split_file',
                            f'{ds}_test_eval100', '--preds', os.path.join(a.results, 'preds', f'{run}_{ds}_test_eval100_fp16.npz'),
                            '--decode', dec, '--lams', '5', '--out', f, *extra], check=True, stdout=subprocess.DEVNULL)
            d = json.load(open(f))
            r, g = d['routing']['lam5'], d['gt_status_counts']
            lines.append(f'| {ds} | {label} | {g["TRUCK"]} / {g["BOAT"]} / {g["FOOT"]} / {g["ISOLATED"]} | '
                         + ' | '.join(f'{r[c]:.3f}' for c in COLS) + ' |')
        print('done', ds, flush=True)
    open(os.path.join(a.out, 'ablation.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
