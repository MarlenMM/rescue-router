"""Off-road gate sweep (PLAN §15 "Off-road gate", 0i): stranded-house metrics of the predicted maps for a range of
gates g (truck may leave the road only where p_wet < g), on val30 (where the gate is chosen) and eval100. CPU only.

    python src/gate_sweep.py                    # both: FloodNet 50-shot -> results/routing/gate/, RescueNet 20-shot
                                                #       -> results/routing/gate_rescuenet/ (gate.md + one JSON per gate)
Each point is src/route_eval.py with the run's val30-tuned decode, lam 5, router v3 defaults; 'none' = no gate.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEPS = {  # name: (dataset, run, out dir, gates)
    'fn50': ('floodnet', 'floodnet_50shot_s0_stratified', 'gate', ['none', 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0]),
    'rn20': ('rescuenet', 'rescuenet_20shot_s0_stratified', 'gate_rescuenet',
             ['none', 0.5, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]),
}
COLS = ['stranded_precision', 'stranded_recall', 'stranded_f1', 'reach_agreement', 'rsv_mean', 'unsafe_route_rate',
        'cost_ratio_median', 'infeasible_rate']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--routing', default='results/routing')
    ap.add_argument('--sweeps', default='fn50,rn20')
    a = ap.parse_args()
    for name in a.sweeps.split(','):
        ds, run, sub, gates = SWEEPS[name]
        out = os.path.join(a.routing, sub)
        os.makedirs(out, exist_ok=True)
        dec = os.path.join(out, f'decode_{run}.json')
        json.dump(json.load(open(os.path.join(a.results, 'decode', f'{run}.json')))['chosen'], open(dec, 'w'))
        lines = [f'# Off-road gate sweep: {run} (tuned decode, lam 5, router v3)', '',
                 '| split | gate g | stranded P | R | F1 | agreement | RSV | unsafe | cost ratio | infeasible | GT stranded '
                 '| pred stranded |', '|' + '---|' * 12]
        for split, sf, cache in (('val30', f'{ds}_val_tune30', f'{run}_{ds}_val_tune30.npz'),
                                 ('eval100', f'{ds}_test_eval100', f'{run}_{ds}_test_eval100_fp16.npz')):
            for g in gates:
                f = os.path.join(out, f'{name}_{split}_g{g}.json')
                subprocess.run([sys.executable, os.path.join(HERE, 'route_eval.py'), '--dataset', ds, '--split_file', sf,
                                '--preds', os.path.join(a.results, 'preds', cache), '--decode', dec, '--lams', '5',
                                '--offroad_maxwet', str(-1 if g == 'none' else g), '--out', f], check=True,
                               stdout=subprocess.DEVNULL)
                d = json.load(open(f))
                r = d['routing']['lam5']
                gt_str = sum(v for k, v in d['gt_status_counts'].items() if k != 'TRUCK')
                pr_str = sum(sum(p['pred_status'][1:]) for p in d['per_image'] if 'pred_status' in p)
                lines.append(f'| {split} | {g} | ' + ' | '.join('–' if r[c] is None else f'{r[c]:.3f}' for c in COLS)
                             + f' | {gt_str} | {pr_str} |')
            print('done', name, split, flush=True)
        open(os.path.join(out, 'gate.md'), 'w').write('\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
