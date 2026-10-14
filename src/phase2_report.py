"""Phase 2 tables from the per-run decode JSONs (run_decode.py) and routing (route_eval.py). CPU only.

    python src/phase2_report.py --results results/phase2/rr --out results/phase2/report [--runs <glob>] [--routing]

Per run: shot, training seed, support draw; val30 (chosen chain, no SAM); eval100 mIoU with tau .5, the label-free
prior-matched thresholds, the chosen chain and the chain + SAM; per-class IoU and drivable precision of the reported
decode; with --routing the stranded F1 and false-safe rate of router v3 on eval100 (route_eval.py, chosen chain;
cached as <results>/../routing/<run>.json). Grouped by shot: mean, sd, n. When a shot has a full seed x draw grid,
the variance is split into training seed, support draw and remainder (two-way ANOVA without replication: the
interaction is not separable from noise).
"""
import argparse
import glob
import itertools
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NAMES = ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water']
RN_NAMES = ['road_clear', 'road_blocked', 'bldg_intact', 'bldg_damaged', 'water']     # RescueNet's channels, same slots


def parse(run):
    m = re.match(r'(floodnet|rescuenet)_(\d+)shot_s(\d+)_(\w+?)(?:_sup(\d+))?(?:_(.+))?$', run)
    ds, shot, seed, sel, sup, tag = m.groups()
    return dict(dataset=ds, shot=int(shot), seed=int(seed), draw=int(sup) if sup else int(seed), sel=sel, tag=tag or '')


def routing(results, run, dec, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    f = os.path.join(out_dir, f'{run}.json')
    if not os.path.exists(f):
        preds = next(p for p in (os.path.join(results, 'preds', f'{run}_floodnet_test_eval100{s}.npz') for s in ('_fp16', ''))
                     if os.path.exists(p))
        tmp = os.path.join(out_dir, f'{run}_decode.json')
        json.dump(dec, open(tmp, 'w'))
        subprocess.run([sys.executable, os.path.join(HERE, 'route_eval.py'), '--dataset', 'floodnet', '--split_file',
                        'floodnet_test_eval100', '--preds', preds, '--decode', tmp, '--lams', '5', '--out', f],
                       check=True, stdout=subprocess.DEVNULL)
    r = json.load(open(f))['routing']['lam5']
    return dict(stranded_f1=r['stranded_f1'], false_safe=r['false_safe_rate'], infeasible=r['infeasible_rate'])


def anova2(grid):
    """grid[draw][seed] (full, no replication) -> mean squares and variance components"""
    Y = np.asarray(grid, float)
    a, b = Y.shape
    m = Y.mean()
    ss_r = b * ((Y.mean(1) - m) ** 2).sum()
    ss_c = a * ((Y.mean(0) - m) ** 2).sum()
    ss_e = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + m) ** 2).sum()
    ms_r, ms_c, ms_e = ss_r / (a - 1), ss_c / (b - 1), ss_e / max((a - 1) * (b - 1), 1)
    return dict(sd_draw=float(np.sqrt(max((ms_r - ms_e) / b, 0))), sd_seed=float(np.sqrt(max((ms_c - ms_e) / a, 0))),
                sd_rest=float(np.sqrt(ms_e)), sd_total=float(Y.std(ddof=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/phase2/rr')
    ap.add_argument('--out', default='results/phase2/report')
    ap.add_argument('--runs', default='*shot_s*', help='glob over decode/<run>.json')
    ap.add_argument('--routing', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.results, 'decode', f'{a.runs}.json'))):
        d = json.load(open(f))
        if 'splits' not in d and d.get('eval100'):      # decode_tune.py format (RescueNet): its own chosen rule, no SAM
            v = d['chosen']['variant']
            rows.append(dict(run=d['run'], **parse(d['run']), val30=d['val30'][v]['miou'], tau05=d['eval100']['default']['miou'],
                             prior=None, chosen=d['eval100'][v]['miou'], chosen_sam=None, decode=v,
                             drivable_precision=d['eval100'][v]['drivable_precision'],
                             **{n: d['eval100'][v][f'iou_{c}'] for n, c in zip(NAMES, RN_NAMES)}))
            continue
        if 'splits' not in d:                     # neither format
            continue
        run, ev = d['run'], d['splits'].get('floodnet_test_eval100')
        if not ev:
            continue
        rep = 'chosen_sam' if 'chosen_sam' in ev else 'chosen'
        row = dict(run=run, **parse(run), val30=d['val30']['chosen_no_sam']['miou'],
                   tau05=ev['tau0.5']['miou'], prior=ev['prior']['miou'], chosen=ev['chosen']['miou'],
                   chosen_sam=ev.get('chosen_sam', {}).get('miou'), decode=rep,
                   drivable_precision=ev[rep]['drivable_precision'], **{n: ev[rep][f'iou_{n}'] for n in NAMES})
        conf = d['splits'].get('floodnet_val_confirm100')
        if conf:
            row['confirm100'] = conf['chosen']['miou']
        if a.routing:
            row.update(routing(a.results, run, d['chosen'], os.path.join(os.path.dirname(a.out), 'routing')))
        rows.append(row)
    metrics = [k for k in ('chosen', 'chosen_sam', 'prior', 'tau05', 'val30', 'confirm100', 'stranded_f1', 'false_safe')
               if any(r.get(k) is not None for r in rows)]
    md = ['# Phase 2 runs (eval100 unless noted)', '',
          '| run | shot | seed | draw | val30 | τ .5 | prior (50 labels) | chosen | + SAM | ' +
          ' | '.join(NAMES) + ' | drv P' + (' | stranded F1 | false safe' if a.routing else '') + ' |',
          '|' + '---|' * (14 + (2 if a.routing else 0))]
    f3 = lambda x: '–' if x is None else f'{x:.3f}'                     # noqa: E731
    for r in sorted(rows, key=lambda r: (r['dataset'], r['shot'], r['draw'], r['seed'], r['tag'])):
        md.append(f"| {r['run']} | {r['shot']} | {r['seed']} | {r['draw']} | {f3(r['val30'])} | {f3(r['tau05'])} | "
                  f"{f3(r['prior'])} | {f3(r['chosen'])} | {f3(r['chosen_sam'])} | " + ' | '.join(f3(r[n]) for n in NAMES)
                  + f" | {f3(r['drivable_precision'])}" + (f" | {f3(r.get('stranded_f1'))} | {f3(r.get('false_safe'))}"
                                                           if a.routing else '') + ' |')
    summary = {}
    md += ['', '| shot (tag) | n | ' + ' | '.join(f'{k} mean ± sd' for k in metrics) + ' |', '|' + '---|' * (2 + len(metrics))]
    key = lambda r: (r['dataset'], r['shot'], r['tag'])                  # noqa: E731
    for (ds, shot, tag), grp in itertools.groupby(sorted(rows, key=key), key=key):
        grp = list(grp)
        shot = shot if ds == 'floodnet' else f'{ds} {shot}'
        s = {k: [r[k] for r in grp if r.get(k) is not None] for k in metrics}
        summary[f'{shot}{"_" + tag if tag else ""}'] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                                             n=len(v)) for k, v in s.items() if v}
        md.append(f'| {shot}{" (" + tag + ")" if tag else ""} | {len(grp)} | ' + ' | '.join(
            (f'{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}' if len(v) > 1 else f'{v[0]:.3f}') if v else '–' for v in s.values()) + ' |')
        seeds, draws = sorted({r['seed'] for r in grp}), sorted({r['draw'] for r in grp})
        cell = {(r['draw'], r['seed']): r for r in grp}
        if len(seeds) > 1 and len(draws) > 1 and len(cell) == len(seeds) * len(draws):
            for k in ('chosen', 'chosen_sam', 'stranded_f1'):
                if all(cell[(d, s)].get(k) is not None for d in draws for s in seeds):
                    v = anova2([[cell[(d, s)][k] for s in seeds] for d in draws])
                    summary[f'{shot}{"_" + tag if tag else ""}'][f'anova_{k}'] = v
                    md.append(f'|  ↳ {k}: sd from support draw {v["sd_draw"]:.3f}, training seed {v["sd_seed"]:.3f}, '
                              f'rest {v["sd_rest"]:.3f} (total {v["sd_total"]:.3f}) | | |')
    json.dump(dict(rows=rows, summary=summary), open(os.path.join(a.out, 'phase2.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'phase2.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
