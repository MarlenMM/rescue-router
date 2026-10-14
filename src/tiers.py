"""Confidence tiers for truck-reachable houses and the calibration of p_wet (Batch 2). CPU only, from caches.

    python src/tiers.py [--out results/routing/tiers]

The predicted map is routed twice: with the tuned off-road gate (per dataset) and with a cautious gate g_c (off-road
only where p_wet < g_c). A house truck-reachable even on the cautious map is a "confident truck" house; one
reachable only with the tuned gate is "probably truck" (check first); the rest keep their status (boat, on
foot, no route). Truck reach can only shrink with a smaller gate, so the tiers nest.
g_c is chosen per dataset on its own val30 (FloodNet 50-shot, RescueNet 20-shot): the largest candidate below the
tuned gate whose confident-truck tier is >= 99 % truck-reachable on the labelled map (at most 1 wrong dispatch in
100); if none reaches it, the candidate with the highest share. eval100 uses it unchanged. Calibration: per routing cell, predicted p_wet = mean(p1 + p4) against the labelled wet share of
the cell, over all cells and over the cells decoded 'other' (where the gate acts).
Writes tiers.json, tiers.md and figures/calibration_pwet.png (in the report folder).
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import router as R  # noqa: E402
from route_eval import PRED_CAUTIOUS_MAXWET, PRED_OFFROAD_MAXWET  # noqa: E402
from rr.metrics import gt_classes  # noqa: E402
from rr.final_decode import decode_photo, photo  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

CANDIDATES = (0.25, 0.2, 0.15, 0.1, 0.05)
TARGET = 0.99
RUNS = {'floodnet': 'floodnet_50shot_s0_stratified', 'rescuenet': 'rescuenet_20shot_s0_stratified'}
SPLITS = {'val30': ('{ds}_val_tune30', 'val', '{run}_{ds}_val_tune30.npz'),
          'eval100': ('{ds}_test_eval100', 'test', '{run}_{ds}_test_eval100_fp16.npz'),
          'confirm100': ('{ds}_val_confirm100', 'val', '{run}_{ds}_val_confirm100_fp16.npz')}   # FloodNet, Phase 2


def collect(ds, split, a, gates):
    """per building: labelled-map status and predicted status for every gate; per cell: p_wet vs labelled wet"""
    run = RUNS[ds]
    sf, folder, npz = (x.format(ds=ds, run=run) for x in SPLITS[split])
    z = np.load(os.path.join(a.results, 'preds', npz))
    P = dict(zip([str(i) for i in z['ids']], z['probs']))
    dec = {k: v for k, v in json.load(open(os.path.join(a.results, 'decode', f'{run}.json')))['chosen'].items()
           if k != 'variant'}
    rows, cal = [], {'all': ([], []), 'other': ([], [])}
    for i in [l.strip() for l in open(os.path.join(a.repo, 'splits', f'{sf}.txt')) if l.strip()]:
        gt = gt_classes(np.asarray(Image.open(os.path.join(a.data, f'{ds}_r768', folder, 'labels', f'{i}.png'))))
        gcls, gprob = R.cell_classes(gt), R.onehot_probs(gt)
        pred = decode_photo(P[i].astype(np.float32) / 255, dec, img=photo(a.data, ds, folder, i), split_file=sf, photo_id=i,
                            root=a.repo)
        pcls, pprob = R.cell_classes(pred), R.cell_probs(P[i])
        wet_p, wet_g = pprob[[1, 4]].sum(0).ravel(), gprob[[1, 4]].sum(0).ravel()
        for k, m in (('all', np.ones_like(wet_p, bool)), ('other', pcls.ravel() == R.OTHER)):
            cal[k][0].append(wet_p[m]), cal[k][1].append(wet_g[m])
        base = R.base_cells(gcls, ds)
        if base is None:
            continue
        blds = R.buildings_from_gt(gt, ds)
        gsol = R.Grid(gcls, gprob, ds).solve(base)
        psol = {g: R.Grid(pcls, pprob, ds, offroad_maxwet=g).solve(base) for g in gates}
        for b in blds:
            rows.append(dict(image=i, id=b['id'], gt=gsol.status(b), pred={g: psol[g].status(b) for g in gates}))
    return rows, {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in cal.items()}


def tier_table(rows, g_t, g_c):
    out = {}
    for name in ('CONFIDENT_TRUCK', 'PROBABLE_TRUCK', 'BOAT', 'FOOT', 'ISOLATED'):
        sel = [r for r in rows if R.tier(r['pred'][g_t], r['pred'][g_c]) == name]
        n = len(sel)
        truck = sum(r['gt'] == R.TRUCK for r in sel)
        out[name] = dict(n=n, gt_truck=truck, gt_truck_share=truck / n if n else None,
                         gt_status=[sum(r['gt'] == s for r in sel) for s in range(len(R.STATUS))])
    return out


def calibration(p, g, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = k == b
        if m.any():
            rows.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=int(m.sum()),
                             mean_pred=float(p[m].mean()), mean_true=float(g[m].mean())))
            ece += m.mean() * abs(p[m].mean() - g[m].mean())
    return dict(ece=float(ece), bins=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--data', default=RR_DATA)
    ap.add_argument('--repo', default=os.path.dirname(HERE))
    ap.add_argument('--out', default='results/routing/tiers')
    ap.add_argument('--fig', default='results/kaggle_v2/report/figures/calibration_pwet.png')
    ap.add_argument('--choose_on', default='val30',
                    help='splits whose houses choose the cautious gate (PLAN §16: val30,confirm100 once predicted)')
    a = ap.parse_args()
    choose = a.choose_on.split(',')
    os.makedirs(a.out, exist_ok=True)
    out = dict(target=TARGET, gates={}, tiers={}, calibration={})
    md = []
    data = {}
    for ds in RUNS:
        g_t = PRED_OFFROAD_MAXWET[ds]
        cands = tuple(g for g in CANDIDATES if g < g_t)
        for sp in SPLITS:
            sf, _, npz = (x.format(ds=ds, run=RUNS[ds]) for x in SPLITS[sp])
            if sp == 'confirm100' and not os.path.exists(os.path.join(a.results, 'preds', npz)):
                continue                                             # until Phase 2 predicts it
            data[(ds, sp)] = collect(ds, sp, a, (g_t,) + cands)
        rows, sweep = [r for sp in choose if (ds, sp) in data for r in data[(ds, sp)][0]], []
        for g in cands:                                              # choose the cautious gate on these houses
            t = tier_table(rows, g_t, g)['CONFIDENT_TRUCK']
            sweep.append(dict(gate=g, n=t['n'], share=t['gt_truck_share']))
        ok = [s for s in sweep if s['share'] is not None and s['share'] >= TARGET]
        g_c = max(s['gate'] for s in ok) if ok else max(sweep, key=lambda s: (s['share'] or 0, s['gate']))['gate']
        if choose == ['val30']:
            assert g_c == PRED_CAUTIOUS_MAXWET[ds], f'{ds}: chosen cautious gate {g_c}, route_eval says {PRED_CAUTIOUS_MAXWET[ds]}'
        elif g_c != PRED_CAUTIOUS_MAXWET[ds]:
            print(f'{ds}: cautious gate chosen on {a.choose_on}: {g_c} (published {PRED_CAUTIOUS_MAXWET[ds]}); '
                  'if adopted, update route_eval.PRED_CAUTIOUS_MAXWET and rebuild', flush=True)
        out['gates'][ds] = dict(tuned=g_t, cautious=g_c, reached_target=bool(ok), val30_sweep=sweep)
        md += [f'**{ds}**: tuned gate {g_t}; cautious gate chosen on {a.choose_on}: **{g_c}** '
               + ('(largest candidate with confident-truck share >= 99 %)' if ok else '(no candidate reaches 99 %: highest share)'),
               '', f'| gate ({a.choose_on} sweep) | confident-truck houses | truly truck-reachable |', '|---|---|---|']
        md += [f"| {s['gate']} | {s['n']} | {'–' if s['share'] is None else f'{s['share']:.1%}'} |" for s in sweep]
        md.append('')
    md += ['| run | split | tier | houses | truly truck-reachable (labelled map) | labelled map: truck / boat / foot / none |',
           '|---|---|---|---|---|---|']
    for (ds, sp), (rows, cal) in data.items():
        g_t, g_c = out['gates'][ds]['tuned'], out['gates'][ds]['cautious']
        t = tier_table(rows, g_t, g_c)
        out['tiers'][f'{ds}_{sp}'] = t
        out['calibration'][f'{ds}_{sp}'] = {k: calibration(*v) for k, v in cal.items()}
        for name, v in t.items():
            if v['n']:
                md.append(f"| {RUNS[ds].replace('_s0_stratified', '')} | {sp} | {name} | {v['n']} | "
                          f"{v['gt_truck_share']:.1%} | {' / '.join(map(str, v['gt_status']))} |")
    md += ['', '| run | split | cells | ECE (p_wet vs labelled wet share) |', '|---|---|---|---|']
    for key, c in out['calibration'].items():
        for k in ('all', 'other'):
            md.append(f"| {key} | | {'all cells' if k == 'all' else 'cells decoded other (gate domain)'} | {c[k]['ece']:.3f} |")
    json.dump(out, open(os.path.join(a.out, 'tiers.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'tiers.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, ds in zip(axs, RUNS):
        for sp, ls in (('val30', '--'), ('eval100', '-')):
            for k, col in (('all', 'tab:blue'), ('other', 'tab:orange')):
                b = out['calibration'][f'{ds}_{sp}'][k]['bins']
                ax.plot([x['mean_pred'] for x in b], [x['mean_true'] for x in b], ls, color=col, marker='o', ms=3,
                        label=f"{sp}, {'all cells' if k == 'all' else 'decoded other'} "
                              f"(ECE {out['calibration'][f'{ds}_{sp}'][k]['ece']:.3f})")
        ax.plot([0, 1], [0, 1], ':', color='grey')
        ax.axvline(out['gates'][ds]['tuned'], color='k', lw=0.8)
        ax.axvline(out['gates'][ds]['cautious'], color='k', lw=0.8, ls='--')
        ax.set(title=RUNS[ds].replace('_s0_stratified', ''), xlabel='predicted p_wet (cell mean)',
               ylabel='labelled wet share of the cell', xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
    fig.suptitle('Calibration of p_wet; gates per dataset: tuned (solid), cautious (dashed)', fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.fig), exist_ok=True)
    fig.savefig(a.fig, dpi=130)


if __name__ == '__main__':
    main()
