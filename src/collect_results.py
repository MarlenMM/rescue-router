"""Collect the queue results (PLAN §15 step 2): per-run val30 (best step) + eval100 tables and the
data-efficiency curve. CPU only, reads metrics/*.json and logs/*.json of a downloaded queue output.

    python src/collect_results.py --results results/kaggle_v2/rr --out results/kaggle_v2/report

Writes results.md (tables), results.json and curve_floodnet.png.
"""

import argparse
import glob
import json
import os
import re

COLS = ['iou_0', 'iou_1', 'iou_2', 'iou_3', 'iou_4', 'miou', 'drivable_precision', 'drivable_recall', 'wet_iou']
NAMES = {'floodnet': ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water'],
         'rescuenet': ['road_clear', 'road_blocked', 'bldg_intact', 'bldg_damaged', 'water']}
# FloodNet paper Table 4 (test, 713x713, 1,445 training labels), PLAN §5.2
PSPNET = {'road_clear': 0.9118, 'road_flooded': 0.8216, 'bldg_flooded': 0.6893, 'bldg_dry': 0.8975, 'water': 0.9200}
# reference palette, categorical slots 1-5 (light mode)
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
TEXT, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def row(m, names):
    return [m.get(f'iou_{n}') for n in names] + [m.get(k) for k in COLS[5:]]


def fmt(v):
    return '–' if v is None else f'{v:.3f}'


def collect(res_dir):
    runs = {}
    for f in sorted(glob.glob(os.path.join(res_dir, 'metrics', '*.json'))):
        name = os.path.basename(f)[:-5]
        if '_step' in name:                       # per-snapshot val of a --defer_val run (select_ckpt.py)
            continue
        m = json.load(open(f))
        mo = re.match(r'(floodnet|rescuenet)_(\d+)shot_s(\d+)_(\w+?)(?:_(floodnet|rescuenet)_test_eval100.*)?$', name)
        if not mo:
            continue
        ds, shot = mo.group(1), int(mo.group(2))
        run_id = f'{ds}_{shot}shot_s{mo.group(3)}_{mo.group(4)}'
        r = runs.setdefault(run_id, dict(dataset=ds, shot=shot))
        if mo.group(5):
            r['eval100'] = m
        else:
            r['val30'] = m
        lf = os.path.join(res_dir, 'logs', f'{run_id}.json')
        if os.path.exists(lf) and 'log' not in r:
            lg = json.load(open(lf))
            r['log'] = dict(finished=lg.get('finished'), timing=lg.get('timing'),
                            val_history=[dict(step=v['step'], miou=v['miou']) for v in lg.get('val', [])])
    return runs


def table(runs, ds, split):
    names = NAMES[ds]
    head = ['shot', 'best step' if split == 'val30' else 'n img'] + names + ['mIoU', 'drv P', 'drv R', 'wet IoU']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in sorted((r for r in runs.values() if r['dataset'] == ds), key=lambda r: r['shot']):
        m = r.get(split)
        if not m:
            lines.append(f'| {r["shot"]} | missing |' + ' |' * (len(head) - 2))
            continue
        extra = m.get('best_step') if split == 'val30' else m.get('n_images')
        lines.append('| ' + ' | '.join([str(r['shot']), str(extra)] + [fmt(v) for v in row(m, names)]) + ' |')
    return '\n'.join(lines)


def spread(ys, gap=0.04):
    """Nudge end-label heights apart (keeps order) so direct labels never overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


def plot(runs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fl = sorted((r for r in runs.values() if r['dataset'] == 'floodnet'), key=lambda r: r['shot'])
    if not fl:
        return
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': TEXT, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, split, title in ((axes[0], 'val30', 'val30 (best checkpoint)'), (axes[1], 'eval100', 'eval100 (held-out test)')):
        pts = [(r['shot'], r[split]) for r in fl if split in r]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ends = []
        for k, n in enumerate(NAMES['floodnet']):
            ys = [p[1][f'iou_{n}'] for p in pts]
            ax.plot(xs, ys, color=SERIES[k], lw=2, marker='o', ms=6, label=n)
            ends.append((n, ys[-1], MUTED, 'normal'))
        ax.plot(xs, [p[1]['miou'] for p in pts], color=TEXT, lw=2.5, marker='s', ms=7, label='mIoU (5 ch)')
        ends.append(('mIoU', pts[-1][1]['miou'], TEXT, 'bold'))
        for (n, _, c, w), y in zip(ends, spread([e[1] for e in ends])):
            ax.annotate(n, (xs[-1], y), xytext=(8, 0), textcoords='offset points', va='center', fontsize=8, color=c,
                        weight=w)
        if split == 'eval100':
            ax.axhline(PSPNET['road_flooded'], color=SERIES[1], lw=1.2, ls='--')
            ax.text(xs[0], PSPNET['road_flooded'] + 0.015, 'PSPNet road_flooded, 1,445 labels (paper, full test)',
                    fontsize=8, color=MUTED)
        ax.set_xscale('log')
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlim(xs[0] * 0.8, xs[-1] * 2.2)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_xlabel('labelled support images (shot)')
        ax.set_title(title, color=TEXT, fontsize=10, loc='left')
    axes[0].set_ylabel('IoU')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('FloodNet data-efficiency curve (Chameleon, recipe v2, seed 0)', x=0.01, ha='left', color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--out', default='results/kaggle_v2/report')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = collect(a.results)
    md = []
    for ds in ('floodnet', 'rescuenet'):
        for split in ('val30', 'eval100'):
            md += [f'### {ds} {split} (τ = 0.5)', '', table(runs, ds, split), '']
    open(os.path.join(a.out, 'results.md'), 'w').write('\n'.join(md))
    json.dump(runs, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    plot(runs, os.path.join(a.out, 'curve_floodnet.png'))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
