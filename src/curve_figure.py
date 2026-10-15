"""Learning curves: Chameleon vs the same-budget baseline by number of labelled photos (PLAN §16.5). CPU only.

eval100 for every point (both models have it at every k; the full-test pairs are in baselines.md). curve.png plots
every series at its k training photos; curve_total.png plots Chameleon's tuned decodes at k + 30 (the val30 photos that
chose their thresholds), the other view of "same budget" (e.g. Chameleon 20 + 30 vs the baseline on 50). Baseline: argmax
(k labels, no tuning). Chameleon: its chosen decode (k + 30 tuning labels): the published runs (first recipe at
1-20 shots, the submitted model at 50) and, with --phase2, the one-recipe `p2` runs. 95 % bootstrap intervals over
photos. FloodNet and RescueNet panels; all 1,445 FloodNet photos plotted at 1,445. curve_routing.png: the routing
safety metrics (stranded-house F1, false-safe rate) by k from baseline_report.py's paired eval100 routing.

    python src/curve_figure.py [--phase2 results/phase2/rr] [--out results/baselines/report/curve]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baseline_report as BR  # noqa: E402
from bootstrap_ci import boot  # noqa: E402
from rr.final_decode import decode_photo  # noqa: E402
from rr.metrics import confusion  # noqa: E402

ALL = {'floodnet': 1445, 'rescuenet': 419}


def point(cms, fn, B=1000):
    r, _ = boot(cms, sorted(cms), fn, B, np.random.default_rng(0))
    return dict(value=float(r['value']), lo=float(r['lo']), hi=float(r['hi']), n=r['n'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baselines', default='results/baselines')
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--phase2', default=None)
    ap.add_argument('--label_free', default='results/kaggle_v2/report/label_free',
                    help='run_decode.py outputs (decode/<run>.json with prior_tau) for the published 1-20-shot runs')
    ap.add_argument('--out', default='results/baselines/report/curve')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    BR.label_free_decodes(a.results, a.label_free)
    gts = BR.Labels()
    res = {}
    for ds in ('floodnet', 'rescuenet'):
        sf, fn = f'{ds}_test_eval100', BR.miou_fn(ds)
        g = gts(ds, sf)
        series = {'baseline': {}, 'baseline from the FloodNet model': {}, 'chameleon (published)': {},
                  'chameleon (one recipe, Phase 2)': {}, 'chameleon label-free (k labels)': {}}
        runs = [BR.parse(r) for r in sorted({f[:-len(f'_{sf}.npz')] for f in os.listdir(os.path.join(a.baselines, 'preds'))
                                              if f.endswith(f'_{sf}.npz')})]
        for b in runs:
            if b and b['dataset'] == ds and b['arch'] == 'deeplabv3_r50' and (b['seed'], b['draw']) == (0, 0) \
                    and b['tag'] in ('', 'ft50') and not b['tta']:
                P = BR.probs(BR.cache(a.baselines, b['run'], sf))
                name = 'baseline' if not b['tag'] else 'baseline from the FloodNet model'
                series[name][b['k'] or ALL[ds]] = point(
                    {i: confusion(decode_photo(P[i], {'argmax': True}), g[i]) for i in g}, fn)
        for k in (1, 5, 10, 20, 50, 100):
            for label, folder, run in (('chameleon (published)', a.results, f'{ds}_{k}shot_s0_stratified'),
                                       ('chameleon (one recipe, Phase 2)', a.phase2, f'{ds}_{k}shot_s0_stratified_p2')):
                if folder and os.path.exists(os.path.join(folder, 'decode', f'{run}.json')) and BR.cache(folder, run, sf):
                    series[label][k] = point(BR.cham_cms(os.path.abspath(folder), run, ds, sf, g), fn)
            # label-fair: Chameleon decoded with thresholds from its k training labels only (baseline_report.label_free_tau:
            # FloodNet run_decode.py 'prior_tau' after the guided filter, RescueNet the same matching on the raw
            # probabilities), Phase 2 run first
            for folder, run in ((a.phase2, f'{ds}_{k}shot_s0_stratified_p2'), (a.results, f'{ds}_{k}shot_s0_stratified')):
                src = folder and BR.cache(folder, run, sf)
                tau = src and BR.label_free_tau(folder, run, a.label_free, ds, k, 0)
                if tau:
                    series['chameleon label-free (k labels)'][k] = point(BR.cham_label_free_cms(folder, run, ds, sf, g, tau), fn)
                    break
        if a.phase2 and ds == 'floodnet':                            # the submitted model is the p2 recipe at 50
            series['chameleon (one recipe, Phase 2)'].setdefault(50, series['chameleon (published)'].get(50))
        res[ds] = {s: {str(k): v for k, v in sorted(pts.items())} for s, pts in series.items() if pts}
        print(ds, {s: {k: round(v['value'], 3) for k, v in pts.items()} for s, pts in res[ds].items()}, flush=True)
    json.dump(res, open(os.path.join(a.out, 'curve.json'), 'w'), indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style = {'baseline': ('tab:orange', 'o'), 'baseline from the FloodNet model': ('tab:red', '^'),
             'chameleon (published)': ('tab:blue', 's'),
             'chameleon (one recipe, Phase 2)': ('tab:green', 'D'), 'chameleon label-free (k labels)': ('tab:purple', 'v')}
    # extra labelled photos each series has used beyond its k training photos: Chameleon's tuned decodes chose their
    # thresholds on the 30 val30 photos; the FloodNet-pretrained baseline saw 50 FloodNet photos (another disaster's)
    extra = {'chameleon (published)': 30, 'chameleon (one recipe, Phase 2)': 30}
    for total, fname, xlabel in ((False, 'curve.png', 'labelled training photos k (log scale)'),
                                 (True, 'curve_total.png', 'labelled photos used, training + tuning (log scale)')):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, ds in zip(axs, res):
            for s, pts in res[ds].items():
                ks = [int(k) + (extra.get(s, 0) if total else 0) for k in pts]
                v = np.array([[pts[k]['value'], pts[k]['lo'], pts[k]['hi']] for k in pts])
                ax.errorbar(ks, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=style[s][0], marker=style[s][1],
                            capsize=3, ms=5, lw=1.5, label=s + (' (k labels)' if s == 'baseline' else
                                                                 ' (k + 50 FloodNet labels)' if 'FloodNet model' in s else
                                                                 '' if 'label-free' in s else
                                                                 ' (k + 30, plotted at k + 30)' if total else ' (k + 30 labels)'))
            if ds == 'floodnet':
                ax.axhline(0.848, color='grey', ls=':', lw=1)
                ax.text(1.1, 0.855, 'PSPNet, 1,445 labels (paper, full test)', fontsize=7, color='grey')
            ticks = [1, 5, 10, 20, 50, 100, ALL[ds]]
            ax.set_xscale('log')
            ax.set_xticks(ticks, [f'{t:,}' for t in ticks])
            ax.minorticks_off()
            ax.set_xlim(0.8, ALL[ds] * 1.3)
            ax.set(xlabel=xlabel, ylabel='mIoU, eval100 (5 classes)',
                   title={'floodnet': 'FloodNet (Hurricane Harvey)', 'rescuenet': 'RescueNet (Hurricane Michael)'}[ds],
                   ylim=(0, 0.9))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc='lower right')
        fig.suptitle('Chameleon vs a standard segmenter (DeepLabV3-ResNet50) trained on the same photos; 95 % CIs over photos'
                     + ('; x = all labelled photos used' if total else ''), fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, fname), dpi=140)
        plt.close(fig)
        print('wrote', os.path.join(a.out, fname))

    routing_figure(os.path.join(os.path.dirname(a.out.rstrip('/')), 'baselines.json'), a.out)


def routing_figure(baselines_json, out):
    """curve_routing.png: the routing safety metrics by number of labels, both models, from baseline_report.py's paired
    eval100 routing (router v3, the published gate; baseline argmax maps, Chameleon's chosen decode of the same k):
    stranded-house F1 (higher is better) and false-safe rate (lower is better), 95 % intervals over photos"""
    if not os.path.exists(baselines_json):
        return
    rows = [r for r in json.load(open(baselines_json)) if r['arch'] == 'deeplabv3_r50' and (r['seed'], r['draw']) == (0, 0)
            and not r['tag'] and not r['tta'] and 'routing_eval100' in r]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
    for col, ds in enumerate(('floodnet', 'rescuenet')):
        for row, (m, name) in enumerate((('stranded_f1', 'stranded-house F1 (higher is better)'),
                                         ('false_safe', 'false-safe rate (lower is better)'))):
            ax = axs[row, col]
            for who, color, marker, dx in (('baseline', 'tab:orange', 'o', 0.97), ('chameleon', 'tab:blue', 's', 1.03)):
                pts = sorted((r['k'] or ALL[ds], r['routing_eval100'][m].get(who)) for r in rows if r['dataset'] == ds)
                pts = [(k, v) for k, v in pts if v]
                if not pts:
                    continue
                k = np.array([p[0] for p in pts]) * dx
                v = np.array([[p[1]['value'], p[1]['lo'], p[1]['hi']] for p in pts])
                ax.errorbar(k, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=color, marker=marker, capsize=3,
                            ms=5, lw=1.5, label='baseline (k labels)' if who == 'baseline' else 'Chameleon (k + 30 labels)')
            ticks = [1, 5, 10, 20, 50, 100, ALL[ds]]
            ax.set_xscale('log')
            ax.set_xticks(ticks, [f'{t:,}' for t in ticks])
            ax.minorticks_off()
            ax.set_xlim(0.8, ALL[ds] * 1.3)
            ax.set_ylim(0, 1 if m == 'stranded_f1' else 0.3)
            ax.grid(alpha=0.3)
            ax.set_ylabel(name, fontsize=8)
            if row == 0:
                ax.set_title({'floodnet': 'FloodNet (Hurricane Harvey)', 'rescuenet': 'RescueNet (Hurricane Michael)'}[ds])
            else:
                ax.set_xlabel('labelled training photos k (log scale)')
            ax.legend(fontsize=7, loc='lower right' if m == 'stranded_f1' else 'upper right')
    fig.suptitle('Routing on the predicted maps vs the labelled map (router v3, eval100; 95 % CIs over photos)\n'
                 'RescueNet eval100 holds few stranded houses: wide intervals', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'curve_routing.png'), dpi=140)
    plt.close(fig)
    print('wrote', os.path.join(out, 'curve_routing.png'))


if __name__ == '__main__':
    main()
