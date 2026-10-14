"""Learning curves: Chameleon vs the same-budget baseline by number of labelled photos (PLAN §16.5). CPU only.

eval100 for every point (both models have it at every k; the full-test pairs are in baselines.md). Baseline: argmax
(k labels, no tuning). Chameleon: its chosen decode (k + 30 tuning labels): the published runs (first recipe at
1-20 shots, the submitted model at 50) and, with --phase2, the one-recipe `p2` runs. 95 % bootstrap intervals over
photos. FloodNet and RescueNet panels; all 1,445 FloodNet photos plotted at 1,445.

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
import postproc_sweep as PS  # noqa: E402
from bootstrap_ci import boot  # noqa: E402
from PIL import Image  # noqa: E402
from run_decode import GF  # noqa: E402
from rr.final_decode import decode_photo  # noqa: E402
from rr.metrics import confusion, decode  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

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
    gts = BR.Labels()
    res = {}
    for ds in ('floodnet', 'rescuenet'):
        sf, fn = f'{ds}_test_eval100', BR.miou_fn(ds)
        g = gts(ds, sf)
        series = {'baseline': {}, 'chameleon (published)': {}, 'chameleon (one recipe, Phase 2)': {},
                  'chameleon label-free (k labels)': {}}
        runs = [BR.parse(r) for r in sorted({f[:-len(f'_{sf}.npz')] for f in os.listdir(os.path.join(a.baselines, 'preds'))
                                              if f.endswith(f'_{sf}.npz')})]
        for b in runs:
            if b and b['dataset'] == ds and b['arch'] == 'deeplabv3_r50' and (b['seed'], b['draw']) == (0, 0) \
                    and not b['tag'] and not b['tta']:
                P = BR.probs(BR.cache(a.baselines, b['run'], sf))
                series['baseline'][b['k'] or ALL[ds]] = point(
                    {i: confusion(decode_photo(P[i], {'argmax': True}), g[i]) for i in g}, fn)
        for k in (1, 5, 10, 20, 50, 100):
            for label, folder, run in (('chameleon (published)', a.results, f'{ds}_{k}shot_s0_stratified'),
                                       ('chameleon (one recipe, Phase 2)', a.phase2, f'{ds}_{k}shot_s0_stratified_p2')):
                if folder and os.path.exists(os.path.join(folder, 'decode', f'{run}.json')) and BR.cache(folder, run, sf):
                    series[label][k] = point(BR.cham_cms(os.path.abspath(folder), run, ds, sf, g), fn)
            # label-fair: Chameleon decoded with thresholds from its k training labels only (run_decode.py 'prior_tau':
            # guided filter -> class shares of the supports matched on the photos), Phase 2 run first
            for folder, run in ((a.phase2, f'{ds}_{k}shot_s0_stratified_p2'), (a.label_free, f'{ds}_{k}shot_s0_stratified'),
                                (a.phase2, f'{ds}_{k}shot_s0_stratified')):
                f = folder and os.path.join(folder, 'decode', f'{run}.json')
                src = folder and BR.cache(folder, run, sf) or BR.cache(a.results, run, sf)
                if ds == 'floodnet' and f and os.path.exists(f) and 'prior_tau' in json.load(open(f)) and src:
                    tau = json.load(open(f))['prior_tau']
                    P = BR.probs(src)
                    cms = {}
                    for i in g:
                        img = np.asarray(Image.open(os.path.join(RR_DATA, 'floodnet_r768', 'test', 'images', f'{i}.jpg'))
                                         .convert('RGB'))
                        cms[i] = confusion(decode(PS.guided_filter(img.mean(2).astype(np.float32) / 255, P[i], *GF),
                                                  tau=tau), g[i])
                    series['chameleon label-free (k labels)'][k] = point(cms, fn)
                    break
        if a.phase2 and ds == 'floodnet':                            # the submitted model is the p2 recipe at 50
            series['chameleon (one recipe, Phase 2)'].setdefault(50, series['chameleon (published)'].get(50))
        res[ds] = {s: {str(k): v for k, v in sorted(pts.items())} for s, pts in series.items() if pts}
        print(ds, {s: {k: round(v['value'], 3) for k, v in pts.items()} for s, pts in res[ds].items()}, flush=True)
    json.dump(res, open(os.path.join(a.out, 'curve.json'), 'w'), indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    style = {'baseline': ('tab:orange', 'o'), 'chameleon (published)': ('tab:blue', 's'),
             'chameleon (one recipe, Phase 2)': ('tab:green', 'D'), 'chameleon label-free (k labels)': ('tab:purple', 'v')}
    for ax, ds in zip(axs, res):
        for s, pts in res[ds].items():
            ks = [int(k) for k in pts]
            v = np.array([[pts[k]['value'], pts[k]['lo'], pts[k]['hi']] for k in pts])
            ax.errorbar(ks, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=style[s][0], marker=style[s][1],
                        capsize=3, ms=5, lw=1.5, label=s + (' (k labels)' if s == 'baseline' else
                                                             '' if 'label-free' in s else ' (k + 30 labels)'))
        if ds == 'floodnet':
            ax.axhline(0.848, color='grey', ls=':', lw=1)
            ax.text(1.1, 0.855, 'PSPNet, 1,445 labels (paper, full test)', fontsize=7, color='grey')
        ticks = [1, 5, 10, 20, 50, 100, ALL[ds]]
        ax.set_xscale('log')
        ax.set_xticks(ticks, [f'{t:,}' for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(0.8, ALL[ds] * 1.3)
        ax.set(xlabel='labelled photos (log scale)', ylabel='mIoU, eval100 (5 classes)',
               title={'floodnet': 'FloodNet (Hurricane Harvey)', 'rescuenet': 'RescueNet (Hurricane Michael)'}[ds],
               ylim=(0, 0.9))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')
    fig.suptitle('Chameleon vs a standard segmenter (DeepLabV3-ResNet50) trained on the same photos; 95 % CIs over photos',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'curve.png'), dpi=140)
    print('wrote', os.path.join(a.out, 'curve.png'))


if __name__ == '__main__':
    main()
