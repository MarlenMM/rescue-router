"""RescueNet fairness check (PLAN §16.9, registered 17 Oct before its runs): the RescueNet counterpart of the viewer's
FloodNet model, the seed ensemble (3 × DeepLabV3-ResNet50 on the same 20 draw-0 photos, seeds 0-2, flips averaged,
src/baseline_avg.py) with thresholds tuned on RescueNet val30 (the baseline_report rule), against Chameleon.

  (a) eval100 (the photos the viewer shows): vs the viewer's Chameleon RescueNet model (published 20-shot run)
  (b) full RescueNet test: vs the one-recipe Chameleon of the same draw (Phase 2, rescuenet_20shot_s0_stratified_p2)
Paired bootstrap (2,000 resamples over photos, the same for every model): mIoU, and with router v3 (published gates)
stranded F1 and false-safe rate. Also the ensemble's argmax decode and the single seed-0 network (the viewer's model b
until now). The rule: RescueNet's viewer default becomes the ensemble if its eval100 mIoU (tuned) is higher than the
viewer's Chameleon model's in (a).

    python experiments/rescuenet_ensemble.py     # -> results/baselines/report/rescuenet_ensemble.{json,md}
"""
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
import postproc_sweep as PS  # noqa: E402
from baseline_report import Labels, cache, cham_cms, miou_fn, paired, per_photo_status, probs  # noqa: E402
from bootstrap_ci import f1_from  # noqa: E402
from router import false_safe_from  # noqa: E402
from rr.metrics import confusion, decode  # noqa: E402

DS = 'rescuenet'
BL = os.path.join(REPO, 'results', 'baselines')
ENS = 'baseline_deeplabv3_r50_rescuenet_20shot_s0_ens3_tta'
SINGLE = 'baseline_deeplabv3_r50_rescuenet_20shot_s0'
CHAM = {'eval100': (os.path.join(REPO, 'results', 'kaggle_v2', 'rr'), 'rescuenet_20shot_s0_stratified',
                    'the viewer\'s Chameleon (published 20-shot)',
                    os.path.join(REPO, 'results', 'routing', 'pred_rescuenet_20shot_s0_stratified_tuned.json')),
        'full': (os.path.join(REPO, 'results', 'phase2', 'rr'), 'rescuenet_20shot_s0_stratified_p2',
                 'Chameleon, one recipe (Phase 2, same draw)',
                 os.path.join(REPO, 'results', 'phase2', 'routing', 'rescuenet_20shot_s0_stratified_p2_full.json'))}
SPLIT = {'eval100': 'rescuenet_test_eval100', 'full': 'rescuenet_test_full'}
REP = os.path.join(BL, 'report')


def tuned_tau():
    """the ensemble's thresholds on RescueNet val30, as baseline_report tunes every baseline (decode_tune rule)"""
    f = os.path.join(BL, 'decode', f'{ENS}.json')
    if os.path.exists(f):
        return json.load(open(f))['chosen']['tau']
    gv = Labels()(DS, f'{DS}_val_tune30')
    Pv = probs(cache(BL, ENS, f'{DS}_val_tune30'))
    S = PS.Scorer([Pv[i] for i in gv], [gv[i] for i in gv], 6)
    tau, _ = PS.tune_tau(S)
    S.close()
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(dict(run=ENS, chosen=dict(tau=[float(t) for t in tau]),
                   rule='per-class thresholds tuned on RescueNet val30 (baseline_report rule; experiments/rescuenet_ensemble.py)'),
              open(f, 'w'), indent=1)
    return [float(t) for t in tau]


def route(run, sf, dec, out):
    if not os.path.exists(out):
        dj = out.replace('.json', '_decode.json')
        json.dump(dec, open(dj, 'w'))
        subprocess.run([sys.executable, os.path.join(REPO, 'src', 'route_eval.py'), '--dataset', DS, '--split_file', sf,
                        '--preds', cache(BL, run, sf), '--decode', dj, '--lams', '5', '--out', out], check=True,
                       stdout=subprocess.DEVNULL)
    return per_photo_status(out)


def main():
    tau = tuned_tau()
    labels, fn, out = Labels(), miou_fn(DS), dict(ensemble=ENS, tau=tau)
    for key in ('eval100', 'full'):
        sf = SPLIT[key]
        g = labels(DS, sf)
        ids = sorted(g)
        Pe = probs(cache(BL, ENS, sf))
        m = {'ensemble, tuned': {i: confusion(decode(Pe[i], tau=tau), g[i]) for i in ids},
             'ensemble, argmax': {i: confusion(np.concatenate([Pe[i], 1 - Pe[i].sum(0, keepdims=True)]).argmax(0), g[i])
                                  for i in ids}}
        del Pe
        Ps = probs(cache(BL, SINGLE, sf))
        m['one network (seed 0), argmax'] = {i: confusion(np.concatenate([Ps[i], 1 - Ps[i].sum(0, keepdims=True)]).argmax(0), g[i])
                                             for i in ids}
        del Ps
        cf, cr, clabel, croute = CHAM[key]
        cc = cham_cms(cf, cr, DS, sf, g)
        res = dict(split=sf, n=len(ids), chameleon=clabel, miou={})
        for name, cms in m.items():
            res['miou'][name] = paired(cms, cc, ids, fn, 2000)     # 'baseline' = this model, 'diff' = model − Chameleon
        se = route(ENS, sf, dict(tau=tau), os.path.join(REP, f'routing_{"" if key == "eval100" else "full_"}{ENS}_tuned.json'))
        sc = per_photo_status(croute)
        common = sorted(set(se) & set(sc))
        res['routing'] = {mt: paired(se, sc, common, f, 2000) for mt, f in (('stranded_f1', f1_from), ('false_safe', false_safe_from))}
        res['routing_n'] = len(common)
        out[key] = res
        print(key, {n: round(r['diff']['value'], 4) for n, r in res['miou'].items()}, flush=True)
    a = out['eval100']['miou']['ensemble, tuned']
    out['rule'] = dict(ensemble=a['baseline']['value'], chameleon=a['chameleon']['value'],
                       default='b (ensemble)' if a['baseline']['value'] > a['chameleon']['value'] else 'a (Chameleon)')
    json.dump(out, open(os.path.join(REP, 'rescuenet_ensemble.json'), 'w'), indent=1, default=float)
    f3 = lambda d: f"{d['value']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"          # noqa: E731
    fd = lambda d: f"{d['value']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"       # noqa: E731
    md = ['# RescueNet fairness check: the seed ensemble with flips vs Chameleon (20 photos, draw 0; PLAN §16.9)', '',
          f'Ensemble `{ENS}` (seeds 0-2, flips), tuned thresholds {tau} (RescueNet val30). Paired, 2,000 resamples; '
          'difference = model − Chameleon.', '']
    for key in ('eval100', 'full'):
        r = out[key]
        md += [f"## {r['split']} ({r['n']} photos) vs {r['chameleon']}", '', '| model | mIoU | Chameleon | model − Chameleon |',
               '|---|---|---|---|']
        md += [f"| {n} | {f3(x['baseline'])} | {f3(x['chameleon'])} | **{fd(x['diff'])}** |" for n, x in r['miou'].items()]
        md += ['', f"Routing (ensemble, tuned; {r['routing_n']} photos): " + '; '.join(
            f"{mt.replace('_', ' ')} {x['baseline']['value']:.3f} vs {x['chameleon']['value']:.3f}, {fd(x['diff'])}"
            for mt, x in r['routing'].items()), '']
    md += [f"**Rule (registered):** eval100 mIoU ensemble {out['rule']['ensemble']:.3f} vs the viewer's Chameleon "
           f"{out['rule']['chameleon']:.3f} → RescueNet's viewer default: **{out['rule']['default']}**."]
    open(os.path.join(REP, 'rescuenet_ensemble.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
