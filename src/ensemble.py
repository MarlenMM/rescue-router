"""Product-model choice: Chameleon, same-budget baselines, and their probability averages (PLAN §16, A5; PROTOCOL.md
"Model choice"). CPU only, from the caches.

Candidates (FloodNet, all trained on the same 50 photos):
  - 'chameleon': the submitted model with its submitted decode (guided filter -> val30 tau -> SAM);
  - '<baseline> argmax': each baseline run's own decode (argmax over the 5 classes + other; no tuning labels);
  - 'w=<w> <baseline>': P = w * Chameleon (guided-filtered, as in its decode) + (1 - w) * baseline softmax, per-class
    thresholds tuned on val30 (decode_tune rule, drivable-precision floor as for the submitted model), no SAM;
    w = 0 is the baseline with tuned thresholds, w = 1 Chameleon without SAM.
--choose_on val30 (default, the 14 Oct table): w chosen by its tuned val30 score, first baseline only.
--choose_on val30,confirm100 (the pre-registered rule): every candidate scored by mIoU of the confusion summed over
the two splits; the highest is the product model (ties: fewer labels used, i.e. argmax before tuned, then the
simpler model). Every candidate is reported on eval100 and the full test; the choice is marked.

    python src/ensemble.py [--baseline baseline_deeplabv3_r50_50shot_s0[,...]] [--choose_on val30,confirm100]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import postproc_sweep as PS  # noqa: E402
from rr.final_decode import decode_photo, load_chosen  # noqa: E402
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

CHAM = 'floodnet_50shot_s0_stratified'
NAMES = PS.NAMES['floodnet']
WS = (0.0, 0.25, 0.5, 0.75, 1.0)
SPLITS = {'val30': ('floodnet_val_tune30', 'val'), 'confirm100': ('floodnet_val_confirm100', 'val'),
          'eval100': ('floodnet_test_eval100', 'test'), 'full': ('floodnet_test_full', 'test')}
REPORT = ('eval100', 'full')


def cache(folder, run, sf):
    for s in ('', '_fp16'):
        p = os.path.join(folder, 'preds', f'{run}_{sf}{s}.npz')
        if os.path.exists(p):
            return p
    return None


def photos(a, key):
    """yields per photo: (Chameleon guided-filtered probs, Chameleon submitted class map, {baseline: probs}, label);
    None if a cache is missing. One photo at a time: the full test does not fit in memory as floats."""
    sf, split = SPLITS[key]
    sf_c = a.confirm_split if key == 'confirm100' and a.confirm_split else sf
    pc = cache(a.results, CHAM, sf_c)
    fb = {b: cache(a.baselines, b, sf_c) for b in a.baselines_list}
    if pc is None or None in fb.values():
        return None
    return _photos(a, pc, fb, sf_c)


def _photos(a, pc, fb, sf_c):
    zc = np.load(pc)
    pb = {}
    for b, f in fb.items():
        zb = np.load(f)
        pb[b] = {str(i): P for i, P in zip(zb['ids'], zb['probs'])}
    dec = load_chosen(a.results, CHAM)
    root = os.path.join(RR_DATA, 'floodnet_r768', 'val' if '_val_' in sf_c else 'test')
    for i, P in zip(zc['ids'], zc['probs']):
        i = str(i)
        img = np.asarray(Image.open(os.path.join(root, 'images', f'{i}.jpg')).convert('RGB'))
        P = P.astype(np.float32) / 255
        Pc = np.clip(PS.guided_filter(img.mean(2).astype(np.float32) / 255, P, 2, 1e-3), 0, 1)
        maps = [s for s in (sf_c, 'floodnet_test_full') if s in (dec.get('classmaps') or {})]
        cls_c = decode_photo(P, dec, img=img, split_file=maps[0] if maps else sf_c, photo_id=i, root=os.path.dirname(HERE))
        g = gt_classes(np.asarray(Image.open(os.path.join(root, 'labels', f'{i}.png'))))
        yield Pc, cls_c, {b: pb[b][i].astype(np.float32) / 255 for b in pb}, g


def argmax6(P):
    return np.concatenate([P, 1 - P.sum(0, keepdims=True)]).argmax(0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--baselines', default='results/baselines')
    ap.add_argument('--baseline', default='baseline_deeplabv3_r50_50shot_s0', help='comma list of baseline runs')
    ap.add_argument('--choose_on', default='val30', help="'val30' (14 Oct table) or 'val30,confirm100' (protocol)")
    ap.add_argument('--confirm_split', default=None, help='test only: stand-in split file for confirm100')
    ap.add_argument('--out', default='results/baselines/report')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a.baselines_list = a.baseline.split(',')
    choose = a.choose_on.split(',')
    protocol = choose != ['val30']
    keys = list(dict.fromkeys(['val30'] + choose + list(REPORT)))
    missing = [k for k in keys if photos(a, k) is None]
    if set(missing) & set(choose):
        sys.exit(f'no Chameleon / baseline caches for {missing} (confirm100 comes with the Phase 2 curve session)')
    # candidates: name -> function(photo tuple) -> class map
    cands, info = {}, {}
    if protocol:
        cands['chameleon'] = lambda x: x[1]
        info['chameleon'] = dict(labels=80, kind='submitted chain (guided filter, tau, SAM)')
    for b in a.baselines_list if protocol else a.baselines_list[:1]:
        if protocol:
            cands[f'{b} argmax'] = (lambda b: lambda x: argmax6(x[2][b]))(b)
            info[f'{b} argmax'] = dict(labels=50, kind='own argmax')
        val = list(photos(a, 'val30'))
        for w in WS:
            S = PS.Scorer([w * c + (1 - w) * p[b] for c, _, p, _ in val], [g for *_, g in val], 6)
            tau, rv = PS.tune_tau(S)
            S.close()
            name = f'w={w} {b}'
            cands[name] = (lambda w, b, tau: lambda x: decode(w * x[0] + (1 - w) * x[2][b], tau=tau))(w, b, tau)
            info[name] = dict(labels=80, kind='mix, val30-tuned tau', w=w, tau=[float(t) for t in tau], val30_tuned=rv['miou'])
    res = {name: dict(info[name]) for name in cands}
    for k in keys:
        if k in missing:
            continue
        cms = {name: np.zeros((6, 6), np.int64) for name in cands}
        for x in photos(a, k):
            for name, f in cands.items():
                cms[name] += confusion(f(x), x[3])
        for name, cm in cms.items():
            r = summarize(cm, NAMES, WET['floodnet'])
            res[name][k] = dict(miou=r['miou'], drivable_precision=r['drivable_precision'], cm=cm.tolist(),
                                **{n: r[f'iou_{n}'] for n in NAMES})
    for name in cands:
        if protocol:
            pooled = sum(np.array(res[name][k]['cm']) for k in choose)
            res[name]['choice_score'] = summarize(pooled, NAMES, WET['floodnet'])['miou']
        print(name, {k: round(v['miou'], 4) for k, v in res[name].items() if isinstance(v, dict) and 'miou' in v}, flush=True)
    if protocol:
        order = list(cands)                          # insertion order = tie-break (fewer labels / simpler first)
        best = max(order, key=lambda n: (round(res[n]['choice_score'], 6), -info[n]['labels'], -order.index(n)))
    else:
        best = max((n for n in cands), key=lambda n: info[n]['val30_tuned'])
    tag = '' if not protocol else '_choice'
    json.dump(dict(baselines=a.baselines_list, choose_on=choose, chosen=best, results=res),
              open(os.path.join(a.out, f'ensemble_{a.baselines_list[0]}{tag}.json'), 'w'), indent=1)
    cols = ([f'{"+".join(choose)} (choice)'] if protocol else ['val30 (tuned)']) + ['eval100', 'full test']
    md = [f'# Product-model candidates (Chameleon, {", ".join(a.baselines_list)}); choice on {" + ".join(choose)}', '',
          '| candidate | labels | ' + ' | '.join(cols) + ' | ' + ' | '.join(NAMES) + ' (full test) | drv P |',
          '|' + '---|' * (2 + len(cols) + len(NAMES) + 1)]
    for n in cands:
        r, full = res[n], res[n].get('full', {})
        first = r['choice_score'] if protocol else r['val30_tuned']
        md.append(f"| {n}{' **(choice)**' if n == best else ''} | {info[n]['labels']} | {first:.3f} | "
                  f"{r.get('eval100', {}).get('miou', float('nan')):.3f} | **{full.get('miou', float('nan')):.3f}** | "
                  + ' | '.join(f'{full.get(x, float("nan")):.3f}' for x in NAMES) + f" | {full.get('drivable_precision', float('nan')):.3f} |")
    md += ['', 'w = 1 is Chameleon without SAM (its submitted chain with SAM is the "chameleon" row); w = 0 the baseline '
           'with val30-tuned thresholds.' + (' Choice: highest mIoU of the confusion summed over the choice splits.'
                                              if protocol else ' Choice: w with the best tuned val30 score.')]
    open(os.path.join(a.out, f'ensemble_{a.baselines_list[0]}{tag}.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
