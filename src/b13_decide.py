"""Batch 13.3: score every GPU variant on val30 with the fixed adoption rule (PLAN §15 "Batch 13"). CPU only.

    python src/b13_decide.py --dirs results/b13/r1a/rr,results/b13/r1b/rr,results/b13/colab [--out results/b13/decide]

For every val30 cache found (training variants: each snapshot + EMA; prediction variants of the published model):
  * tuned-tau decode re-tuned on val30 (decode_tune rule: drivable precision not below the tau 0.5 decode's);
  * the stacked no-cost decode (guided filter r4 eps 1e-3 -> two-step t_w .2 -> clean-up), tau and clean-up re-tuned.
Per training run the snapshot is chosen as in the published pipeline (select_ckpt: best val30 mIoU at tau .5 among
the plain snapshots); its EMA (I17) is a separate row. Deltas: vs the published model (.4835 tuned tau / .5038
stacked, the rule's reference) and vs the I34 control run (attribution: same 1000-step schedule).
Rule: no running cost -> kept if delta >= 0 and drivable precision not lower; running cost -> delta >= +.01.
Writes <out>/decide.json and <out>/decide.md.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import postproc_sweep as PS  # noqa: E402
from decode_tune import load  # noqa: E402

REF = dict(tau=0.48354, stacked=0.50382, dp_tau=0.5686, dp_stacked=0.5991)
COST = {'v2': '2x prediction', 'st128': '2x prediction', 'sc1-1.25': '3x prediction', 'self': '2x prediction',
        'wavg': 'none', 'cw': 'none', 'comp': 'none'}
IDEA = {'b13ctl': 'I34 control (1000 steps, freeze_rest)', 'b13rare': 'I12+I13 rare-class weights + crops',
        'b13dice': 'I14 Dice term', 'b13ld': 'I5 label decoder tuned (+I7 head)', 'b13reg': 'I15+I18+I19 regularisers',
        'b13lr2': 'I16 lr 2x', 'b13rp': 'I6 relpos tuned', 'b13bal': 'I4 class-balanced supports',
        'b13ep4': 'I10 authors episodes (2+2, 500 steps)', 'b13sa': 'I8 separate alpha',
        'cw13331': 'I2 support crop weights 1,3,3,3,1', 'wavg1500-2000': 'I23 weight average step 1500+2000',
        'v2': 'I1 two support views', 'comp': 'I36 torch.compile', 'st128': 'I24 stride 128', 'sc1-1.25': 'I25 scales 1+1.25'}


def stacked_tune(probs, gts, imgs, workers):
    pp = [PS.guided_filter(I, P, 4, 1e-3) for I, P in zip(imgs, probs)]
    S = PS.Scorer(pp, gts, workers)
    tau, r = PS.tune_tau(S, 'combo', dict(t_w=0.2), start=[0.25, 0.35, 0.6, 0.3, 0.45])
    best = (0, 0, r)
    for k in (0, 800, 1600, 3200):
        for rr in (0, 1, 4):
            if k or rr:
                x = S('combo', tau=tau, t_w=0.2, k=k, r=rr)
                if x['drivable_precision'] >= PS.P_MIN - 1e-9 and x['miou'] > best[2]['miou']:
                    best = (k, rr, x)
    S.close()
    return dict(tau=tau, k=best[0], r=best[1]), best[2]


def score(path, root, imgs_cache, workers, stacked=True):
    probs, gts = load(path, root, 'val')
    probs = [P.astype(np.float32) / 255 for P in probs]
    S = PS.Scorer(probs, gts, workers)
    r05 = S('tau', tau=0.5)
    tau, rt = PS.tune_tau(S)
    S.close()
    out = dict(miou05=r05['miou'], tau=tau, miou_tau=rt['miou'], dp_tau=rt['drivable_precision'], val30_tau=rt)
    if stacked:
        ids = [str(i) for i in np.load(path)['ids']]
        imgs = [imgs_cache.setdefault(i, np.asarray(Image.open(os.path.join(root, 'val', 'images', f'{i}.jpg')).convert('L'),
                                                     np.float32) / 255) for i in ids]
        p, rs = stacked_tune(probs, gts, imgs, workers)
        out.update(stacked_params=p, miou_stacked=rs['miou'], dp_stacked=rs['drivable_precision'], val30_stacked=rs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--out', default='results/b13/decide')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f_json = os.path.join(a.out, 'decide.json')
    done = json.load(open(f_json)) if os.path.exists(f_json) else {}
    root = os.path.join(a.data, 'floodnet_r768')
    imgs = {}
    caches = sorted(c for d in a.dirs.split(',') for c in glob.glob(os.path.join(d, 'preds', '*floodnet_val_tune30*.npz')))
    rows = []
    for c in caches:
        name = os.path.basename(c)[:-4]
        m = re.match(r'floodnet_50shot_s0_(stratified|balanced)_(b13\w+?)(_ema)?_step(\d+)_floodnet_val_tune30_fp16$', name)
        v = re.match(r'floodnet_50shot_s0_stratified_floodnet_val_tune30_fp16_(.+)$', name)
        if m:
            row = dict(kind='train', run=m.group(2), ema=bool(m.group(3)), step=int(m.group(4)))
        elif v:
            row = dict(kind='predict', run=v.group(1), ema=False, step=None)
        else:
            continue
        row.update(cache=name, idea=IDEA.get(row['run'], row['run']))
        if name not in done:
            done[name] = score(c, root, imgs, a.workers, stacked=False)
            json.dump(done, open(f_json, 'w'), indent=1)
        row.update(done[name])
        rows.append(row)
        print(name, round(row['miou05'], 4), round(row['miou_tau'], 4), flush=True)
    # per training run: the select_ckpt choice (best tau-.5 val30 among plain snapshots) + its EMA; stacked decode on both
    chosen = []
    for run in sorted({r['run'] for r in rows if r['kind'] == 'train'}):
        plain = [r for r in rows if r['run'] == run and not r['ema']]
        best = max(plain, key=lambda r: r['miou05'])
        chosen.append(dict(best, label=f"{IDEA.get(run, run)} (step {best['step']})"))
        chosen += [dict(r, label=f"I17 EMA of {run} (step {r['step']})") for r in rows if r['run'] == run and r['ema']]
    chosen += [dict(r, label=IDEA.get(r['run'], r['run'])) for r in rows if r['kind'] == 'predict']
    for r in chosen:
        if 'miou_stacked' not in done[r['cache']]:
            done[r['cache']] = score(os.path.join(next(d for d in a.dirs.split(',') if os.path.exists(os.path.join(d, 'preds', r['cache'] + '.npz'))),
                                                  'preds', r['cache'] + '.npz'), root, imgs, a.workers, stacked=True)
            json.dump(done, open(f_json, 'w'), indent=1)
        r.update(done[r['cache']])
    ctl = next((r for r in chosen if r['run'] == 'b13ctl' and not r['ema']), None)
    lines = ['| variant | cost | val30 τ.5 | tuned τ | Δ ref | stacked | Δ ref | Δ ctl (stacked) | drv P (stacked) | kept |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in chosen:
        cost = next((v for k, v in COST.items() if r['run'].startswith(k)), 'none') if r['kind'] == 'predict' else 'none'
        d = r['miou_stacked'] - REF['stacked']
        keep = (d >= 0 and r['dp_stacked'] >= REF['dp_stacked'] - 1e-9) if cost == 'none' else d >= 0.01
        r.update(cost=cost, delta_ref=d, keep=bool(keep),
                 delta_ctl=(r['miou_stacked'] - ctl['miou_stacked']) if ctl else None)
        lines.append(f"| {r['label']} | {cost} | {r['miou05']:.4f} | {r['miou_tau']:.4f} | {r['miou_tau'] - REF['tau']:+.4f} | "
                     f"{r['miou_stacked']:.4f} | {d:+.4f} | {'' if ctl is None else f'{r['delta_ctl']:+.4f}'} | "
                     f"{r['dp_stacked']:.3f} | {'yes' if keep else 'no'} |")
    json.dump(dict(scores=done, chosen=chosen), open(os.path.join(a.out, 'decide_rows.json'), 'w'), indent=1, default=str)
    open(os.path.join(a.out, 'decide.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
