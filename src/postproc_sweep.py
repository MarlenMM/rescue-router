"""Batch 13.1, CPU: the free (no-GPU) ideas on the FloodNet 50-shot val30 caches (PLAN §15 "Batch 13"). CPU only.

    python src/postproc_sweep.py [--only i22,i27,...] [--out results/b13/postproc]

Every variant is scored on val30 with its own decode re-tuned on val30 (decode_tune.tune: per-channel tau by
coordinate ascent, drivable precision not below the tau 0.5 decode's) and compared with the current model's
tuned val30 (.4835, drivable precision .5686). Ideas:
  I22  average the step-500/1000/1500/2000 snapshots' val30 probabilities (several subsets);
  I27  additive per-class bias offsets (candidate if p_c + b_c >= .5, winner = largest p_c + b_c);
  I28  two-step decode: wet vs dry first (W = max(p1,p2,p4) vs D = max(p0,p3), wet iff W >= t_w * (W + D)),
       then the class inside the group by the tuned tau; 'other' if no channel passes its tau;
  I29  clean-up after the tuned decode: components smaller than k px are filled from the nearest larger
       component, then a closing of radius r on the road/water classes (closed pixels that were 'other' only);
  I30  edge-aware refinement: a guided filter (He et al. 2010) on each probability channel, guided by the photo
       (grey), then the decode re-tuned (dense CRF: pydensecrf does not build for Python 3.12, guided filter only);
  I32  p_wet calibration (reliability table + ECE) of the routers' wet probability, for the record: the gates are
       thresholds tuned on val30, so any monotone recalibration leaves the statuses unchanged; re-tune if the model changes.
Writes <out>/postproc.json (all numbers) and <out>/postproc.md (the table for the "tried / kept / dropped" log).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decode_tune import NAMES, TAU_GRID, load  # noqa: E402
from rr.metrics import OTHER_IDX, WET, confusion, decode, summarize  # noqa: E402

DS = 'floodnet'
RUN = 'floodnet_50shot_s0_stratified'
STEPS = (500, 1000, 1500, 2000)
REF = dict(miou=0.48354070568476215, drivable_precision=0.5686109269701369)    # current model, tuned decode
P_MIN = 0.5671217626701424                        # the tau 0.5 decode's val30 drivable precision (decode_tune constraint)


# ----------------------------------------------------------------------------- decoders
def decode_bias(P, bias):
    s = P + np.asarray(bias, np.float32).reshape(-1, 1, 1)
    return np.where(s.max(0) >= 0.5, s.argmax(0), OTHER_IDX).astype(np.uint8)


def decode_two_step(P, tau, t_w):
    s = P / np.asarray(tau, np.float32).reshape(-1, 1, 1)
    W, D = P[[1, 2, 4]].max(0), P[[0, 3]].max(0)
    wet = W >= t_w * (W + D + 1e-9)
    s_wet, s_dry = s.copy(), s.copy()
    s_wet[[0, 3]] = -1
    s_dry[[1, 2, 4]] = -1
    s = np.where(wet[None], s_wet, s_dry)
    out = np.where(s.max(0) >= 1, s.argmax(0), OTHER_IDX).astype(np.uint8)
    any_pass = (P / np.asarray(tau, np.float32).reshape(-1, 1, 1)).max(0) >= 1
    return np.where(any_pass & (out == OTHER_IDX), decode(P, tau), out)    # the group had no candidate: plain rule


def cleanup(cls, k=0, r=0, close_classes=(0, 1, 4)):
    out = cls.copy()
    if k:
        hole = np.zeros(cls.shape, bool)
        for c in range(OTHER_IDX + 1):
            lab, n = ndimage.label(cls == c)
            if n:
                size = np.bincount(lab.ravel())
                small = size < k
                small[0] = False
                hole |= small[lab]
        if hole.any() and not hole.all():
            _, (iy, ix) = ndimage.distance_transform_edt(hole, return_indices=True)
            out = out[iy, ix]
    if r:
        st = ndimage.generate_binary_structure(2, 1)
        for c in close_classes:
            m = ndimage.binary_closing(out == c, st, iterations=r)
            out[m & (out == OTHER_IDX)] = c
    return out


def box(x, r):
    return ndimage.uniform_filter(x, size=2 * r + 1, mode='reflect')


def guided_filter(I, P, r, eps):
    """I: (H, W) guide in [0,1]; P: (C, H, W) -> filtered (C, H, W) (grey-guide guided filter)."""
    mI = box(I, r)
    vI = box(I * I, r) - mI * mI
    out = np.empty_like(P)
    for c in range(P.shape[0]):
        mp = box(P[c], r)
        a = (box(I * P[c], r) - mI * mp) / (vI + eps)
        b = mp - a * mI
        out[c] = box(a, r) * I + box(b, r)
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------------------- scoring
_G = {}


def _init(probs, gts):
    _G['probs'], _G['gts'] = probs, gts


def _cm_one(args):
    i, kind, params = args
    P, g = _G['probs'][i], _G['gts'][i]
    if kind == 'tau':
        pred = decode(P, **params)
    elif kind == 'bias':
        pred = decode_bias(P, params['bias'])
    elif kind == 'two':
        pred = decode_two_step(P, params['tau'], params['t_w'])
    elif kind == 'clean':
        pred = cleanup(decode(P, tau=params['tau']), params['k'], params['r'])
    elif kind == 'combo':                       # two-step decode, then clean-up
        pred = decode_two_step(P, params['tau'], params['t_w']) if params.get('t_w') else decode(P, tau=params['tau'])
        pred = cleanup(pred, params.get('k', 0), params.get('r', 0))
    else:
        raise ValueError(kind)
    return confusion(pred, g)


class Scorer:
    def __init__(self, probs, gts, workers=10):
        self.n = len(probs)
        self.pool = ProcessPoolExecutor(workers, initializer=_init, initargs=(probs, gts))

    def __call__(self, kind, **params):
        cm = sum(self.pool.map(_cm_one, [(i, kind, params) for i in range(self.n)]))
        return summarize(cm, NAMES[DS], WET[DS])

    def close(self):
        self.pool.shutdown()


def tune_tau(S, kind='tau', extra=None, grid=TAU_GRID, start=None):
    """decode_tune.tune's coordinate ascent (3 passes, drivable precision >= P_MIN) for any decoder with a tau."""
    extra = extra or {}
    tau = list(start or [0.5] * 5)
    best = S(kind, tau=tau, **extra)
    if best['drivable_precision'] < P_MIN - 1e-9:
        best = dict(best, miou=-1)
    for _ in range(3):
        changed = False
        for c in range(5):
            for t in grid:
                cand = tau[:c] + [t] + tau[c + 1:]
                r = S(kind, tau=cand, **extra)
                if r['miou'] > best['miou'] + 1e-6 and r['drivable_precision'] >= P_MIN - 1e-9:
                    best, tau, changed = r, cand, True
        if not changed:
            break
    return tau, best


def tune_bias(S):
    grid = [-0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3]
    b = [0.0] * 5
    best = S('bias', bias=b)
    for _ in range(3):
        changed = False
        for c in range(5):
            for t in grid:
                cand = b[:c] + [t] + b[c + 1:]
                r = S('bias', bias=cand)
                if r['miou'] > best['miou'] + 1e-6 and r['drivable_precision'] >= P_MIN - 1e-9:
                    best, b, changed = r, cand, True
        if not changed:
            break
    return b, best


def row(name, idea, params, r, cost='none'):
    d = r['miou'] - REF['miou']
    keep = (d >= 0 and r['drivable_precision'] >= REF['drivable_precision'] - 1e-9) if cost == 'none' else d >= 0.01
    return dict(name=name, idea=idea, params=params, cost=cost, val30=r, delta=d, keep=bool(keep))


def load_cache(path, root):
    probs, gts = load(path, root, 'val')
    return [P.astype(np.float32) / 255 for P in probs], gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--out', default='results/b13/postproc')
    ap.add_argument('--only', default='base,i22,i27,i28,i29,i30,i32,combo')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f_json = os.path.join(a.out, 'postproc.json')
    rows = json.load(open(f_json)) if os.path.exists(f_json) else []
    only = a.only.split(',')
    rows = [r for r in rows if r['idea'].lower() not in only]
    root = os.path.join(a.data, f'{DS}_r768')
    pred = lambda s: os.path.join(a.results, 'preds', f'{RUN}_step{s}_floodnet_val_tune30_fp16.npz')
    main_cache = os.path.join(a.results, 'preds', f'{RUN}_floodnet_val_tune30.npz')
    probs, gts = load_cache(main_cache, root)
    ids = [str(i) for i in np.load(main_cache)['ids']]
    t0 = time.time()

    S = Scorer(probs, gts, a.workers)
    tau0, r0 = tune_tau(S)
    print(f'base: tau {tau0} miou {r0["miou"]:.4f} drvP {r0["drivable_precision"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
    if 'base' in only:
        rows.append(row('current model, tuned tau (reference)', 'base', dict(tau=tau0), r0))

    if 'i27' in only:
        b, r = tune_bias(S)
        rows.append(row('bias offsets (alone)', 'I27', dict(bias=b), r))
        print('I27', b, round(r['miou'], 4), flush=True)
    if 'i28' in only:
        best = None
        for t_w in (0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
            tau, r = tune_tau(S, 'two', dict(t_w=t_w), start=tau0)
            print('I28 t_w', t_w, tau, round(r['miou'], 4), flush=True)
            if best is None or r['miou'] > best[2]['miou']:
                best = (t_w, tau, r)
        rows.append(row('two-step decode (wet vs dry, then class)', 'I28', dict(t_w=best[0], tau=best[1]), best[2]))
    if 'i29' in only:
        best = None
        for k in (0, 50, 100, 200, 400, 800, 1600, 3200, 6400):
            for r_ in (0, 1, 2, 4, 6, 8):
                if not k and not r_:
                    continue
                r = S('clean', tau=tau0, k=k, r=r_)
                if r['drivable_precision'] >= P_MIN - 1e-9 and (best is None or r['miou'] > best[2]['miou']):
                    best = (k, r_, r)
        print('I29', best[:2], round(best[2]['miou'], 4), flush=True)
        rows.append(row('clean-up (small specks filled, road/water closing)', 'I29', dict(tau=tau0, k=best[0], r=best[1]), best[2]))
    S.close()

    if 'i22' in only:
        snaps = {s: load_cache(pred(s), root)[0] for s in STEPS}
        for sub in ((1000, 1500), (1000, 1500, 2000), (500, 1000, 1500), (500, 1000, 1500, 2000), (1500, 2000)):
            avg = [np.mean([snaps[s][i] for s in sub], 0) for i in range(len(gts))]
            S2 = Scorer(avg, gts, a.workers)
            tau, r = tune_tau(S2)
            S2.close()
            print('I22', sub, tau, round(r['miou'], 4), flush=True)
            rows.append(row(f'average of snapshots {"+".join(map(str, sub))}', 'I22', dict(steps=sub, tau=tau), r,
                            cost=f'{len(sub)}x prediction'))
        del snaps
    if 'i30' in only:
        imgs = [np.asarray(Image.open(os.path.join(root, 'val', 'images', f'{i}.jpg')).convert('L'), np.float32) / 255
                for i in ids]
        for rad, eps in ((2, 1e-2), (2, 1e-3), (4, 1e-2), (4, 1e-3), (8, 1e-2), (8, 1e-3), (16, 1e-2), (16, 1e-1), (32, 1e-2)):
            filt = [guided_filter(I, P, rad, eps) for I, P in zip(imgs, probs)]
            S2 = Scorer(filt, gts, a.workers)
            tau, r = tune_tau(S2, start=tau0)
            S2.close()
            print('I30 r', rad, 'eps', eps, tau, round(r['miou'], 4), flush=True)
            rows.append(row(f'guided filter r={rad} eps={eps:g}', 'I30', dict(r=rad, eps=eps, tau=tau), r))
    if 'combo' in only:
        # the no-cost winners stacked: guided filter (I30) -> two-step decode (I28) -> clean-up (I29), each re-tuned in order
        imgs = [np.asarray(Image.open(os.path.join(root, 'val', 'images', f'{i}.jpg')).convert('L'), np.float32) / 255
                for i in ids]
        for gf in (None, (2, 1e-2), (4, 1e-2), (4, 1e-3)):
            pp = [guided_filter(I, P, *gf) for I, P in zip(imgs, probs)] if gf else probs
            S2 = Scorer(pp, gts, a.workers)
            best = None
            for t_w in (0.2, 0.25, 0.3, 0.4):
                tau, r = tune_tau(S2, 'combo', dict(t_w=t_w), start=tau0)
                if best is None or r['miou'] > best[2]['miou']:
                    best = (t_w, tau, r)
            t_w, tau, r2 = best
            bk = (0, 0, r2)
            for k in (0, 200, 800, 1600, 3200):
                for r_ in (0, 1, 2, 4, 6):
                    if not k and not r_:
                        continue
                    r = S2('combo', tau=tau, t_w=t_w, k=k, r=r_)
                    if r['drivable_precision'] >= P_MIN - 1e-9 and r['miou'] > bk[2]['miou']:
                        bk = (k, r_, r)
            S2.close()
            name = (f'guided filter r={gf[0]} eps={gf[1]:g} + ' if gf else '') + f'two-step t_w={t_w} + clean-up k={bk[0]} r={bk[1]}'
            print('COMBO', name, tau, round(bk[2]['miou'], 4), flush=True)
            rows.append(row(name, 'combo', dict(gf=gf, t_w=t_w, tau=tau, k=bk[0], r=bk[1]), bk[2]))
    if 'i32' in only:
        # p_wet = the routers' wet probability: max over the wet channels (road_flooded, bldg_flooded, water)
        bins = np.linspace(0, 1, 11)
        tot, conf, acc = np.zeros(10), np.zeros(10), np.zeros(10)
        for P, g in zip(probs, gts):
            v = g != 255
            pw = P[list(WET[DS])].max(0)[v]
            y = np.isin(g[v], WET[DS])
            b = np.clip(np.digitize(pw, bins) - 1, 0, 9)
            tot += np.bincount(b, minlength=10)
            conf += np.bincount(b, pw, minlength=10)
            acc += np.bincount(b, y, minlength=10)
        ece = float(np.sum(np.abs(acc - conf)) / tot.sum())
        rel = [dict(bin=f'{bins[i]:.1f}-{bins[i + 1]:.1f}', n=int(tot[i]), mean_p=float(conf[i] / max(tot[i], 1)),
                    frac_wet=float(acc[i] / max(tot[i], 1))) for i in range(10)]
        rows.append(dict(name='p_wet calibration (val30)', idea='I32', params=dict(ece=ece, reliability=rel), cost='none',
                         val30=None, delta=None, keep=None,
                         note='gates are val30-tuned thresholds: monotone recalibration changes no status; re-tune the gates if the model changes'))
        print('I32 ECE', round(ece, 4), flush=True)

    json.dump(rows, open(f_json, 'w'), indent=1)
    lines = ['| idea | variant | cost | val30 tuned mIoU | Δ vs .4835 | drivable P | kept |', '|---|---|---|---|---|---|---|']
    for r in rows:
        if r['val30'] is None:
            lines.append(f"| {r['idea']} | {r['name']} | {r['cost']} | ECE {r['params']['ece']:.3f} | | | {r['note']} |")
            continue
        lines.append(f"| {r['idea']} | {r['name']} | {r['cost']} | {r['val30']['miou']:.4f} | {r['delta']:+.4f} | "
                     f"{r['val30']['drivable_precision']:.3f} | {'yes' if r['keep'] else 'no'} |")
    open(os.path.join(a.out, 'postproc.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
