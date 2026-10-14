"""Two-model agreement as a dispatch-safety signal (PLAN §16.5, C2 follow-up). CPU only, from the caches.

The dangerous routing error is a house shown truck-reachable that is not (false safe). The published answer is a
confidence tier from one model: truck-reachable even with a cautious off-road gate (0.05) -> "confident". Here a second,
independently trained model is the check: a house is "confident" only when both models' maps route a truck to it.
Every rule is a house-level dispatch policy scored against the labelled map (router v3, the published parameters):
how many houses it calls truck-reachable, how many of those truly are (1 - false-safe rate), and the stranded F1 of
the status it implies (a house is treated as stranded unless the rule calls it truck-reachable).

Models: 'chameleon' (the submitted model and decode) and baseline runs (results/baselines, argmax decode).
Exploratory: any adoption follows PROTOCOL.md (choice on val30 + confirm100, report on eval100 / full test).

    python src/consensus.py [--a chameleon] [--b baseline_deeplabv3_r50_50shot_s0] [--splits val30,eval100,full]
"""
import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import router as R  # noqa: E402
from bootstrap_ci import f1_from  # noqa: E402
from route_eval import PRED_CAUTIOUS_MAXWET, PRED_OFFROAD_MAXWET  # noqa: E402
from rr.final_decode import decode_photo, load_chosen, photo  # noqa: E402
from rr.metrics import gt_classes  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

REPO = os.path.dirname(HERE)
SPLITS = {'val30': ('floodnet_val_tune30', 'val'), 'confirm100': ('floodnet_val_confirm100', 'val'),
          'eval100': ('floodnet_test_eval100', 'test'), 'full': ('floodnet_test_full', 'test')}
G_T, G_C = PRED_OFFROAD_MAXWET['floodnet'], PRED_CAUTIOUS_MAXWET['floodnet']


def model(name, a):
    """(prediction-cache finder, decode) of a model name"""
    if name == 'chameleon':
        folder, run = a.results, 'floodnet_50shot_s0_stratified'
        dec = load_chosen(folder, run)
    else:
        folder, run, dec = a.baselines, name, {'argmax': True}
    return folder, run, dec


def cache(folder, run, sf):
    for s in ('', '_fp16'):
        p = os.path.join(folder, 'preds', f'{run}_{sf}{s}.npz')
        if os.path.exists(p):
            return p
    return None


def photo_rows(job):
    """per GT building of one photo: labelled-map status, and each model's status at the tuned and cautious gates"""
    i, sf, split, maps = job
    gt = gt_classes(np.asarray(Image.open(os.path.join(RR_DATA, 'floodnet_r768', split, 'labels', f'{i}.png'))))
    gcls = R.cell_classes(gt)
    base = R.base_cells(gcls, 'floodnet')
    if base is None:
        return []
    blds = R.buildings_from_gt(gt, 'floodnet')
    gsol = R.Grid(gcls, R.onehot_probs(gt), 'floodnet').solve(base)
    sols = {}
    for m, (cls, probs) in maps.items():
        pc, pp = R.cell_classes(cls), R.cell_probs(probs)
        sols[m] = {g: R.Grid(pc, pp, 'floodnet', offroad_maxwet=g).solve(base) for g in (G_T, G_C)}
    return [dict(image=i, gt=gsol.status(b), **{f'{m}@{g}': s[g].status(b) for m, s in sols.items() for g in s})
            for b in blds]


def jobs_for(a, names, sf, split):
    """yields (photo, split file, folder, {model: (class map, uint8 probs)}) one photo at a time"""
    loaded = []
    for name in names:
        folder, run, dec = model(name, a)
        z = np.load(cache(folder, run, sf))
        loaded.append((name, dec, dict(zip([str(x) for x in z['ids']], z['probs']))))
    for i in loaded[0][2]:
        maps = {}
        for name, dec, P in loaded:
            sfm = sf if sf in (dec.get('classmaps') or {}) else 'floodnet_test_full'
            maps[name] = (decode_photo(P[i].astype(np.float32) / 255, dec, img=photo(RR_DATA, 'floodnet', split, i),
                                       split_file=sfm, photo_id=i, root=REPO), P[i])
        yield i, sf, split, maps


def score(rows, truck_rule):
    """the house-level policy: TRUCK if truck_rule(row) else stranded; confusion in the router's 4x4 layout"""
    cm = np.zeros((4, 4), np.int64)
    for r in rows:
        cm[r['gt'], R.TRUCK if truck_rule(r) else R.ISOLATED] += 1
    n = int(cm[:, 0].sum())
    return dict(n_truck=n, truly_truck=float(cm[0, 0] / n) if n else None, false_safe=R.false_safe_from(cm),
                stranded_f1=f1_from(cm), n_houses=int(cm.sum()), n_gt_truck=int(cm[0].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='chameleon')
    ap.add_argument('--b', default='baseline_deeplabv3_r50_50shot_s0')
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--baselines', default='results/baselines')
    ap.add_argument('--splits', default='val30,eval100,full')
    ap.add_argument('--out', default='results/routing/consensus')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    A, B, T = f'{a.a}@{G_T}', f'{a.b}@{G_T}', R.TRUCK
    rules = {f'{a.a} (tuned gate {G_T})': lambda r: r[A] == T,
             f'{a.a} cautious tier (gate {G_C})': lambda r: r[f'{a.a}@{G_C}'] == T,
             f'{a.b} (tuned gate {G_T})': lambda r: r[B] == T,
             f'{a.b} cautious (gate {G_C})': lambda r: r[f'{a.b}@{G_C}'] == T,
             'both (tuned gates)': lambda r: r[A] == T and r[B] == T,
             f'{a.a} cautious and {a.b} tuned': lambda r: r[f'{a.a}@{G_C}'] == T and r[B] == T,
             'both cautious': lambda r: r[f'{a.a}@{G_C}'] == T and r[f'{a.b}@{G_C}'] == T,
             'either (tuned gates)': lambda r: r[A] == T or r[B] == T}
    res = {}
    for key in a.splits.split(','):
        sf, split = SPLITS[key]
        if not all(cache(*model(m, a)[:2], sf) for m in (a.a, a.b)):
            print('skip', key, '(no cache)')
            continue
        rows, jobs = [], jobs_for(a, (a.a, a.b), sf, split)
        with ProcessPoolExecutor(a.workers) as ex:              # bounded: 4 photos per worker in flight
            while batch := list(itertools.islice(jobs, 4 * a.workers)):
                rows += [r for rs in ex.map(photo_rows, batch) for r in rs]
        res[key] = {name: score(rows, f) for name, f in rules.items()}
        json.dump(rows, open(os.path.join(a.out, f'houses_{key}.json'), 'w'))
        print(key, {n: (v['n_truck'], round(v['truly_truck'] or 0, 3)) for n, v in res[key].items()}, flush=True)
    json.dump(res, open(os.path.join(a.out, 'consensus.json'), 'w'), indent=1)
    md = [f'# Two-model agreement as a dispatch check: {a.a} and {a.b} (FloodNet, router v3)', '',
          'A rule calls a house truck-reachable; "truly" = share of those that are truck-reachable on the labelled map '
          '(1 − false-safe rate); F1 = stranded-house F1 of the status the rule implies. Exploratory (PROTOCOL.md).', '']
    for key, r in res.items():
        md += [f'## {key} ({next(iter(r.values()))["n_houses"]} houses, {next(iter(r.values()))["n_gt_truck"]} truck-reachable '
               'on the labelled map)', '', '| rule | houses called truck-reachable | truly | false safe | stranded F1 |',
               '|---|---|---|---|---|']
        md += [f"| {n} | {v['n_truck']} | {v['truly_truck']:.3f} | {v['false_safe']:.3f} | {v['stranded_f1']:.3f} |"
               for n, v in r.items()]
        md.append('')
    open(os.path.join(a.out, 'consensus.md'), 'w').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
