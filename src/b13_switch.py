"""Batch 13.6: make the Batch 13 model the FloodNet 50-shot headline, in place, reversibly. CPU.

    python src/b13_switch.py install --final results/b13/final_run/rr --decode results/b13/r2/decode_wavg.json
    python src/b13_switch.py restore            # back to the published files (from the archive)

install:
  1. archive (once): results/kaggle_v2/rr/{preds,ckpt,decode,metrics,logs} files of floodnet_50shot_s0_stratified,
     results/kaggle_v2/report, results/routing, app/scenes -> results/published_7oct.tar (restore puts them back);
  2. new caches under the published names: val30, full test (parts + merged), eval100 (the eval100 photos of the full
     test), RescueNet transfer; checkpoint = the weight-averaged Batch 13 checkpoint;
  3. decode/floodnet_50shot_s0_stratified.json: frozen Batch 13 chain (guided filter -> tau -> clean-up -> SAM grow),
     with precomputed class maps per split (classmaps/classmaps_<split>.npz; SAM is the slow step) and its metrics;
  4. the learning curve keeps the published baseline-recipe 50-shot point: its caches stay as run
     floodnet_50shot_s0_stratified_pub (for the curve / write-up comparison).
Then: python src/rebuild_all.py -j 4 --only <all but tta> (the flip report belongs to the published model).
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tarfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr.final_decode import decode_photo  # noqa: E402
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402

R = 'results/kaggle_v2/rr'
FN50 = 'floodnet_50shot_s0_stratified'
ARCH = 'results/published_7oct.tar'
NAMES = ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water']
CKPT = 'floodnet_50shot_s0_stratified_b13ctl_wavg750-1000'


def archive():
    if os.path.exists(ARCH):
        print('archive exists:', ARCH)
        return
    with tarfile.open(ARCH, 'w') as t:
        for sub in ('preds', 'ckpt', 'decode', 'metrics', 'logs'):
            for f in glob.glob(f'{R}/{sub}/{FN50}*'):
                t.add(f)
        for d in ('results/kaggle_v2/report', 'results/routing', 'app/scenes'):
            t.add(d)
    print('archived ->', ARCH, os.path.getsize(ARCH) // 2 ** 20, 'MB')


def restore():
    with tarfile.open(ARCH) as t:
        t.extractall('.', filter='data')
    for f in glob.glob(f'{R}/classmaps/*.npz') + glob.glob(f'{R}/*/{FN50}_pub*'):
        os.remove(f)
    print('restored the published files from', ARCH)


def classmaps(split_file, cache, ds, folder, dec, data):
    z = np.load(cache)
    ids = [str(i) for i in z['ids']]
    maps, cm = [], np.zeros((6, 6), np.int64)
    for i, P in zip(ids, z['probs']):
        img = np.asarray(Image.open(os.path.join(data, f'{ds}_r768', folder, 'images', f'{i}.jpg')).convert('RGB'))
        c = decode_photo(P.astype(np.float32) / 255, dec, img=img)
        maps.append(c)
        if ds == 'floodnet':
            cm += confusion(c, gt_classes(np.asarray(Image.open(os.path.join(data, f'{ds}_r768', folder, 'labels', f'{i}.png')))))
    os.makedirs(f'{R}/classmaps', exist_ok=True)
    f = f'{R}/classmaps/classmaps_{FN50}_{split_file}.npz'
    np.savez_compressed(f, ids=np.array(ids), maps=np.stack(maps))
    print(split_file, len(ids), 'class maps ->', f, flush=True)
    return f, (summarize(cm, NAMES, WET['floodnet']) if ds == 'floodnet' else None)


def install(a):
    archive()
    src = lambda name: os.path.join(a.final, 'preds', name)
    # keep the published baseline-recipe caches for the learning curve as run <FN50>_pub
    for f in glob.glob(f'{R}/preds/{FN50}_*.npz'):
        base = os.path.basename(f)
        if any(t in base for t in ('_tta4', '_step', 'classmaps', '_pub')):
            continue
        shutil.copy(f, f'{R}/preds/{FN50}_pub' + base[len(FN50):])
    for f in glob.glob(f'{R}/metrics/{FN50}*.json') + [f'{R}/decode/{FN50}.json', f'{R}/logs/{FN50}.json']:
        base = os.path.basename(f)
        if os.path.exists(f) and '_pub' not in base and '_step' not in base:
            shutil.copy(f, os.path.join(os.path.dirname(f), f'{FN50}_pub' + base[len(FN50):]))
    # new caches under the published names
    p1 = np.load(src(f'{CKPT}_floodnet_test_full_part1_fp16.npz'))
    p2 = np.load(src(f'{CKPT}_floodnet_test_full_part2_fp16.npz'))
    shutil.copy(src(f'{CKPT}_floodnet_test_full_part1_fp16.npz'), f'{R}/preds/{FN50}_floodnet_test_full_part1_fp16.npz')
    shutil.copy(src(f'{CKPT}_floodnet_test_full_part2_fp16.npz'), f'{R}/preds/{FN50}_floodnet_test_full_part2_fp16.npz')
    ids = np.concatenate([p1['ids'], p2['ids']])
    probs = np.concatenate([p1['probs'], p2['probs']])
    np.savez_compressed(f'{R}/preds/{FN50}_floodnet_test_full_fp16.npz', probs=probs, ids=ids, cm=p1['cm'] + p2['cm'])
    e100 = [l.strip() for l in open('splits/floodnet_test_eval100.txt') if l.strip()]
    pos = {str(i): k for k, i in enumerate(ids)}
    sel = [pos[i] for i in e100]
    np.savez_compressed(f'{R}/preds/{FN50}_floodnet_test_eval100_fp16.npz', probs=probs[sel], ids=np.array(e100), cm=np.zeros((6, 6), np.int64))
    shutil.copy(src(f'{CKPT}_floodnet_val_tune30_fp16.npz'), f'{R}/preds/{FN50}_floodnet_val_tune30.npz')
    shutil.copy(src(f'{CKPT}_rescuenet_test_eval100_fp16.npz'), f'{R}/preds/{FN50}_rescuenet_test_eval100_fp16.npz')
    shutil.copy(a.ckpt, f'{R}/ckpt/{FN50}.ckpt')
    # frozen decode with precomputed class maps
    best = max(json.load(open(a.decode)), key=lambda r: r['miou'])
    chain = dict(tau=best['tau'], gf=best['gf'], sam=dict(mode='grow', min_area=50))
    if best['k'] or best['r']:
        chain['clean'] = [best['k'], best['r']]
    maps, val = {}, {}
    for sf, cache, ds, folder in (('floodnet_val_tune30', f'{R}/preds/{FN50}_floodnet_val_tune30.npz', 'floodnet', 'val'),
                                  ('floodnet_test_full', f'{R}/preds/{FN50}_floodnet_test_full_fp16.npz', 'floodnet', 'test'),
                                  ('rescuenet_test_eval100', f'{R}/preds/{FN50}_rescuenet_test_eval100_fp16.npz', 'rescuenet', 'test')):
        maps[sf], val[sf] = classmaps(sf, cache, ds, folder, chain, a.data)
    maps['floodnet_test_eval100'] = maps['floodnet_test_full']      # same photos (ids looked up in the full-test maps)
    # tau 0.5 ('default') and the chain ('b13') on val30 / eval100 (decode_tune --table format) + metrics JSONs (tau 0.5)
    e100 = set(e100)
    def plain(cache, folder, keep=None):
        z = np.load(cache)
        cm, cmb = np.zeros((6, 6), np.int64), np.zeros((6, 6), np.int64)
        mp = dict(zip([str(i) for i in np.load(maps[sf_of[cache]])['ids']], np.load(maps[sf_of[cache]])['maps']))
        for i, P in zip([str(x) for x in z['ids']], z['probs']):
            if keep and i not in keep:
                continue
            g = gt_classes(np.asarray(Image.open(os.path.join(a.data, 'floodnet_r768', folder, 'labels', f'{i}.png'))))
            cm += confusion(decode(P.astype(np.float32) / 255, tau=0.5), g)
            cmb += confusion(mp[i], g)
        return summarize(cm, NAMES, WET['floodnet']), dict(summarize(cmb, NAMES, WET['floodnet']), confusion=cmb.tolist())
    sf_of = {f'{R}/preds/{FN50}_floodnet_val_tune30.npz': 'floodnet_val_tune30',
             f'{R}/preds/{FN50}_floodnet_test_full_fp16.npz': 'floodnet_test_full'}
    v05, vb = plain(f'{R}/preds/{FN50}_floodnet_val_tune30.npz', 'val')
    e05, eb = plain(f'{R}/preds/{FN50}_floodnet_test_full_fp16.npz', 'test', e100)
    f05, fb = plain(f'{R}/preds/{FN50}_floodnet_test_full_fp16.npz', 'test')
    d = dict(run=FN50, dataset='floodnet', frozen=True, batch13=dict(checkpoint=CKPT, decode_source=a.decode),
             params=dict(chosen='b13', tau=best['tau'], gf=best['gf'], clean=chain.get('clean'), sam='grow, min area 50',
                         bldg_t=None, ctx=None),
             chosen=dict(chain, classmaps=maps, variant='b13'),
             val30={'default': v05, 'b13': vb}, eval100={'default': e05, 'b13': eb}, full_test={'default': f05, 'b13': fb})
    json.dump(d, open(f'{R}/decode/{FN50}.json', 'w'), indent=1)
    meta = dict(run_id=FN50, eval_shot=50, tau=0.5, amp='fp16', gpu='Tesla T4', batch13=True, checkpoint=CKPT)
    json.dump(dict(v05, split='floodnet_val_tune30', n_images=30, **meta, best_step='750+1000 weight average'),
              open(f'{R}/metrics/{FN50}.json', 'w'), indent=1)
    json.dump(dict(e05, split='floodnet_test_eval100', n_images=100, **meta),
              open(f'{R}/metrics/{FN50}_floodnet_test_eval100_fp16.json', 'w'), indent=1)
    for k in (1, 2):
        mf = os.path.join(a.final, 'metrics', f'{CKPT}_floodnet_test_full_part{k}_fp16.json')
        if os.path.exists(mf):
            shutil.copy(mf, f'{R}/metrics/{FN50}_floodnet_test_full_part{k}_fp16.json')
    mf = os.path.join(a.final, 'metrics', f'{CKPT}_rescuenet_test_eval100_fp16.json')
    if os.path.exists(mf):
        shutil.copy(mf, f'{R}/metrics/{FN50}_rescuenet_test_eval100_fp16.json')
    print('val30 τ.5', round(v05['miou'], 4), 'chain', round(vb['miou'], 4), '| eval100', round(e05['miou'], 4), round(eb['miou'], 4),
          '| full', round(f05['miou'], 4), round(fb['miou'], 4))
    print('installed; val30', round(val['floodnet_val_tune30']['miou'], 4), 'full test', round(val['floodnet_test_full']['miou'], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['install', 'restore', 'archive'])
    ap.add_argument('--final', default='results/b13/final_run/rr')
    ap.add_argument('--decode', default='results/b13/r2/decode_wavg_noclean.json')
    ap.add_argument('--ckpt', default=f'results/b13/{CKPT}.ckpt')
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    a = ap.parse_args()
    {'install': lambda: install(a), 'restore': restore, 'archive': archive}[a.cmd]()


if __name__ == '__main__':
    main()
