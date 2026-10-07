"""Lossless packing of the prediction caches for the data zip (Batch 8, PLAN §2.6). CPU only.

    python src/cache_pack.py pack   --out <dir>   [--results results/kaggle_v2/rr]   # preds/*.npz -> <dir>/preds/*.ptar
    python src/cache_pack.py unpack --src <dir>   [--results results/kaggle_v2/rr]   # back to identical .npz files

A cache (preds/<name>.npz: probs uint8 (N, 5, 576, 768), ids, optional cm) becomes <name>.ptar, a plain tar with
manifest.json, any other arrays as .npy, and one lossless WebP per image (the 5 channels stacked vertically as a
grayscale 2880x768 image). Its prediction suits the smooth probability maps: 53 % of the compressed npz (PNG: 63 %),
and exact (uint8 in, uint8 out). `unpack` rebuilds the npz and checks every array against the pack's SHA-256 manifest, so the
caches the scripts read are bit-identical to the GPU outputs. Which caches go into the data zip: PACK below.
"""

import argparse
import hashlib
import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

# the caches every reported number needs (PLAN §2.6): eval100 + val30 of all 7 runs, the full FloodNet test of the
# headline model, and its zero-label transfer to RescueNet. Not packed: step snapshots (checkpoint selection, kept in
# the logs), the full-test halves (merged), test-time flips (not adopted; reproducible with kaggle/flips).
RUNS = ['floodnet_1shot', 'floodnet_5shot', 'floodnet_10shot', 'floodnet_20shot', 'floodnet_50shot',
        'rescuenet_20shot', 'rescuenet_50shot']
PACK = ([f'{r}_s0_stratified_{r.split("_")[0]}_test_eval100_fp16' for r in RUNS]
        + [f'{r}_s0_stratified_{r.split("_")[0]}_val_tune30' for r in RUNS]
        + ['floodnet_50shot_s0_stratified_floodnet_test_full_fp16',
           'floodnet_50shot_s0_stratified_rescuenet_test_eval100_fp16']
        # Batch 13: the published FloodNet 50-shot model's caches (run <...>_pub), for the paired comparison
        + ['floodnet_50shot_s0_stratified_pub_floodnet_test_full_fp16', 'floodnet_50shot_s0_stratified_pub_floodnet_test_eval100_fp16',
           'floodnet_50shot_s0_stratified_pub_floodnet_val_tune30'])


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mtime = len(data), 0
    tar.addfile(info, io.BytesIO(data))


def pack_one(job):
    src, dst = job
    z = np.load(src)
    probs = z['probs']
    ids = [str(i) for i in z['ids']]
    man = dict(shape=list(probs.shape), dtype=str(probs.dtype), ids=ids, sha256={'probs': sha(probs)}, extra=[])
    with tarfile.open(dst, 'w') as tar:
        for k in z.files:
            if k not in ('probs', 'ids'):
                b = io.BytesIO()
                np.save(b, z[k])
                add(tar, f'{k}.npy', b.getvalue())
                man['extra'].append(k)
                man['sha256'][k] = sha(z[k])
        for i, p in zip(ids, probs):
            b = io.BytesIO()
            Image.fromarray(p.reshape(-1, p.shape[-1])).save(b, 'WEBP', lossless=True, quality=100, method=4)
            add(tar, f'{i}.webp', b.getvalue())
        add(tar, 'manifest.json', json.dumps(man).encode())
    return os.path.basename(dst), os.path.getsize(src), os.path.getsize(dst)


def unpack_one(job):
    src, dst = job
    with tarfile.open(src) as tar:
        man = json.loads(tar.extractfile('manifest.json').read())
        C, H, W = man['shape'][1:]
        probs = np.stack([np.asarray(Image.open(io.BytesIO(tar.extractfile(f'{i}.webp').read())).convert('L')).reshape(C, H, W)
                          for i in man['ids']])
        extra = {k: np.load(io.BytesIO(tar.extractfile(f'{k}.npy').read())) for k in man['extra']}
    assert sha(probs) == man['sha256']['probs'], f'{src}: probabilities differ from the original'
    for k, v in extra.items():
        assert sha(v) == man['sha256'][k], f'{src}: {k} differs'
    np.savez_compressed(dst, probs=probs, ids=np.array(man['ids']), **extra)
    return os.path.basename(dst), probs.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['pack', 'unpack'])
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--out', help='pack: output folder (gets preds/)')
    ap.add_argument('--src', help='unpack: folder holding preds/*.ptar')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    preds = os.path.join(a.results, 'preds')
    if a.cmd == 'pack':
        os.makedirs(os.path.join(a.out, 'preds'), exist_ok=True)
        jobs = [(os.path.join(preds, f'{n}.npz'), os.path.join(a.out, 'preds', f'{n}.ptar')) for n in PACK]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(pack_one, jobs))
        for n, s0, s1 in rows:
            print(f'{n:70s} {s0 / 1e6:7.1f} MB -> {s1 / 1e6:7.1f} MB')
        print(f'total {sum(r[1] for r in rows) / 1e6:.0f} MB npz -> {sum(r[2] for r in rows) / 1e6:.0f} MB packed')
    else:
        os.makedirs(preds, exist_ok=True)
        srcs = sorted(f for f in os.listdir(os.path.join(a.src, 'preds')) if f.endswith('.ptar'))
        jobs = [(os.path.join(a.src, 'preds', f), os.path.join(preds, f[:-5] + '.npz')) for f in srcs]
        with ProcessPoolExecutor(a.workers) as ex:
            for n, k in ex.map(unpack_one, jobs):
                print(f'{n}: {k} images, verified')


if __name__ == '__main__':
    main()
