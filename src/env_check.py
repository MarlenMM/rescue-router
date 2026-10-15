"""GPU environment check before a long Kaggle session (PLAN §16.6): the submitted model must reproduce its published
T4 predictions on a few photos, or the session stops before spending its quota on runs that would not be comparable.

    python src/env_check.py ref --cache results/kaggle_v2/rr/preds/floodnet_50shot_s0_stratified_floodnet_test_full_part1_fp16.npz --out ref.npz
    python src/env_check.py check --ref ref.npz --cache <the new cache of the same photos>

The reference keeps the first --n photos as 4x4 block means (uint8 probabilities, ~0.2 MB for 3 photos), small enough
to embed in the kernel. `check` compares block means photo by photo (matched by id): the 8 Oct T4 runs repeat to
within 2/255 even across devices (Apple MPS, PLAN §16.4), so max |diff| <= 8/255 and mean |diff| <= 1/255 pass; a
different package, image or decode gives far more. Exit code 0 = pass, 1 = fail.
"""
import argparse
import sys

import numpy as np

BLOCK = 4
MAX_DIFF, MEAN_DIFF = 8.0, 1.0                       # on the 0-255 scale


def blocks(p):
    """(N, C, H, W) uint8 -> (N, C, H/4, W/4) float32 block means"""
    n, c, h, w = p.shape
    return p.reshape(n, c, h // BLOCK, BLOCK, w // BLOCK, BLOCK).astype(np.float32).mean((3, 5))


def make_ref(cache, n):
    z = np.load(cache)
    return dict(ids=z['ids'][:n], blocks=np.round(blocks(z['probs'][:n])).astype(np.uint8))


def compare(ref, cache):
    """-> (ok, max |diff|, mean |diff|, per-photo max) on the photos both hold; no common photo = fail"""
    z = np.load(cache)
    pos = {str(i): k for k, i in enumerate(z['ids'])}
    ids = [str(i) for i in ref['ids'] if str(i) in pos]
    if not ids:
        return False, float('inf'), float('inf'), {}
    new = blocks(z['probs'][[pos[i] for i in ids]])
    old = ref['blocks'][[k for k, i in enumerate(ref['ids']) if str(i) in pos]].astype(np.float32)
    d = np.abs(new - old)
    per = {i: float(d[k].max()) for k, i in enumerate(ids)}
    ok = bool(np.isfinite(d).all() and d.max() <= MAX_DIFF and d.mean() <= MEAN_DIFF and len(ids) == len(ref['ids']))
    return ok, float(d.max()), float(d.mean()), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['ref', 'check'])
    ap.add_argument('--cache', required=True)
    ap.add_argument('--ref', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--n', type=int, default=3)
    a = ap.parse_args()
    if a.cmd == 'ref':
        r = make_ref(a.cache, a.n)
        np.savez_compressed(a.out, **r)
        print(f'reference: {len(r["ids"])} photos {list(r["ids"])}, blocks {r["blocks"].shape} -> {a.out}')
        return
    z = np.load(a.ref)
    ok, mx, mean, per = compare(dict(ids=z['ids'], blocks=z['blocks']), a.cache)
    print(f'ENVCHECK {"PASS" if ok else "FAIL"}: max |diff| {mx:.2f}/255, mean {mean:.3f}/255 over {len(per)} photos '
          f'(limits {MAX_DIFF:g} / {MEAN_DIFF:g}); per photo {per}', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
