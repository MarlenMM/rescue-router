"""Device control for the baselines moved from the Mac (MPS) to a Kaggle T4 (CUDA) on 16 Oct (PLAN §16.8): the same
RescueNet 20-photo baseline (draw 0, seed 0, same script and recipe) trained on both; paired bootstrap of the
full-test and eval100 mIoU (argmax decode, 2,000 resamples over photos, the same resamples for both).

    python experiments/device_control.py      # -> results/baselines/report/device_control.{json,md}
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
from baseline_report import Labels, miou_fn  # noqa: E402
from bootstrap_ci import boot  # noqa: E402
from rr.metrics import confusion  # noqa: E402

RUN = 'baseline_deeplabv3_r50_rescuenet_20shot_s0'
FOLDERS = {'MPS (Mac, 16 Oct 01:41)': 'results/baselines', 'CUDA (Kaggle T4, rr-p2-baselines)': 'results/baselines_cuda_control'}


def argmax_cms(path, gts):
    z = np.load(path)
    out = {}
    for i, P in zip(z['ids'], z['probs']):
        P = P.astype(np.float32) / 255
        pred = np.concatenate([P, 1 - P.sum(0, keepdims=True)]).argmax(0).astype(np.uint8)
        out[str(i)] = confusion(pred, gts[str(i)])
    return out


def main():
    labels, fn, out, md = Labels(), miou_fn('rescuenet'), {}, []
    for sf in ('rescuenet_test_full', 'rescuenet_test_eval100'):
        gts = labels('rescuenet', sf)
        cms = {k: argmax_cms(os.path.join(REPO, f, 'preds', f'{RUN}_{sf}.npz'), gts) for k, f in FOLDERS.items()}
        ids = sorted(set.intersection(*(set(c) for c in cms.values())))
        v = {k: boot(c, ids, fn, 2000, np.random.default_rng(0)) for k, c in cms.items()}
        (ka, (ra, da)), (kb, (rb, db)) = v.items()
        d = db - da
        out[sf] = dict(n=len(ids), **{k: r for k, (r, _) in v.items()},
                       diff=dict(value=rb['value'] - ra['value'], lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5))))
        md.append(f"| {sf} ({len(ids)}) | {ra['value']:.3f} [{ra['lo']:.3f}, {ra['hi']:.3f}] | "
                  f"{rb['value']:.3f} [{rb['lo']:.3f}, {rb['hi']:.3f}] | **{out[sf]['diff']['value']:+.3f} "
                  f"[{out[sf]['diff']['lo']:+.3f}, {out[sf]['diff']['hi']:+.3f}]** |")
    rep = os.path.join(REPO, 'results', 'baselines', 'report')
    json.dump(out, open(os.path.join(rep, 'device_control.json'), 'w'), indent=1)
    md = [f'# Device control: {RUN} trained on MPS and on CUDA (same script, recipe, seed and photos)', '',
          f'| photos | {" | ".join(FOLDERS)} | CUDA − MPS [95 % CI] |', '|---|---|---|---|'] + md
    open(os.path.join(rep, 'device_control.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
