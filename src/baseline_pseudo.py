"""Pseudo-labels for self-training the baseline (PLAN §16.5). No labels are read: the teacher (a baseline_seg.py
checkpoint trained on k labelled photos) predicts the unlabelled photos of the support split, averaged over the four
flips; a pixel keeps the teacher's class (5 classes + other) when its probability is >= --conf, else 255 (ignored by
the student). The supports themselves are left out (they have real labels), and so is every photo of an evaluation
split.

    python src/baseline_pseudo.py --teacher baseline_deeplabv3_r50_50shot_s0 [--conf 0.9] [--device cpu]
    -> results/baselines/pseudo/<teacher>_c90/{ids.txt, labels/<id>.png, meta.json}

then: python src/baseline_seg.py --shot 50 --pseudo results/baselines/pseudo/<teacher>_c90
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baseline_seg import DATASETS, REPO, RR_DATA, build, predict  # noqa: E402

EVAL_SPLITS = ('val_tune30', 'val_confirm100', 'test_full')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True, help='run name of a baseline_seg.py checkpoint')
    ap.add_argument('--arch', default=None, help='default: from the run name')
    ap.add_argument('--dataset', default='floodnet', choices=list(DATASETS))
    ap.add_argument('--conf', type=float, default=0.9)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--baselines', default='results/baselines')
    ap.add_argument('--limit', type=int, default=0, help='first N photos only (test)')
    a = ap.parse_args()
    arch = a.arch or next(x for x in ('deeplabv3_r50', 'deeplabv3_mbv3', 'lraspp_mbv3') if x in a.teacher)
    ds = DATASETS[a.dataset]
    log = json.load(open(os.path.join(a.baselines, 'logs', f'{a.teacher}.json')))
    sup = set(log['supports']) if isinstance(log['supports'], list) else set()
    read = lambda f: open(os.path.join(REPO, 'splits', f)).read().split()        # noqa: E731
    held = set().union(*(read(f'{a.dataset}_{s}.txt') for s in EVAL_SPLITS
                         if os.path.exists(os.path.join(REPO, 'splits', f'{a.dataset}_{s}.txt'))))
    ids = [i for i in read(f'{a.dataset}_support_pool.txt') if i not in sup and i not in held]
    if a.limit:
        ids = ids[:a.limit]
    out = os.path.join(a.baselines, 'pseudo', f'{a.teacher}_c{round(a.conf * 100)}')
    os.makedirs(os.path.join(out, 'labels'), exist_ok=True)
    model = build(arch, a.device, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(a.baselines, 'ckpt', f'{a.teacher}.pt'), map_location=a.device))
    kept, counts, t0 = 0, np.zeros(6, np.int64), time.time()
    for n, i in enumerate(ids, 1):
        x = load_photo(ds['support_split'], i, a.dataset)
        P = predict(model, x, a.device, tta=True)
        full = np.concatenate([P, 1 - P.sum(0, keepdims=True)])
        cls, conf = full.argmax(0).astype(np.uint8), full.max(0)
        cls[conf < a.conf] = 255
        Image.fromarray(cls).save(os.path.join(out, 'labels', f'{i}.png'))
        kept += int((cls != 255).sum())
        counts += np.bincount(cls[cls != 255], minlength=6)
        if n % 100 == 0:
            print(f'{n}/{len(ids)} photos, {(time.time() - t0) / n:.2f} s/photo, kept {kept / (n * cls.size):.1%}', flush=True)
    open(os.path.join(out, 'ids.txt'), 'w').write('\n'.join(ids) + '\n')
    meta = dict(teacher=a.teacher, dataset=a.dataset, conf=a.conf, tta='identity + h/v/hv flips', n_photos=len(ids),
                excluded=dict(supports=len(sup), eval_splits=list(EVAL_SPLITS)),
                kept_pixel_share=kept / (len(ids) * cls.size),
                kept_class_share={k: float(c / max(kept, 1)) for k, c in zip(ds['names'] + ['other'], counts)},
                s_per_photo=round((time.time() - t0) / len(ids), 2))
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


def load_photo(split, i, dataset):
    root = os.path.join(RR_DATA, f'{dataset}_r768', split)
    return torch.from_numpy(np.array(Image.open(os.path.join(root, 'images', f'{i}.jpg')).convert('RGB'))).permute(2, 0, 1)


if __name__ == '__main__':
    main()
