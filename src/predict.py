"""Stage-2 tiled inference for one checkpoint on one split; writes the prediction cache (PLAN §4.3).

    python src/predict.py --ckpt <drive>/ckpt/floodnet_20shot_s0_stratified.ckpt \
        --dataset floodnet --split_file floodnet_test_eval100 --split test --eval_shot 20

Output: preds/<run_id>_<split_file>[_es<k>][_fp16].npz (uint8 probabilities, ids, confusion) and
metrics/<same>.json. --limit N predicts only the first N images (timing). --amp fp16 runs inference
under fp16 autocast (T4); the PLAN §0.1 check compares one checkpoint's fp16 and fp32 metrics.
--query_dataset rescuenet (with a FloodNet checkpoint and --split_file rescuenet_test_eval100): zero-label
transfer (PLAN §3.4): the supports stay the checkpoint's own FloodNet ones, the queries come from the
other dataset; its metrics JSON is over FloodNet channel names, so score super-classes from the cache.
--flip h|v|hv: test-time flip (Batch 7): the query is flipped horizontally / vertically / both before tiling and
the probabilities are flipped back, so the cache (suffix _flip<f>) is aligned with the photo; src/tta.py averages
these with the unflipped cache. The 384/192 tile grid is symmetric at 768x576, so the tiling itself is mirrored.
Batch 13 options (defaults = the published predictions bit for bit; any non-default adds _<tag> to the cache name,
--tag sets it): --views K support-view ensemble (I1: K different support crops per labelled photo, probabilities
averaged), --crop_weights w0,..,w4 (I2: support-crop channel weights), --stride 128 (I24: denser tiles),
--scales 1,1.25 (I25: multi-scale), --compile (I36: torch.compile of the encoder blocks).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr import device as D  # noqa: E402
from finetune import NAMES, SUPPORT_SPLIT, setup_chameleon  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataset', default='floodnet')
    ap.add_argument('--split_file', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--query_dataset', default=None, help='dataset of the query images (default: --dataset)')
    ap.add_argument('--eval_shot', type=int, default=0, help='0 = the shot the checkpoint was trained with')
    ap.add_argument('--support_file', default=None, help='default: supports/<dataset>_<sel>_s<seed> of the run')
    ap.add_argument('--res', default='r768')
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no_save', action='store_true')
    ap.add_argument('--force', action='store_true', help='recompute even if the cache exists')
    ap.add_argument('--chunk', type=int, default=6)
    ap.add_argument('--channel_chunk', type=int, default=5)
    ap.add_argument('--amp', default='none', choices=['none', 'fp16'])
    ap.add_argument('--flip', default='none', choices=['none', 'h', 'v', 'hv'])
    ap.add_argument('--views', type=int, default=1, help='I1: support views averaged')
    ap.add_argument('--crop_weights', default='', help='I2: support-crop channel weights, e.g. 1,3,3,3,1')
    ap.add_argument('--stride', type=int, default=192, help='I24: tile stride')
    ap.add_argument('--scales', default='1', help='I25: comma list of query scales, e.g. 1,1.25')
    ap.add_argument('--compile', action='store_true', help='I36: torch.compile the image-encoder blocks')
    ap.add_argument('--tag', default='', help='suffix of the cache name for non-default options')
    ap.add_argument('--data', default='/content/rr_data')
    ap.add_argument('--runs', default='/content/runs_data')
    ap.add_argument('--drive', default='/content/drive/MyDrive/AIC200-my_work/rescue_router')
    ap.add_argument('--mt_ckpt', default='/content/generalist.ckpt')
    ap.add_argument('--chameleon', default='/content/AIC200-Project-StructuredPrediction/chameleon')
    a = ap.parse_args()
    sys.path.insert(0, a.chameleon)
    from rr.data import build_run_folder, load_query, read_ids
    from rr.runs import support_seed_of
    from rr.infer import compile_encoder, encode_support, predict_multiscale, predict_views, set_support_crop
    from rr.metrics import WET, confusion, decode, gt_classes, summarize

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)['hyper_parameters']['config']
    ck_shot = ck.eval_shot if (getattr(ck, 'eval_shot', 0) or 0) > 0 else ck.shot    # = ck.shot except I20 runs
    shot = a.eval_shot or ck_shot
    run_id = os.path.splitext(os.path.basename(a.ckpt))[0]
    tag = f'{run_id}_{a.split_file}' + ('' if shot == ck_shot else f'_es{shot}') + (f'_n{a.limit}' if a.limit else '') \
        + ('' if a.amp == 'none' else f'_{a.amp}') + ('' if a.flip == 'none' else f'_flip{a.flip}')
    scales = tuple(float(x) for x in a.scales.split(','))
    crop_w = [float(x) for x in a.crop_weights.split(',')] if a.crop_weights else None
    nondefault = a.views != 1 or crop_w or a.stride != 192 or scales != (1.0,) or a.compile
    if nondefault or a.tag:
        tag += '_' + (a.tag or '_'.join(filter(None, [f'v{a.views}' if a.views != 1 else '',
                                                        'cw' + ''.join(f'{w:g}' for w in crop_w) if crop_w else '',
                                                        f'st{a.stride}' if a.stride != 192 else '',
                                                        'sc' + '-'.join(f'{x:g}' for x in scales) if scales != (1.0,) else '',
                                                        'comp' if a.compile else ''])))
    if not (a.no_save or a.force) and os.path.exists(os.path.join(a.drive, 'preds', f'{tag}.npz')):
        print(f'[{tag}] cached already, skipping')
        return
    seed, sel = support_seed_of(ck), run_id.split('_')[3] if len(run_id.split('_')) > 3 else 'stratified'
    sup_file = a.support_file or os.path.join(a.drive, 'supports', f'{a.dataset}_{sel}_s{seed}.txt')
    supports = read_ids(sup_file)[:shot]
    data_root = os.path.join(a.data, f'{a.dataset}_{a.res}')
    query_root = os.path.join(a.data, f'{a.query_dataset or a.dataset}_{a.res}')
    run_root = build_run_folder(os.path.join(a.runs, f'{run_id}_infer'), data_root, supports,
                                split_support=SUPPORT_SPLIT[a.dataset])
    su = setup_chameleon(a.chameleon, run_root)
    ov = su.derive_stage2_overrides(a.ckpt, verbose=False)
    ov.update(shot=shot, eval_shot=shot, chunk_size=a.chunk, channel_chunk_size=a.channel_chunk, n_tasks=5,
              loss_type='bce', img_size=384, base_size=list(ck.base_size))
    cfg = su.build_config(stage=2, dataset='custom', **ov)
    from rr.lora import is_lora_ckpt, load_chameleon_lora
    if is_lora_ckpt(a.ckpt):                    # I9 checkpoints carry LoRA adapters the starter cannot load
        model, cfg = load_chameleon_lora(su, cfg, a.mt_ckpt, a.ckpt)
    else:
        model, cfg = su.load_chameleon(cfg, a.mt_ckpt, a.ckpt, verbose=False)
    model.eval()

    ids = read_ids(os.path.join(a.drive, 'splits', f'{a.split_file}.txt'))
    if a.limit:
        ids = ids[:a.limit]
    if a.compile:
        print(f'compiled {compile_encoder(model)} encoder blocks', flush=True)
    D.reset_peak()
    dims = {'none': [], 'h': [-1], 'v': [-2], 'hv': [-2, -1]}[a.flip]
    cm, probs = np.zeros((6, 6), np.int64), []
    if a.views == 1:
        set_support_crop(0, crop_w)
        t_sup = encode_support(model, amp=a.amp)
        sup_peak = D.peak_gb()
        t0 = time.time()
        for i in ids:
            X, lab = load_query(query_root, a.split, i)
            if dims:
                X = torch.flip(X, dims)
            P = predict_multiscale(model, X, scales, a.stride, a.amp)
            P = (torch.flip(P, dims) if dims else P).numpy()
            cm += confusion(decode(P, a.tau), gt_classes(lab))
            probs.append(np.round(P * 255).astype(np.uint8))
    else:
        t0 = time.time()
        Ps, labs, t_sup = predict_views(model, lambda i: load_query(query_root, a.split, i), ids, a.views, crop_w,
                                        scales, a.stride, a.amp, dims)
        sup_peak = D.peak_gb()
        for P, lab in zip(Ps, labs):
            cm += confusion(decode(P, a.tau), gt_classes(lab))
            probs.append(np.round(P * 255).astype(np.uint8))
    D.sync()
    res = summarize(cm, NAMES[a.dataset], WET[a.dataset])
    res.update(run_id=run_id, split=a.split_file, query_dataset=a.query_dataset or a.dataset, eval_shot=shot, tau=a.tau, n_images=len(ids), amp=a.amp,
               flip=a.flip, views=a.views, crop_weights=crop_w, stride=a.stride, scales=list(scales), compiled=a.compile,
               support_encode_s=round(t_sup, 2), s_per_image=round((time.time() - t0) / max(len(ids), 1), 2),
               support_peak_gb=round(sup_peak, 2),
               infer_peak_gb=round(D.peak_gb(), 2),
               gpu=D.name())
    print(json.dumps(res, indent=1), flush=True)
    if not a.no_save:
        os.makedirs(os.path.join(a.drive, 'preds'), exist_ok=True)
        os.makedirs(os.path.join(a.drive, 'metrics'), exist_ok=True)
        np.savez_compressed(os.path.join(a.drive, 'preds', f'{tag}.npz'), probs=np.stack(probs), ids=np.array(ids), cm=cm)
        json.dump(res, open(os.path.join(a.drive, 'metrics', f'{tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
