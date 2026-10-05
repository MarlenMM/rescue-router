"""Stage-2 tiled inference for one checkpoint on one split; writes the prediction cache (PLAN §4.3).

    python src/predict.py --ckpt <drive>/ckpt/floodnet_20shot_s0_stratified.ckpt \
        --dataset floodnet --split_file floodnet_test_eval100 --split test --eval_shot 20

Output: preds/<run_id>_<split_file>[_es<k>].npz (uint8 probabilities, ids, confusion) and
metrics/<run_id>_<split_file>[_es<k>].json. --limit N predicts only the first N images (timing).
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
from finetune import NAMES, SUPPORT_SPLIT, setup_chameleon  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataset', default='floodnet')
    ap.add_argument('--split_file', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--eval_shot', type=int, default=0, help='0 = the shot the checkpoint was trained with')
    ap.add_argument('--support_file', default=None, help='default: supports/<dataset>_<sel>_s<seed> of the run')
    ap.add_argument('--res', default='r768')
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no_save', action='store_true')
    ap.add_argument('--chunk', type=int, default=6)
    ap.add_argument('--channel_chunk', type=int, default=5)
    ap.add_argument('--data', default='/content/rr_data')
    ap.add_argument('--runs', default='/content/runs_data')
    ap.add_argument('--drive', default='/content/drive/MyDrive/AIC200-my_work/rescue_router')
    ap.add_argument('--mt_ckpt', default='/content/generalist.ckpt')
    ap.add_argument('--chameleon', default='/content/AIC200-Project-StructuredPrediction/chameleon')
    a = ap.parse_args()
    sys.path.insert(0, a.chameleon)
    from rr.data import build_run_folder, load_query, read_ids
    from rr.infer import encode_support, tiled_predict
    from rr.metrics import WET, confusion, decode, gt_classes, summarize

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)['hyper_parameters']['config']
    shot = a.eval_shot or ck.shot
    run_id = os.path.splitext(os.path.basename(a.ckpt))[0]
    seed, sel = ck.seed, run_id.split('_')[3] if len(run_id.split('_')) > 3 else 'stratified'
    sup_file = a.support_file or os.path.join(a.drive, 'supports', f'{a.dataset}_{sel}_s{seed}.txt')
    supports = read_ids(sup_file)[:shot]
    data_root = os.path.join(a.data, f'{a.dataset}_{a.res}')
    run_root = build_run_folder(os.path.join(a.runs, f'{run_id}_infer'), data_root, supports,
                                split_support=SUPPORT_SPLIT[a.dataset])
    su = setup_chameleon(a.chameleon, run_root)
    ov = su.derive_stage2_overrides(a.ckpt, verbose=False)
    ov.update(shot=shot, eval_shot=shot, chunk_size=a.chunk, channel_chunk_size=a.channel_chunk, n_tasks=5,
              loss_type='bce', img_size=384, base_size=list(ck.base_size))
    cfg = su.build_config(stage=2, dataset='custom', **ov)
    model, cfg = su.load_chameleon(cfg, a.mt_ckpt, a.ckpt, verbose=False)
    model.eval()

    ids = read_ids(os.path.join(a.drive, 'splits', f'{a.split_file}.txt'))
    if a.limit:
        ids = ids[:a.limit]
    torch.cuda.reset_peak_memory_stats()
    t_sup = encode_support(model)
    sup_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    cm, probs, t0 = np.zeros((6, 6), np.int64), [], time.time()
    for i in ids:
        X, lab = load_query(data_root, a.split, i)
        P = tiled_predict(model, X).numpy()
        cm += confusion(decode(P, a.tau), gt_classes(lab))
        probs.append(np.round(P * 255).astype(np.uint8))
    torch.cuda.synchronize()
    res = summarize(cm, NAMES[a.dataset], WET[a.dataset])
    res.update(run_id=run_id, split=a.split_file, eval_shot=shot, tau=a.tau, n_images=len(ids),
               support_encode_s=round(t_sup, 2), s_per_image=round((time.time() - t0) / max(len(ids), 1), 2),
               support_peak_gb=round(sup_peak, 2),
               infer_peak_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
               gpu=torch.cuda.get_device_name(0))
    print(json.dumps(res, indent=1), flush=True)
    if not a.no_save:
        tag = f'{run_id}_{a.split_file}' + ('' if shot == ck.shot else f'_es{shot}') + (f'_n{a.limit}' if a.limit else '')
        os.makedirs(os.path.join(a.drive, 'preds'), exist_ok=True)
        os.makedirs(os.path.join(a.drive, 'metrics'), exist_ok=True)
        np.savez_compressed(os.path.join(a.drive, 'preds', f'{tag}.npz'), probs=np.stack(probs), ids=np.array(ids), cm=cm)
        json.dump(res, open(os.path.join(a.drive, 'metrics', f'{tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
