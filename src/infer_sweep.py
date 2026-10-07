"""I33 (Batch 13): many prediction variants of one split with ONE model load (the 3 GB generalist loads once). GPU.

    python src/infer_sweep.py --ckpt <drive>/ckpt/floodnet_50shot_s0_stratified.ckpt --split_file floodnet_val_tune30 \
        --split val --variants '[{"name": "v2", "views": 2}, {"name": "st128", "stride": 128}, ...]' [predict.py paths]

Variant keys: name (cache suffix), ckpt (another fine-tuned checkpoint of the same architecture, e.g. a snapshot or
an avg_ckpt.py weight average: swapped in without rebuilding the model), views (I1), crop_weights (I2), stride (I24),
scales (I25), compile (I36; applied once and kept for the later variants, so put it last). Each variant writes
preds/<run>_<split_file>_fp16_<name>.npz + metrics/<same>.json like predict.py (existing caches are skipped), plus
the timing (support encoding, s per photo) and the peak memory. A failed variant (e.g. CUDA OOM) is logged and the
sweep goes on.
"""

import argparse
import json
import os
import sys
import time
import traceback

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
    ap.add_argument('--split', default='val')
    ap.add_argument('--variants', required=True, help='JSON list (or a path to a JSON file)')
    ap.add_argument('--amp', default='fp16', choices=['none', 'fp16'])
    ap.add_argument('--chunk', type=int, default=1)
    ap.add_argument('--channel_chunk', type=int, default=1)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--res', default='r768')
    ap.add_argument('--data', default='/content/rr_data')
    ap.add_argument('--runs', default='/content/runs_data')
    ap.add_argument('--drive', default='/content/drive/MyDrive/AIC200-my_work/rescue_router')
    ap.add_argument('--mt_ckpt', default='/content/generalist.ckpt')
    ap.add_argument('--chameleon', default='/content/AIC200-Project-StructuredPrediction/chameleon')
    a = ap.parse_args()
    variants = json.load(open(a.variants)) if os.path.exists(a.variants) else json.loads(a.variants)
    sys.path.insert(0, a.chameleon)
    from rr.data import build_run_folder, load_query, read_ids
    from rr.infer import compile_encoder, load_finetuned_weights, predict_views
    from rr.metrics import WET, confusion, decode, gt_classes, summarize

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)['hyper_parameters']['config']
    run_id = os.path.splitext(os.path.basename(a.ckpt))[0]
    sel = run_id.split('_')[3] if len(run_id.split('_')) > 3 else 'stratified'
    supports = read_ids(os.path.join(a.drive, 'supports', f'{a.dataset}_{sel}_s{ck.seed}.txt'))[:ck.shot]
    data_root = os.path.join(a.data, f'{a.dataset}_{a.res}')
    run_root = build_run_folder(os.path.join(a.runs, f'{run_id}_sweep'), data_root, supports,
                                split_support=SUPPORT_SPLIT[a.dataset])
    su = setup_chameleon(a.chameleon, run_root)
    ov = su.derive_stage2_overrides(a.ckpt, verbose=False)
    ov.update(shot=ck.shot, eval_shot=ck.shot, chunk_size=a.chunk, channel_chunk_size=a.channel_chunk, n_tasks=5,
              loss_type='bce', img_size=384, base_size=list(ck.base_size))
    cfg = su.build_config(stage=2, dataset='custom', **ov)
    t_load = time.time()
    model, cfg = su.load_chameleon(cfg, a.mt_ckpt, a.ckpt, verbose=False)
    model.eval()
    print(f'model loaded once in {time.time() - t_load:.0f} s', flush=True)
    ids = read_ids(os.path.join(a.drive, 'splits', f'{a.split_file}.txt'))
    if a.limit:
        ids = ids[:a.limit]
    os.makedirs(os.path.join(a.drive, 'preds'), exist_ok=True)
    os.makedirs(os.path.join(a.drive, 'metrics'), exist_ok=True)
    cur_ckpt, compiled = a.ckpt, False
    for v in variants:
        name = v['name']
        tag = f'{run_id}_{a.split_file}' + (f'_n{a.limit}' if a.limit else '') + ('' if a.amp == 'none' else f'_{a.amp}') + f'_{name}'
        f_npz = os.path.join(a.drive, 'preds', f'{tag}.npz')
        if os.path.exists(f_npz):
            print(f'[{tag}] cached already, skipping', flush=True)
            continue
        try:
            want = v.get('ckpt', a.ckpt)
            if want != cur_ckpt:
                print(f'[{name}] swapped in {load_finetuned_weights(model, want)} tensors from {os.path.basename(want)}', flush=True)
                cur_ckpt = want
            if v.get('compile') and not compiled:
                print(f'[{name}] compiled {compile_encoder(model)} encoder blocks', flush=True)
                compiled = True
            scales = tuple(float(x) for x in v.get('scales', [1.0]))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            Ps, labs, t_sup = predict_views(model, lambda i: load_query(data_root, a.split, i), ids, v.get('views', 1),
                                            v.get('crop_weights'), scales, v.get('stride', 192), a.amp)
            torch.cuda.synchronize()
            dt = time.time() - t0 - t_sup
            cm = np.zeros((6, 6), np.int64)
            probs = []
            for P, lab in zip(Ps, labs):
                cm += confusion(decode(P, a.tau), gt_classes(lab))
                probs.append(np.round(P * 255).astype(np.uint8))
            res = summarize(cm, NAMES[a.dataset], WET[a.dataset])
            res.update(run_id=run_id, variant=v, split=a.split_file, ckpt=os.path.basename(cur_ckpt), tau=a.tau,
                       n_images=len(ids), amp=a.amp, support_encode_s=round(t_sup, 2),
                       s_per_image=round(dt / max(len(ids), 1), 2),
                       peak_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2), gpu=torch.cuda.get_device_name(0))
            np.savez_compressed(f_npz, probs=np.stack(probs), ids=np.array(ids), cm=cm)
            json.dump(res, open(os.path.join(a.drive, 'metrics', f'{tag}.json'), 'w'), indent=1)
            print(f'[{name}] miou@.5 {res["miou"]:.4f}  {res["s_per_image"]} s/photo  peak {res["peak_gb"]} GB', flush=True)
            del Ps, probs
        except Exception:
            print(f'[{name}] FAILED\n{traceback.format_exc()}', flush=True)
            torch.cuda.empty_cache()
            model.model.reset_support()


if __name__ == '__main__':
    main()
