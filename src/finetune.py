"""One resumable Chameleon fine-tuning run (PLAN §4). GPU.

    python src/finetune.py --dataset floodnet --shot 20 --seed 0 --n_steps 2000

Writes to <drive>/:  ckpt/<run_id>.ckpt (starter format, ~20 MB), ckpt/<run_id>_last.pt (resume
state), metrics/<run_id>.json (final val metrics), logs/<run_id>.json (config, GPU, timings,
loss every 50 steps, val history), preds/<run_id>_val.npz (uint8 val probabilities, final step).
A run whose .ckpt and metrics .json both exist is skipped. Interrupted runs resume from _last.pt.

--amp fp16: forward + loss under fp16 autocast with a GradScaler (trainable params stay fp32; the
scaler state is part of _last.pt), and the val inference runs in fp16 too (rr.infer).
"""

import argparse
import contextlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NAMES = {'floodnet': ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water'],
         'rescuenet': ['road_clear', 'road_blocked', 'bldg_intact', 'bldg_damaged', 'water']}
SUPPORT_SPLIT = {'floodnet': 'train', 'rescuenet': 'val'}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='floodnet', choices=list(NAMES))
    ap.add_argument('--shot', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sel', default='stratified')
    ap.add_argument('--n_steps', type=int, default=2000)
    ap.add_argument('--gbs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--channel_sampling', type=int, default=-1)
    ap.add_argument('--grad_ckpt', action='store_true', help='checkpoint image-encoder blocks (saves memory)')
    ap.add_argument('--tf32', action='store_true', help='TF32 matmuls (A100/L4 tensor cores; no effect on T4)')
    ap.add_argument('--amp', default='none', choices=['none', 'fp16'], help='mixed precision (fp16 + GradScaler)')
    ap.add_argument('--res', default='r768')
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--val_every', type=int, default=250)
    ap.add_argument('--ckpt_every', type=int, default=500)
    ap.add_argument('--val_split', default=None, help='split file for validation (default: floodnet_val_tune30)')
    ap.add_argument('--chunk', type=int, default=6, help='query tiles per forward at eval')
    ap.add_argument('--channel_chunk', type=int, default=5)
    ap.add_argument('--stop_after', type=int, default=0, help='exit after this many steps (timing test)')
    ap.add_argument('--data', default='/content/rr_data')
    ap.add_argument('--runs', default='/content/runs_data')
    ap.add_argument('--drive', default='/content/drive/MyDrive/AIC200-my_work/rescue_router')
    ap.add_argument('--mt_ckpt', default='/content/generalist.ckpt')
    ap.add_argument('--chameleon', default='/content/AIC200-Project-StructuredPrediction/chameleon')
    ap.add_argument('--code_version', default=os.environ.get('RR_CODE_VERSION', 'unknown'))
    return ap.parse_args()


def setup_chameleon(chameleon_dir, run_root):
    sys.path.insert(0, chameleon_dir)
    os.chdir(chameleon_dir)
    import starter_utils as su
    from downstream.custom import register_custom
    from rr.compat import patch_matching_fp32, patch_scipy_interp2d, patch_vit_attention_fp16
    from rr.data import FloodDataset, FloodLearner
    if patch_scipy_interp2d():
        print('patched scipy.interpolate.interp2d -> RectBivariateSpline (SciPy >= 1.14)', flush=True)
    patch_vit_attention_fp16()
    patch_matching_fp32()
    register_custom(dataset_cls=FloodDataset, learner_cls=FloodLearner, n_tasks=5)
    su.write_data_paths('/content/unused_demo_data', extra={'custom': run_root}, chameleon_root=chameleon_dir)
    return su


def evaluate(model, data_root, split, ids, names, wet, tau=0.5, keep_probs=False, amp='none'):
    from rr.data import load_query
    from rr.infer import encode_support, tiled_predict
    from rr.metrics import confusion, decode, gt_classes, summarize
    model.eval()
    t_sup = encode_support(model, amp=amp)
    cm = np.zeros((6, 6), np.int64)
    probs_all, t0 = [], time.time()
    for i in ids:
        X, lab = load_query(data_root, split, i)
        P = tiled_predict(model, X, amp=amp).numpy()
        cm += confusion(decode(P, tau), gt_classes(lab))
        if keep_probs:
            probs_all.append(np.round(P * 255).astype(np.uint8))
    torch.cuda.synchronize()
    t_img = (time.time() - t0) / max(len(ids), 1)
    model.model.reset_support()
    model.train()
    res = summarize(cm, names, wet)
    res.update(support_encode_s=round(t_sup, 2), s_per_image=round(t_img, 2), n_images=len(ids))
    return res, cm, probs_all


def main():
    a = parse()
    sys.path.insert(0, a.chameleon)            # rr.data imports chameleon's downstream.custom
    from rr.data import build_run_folder, read_ids
    from rr.metrics import WET
    run_id = f'{a.dataset}_{a.shot}shot_s{a.seed}_{a.sel}' + ('' if a.res == 'r768' else f'_{a.res}')
    for d in ('ckpt', 'metrics', 'logs', 'preds'):
        os.makedirs(os.path.join(a.drive, d), exist_ok=True)
    f_ckpt = os.path.join(a.drive, 'ckpt', f'{run_id}.ckpt')
    f_last = os.path.join(a.drive, 'ckpt', f'{run_id}_last.pt')
    f_metrics = os.path.join(a.drive, 'metrics', f'{run_id}.json')
    f_log = os.path.join(a.drive, 'logs', f'{run_id}.json')
    if os.path.exists(f_ckpt) and os.path.exists(f_metrics) and not a.stop_after:
        print(f'[{run_id}] done already, skipping')
        return

    data_root = os.path.join(a.data, f'{a.dataset}_{a.res}')
    supports = read_ids(os.path.join(a.drive, 'supports', f'{a.dataset}_{a.sel}_s{a.seed}.txt'))[:a.shot]
    val_file = a.val_split or ('floodnet_val_tune30' if a.dataset == 'floodnet' else None)
    val_ids = read_ids(os.path.join(a.drive, 'splits', f'{val_file}.txt')) if val_file else []
    val_split = 'val'
    run_root = build_run_folder(os.path.join(a.runs, run_id), data_root, supports,
                                split_support=SUPPORT_SPLIT[a.dataset])
    su = setup_chameleon(a.chameleon, run_root)
    from train.optim import get_optimizer
    from dataset.utils import to_device

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = a.tf32
    H, W = {'r768': (576, 768), 'r1024': (768, 1024)}[a.res]
    cfg = su.build_config(stage=1, dataset='custom', n_tasks=5, loss_type='bce', shot=a.shot, eval_shot=a.shot,
                          img_size=384, base_size=[H, W], global_batch_size=a.gbs, n_steps=a.n_steps, lr=a.lr,
                          channel_sampling=a.channel_sampling, seed=a.seed, chunk_size=a.chunk,
                          channel_chunk_size=a.channel_chunk)
    model, cfg = su.load_chameleon(cfg, a.mt_ckpt, verbose=False)
    if a.grad_ckpt:
        from rr.compat import enable_grad_checkpointing
        print(f'[{run_id}] gradient checkpointing on {enable_grad_checkpointing(model)} encoder blocks', flush=True)
    optimizer, sched = get_optimizer(cfg, model.model)
    names = set(su.finetuned_parameter_names(model))
    from rr.infer import amp_context
    scaler = torch.amp.GradScaler('cuda', enabled=a.amp == 'fp16')
    autocast = (lambda: amp_context(a.amp)) if a.amp != 'none' else contextlib.nullcontext

    log = dict(run_id=run_id, code_version=a.code_version, gpu=torch.cuda.get_device_name(0),
               gpu_mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
               args=vars(a), supports=supports, val_split=val_file, losses=[], val=[], timing={},
               started=time.strftime('%Y-%m-%d %H:%M:%S'))
    start = 0
    if os.path.exists(f_last):
        st = torch.load(f_last, map_location='cpu', weights_only=False)
        missing = model.load_state_dict(st['state_dict'], strict=False)
        optimizer.load_state_dict(st['optimizer'])
        if st.get('scaler'):
            scaler.load_state_dict(st['scaler'])
        start, log = st['step'], st['log']
        torch.set_rng_state(st['rng_torch']); np.random.set_state(st['rng_np'])
        print(f'[{run_id}] resumed at step {start}', flush=True)

    def save_last(step):
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}
        tmp = f_last + '.tmp'
        torch.save(dict(state_dict=sd, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step, log=log,
                        rng_torch=torch.get_rng_state(), rng_np=np.random.get_state()), tmp)
        os.replace(tmp, f_last)

    def write_log():
        json.dump(log, open(f_log, 'w'), indent=1)

    loader = model.learner.get_train_loader()
    it = iter(loader)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t_win, run_loss, n_skip = time.time(), [], 0
    for step in range(start, a.n_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = to_device(batch, 'cuda')
        with autocast():
            Y_pred, Y_q, M_q, _ = model(batch)
            loss, _ = model.learner.compute_loss(Y_pred, Y_q, M_q)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scale = scaler.get_scale()
        scaler.step(optimizer)                  # skipped if the fp16 grads overflowed
        scaler.update()
        n_skip += scaler.is_enabled() and scaler.get_scale() < scale
        sched.step(step)
        run_loss.append(loss.item())
        done = step + 1
        if done % a.log_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t_win
            log['losses'].append(dict(step=done, loss=float(np.mean(run_loss)), s_per_50=round(dt * 50 / a.log_every, 2),
                                      **(dict(scale=scaler.get_scale(), skipped=int(n_skip)) if scaler.is_enabled() else {})))
            log['timing'].setdefault('train_s_per_50_steps', round(dt * 50 / a.log_every, 2))
            log['timing']['train_peak_gb'] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            print(f'[{run_id}] step {done} loss {np.mean(run_loss):.4f} ({dt:.1f}s / {a.log_every} steps, '
                  f'peak {log["timing"]["train_peak_gb"]} GB' +
                  (f', scale {scaler.get_scale():g}, skipped {n_skip})' if scaler.is_enabled() else ')'), flush=True)
            run_loss, n_skip = [], 0
            write_log()
            t_win = time.time()
        if a.stop_after and done >= a.stop_after:
            print(f'[{run_id}] stop_after reached', flush=True)
            write_log()
            return
        last = done == a.n_steps
        if val_ids and (done % a.val_every == 0 or last):
            res, cm, probs = evaluate(model, data_root, val_split, val_ids, NAMES[a.dataset], WET[a.dataset],
                                      keep_probs=last, amp=a.amp)
            res['step'] = done
            log['val'].append(res)
            log['timing']['eval_s_per_image'] = res['s_per_image']
            log['timing']['support_encode_s'] = res['support_encode_s']
            print(f'[{run_id}] VAL step {done}: ' + ' '.join(f'{k}={v:.3f}' for k, v in res.items()
                                                             if k.startswith('iou') or k == 'miou'), flush=True)
            if last:
                np.savez_compressed(os.path.join(a.drive, 'preds', f'{run_id}_{val_file}.npz'),
                                    probs=np.stack(probs), ids=np.array(val_ids), cm=cm)
            write_log()
            t_win = time.time()
        if done % a.ckpt_every == 0 and not last:
            save_last(done)

    su.save_finetuned(model, f_ckpt)
    final = log['val'][-1] if log['val'] else {}
    json.dump(dict(run_id=run_id, split=val_file, tau=0.5, **final), open(f_metrics, 'w'), indent=1)
    log['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    write_log()
    if os.path.exists(f_last):
        os.remove(f_last)
    print(f'[{run_id}] DONE', flush=True)


if __name__ == '__main__':
    main()
