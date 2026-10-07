"""One resumable Chameleon fine-tuning run (PLAN §4). GPU.

    python src/finetune.py --dataset floodnet --shot 20 --seed 0 --n_steps 2000

Writes to <drive>/:  ckpt/<run_id>.ckpt (starter format, ~5 MB; the checkpoint with the best val
mIoU), ckpt/<run_id>_last.pt (resume state), metrics/<run_id>.json (val metrics of the best step),
logs/<run_id>.json (config, GPU, timings, loss every 50 steps, val history), preds/<run_id>_<val>.npz
(uint8 val probabilities + confusion of the best step).
A run whose .ckpt and metrics .json both exist is skipped. Interrupted runs resume from _last.pt.

--defer_val (50 shots on a T4): no in-process validation, because 50-shot inference (12 GB) does not
fit next to the training state (both 50-shot runs of the first Kaggle queue hit OOM at step 500). At
every val step it saves a snapshot ckpt/<run_id>_step<N>.ckpt instead; select_ckpt.py then
evaluates the snapshots on val30 in fresh processes and keeps the best (same rule as in-process).

--amp fp16: forward + loss under fp16 autocast with a GradScaler (trainable params stay fp32; the
scaler state is part of _last.pt), and the val inference runs in fp16 too (rr.infer).

Batch 13 options (PLAN §15 "Batch 13"; every default reproduces the published recipe exactly; --tag names the
variant run <run_id>_<tag>): --tune label_decoder,relpos,head (I5-I7, the starter's tuning flags, at
--lr_pretrained), --separate_alpha (I8), --accum k gradient accumulation (I11; with --gbs = episode size, I10),
--pos_weight (I12), --rare_crop (I13), --dice (I14), --rot_blur (I15), --attn_dropout (I18), --copy_paste (I19),
--ema d (I17: an exponential moving average of the trained tensors, saved next to each snapshot as
<run_id>_ema_step<N>.ckpt from --ema_from on), --snap_from N (no snapshots before step N), --freeze_rest
(efficiency: gradients only for the trained tensors; the same optimizer updates, ~3 GB less memory, less backward),
--pseudo_list/--pseudo_dir (I20: the training pool also holds pseudo-labelled photos; the checkpoint's shot is then
50 + N for the pool, its eval_shot 50 = the support set predict.py uses), --lora r (I9: LoRA adapters on the
image-encoder qkv, rank r, alpha 2r, lr --lora_lr; rr/lora.py).
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
    ap.add_argument('--lr_schedule', default='cos', choices=['constant', 'cos', 'poly'])
    ap.add_argument('--channel_sampling', type=int, default=-1)
    ap.add_argument('--grad_ckpt', action='store_true', help='checkpoint image-encoder blocks (saves memory)')
    ap.add_argument('--tf32', action='store_true', help='TF32 matmuls (A100/L4 tensor cores; no effect on T4)')
    ap.add_argument('--amp', default='none', choices=['none', 'fp16'], help='mixed precision (fp16 + GradScaler)')
    ap.add_argument('--res', default='r768')
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--val_every', type=int, default=250)
    ap.add_argument('--ckpt_every', type=int, default=500)
    ap.add_argument('--val_split', default=None, help='split file for validation (default: <dataset>_val_tune30)')
    ap.add_argument('--chunk', type=int, default=6, help='query tiles per forward at eval')
    ap.add_argument('--channel_chunk', type=int, default=5)
    ap.add_argument('--defer_val', action='store_true', help='save snapshots at val steps; select_ckpt.py validates')
    ap.add_argument('--stop_after', type=int, default=0, help='exit after this many steps (timing test)')
    ap.add_argument('--data', default='/content/rr_data')
    ap.add_argument('--runs', default='/content/runs_data')
    ap.add_argument('--drive', default='/content/drive/MyDrive/AIC200-my_work/rescue_router')
    ap.add_argument('--mt_ckpt', default='/content/generalist.ckpt')
    ap.add_argument('--chameleon', default='/content/AIC200-Project-StructuredPrediction/chameleon')
    ap.add_argument('--code_version', default=os.environ.get('RR_CODE_VERSION', 'unknown'))
    # Batch 13 options (defaults = the published recipe)
    ap.add_argument('--tag', default='', help='variant name, appended to the run id')
    ap.add_argument('--tune', default='', help='comma list of label_decoder, relpos, head (starter tuning flags)')
    ap.add_argument('--lr_pretrained', type=float, default=None, help='lr of the --tune parts (starter default 2e-4)')
    ap.add_argument('--separate_alpha', action='store_true', help='one matching temperature per channel')
    ap.add_argument('--accum', type=int, default=1, help='gradient accumulation steps')
    ap.add_argument('--pos_weight', default='', help='per-channel BCE positive weights, e.g. 1,3,1,3,1')
    ap.add_argument('--dice', type=float, default=0.0, help='weight of a soft-Dice term')
    ap.add_argument('--rare_crop', type=float, default=0.0, help='probability of a rare-class-centred training crop')
    ap.add_argument('--rot_blur', action='store_true', help='rotation + blur augmentation')
    ap.add_argument('--copy_paste', type=float, default=0.0, help='probability of rare-class copy-paste')
    ap.add_argument('--attn_dropout', type=float, default=None, help='matching-attention dropout (starter 0.5)')
    ap.add_argument('--ema', type=float, default=0.0, help='EMA decay of the trained tensors (0 = off)')
    ap.add_argument('--ema_from', type=int, default=0, help='first step whose snapshot also saves the EMA')
    ap.add_argument('--snap_from', type=int, default=0, help='--defer_val: no snapshots before this step')
    ap.add_argument('--pseudo_list', default='', help='I20: supports/<name>.txt of pseudo-labelled train photos')
    ap.add_argument('--pseudo_dir', default='', help='I20: folder with labels/<id>.png (pseudo_labels.py make)')
    ap.add_argument('--lora', type=int, default=0, help='I9: LoRA rank on the image-encoder qkv (0 = off)')
    ap.add_argument('--lora_lr', type=float, default=2e-4)
    ap.add_argument('--freeze_rest', action='store_true',
                    help='requires_grad only on the trained tensors (same updates; no weight grads for the frozen 3 GB)')
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


def evaluate(model, data_root, split, ids, names, wet, tau=0.5, keep_probs=True, amp='none'):
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
    run_id = f'{a.dataset}_{a.shot}shot_s{a.seed}_{a.sel}' + ('' if a.res == 'r768' else f'_{a.res}') \
        + (f'_{a.tag}' if a.tag else '')
    for d in ('ckpt', 'metrics', 'logs', 'preds'):
        os.makedirs(os.path.join(a.drive, d), exist_ok=True)
    f_ckpt = os.path.join(a.drive, 'ckpt', f'{run_id}.ckpt')
    f_last = os.path.join(a.drive, 'ckpt', f'{run_id}_last.pt')
    f_metrics = os.path.join(a.drive, 'metrics', f'{run_id}.json')
    f_log = os.path.join(a.drive, 'logs', f'{run_id}.json')
    if os.path.exists(f_ckpt) and os.path.exists(f_metrics) and not a.stop_after:
        print(f'[{run_id}] done already, skipping')
        return
    f_snap = lambda step, kind='': os.path.join(a.drive, 'ckpt', f'{run_id}{kind}_step{step}.ckpt')
    if a.defer_val and os.path.exists(f_snap(a.n_steps)) and not os.path.exists(f_last):
        print(f'[{run_id}] training done already (snapshots exist), skipping')
        return

    data_root = os.path.join(a.data, f'{a.dataset}_{a.res}')
    supports = read_ids(os.path.join(a.drive, 'supports', f'{a.dataset}_{a.sel}_s{a.seed}.txt'))[:a.shot]
    val_file = a.val_split or f'{a.dataset}_val_tune30'
    val_ids = read_ids(os.path.join(a.drive, 'splits', f'{val_file}.txt')) if val_file else []
    val_split = 'val'
    run_root = build_run_folder(os.path.join(a.runs, run_id), data_root, supports,
                                split_support=SUPPORT_SPLIT[a.dataset])
    pseudo = read_ids(os.path.join(a.drive, 'supports', f'{a.pseudo_list}.txt')) if a.pseudo_list else []
    for rank, i in enumerate(pseudo, len(supports)):       # I20: training pool = supports + pseudo-labelled photos
        os.symlink(os.path.join(data_root, SUPPORT_SPLIT[a.dataset], 'images', f'{i}.jpg'),
                   os.path.join(run_root, 'train', 'images', f'{rank:03d}_{i}.jpg'))
        os.symlink(os.path.join(a.pseudo_dir, 'labels', f'{i}.png'), os.path.join(run_root, 'train', 'labels', f'{rank:03d}_{i}.png'))
    su = setup_chameleon(a.chameleon, run_root)
    from train.optim import get_optimizer
    from dataset.utils import to_device

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = a.tf32
    H, W = {'r768': (576, 768), 'r1024': (768, 1024)}[a.res]
    tune = [t for t in a.tune.split(',') if t]
    assert set(tune) <= {'label_decoder', 'relpos', 'head'}, tune
    assert not {'label_decoder', 'head'} <= set(tune), 'the head is part of the label decoder'
    extra = {f'{t}_tuning': True for t in tune}
    if a.separate_alpha:
        extra['separate_alpha'] = True
    if a.lr_pretrained is not None:
        extra['lr_pretrained'] = a.lr_pretrained
    cfg = su.build_config(stage=1, dataset='custom', n_tasks=5, loss_type='bce', shot=a.shot + len(pseudo), eval_shot=a.shot,
                          img_size=384, base_size=[H, W], global_batch_size=a.gbs, n_steps=a.n_steps, lr=a.lr,
                          lr_schedule=a.lr_schedule,
                          channel_sampling=a.channel_sampling, seed=a.seed, chunk_size=a.chunk,
                          channel_chunk_size=a.channel_chunk, **extra)
    from rr.data import FloodDataset, FloodLearner
    FloodDataset.rare_crop_p, FloodDataset.rot_blur, FloodDataset.copy_paste_p = a.rare_crop, a.rot_blur, a.copy_paste
    if a.pos_weight:
        FloodLearner.pos_weight = [float(w) for w in a.pos_weight.split(',')]
    FloodLearner.dice = a.dice
    model, cfg = su.load_chameleon(cfg, a.mt_ckpt, verbose=False)
    if a.attn_dropout is not None:
        for attn in model.model.matching_module.matching:
            attn.attn_dropout.p = a.attn_dropout
        cfg.attn_dropout = model.config.attn_dropout = a.attn_dropout
    if a.grad_ckpt:
        from rr.compat import enable_grad_checkpointing
        print(f'[{run_id}] gradient checkpointing on {enable_grad_checkpointing(model)} encoder blocks', flush=True)
    lora_names = []
    if a.lora:                                  # I9: adapters injected before the optimizer is built
        from rr.lora import inject_lora
        lora_names = inject_lora(model, a.lora, 2 * a.lora)
        cfg.rr_lora = model.config.rr_lora
        _names = su.finetuned_parameter_names
        su.finetuned_parameter_names = lambda m: _names(m) + lora_names     # save_finetuned keeps the adapters
    optimizer, sched = get_optimizer(cfg, model.model)
    if lora_names:
        params = dict(model.named_parameters())
        optimizer.add_param_group({'params': [params[n] for n in lora_names], 'lr': a.lora_lr})
        sched.lr_coefs.append(a.lora_lr / a.lr)
    names = set(su.finetuned_parameter_names(model))
    if a.freeze_rest:
        for k, v in model.named_parameters():
            v.requires_grad_(k in names)
        print(f'[{run_id}] trainable tensors: {sum(v.requires_grad for v in model.parameters())}, '
              f'{sum(v.numel() for v in model.parameters() if v.requires_grad) / 1e6:.2f} M values', flush=True)
    from rr.infer import amp_context
    scaler = torch.amp.GradScaler('cuda', enabled=a.amp == 'fp16')
    autocast = (lambda: amp_context(a.amp)) if a.amp != 'none' else contextlib.nullcontext

    log = dict(run_id=run_id, code_version=a.code_version, gpu=torch.cuda.get_device_name(0),
               gpu_mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
               args=vars(a), supports=supports, val_split=val_file, losses=[], val=[], timing={},
               started=time.strftime('%Y-%m-%d %H:%M:%S'))
    start, st = 0, {}
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
                        rng_torch=torch.get_rng_state(), rng_np=np.random.get_state(),
                        **(dict(ema={k: v.cpu() for k, v in ema.items()}) if ema is not None else {})), tmp)
        os.replace(tmp, f_last)

    def write_log():
        json.dump(log, open(f_log, 'w'), indent=1)

    trained = [(k, v) for k, v in model.named_parameters() if k in names]
    ema = {k: v.detach().float().clone() for k, v in trained} if a.ema else None
    if ema is not None and os.path.exists(f_last) and st.get('ema'):
        ema = {k: v.cuda() for k, v in st['ema'].items()}

    def save_ema(step):
        live = {k: v.detach().clone() for k, v in trained}
        with torch.no_grad():
            for k, v in trained:
                v.copy_(ema[k])
        su.save_finetuned(model, f_snap(step, '_ema'))
        with torch.no_grad():
            for k, v in trained:
                v.copy_(live[k])

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
        if a.accum == 1:
            with autocast():
                Y_pred, Y_q, M_q, _ = model(batch)
                loss, _ = model.learner.compute_loss(Y_pred, Y_q, M_q)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
        else:                                   # I11: k micro-batches per optimizer step
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for j in range(a.accum):
                if j:
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(loader)
                        batch = next(it)
                    batch = to_device(batch, 'cuda')
                with autocast():
                    Y_pred, Y_q, M_q, _ = model(batch)
                    loss_j, _ = model.learner.compute_loss(Y_pred, Y_q, M_q)
                scaler.scale(loss_j / a.accum).backward()
                losses.append(loss_j.detach())
            loss = torch.stack(losses).mean()
        scale = scaler.get_scale()
        scaler.step(optimizer)                  # skipped if the fp16 grads overflowed
        scaler.update()
        n_skip += scaler.is_enabled() and scaler.get_scale() < scale
        sched.step(step)
        if ema is not None:
            with torch.no_grad():
                for k, v in trained:
                    ema[k].mul_(a.ema).add_(v.detach().float(), alpha=1 - a.ema)
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
        if a.defer_val and (done % a.val_every == 0 or last) and done >= a.snap_from:
            su.save_finetuned(model, f_snap(done))
            log.setdefault('snapshots', []).append(done)
            if ema is not None and done >= a.ema_from:
                save_ema(done)
                log.setdefault('ema_snapshots', []).append(done)
            print(f'[{run_id}] snapshot at step {done}', flush=True)
            write_log()
            t_win = time.time()
        elif val_ids and (done % a.val_every == 0 or last):
            res, cm, probs = evaluate(model, data_root, val_split, val_ids, NAMES[a.dataset], WET[a.dataset],
                                      amp=a.amp)
            res['step'] = done
            log['val'].append(res)
            log['timing']['eval_s_per_image'] = res['s_per_image']
            log['timing']['support_encode_s'] = res['support_encode_s']
            print(f'[{run_id}] VAL step {done}: ' + ' '.join(f'{k}={v:.3f}' for k, v in res.items()
                                                             if k.startswith('iou') or k == 'miou'), flush=True)
            if res['miou'] > log.get('best', {}).get('miou', -1):     # keep the best-val checkpoint
                log['best'] = dict(res)
                su.save_finetuned(model, f_ckpt)
                np.savez_compressed(os.path.join(a.drive, 'preds', f'{run_id}_{val_file}.npz'),
                                    probs=np.stack(probs), ids=np.array(val_ids), cm=cm, step=done)
                print(f'[{run_id}] new best val mIoU {res["miou"]:.3f} at step {done}', flush=True)
            write_log()
            t_win = time.time()
        if done % a.ckpt_every == 0 and not last:
            save_last(done)

    if not val_ids:
        su.save_finetuned(model, f_ckpt)
    best = log.get('best', {})
    if not a.defer_val:                        # deferred runs get their metrics from select_ckpt.py
        json.dump(dict(run_id=run_id, split=val_file, tau=0.5, best_step=best.get('step'), **best,
                       final=log['val'][-1] if log['val'] else {}), open(f_metrics, 'w'), indent=1)
    log['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    write_log()
    if os.path.exists(f_last):
        os.remove(f_last)
    print(f'[{run_id}] DONE', flush=True)


if __name__ == '__main__':
    main()
