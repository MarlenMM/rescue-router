"""Same-label-budget baseline: a standard supervised segmenter on the same k photos (PLAN §16, A5).

The published comparison (PSPNet, 1,445 labels, .848) is not like for like. Here torchvision's DeepLabV3-ResNet50
(COCO-pretrained, the family of the FloodNet paper's baselines) is fine-tuned on exactly the photos Chameleon used
(supports/<dataset>_stratified_s<draw>.txt, first k; or all 1,445 FloodNet train photos with --shot 0) with the same
384-px training crops, and predicts whole 768x576 photos. No validation labels are used: fixed schedule, last step.

Output = the same prediction-cache format as Chameleon (uint8 per-channel probabilities; here the softmax over the 5
classes + other, the 5 class channels kept), so decode_tune.py / route_eval.py / bootstrap_ci.py run on it unchanged:
  <out>/preds/<run>_<split_file>.npz, <out>/metrics/<same>.json, <out>/logs/<run>.json, <out>/ckpt/<run>.pt
  run = baseline_<arch>[_rescuenet]_<k|all>shot_s<seed>[_sup<draw>][_<tag>][_tta at prediction]

Options (defaults = the 14 Oct runs bit for bit):
  --dataset rescuenet   RescueNet (its 5 slots; supports from its val split, like Chameleon's)
  --support_seed k      support draw k with training seed --seed (default: draw = seed)
  --pseudo DIR          self-training (PLAN §16.5): the batch also holds unlabelled photos of the support split
                        with the teacher's pseudo-labels (DIR/ids.txt, DIR/labels/<id>.png from baseline_pseudo.py,
                        255 = not confident), --pseudo_frac of each batch; run tag 'st' unless --tag is given
  --tta                 prediction averaged over the identity and h / v / hv flips (cache name _tta)
  --accum k             k micro-batches of --bs per optimizer step (self-training: --bs 8 --accum 2 = 8 labelled +
                        8 pseudo-labelled crops per step at the memory of 8)
  --arch deeplabv3_mbv3 / lraspp_mbv3   small models for CPU / in-browser use
  --query_dataset rescuenet   predict another dataset's split with this model (zero-label transfer)
  --init CKPT           start from another baseline checkpoint (transfer: FloodNet model -> k RescueNet photos)
  --save_every k        resumable state (<ckpt>.part: weights, optimizer, schedule, every RNG) every k steps; a rerun
                        after a crash or reboot continues from it and ends with the same weights (test_baseline.py)

    python src/baseline_seg.py --shot 50 [--iters 3000] [--device mps] [--splits floodnet_val_tune30,floodnet_test_full]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr.metrics import OTHER_IDX, WET, confusion, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

REPO = os.path.dirname(HERE)
NAMES = ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water']
DATASETS = {'floodnet': dict(names=NAMES, support_split='train', splits='floodnet_val_tune30,floodnet_test_full'),
            'rescuenet': dict(names=['road_clear', 'road_blocked', 'bldg_intact', 'bldg_damaged', 'water'],
                              support_split='val', splits='rescuenet_val_tune30,rescuenet_test_eval100')}
MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
N_CLS = OTHER_IDX + 1                              # 5 classes + other; 255 = ignore
FLIPS = ((), (-1,), (-2,), (-2, -1))               # --tta views: identity, h, v, hv


def load(split, i, dataset='floodnet', label_dir=None):
    """photo (3, H, W) uint8 and its class map (H, W) uint8 (0-5, 255 ignore); label_dir: pseudo-labels instead"""
    root = os.path.join(RR_DATA, f'{dataset}_r768', split)
    x = torch.from_numpy(np.array(Image.open(os.path.join(root, 'images', f'{i}.jpg')).convert('RGB'))).permute(2, 0, 1)
    if label_dir:
        g = np.asarray(Image.open(os.path.join(label_dir, f'{i}.png')))
    else:
        g = gt_classes(np.asarray(Image.open(os.path.join(root, 'labels', f'{i}.png'))))
    return x, torch.from_numpy(g.astype(np.uint8))


def build(arch, device, pretrained=True):
    from torchvision.models import segmentation as S
    if arch == 'deeplabv3_r50':
        m = S.deeplabv3_resnet50(weights=S.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None,
                                 weights_backbone=None, aux_loss=True)
    elif arch == 'deeplabv3_mbv3':
        m = S.deeplabv3_mobilenet_v3_large(weights=S.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
                                           if pretrained else None, weights_backbone=None, aux_loss=True)
    elif arch == 'lraspp_mbv3':
        m = S.lraspp_mobilenet_v3_large(weights=S.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
                                        if pretrained else None, weights_backbone=None)
    else:
        raise ValueError(arch)
    if arch.startswith('deeplabv3'):               # new 6-class output layers; the rest keeps its COCO weights
        m.classifier[-1] = torch.nn.Conv2d(m.classifier[-1].in_channels, N_CLS, 1)
        m.aux_classifier[-1] = torch.nn.Conv2d(m.aux_classifier[-1].in_channels, N_CLS, 1)
    else:
        m.classifier.low_classifier = torch.nn.Conv2d(m.classifier.low_classifier.in_channels, N_CLS, 1)
        m.classifier.high_classifier = torch.nn.Conv2d(m.classifier.high_classifier.in_channels, N_CLS, 1)
    return m.to(device)


def batches(data, crop, bs, rng):
    """random crop + h/v flip + mild colour jitter, like Chameleon's training episodes"""
    while True:
        xs, ys = [], []
        for k in rng.integers(0, len(data), bs):
            x, y = data[k]
            H, W = y.shape
            t, l = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
            x, y = x[:, t:t + crop, l:l + crop].float() / 255, y[t:t + crop, l:l + crop].long()
            if rng.random() < .5:
                x, y = x.flip(-1), y.flip(-1)
            if rng.random() < .5:
                x, y = x.flip(-2), y.flip(-2)
            x = (x * (0.8 + 0.4 * rng.random()) + 0.1 * (rng.random() - .5)).clamp(0, 1)
            xs.append((x - MEAN) / STD)
            ys.append(y)
        yield torch.stack(xs), torch.stack(ys)


def mixed(labelled, pseudo, crop, bs, n_pseudo, rng):
    """self-training batches: bs - n_pseudo labelled crops + n_pseudo pseudo-labelled crops (one shared rng)"""
    it_l, it_p = batches(labelled, crop, bs - n_pseudo, rng), batches(pseudo, crop, n_pseudo, rng)
    while True:
        (xl, yl), (xp, yp) = next(it_l), next(it_p)
        yield torch.cat([xl, xp]), torch.cat([yl, yp])


@torch.no_grad()
def predict(model, x, device, tta=False):
    """softmax over the 5 classes + other of one photo (3, H, W) uint8 -> the 5 class channels (other = 1 - sum)"""
    model.eval()
    z = ((x.float() / 255 - MEAN) / STD)[None].to(device)
    p = 0
    for dims in FLIPS if tta else ((),):
        out = F.softmax(model(z.flip(dims) if dims else z)['out'], 1)
        p = p + (out.flip(dims) if dims else out)
    p = p / (len(FLIPS) if tta else 1)
    return p[0, :OTHER_IDX].cpu().numpy()


RESUME_FREE = ('stop_after', 'save_every', 'splits', 'tta', 'device')   # may differ between a run and its resume


def atomic_save(obj, path):
    """torch.save via a temporary file, so a crash never leaves a truncated checkpoint behind"""
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


def run_id(a):
    draw = a.seed if a.support_seed < 0 else a.support_seed
    tag = a.tag or ('st' if a.pseudo else '')
    return (f'baseline_{a.arch}' + ('' if a.dataset == 'floodnet' else f'_{a.dataset}') + f'_{a.shot or "all"}shot_s{a.seed}'
            + ('' if draw == a.seed else f'_sup{draw}') + (f'_{tag}' if tag else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='deeplabv3_r50', choices=['deeplabv3_r50', 'deeplabv3_mbv3', 'lraspp_mbv3'])
    ap.add_argument('--dataset', default='floodnet', choices=list(DATASETS))
    ap.add_argument('--shot', type=int, default=50, help='0 = all photos of the support pool (FloodNet: 1,445 train)')
    ap.add_argument('--seed', type=int, default=0, help='training seed (and the support draw unless --support_seed)')
    ap.add_argument('--support_seed', type=int, default=-1, help='support draw (default: --seed)')
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--accum', type=int, default=1, help='micro-batches of --bs per optimizer step')
    ap.add_argument('--crop', type=int, default=384)
    ap.add_argument('--lr', type=float, default=1e-4, help='backbone; heads get 10x')
    ap.add_argument('--pseudo', default='', help='self-training: folder with ids.txt + labels/<id>.png')
    ap.add_argument('--pseudo_frac', type=float, default=0.5, help='share of each batch from the pseudo-labelled photos')
    ap.add_argument('--tag', default='', help='run-name suffix (default "st" with --pseudo)')
    ap.add_argument('--tta', action='store_true', help='flip test-time augmentation at prediction')
    ap.add_argument('--init', default='', help='start from this baseline checkpoint (.pt) instead of the COCO weights '
                                                '(e.g. RescueNet fine-tuned from the FloodNet model)')
    ap.add_argument('--query_dataset', default=None,
                    help='photos of another dataset at prediction (zero-label transfer, e.g. a FloodNet model on '
                         'rescuenet_test_eval100; scored on shared super-classes by transfer_eval.py)')
    ap.add_argument('--save_every', type=int, default=500, help='steps between resumable states (0 = none)')
    ap.add_argument('--stop_after', type=int, default=0, help='stop after this many steps, resumable (tests)')
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--splits', default=None, help='comma list of split files (default per dataset)')
    ap.add_argument('--out', default='results/baselines')
    a = ap.parse_args()
    ds = DATASETS[a.dataset]
    names = ds['names']
    splits = [f for f in (ds['splits'] if a.splits is None else a.splits).split(',') if f]   # '' = train only
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    draw = a.seed if a.support_seed < 0 else a.support_seed
    if a.shot:
        ids = open(os.path.join(REPO, 'supports', f'{a.dataset}_stratified_s{draw}.txt')).read().split()[:a.shot]
    else:
        ids = open(os.path.join(REPO, 'splits', f'{a.dataset}_support_pool.txt')).read().split()
        if a.dataset == 'rescuenet':                # its pool is the val split, which also holds val30
            ids = sorted(set(ids) - set(open(os.path.join(REPO, 'splits', 'rescuenet_val_tune30.txt')).read().split()))
    run = run_id(a)
    for d in ('preds', 'metrics', 'logs', 'ckpt'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    f_ckpt = os.path.join(a.out, 'ckpt', f'{run}.pt')
    model = build(a.arch, a.device)
    log = dict(run=run, args=vars(a), supports=ids if a.shot else f'{len(ids)} train photos', losses=[])
    if os.path.exists(f_ckpt):
        model.load_state_dict(torch.load(f_ckpt, map_location=a.device))
        print(f'[{run}] loaded {f_ckpt}', flush=True)
    else:
        if a.init:                                 # same 6 outputs, so every tensor carries over
            model.load_state_dict(torch.load(a.init, map_location=a.device))
            log['init'] = a.init
        data = [load(ds['support_split'], i, a.dataset) for i in ids]
        if a.pseudo:
            p_ids = open(os.path.join(a.pseudo, 'ids.txt')).read().split()
            assert not set(p_ids) & set(ids), 'pseudo-labelled photos must not include the labelled supports'
            pseudo = [load(ds['support_split'], i, a.dataset, os.path.join(a.pseudo, 'labels')) for i in p_ids]
            n_p = int(round(a.bs * a.pseudo_frac))
            log['pseudo'] = dict(dir=a.pseudo, n_photos=len(p_ids), per_batch=n_p,
                                 meta=json.load(open(os.path.join(a.pseudo, 'meta.json')))
                                 if os.path.exists(os.path.join(a.pseudo, 'meta.json')) else None)
        heads = [p for n, p in model.named_parameters() if 'classifier' in n]
        body = [p for n, p in model.named_parameters() if 'classifier' not in n]
        opt = torch.optim.AdamW([dict(params=body, lr=a.lr), dict(params=heads, lr=10 * a.lr)], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1 - s / a.iters) ** 0.9)
        f_part, start, t_prev = f_ckpt + '.part', 1, 0.0
        if os.path.exists(f_part):                 # resume: every state the remaining steps depend on
            st = torch.load(f_part, map_location='cpu', weights_only=False)   # RNG states must stay on the CPU
            assert st['args'] == {k: v for k, v in vars(a).items() if k not in RESUME_FREE}, \
                f'{f_part} was written with other arguments; delete it to start over'
            model.load_state_dict(st['model'])
            opt.load_state_dict(st['opt'])
            sched.load_state_dict(st['sched'])
            rng.bit_generator.state = st['np_rng']
            torch.set_rng_state(st['torch_rng'])
            if st.get('mps_rng') is not None and a.device == 'mps':
                torch.mps.set_rng_state(st['mps_rng'])
            start, t_prev, log = st['step'] + 1, st['train_s'], st['log']
            print(f'[{run}] resumed from {f_part} at step {st["step"]}', flush=True)
        # the crop iterators draw from rng only, one whole batch per next(): rebuilt here they continue the sequence
        it = mixed(data, pseudo, a.crop, a.bs, n_p, rng) if a.pseudo else batches(data, a.crop, a.bs, rng)
        t0, run_loss = time.time(), []
        model.train()
        for step in range(start, a.iters + 1):
            opt.zero_grad(set_to_none=True)
            losses = []
            for _ in range(a.accum):                  # --accum k: k micro-batches of --bs per step (k = 1: plain)
                x, y = next(it)
                out = model(x.to(a.device))
                y = y.to(a.device)
                loss = F.cross_entropy(out['out'], y, ignore_index=255)
                if 'aux' in out:
                    loss = loss + 0.4 * F.cross_entropy(out['aux'], y, ignore_index=255)
                (loss / a.accum).backward()
                losses.append(loss.item())
            opt.step()
            sched.step()
            run_loss.append(float(np.mean(losses)))
            if step % 100 == 0:
                log['losses'].append(dict(step=step, loss=float(np.mean(run_loss)), s=round(t_prev + time.time() - t0, 1)))
                print(f'[{run}] step {step} loss {np.mean(run_loss):.4f} ({t_prev + time.time() - t0:.0f} s)', flush=True)
                run_loss = []
            stop = a.stop_after and step == a.stop_after and step < a.iters
            if (a.save_every and step % a.save_every == 0 and step < a.iters) or stop:
                atomic_save(dict(step=step, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                                 np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                                 mps_rng=torch.mps.get_rng_state() if a.device == 'mps' else None,
                                 train_s=round(t_prev + time.time() - t0, 1), log=log,
                                 args={k: v for k, v in vars(a).items() if k not in RESUME_FREE}), f_part)
            if stop:
                print(f'[{run}] stopped after step {step} (--stop_after); rerun to continue', flush=True)
                return
        log['train_s'] = round(t_prev + time.time() - t0, 1)
        atomic_save(model.state_dict(), f_ckpt)
        if os.path.exists(f_part):
            os.remove(f_part)
    pred_run = run + ('_tta' if a.tta else '')
    for sf in splits:
        split = 'val' if '_val_' in sf else 'test'
        qids = open(os.path.join(REPO, 'splits', f'{sf}.txt')).read().split()
        probs, cm, t0 = [], np.zeros((6, 6), np.int64), time.time()
        for i in qids:
            x, g = load(split, i, a.query_dataset or a.dataset)
            P = predict(model, x, a.device, a.tta)
            probs.append(np.round(P * 255).astype(np.uint8))
            full = np.concatenate([P, 1 - P.sum(0, keepdims=True)])      # + other
            cm += confusion(full.argmax(0).astype(np.uint8), g.numpy().astype(np.uint8))   # its own argmax decode
        name = f'{pred_run}_{sf}'
        np.savez_compressed(os.path.join(a.out, 'preds', f'{name}.npz'), probs=np.stack(probs), ids=np.array(qids), cm=cm)
        res = summarize(cm, names, WET[a.dataset])
        res.update(n_images=len(qids), s_per_image=round((time.time() - t0) / len(qids), 3))
        json.dump(res, open(os.path.join(a.out, 'metrics', f'{name}.json'), 'w'), indent=1)
        log[f'eval_{sf}' + ('_tta' if a.tta else '')] = res
        print(f'[{name}] mIoU (argmax over the 5 classes + other) {res["miou"]:.3f}  ' +
              ' '.join(f'{n} {res[f"iou_{n}"]:.3f}' for n in names), flush=True)
    f_log = os.path.join(a.out, 'logs', f'{run}.json')
    if not log['losses'] and os.path.exists(f_log):     # prediction-only rerun: keep the training log
        log = {**json.load(open(f_log)), **{k: v for k, v in log.items() if k.startswith('eval_')}}
    json.dump(log, open(f_log, 'w'), indent=1)


if __name__ == '__main__':
    main()
