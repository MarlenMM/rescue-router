"""Same-label-budget baseline: a standard supervised segmenter on the same k FloodNet photos (PLAN §16, A5).

The published comparison (PSPNet, 1,445 labels, .848) is not like for like. Here torchvision's DeepLabV3-ResNet50
(COCO-pretrained, the family of the FloodNet paper's baselines) is fine-tuned on exactly the photos Chameleon used
(supports/floodnet_stratified_s<seed>.txt, first k; or all 1,445 train photos with --shot 0) with the same 384-px
training crops, and predicts whole 768x576 photos. No validation labels are used: fixed schedule, last step.

Output = the same prediction-cache format as Chameleon (uint8 per-channel probabilities; here the softmax over the 5
classes + other, the 5 class channels kept), so decode_tune.py / route_eval.py / bootstrap_ci.py run on it unchanged:
  <out>/preds/baseline_<arch>_<k>shot_s<seed>_<split_file>.npz, <out>/metrics/<same>.json, <out>/logs/<run>.json

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
MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
N_CLS = OTHER_IDX + 1                              # 5 classes + other; 255 = ignore


def load(split, i):
    root = os.path.join(RR_DATA, 'floodnet_r768', split)
    x = torch.from_numpy(np.array(Image.open(os.path.join(root, 'images', f'{i}.jpg')).convert('RGB'))).permute(2, 0, 1)
    g = gt_classes(np.asarray(Image.open(os.path.join(root, 'labels', f'{i}.png'))))
    return x, torch.from_numpy(g.astype(np.int64))


def build(arch, device):
    from torchvision.models import segmentation as S
    if arch == 'deeplabv3_r50':
        m = S.deeplabv3_resnet50(weights=S.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, aux_loss=True)
    elif arch == 'lraspp_mbv3':
        m = S.lraspp_mobilenet_v3_large(weights=S.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
    else:
        raise ValueError(arch)
    if arch == 'deeplabv3_r50':                    # new 6-class output layers; the rest keeps its COCO weights
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
            x, y = x[:, t:t + crop, l:l + crop].float() / 255, y[t:t + crop, l:l + crop]
            if rng.random() < .5:
                x, y = x.flip(-1), y.flip(-1)
            if rng.random() < .5:
                x, y = x.flip(-2), y.flip(-2)
            x = (x * (0.8 + 0.4 * rng.random()) + 0.1 * (rng.random() - .5)).clamp(0, 1)
            xs.append((x - MEAN) / STD)
            ys.append(y)
        yield torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def predict(model, x, device):
    model.eval()
    z = model(((x.float() / 255 - MEAN) / STD)[None].to(device))['out']
    p = F.softmax(z, 1)[0, :OTHER_IDX]             # the 5 class channels (other = 1 - sum)
    return p.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='deeplabv3_r50', choices=['deeplabv3_r50', 'lraspp_mbv3'])
    ap.add_argument('--shot', type=int, default=50, help='0 = all 1,445 train photos')
    ap.add_argument('--seed', type=int, default=0, help='support draw and training seed')
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--crop', type=int, default=384)
    ap.add_argument('--lr', type=float, default=1e-4, help='backbone; heads get 10x')
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--splits', default='floodnet_val_tune30,floodnet_test_full')
    ap.add_argument('--out', default='results/baselines')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    if a.shot:
        ids = open(os.path.join(REPO, 'supports', f'floodnet_stratified_s{a.seed}.txt')).read().split()[:a.shot]
    else:
        ids = open(os.path.join(REPO, 'splits', 'floodnet_support_pool.txt')).read().split()
    run = f'baseline_{a.arch}_{a.shot or "all"}shot_s{a.seed}'
    for d in ('preds', 'metrics', 'logs', 'ckpt'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    f_ckpt = os.path.join(a.out, 'ckpt', f'{run}.pt')
    model = build(a.arch, a.device)
    log = dict(run=run, args=vars(a), supports=ids if a.shot else f'{len(ids)} train photos', losses=[])
    if os.path.exists(f_ckpt):
        model.load_state_dict(torch.load(f_ckpt, map_location=a.device))
        print(f'[{run}] loaded {f_ckpt}', flush=True)
    else:
        data = [load('train', i) for i in ids]
        heads = [p for n, p in model.named_parameters() if 'classifier' in n]
        body = [p for n, p in model.named_parameters() if 'classifier' not in n]
        opt = torch.optim.AdamW([dict(params=body, lr=a.lr), dict(params=heads, lr=10 * a.lr)], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1 - s / a.iters) ** 0.9)
        it, t0, run_loss = batches(data, a.crop, a.bs, rng), time.time(), []
        model.train()
        for step in range(1, a.iters + 1):
            x, y = next(it)
            out = model(x.to(a.device))
            y = y.to(a.device)
            loss = F.cross_entropy(out['out'], y, ignore_index=255)
            if 'aux' in out:
                loss = loss + 0.4 * F.cross_entropy(out['aux'], y, ignore_index=255)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss.append(loss.item())
            if step % 100 == 0:
                log['losses'].append(dict(step=step, loss=float(np.mean(run_loss)), s=round(time.time() - t0, 1)))
                print(f'[{run}] step {step} loss {np.mean(run_loss):.4f} ({time.time() - t0:.0f} s)', flush=True)
                run_loss = []
        log['train_s'] = round(time.time() - t0, 1)
        torch.save(model.state_dict(), f_ckpt)
    for sf in a.splits.split(','):
        split = 'val' if '_val_' in sf else 'test'
        qids = open(os.path.join(REPO, 'splits', f'{sf}.txt')).read().split()
        probs, cm, t0 = [], np.zeros((6, 6), np.int64), time.time()
        for i in qids:
            x, g = load(split, i)
            P = predict(model, x, a.device)
            probs.append(np.round(P * 255).astype(np.uint8))
            full = np.concatenate([P, 1 - P.sum(0, keepdims=True)])      # + other
            cm += confusion(full.argmax(0).astype(np.uint8), g.numpy().astype(np.uint8))   # its own argmax decode
        name = f'{run}_{sf}'
        np.savez_compressed(os.path.join(a.out, 'preds', f'{name}.npz'), probs=np.stack(probs), ids=np.array(qids), cm=cm)
        res = summarize(cm, NAMES, WET['floodnet'])
        res.update(n_images=len(qids), s_per_image=round((time.time() - t0) / len(qids), 3))
        json.dump(res, open(os.path.join(a.out, 'metrics', f'{name}.json'), 'w'), indent=1)
        log[f'eval_{sf}'] = res
        print(f'[{name}] mIoU (argmax over the 5 classes + other) {res["miou"]:.3f}  ' +
              ' '.join(f'{n} {res[f"iou_{n}"]:.3f}' for n in NAMES), flush=True)
    json.dump(log, open(os.path.join(a.out, 'logs', f'{run}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
