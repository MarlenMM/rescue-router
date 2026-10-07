"""I31 (Batch 13): refine the outlines of Chameleon's predicted buildings with SAM (Segment Anything, ViT-B). CPU or GPU.

    python src/sam_refine.py [--sam ~/rr_data/weights/sam_vit_b_01ec64.pth] [--out results/b13/sam]

Second pretrained model: SAM ViT-B (Kirillov et al. 2023, https://github.com/facebookresearch/segment-anything,
weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth, Apache-2.0). Chameleon still decides
every class: SAM only redraws the outline of each building Chameleon found, prompted with that building's box;
the building keeps Chameleon's label (flooded / dry); pixels SAM removes fall back to Chameleon's best non-building
channel (or 'other'). Components below --min_area px are left alone. Scored on val30 with the decode chosen so far
(tuned tau; and the stacked no-cost decode of postproc_sweep.py) against the same decode without SAM.
Modes: 'replace' (SAM outline), 'grow' (only add SAM pixels), 'shrink' (only remove pixels outside SAM).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decode_tune import NAMES, load  # noqa: E402
from postproc_sweep import cleanup, decode_two_step, guided_filter  # noqa: E402
from rr.metrics import OTHER_IDX, WET, confusion, decode, summarize  # noqa: E402

BLD = (2, 3)


def refine(cls, P, masks_for, mode='replace', min_area=50, tau=None):
    """cls (H, W) decoded map, P (C, H, W) probabilities; masks_for(boxes) -> list of (H, W) bool SAM masks."""
    b = np.isin(cls, BLD)
    lab, n = ndimage.label(b)
    if not n:
        return cls
    objs = ndimage.find_objects(lab)
    keep = [k for k in range(n) if (lab[objs[k]] == k + 1).sum() >= min_area]
    if not keep:
        return cls
    H, W = cls.shape
    boxes = []
    for k in keep:
        ys, xs = objs[k]
        boxes.append([max(xs.start - 2, 0), max(ys.start - 2, 0), min(xs.stop + 2, W - 1), min(ys.stop + 2, H - 1)])
    masks = masks_for(np.array(boxes, np.float32))
    # fallback class for pixels SAM takes away from a building: Chameleon's best non-building channel
    t = np.asarray(tau if tau is not None else [0.5] * 5, np.float32).reshape(-1, 1, 1)
    s = P / t
    s[list(BLD)] = -1
    fall = np.where(s.max(0) >= 1, s.argmax(0), OTHER_IDX).astype(np.uint8)
    out = cls.copy()
    for k, m in zip(keep, masks):
        comp = lab == k + 1
        c = np.bincount(cls[comp], minlength=6)[list(BLD)].argmax()
        c = BLD[c]
        ys, xs = objs[k]
        win = np.zeros_like(comp)
        win[max(ys.start - 16, 0):ys.stop + 16, max(xs.start - 16, 0):xs.stop + 16] = True
        m = m & win & ~(b & ~comp)                 # never take over another building
        if mode in ('replace', 'grow'):
            out[m & ~comp] = c
        if mode in ('replace', 'shrink'):
            rem = comp & ~m
            out[rem] = fall[rem]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--sam', default=os.path.expanduser('~/rr_data/weights/sam_vit_b_01ec64.pth'))
    ap.add_argument('--cache', default=None, help='val30 cache (default: the published FloodNet 50-shot one)')
    ap.add_argument('--out', default='results/b13/sam')
    ap.add_argument('--postproc', default='results/b13/postproc', help='postproc_sweep.py output of the same model')
    ap.add_argument('--decode_json', default=None, help='decode_ctl_check.py output of the same model (overrides --postproc)')
    a = ap.parse_args()
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    sam = sam_model_registry['vit_b'](checkpoint=a.sam).to(dev).eval()
    pred = SamPredictor(sam)
    root = os.path.join(a.data, 'floodnet_r768')
    cache = a.cache or os.path.join(a.results, 'preds', 'floodnet_50shot_s0_stratified_floodnet_val_tune30.npz')
    probs, gts = load(cache, root, 'val')
    probs = [p.astype(np.float32) / 255 for p in probs]
    ids = [str(i) for i in np.load(cache)['ids']]
    if a.decode_json:                           # a model's own decode (decode_ctl_check.py): best row = filter + tau + clean-up
        rows_d = json.load(open(a.decode_json))
        best = max(rows_d, key=lambda r: r['miou'])
        tau0 = best['tau']
        combo = dict(best, t_w=None)
        decoders = {
            'tuned tau': lambda P, I: decode(P, tau=tau0),
            'final no-cost decode': lambda P, I: cleanup(decode(guided_filter(I, P, *best['gf']) if best['gf'] else P,
                                                               tau=best['tau']), best['k'], best['r']),
        }
    else:
        pp = json.load(open(os.path.join(a.postproc, 'postproc.json')))
        tau0 = next(r for r in pp if r['idea'] == 'base')['params']['tau']
        combo = max((r for r in pp if r['idea'] == 'combo'), key=lambda r: r['val30']['miou'])['params']
        decoders = {
            'tuned tau': lambda P, I: decode(P, tau=tau0),
            'stacked no-cost decode': lambda P, I: cleanup(decode_two_step(
                guided_filter(I, P, *combo['gf']) if combo['gf'] else P, combo['tau'], combo['t_w']), combo['k'], combo['r']),
        }
    modes = ('none', 'replace', 'grow', 'shrink')
    cms = {(d, m, ma): np.zeros((6, 6), np.int64) for d in decoders for m in modes for ma in (50, 200)}
    t0 = time.time()
    for i, P, g in zip(ids, probs, gts):
        img = np.asarray(Image.open(os.path.join(root, 'val', 'images', f'{i}.jpg')).convert('RGB'))
        grey = img.mean(2).astype(np.float32) / 255
        with torch.no_grad():
            pred.set_image(img)

            def masks_for(boxes):
                tb = pred.transform.apply_boxes_torch(torch.as_tensor(boxes, device=dev), img.shape[:2])
                m, _, _ = pred.predict_torch(point_coords=None, point_labels=None, boxes=tb, multimask_output=False)
                return list(m[:, 0].cpu().numpy())
            for dname, dec in decoders.items():
                base = dec(P, grey)
                tau = combo['tau'] if dname != 'tuned tau' else tau0
                for ma in (50, 200):
                    cms[(dname, 'none', ma)] += confusion(base, g)
                    for m in modes[1:]:
                        cms[(dname, m, ma)] += confusion(refine(base, P, masks_for, m, ma, tau), g)
        print(i, f'{time.time() - t0:.0f}s', flush=True)
    rows = []
    for (d, m, ma), cm in cms.items():
        r = summarize(cm, NAMES['floodnet'], WET['floodnet'])
        rows.append(dict(decode=d, mode=m, min_area=ma, **r))
    json.dump(rows, open(os.path.join(a.out, 'sam.json'), 'w'), indent=1)
    lines = ['| decode | SAM mode | min area | mIoU | bldg flooded | bldg dry | drivable P |', '|---|---|---|---|---|---|---|']
    for r in rows:
        if r['mode'] == 'none' and r['min_area'] == 200:
            continue
        lines.append(f"| {r['decode']} | {r['mode']} | {r['min_area'] if r['mode'] != 'none' else ''} | {r['miou']:.4f} | "
                     f"{r['iou_bldg_flooded']:.3f} | {r['iou_bldg_dry']:.3f} | {r['drivable_precision']:.3f} |")
    open(os.path.join(a.out, 'sam.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
