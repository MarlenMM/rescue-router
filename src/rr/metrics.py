"""Decoding and segmentation metrics (PLAN §3.1, §5.1). Pure NumPy, runs on cached predictions."""

import numpy as np

N_CH = 5
OTHER_IDX = N_CH                      # decoded "other" class
IGNORE = 255


def decode(probs, tau=0.5):
    """probs (C, H, W) in [0,1] -> class map (H, W): argmax channel if its prob >= tau, else OTHER."""
    probs = np.asarray(probs, dtype=np.float32)
    am = probs.argmax(0)
    return np.where(probs.max(0) >= tau, am, OTHER_IDX).astype(np.uint8)


def gt_classes(lab):
    """channel-id label (0-4, 254 other, 255 ignore) -> class map with OTHER_IDX, IGNORE kept."""
    g = lab.astype(np.uint8).copy()
    g[(g >= N_CH) & (g != IGNORE)] = OTHER_IDX
    return g


def confusion(pred, gt, n=N_CH + 1):
    v = gt != IGNORE
    return np.bincount(gt[v].astype(np.int64) * n + pred[v], minlength=n * n).reshape(n, n)


WET = {'floodnet': (1, 2, 4), 'rescuenet': (1, 4)}   # road_flooded/blocked, flooded bldg, water


def summarize(cm, names, wet=WET['floodnet']):
    """Per-channel IoU (of the C task channels), mIoU over them, pixel accuracy, binary routing metrics."""
    tp = np.diag(cm).astype(float)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    out = {f'iou_{n}': float(iou[i]) for i, n in enumerate(names)}
    out['miou'] = float(iou[:len(names)].mean())
    out['pixel_acc'] = float(tp.sum() / max(cm.sum(), 1))
    # drivable = channel 0; wet = union of the wet channels (PLAN §5.1)
    p0, g0 = cm[:, 0].sum(), cm[0, :].sum()
    out['drivable_precision'] = float(cm[0, 0] / max(p0, 1))
    out['drivable_recall'] = float(cm[0, 0] / max(g0, 1))
    wet = list(wet)
    tpw = cm[np.ix_(wet, wet)].sum()
    out['wet_iou'] = float(tpw / max(cm[:, wet].sum() + cm[wet, :].sum() - tpw, 1))
    return out
