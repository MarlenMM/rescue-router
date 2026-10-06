"""Decoding and segmentation metrics (PLAN §3.1, §5.1). NumPy (+ SciPy for the building vote), runs on cached predictions."""

import numpy as np

N_CH = 5
OTHER_IDX = N_CH                      # decoded "other" class
IGNORE = 255


def decode(probs, tau=0.5, bldg_t=None, ctx=None, **_):
    """probs (C, H, W) in [0,1] -> class map (H, W).

    tau: scalar (PLAN §3.1: argmax channel if its prob >= tau, else OTHER) or per-channel (C,):
    channel c is a candidate if p_c >= tau_c, and the candidate with the largest p_c / tau_c wins
    (a lower tau_c both admits and favours channel c). A scalar tau gives the plain rule.
    bldg_t: if set, building-level vote: every connected predicted building (channels 2 or 3) gets
    one label, channel 2 if sum(p2) / (sum(p2) + sum(p3)) >= bldg_t, else 3.
    ctx: (d, f) wet-context rule (FloodNet dry-building fix): every connected predicted building is
    flooded (2) iff >= f of the pixels within d px of it (and nearer to it than to any other
    building) are decoded wet (road_flooded or water), else dry (3). Overrides bldg_t.
    """
    probs = np.asarray(probs, dtype=np.float32)
    s = probs / np.asarray(tau, np.float32).reshape(-1, 1, 1)
    out = np.where(s.max(0) >= 1, s.argmax(0), OTHER_IDX).astype(np.uint8)
    if ctx is not None:
        out = wet_context_vote(out, *ctx)
    elif bldg_t is not None:
        out = building_vote(out, probs, bldg_t)
    return out


def wet_context_vote(cls, d, f, wet=(1, 4)):
    """One flooded/dry label per connected predicted building from the decoded wet pixels around it."""
    from scipy import ndimage
    b = (cls == 2) | (cls == 3)
    lab, n = ndimage.label(b)
    if not n:
        return cls
    dist, (iy, ix) = ndimage.distance_transform_edt(~b, return_indices=True)
    ring = np.where(~b & (dist <= d), lab[iy, ix], 0)          # each pixel belongs to its nearest building
    frac = ndimage.mean(np.isin(cls, wet).astype(np.float32), ring, np.arange(1, n + 1))
    flooded = np.nan_to_num(frac) >= f
    out = cls.copy()
    out[b] = np.where(flooded[lab[b] - 1], 2, 3)
    return out


def building_vote(cls, probs, t):
    """One flooded/dry label per connected predicted building (channels 2 = flooded, 3 = dry)."""
    from scipy import ndimage
    b = (cls == 2) | (cls == 3)
    lab, n = ndimage.label(b)
    if not n:
        return cls
    p2 = ndimage.sum(probs[2], lab, np.arange(1, n + 1))
    p3 = ndimage.sum(probs[3], lab, np.arange(1, n + 1))
    flooded = p2 / np.maximum(p2 + p3, 1e-9) >= t
    out = cls.copy()
    out[b] = np.where(flooded[lab[b] - 1], 2, 3)
    return out


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
