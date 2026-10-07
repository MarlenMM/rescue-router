"""Rescue Router dataset for Chameleon's `custom` task (PLAN §3, §13).

Labels on disk are uint8 PNGs from `prep_data.py`: channel id 0-4, 254 = other, 255 = ignore.
`load_label` turns them into Y = one-hot (C, H, W) and M = 1 except at ignore pixels.

`postprocess` replaces the starter's three behaviours (see PLAN §13):
  * training (eval_mode=False): random square crop of side 320-480 px (scale 0.8-1.2 around the
    inference scale), resized to img_size; h/v flips, 90-degree rotations, brightness/contrast 0.2
    (no hue: water colour matters).
  * support (eval_mode=True, resize=True): one deterministic img_size crop per support image,
    the window that covers the most channels (and the most rare-class pixels), instead of squashing
    the whole 768x576 image to 384x384.
  * query (eval_mode=True, resize=False): the whole image, untouched (we tile it ourselves).

Batch 13 options (class attributes, set by finetune.py / predict.py; the defaults reproduce the above exactly and
draw no extra random numbers):
  * crop_weights (I2): channel weights for the support crop (default CROP_WEIGHTS);
  * support_view (I1): 0 = the best support crop; v >= 1 = the v-th next-best window that overlaps every earlier
    view by less than half (a different view of the same labelled photo for the support-view ensemble);
  * rare_crop_p (I13): with this probability the training crop is centred on a random rare-class pixel (RARE);
  * rot_blur (I15): extra augmentation: rotation by up to +-30 degrees (p .5; label nearest, mask 0 outside) and
    Gaussian blur (p .3, sigma .1-1.5);
  * copy_paste_p (I19): with this probability the rare-class pixels of another random support photo (randomly
    flipped) are pasted into the training photo (image, label and mask), before the crop.
FloodLearner options: pos_weight (I12, per-channel BCE positive weights) and dice (I14, weight of a soft-Dice term).
"""

import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ColorJitter

from downstream.custom import CustomDenseDataset, CustomLearner

N_CH = 5
OTHER, IGNORE = 254, 255
# weight of each channel's pixel share when choosing the support crop (rare, routing-critical first)
CROP_WEIGHTS = torch.tensor([1.0, 3.0, 3.0, 1.0, 1.0])
RARE = (1, 3)                 # road_flooded, bldg_dry (FloodNet): the classes the Batch 13 rare-class options favour


def label_to_onehot(lab, n_ch=N_CH):
    """uint8 (H, W) channel-id map -> Y, M float tensors (C, H, W)."""
    lab = torch.from_numpy(np.array(lab, dtype=np.uint8)).long()
    Y = torch.stack([(lab == c) for c in range(n_ch)]).float()
    M = (lab != IGNORE).float()[None].expand(n_ch, -1, -1).clone()
    return Y, M


def best_crop_offset(Y, M, size, stride=64, weights=None, view=0):
    """Top-left (y, x) of the size x size window covering the most channels, then the most
    weighted pixels. Deterministic; uses only the support image's own label.
    view >= 1: the view-th next-best window that overlaps each earlier view by less than half its area."""
    _, H, W = Y.shape
    weights = CROP_WEIGHTS if weights is None else weights
    best, best_off = None, (0, 0)
    cands = []
    for y in sorted(set(list(range(0, H - size + 1, stride)) + [H - size])):
        for x in sorted(set(list(range(0, W - size + 1, stride)) + [W - size])):
            frac = (Y[:, y:y + size, x:x + size] * M[:, y:y + size, x:x + size]).mean((1, 2))
            score = (float((frac >= 0.01).sum()), float((frac * weights).sum()))
            if view:
                cands.append((score, (y, x)))
            if best is None or score > best:
                best, best_off = score, (y, x)
    if not view:
        return best_off
    chosen = [best_off]
    overlap = lambda a, b: max(0, size - abs(a[0] - b[0])) * max(0, size - abs(a[1] - b[1])) / size ** 2
    for score, off in sorted(cands, key=lambda c: c[0], reverse=True):     # stable: ties keep scan order
        if len(chosen) > view:
            break
        if all(overlap(off, c) < 0.5 for c in chosen):
            chosen.append(off)
    return chosen[min(view, len(chosen) - 1)]


class FloodDataset(CustomDenseDataset):
    scale_range = (320, 480)          # training crop side before resizing to img_size
    crop_weights = None               # Batch 13 options (module docstring); defaults = the published recipe
    support_view = 0
    rare_crop_p = 0.0
    rot_blur = False
    copy_paste_p = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2)   # replaces the starter's hue=0.2

    def load_label(self, stem):
        lab = Image.open(self._find(self.label_dir, stem))
        return label_to_onehot(lab, self.n_channels)

    def postprocess(self, X, Y, M):
        S = self.img_size[0]
        if not self.eval_mode:
            if self.copy_paste_p and np.random.rand() < self.copy_paste_p:
                X, Y, M = self.copy_paste(X, Y, M)
            H, W = X.shape[-2:]
            L = int(np.random.randint(self.scale_range[0], min(self.scale_range[1], H, W) + 1))
            y, x = np.random.randint(0, H - L + 1), np.random.randint(0, W - L + 1)
            if self.rare_crop_p and np.random.rand() < self.rare_crop_p:
                ys, xs = torch.nonzero((Y[list(RARE)] * M[list(RARE)]).sum(0) > 0, as_tuple=True)
                if len(ys):
                    j = int(np.random.randint(len(ys)))
                    y = int(np.clip(int(ys[j]) - L // 2, 0, H - L))
                    x = int(np.clip(int(xs[j]) - L // 2, 0, W - L))
            X, Y, M = (t[:, y:y + L, x:x + L] for t in (X, Y, M))
            if L != S:
                X = F.interpolate(X[None], (S, S), mode='bilinear', align_corners=False, antialias=True)[0]
                Y = F.interpolate(Y[None], (S, S), mode='nearest')[0]
                M = F.interpolate(M[None], (S, S), mode='nearest')[0]
            if np.random.rand() < 0.5:
                X, Y, M = (t.flip(-1) for t in (X, Y, M))
            if np.random.rand() < 0.5:
                X, Y, M = (t.flip(-2) for t in (X, Y, M))
            k = int(np.random.randint(4))
            if k:
                X, Y, M = (torch.rot90(t, k, (-2, -1)) for t in (X, Y, M))
            if np.random.rand() < 0.5:
                X = self.jitter(X).clamp(0, 1)
            if self.rot_blur:
                X, Y, M = self.rotate_blur(X, Y, M)
        elif self.resize:
            y, x = best_crop_offset(Y, M, S, weights=self.crop_weights, view=self.support_view)
            X, Y, M = (t[:, y:y + S, x:x + S] for t in (X, Y, M))
        return X.contiguous(), Y.contiguous(), M.contiguous()

    @staticmethod
    def rotate_blur(X, Y, M):
        """I15: rotation by up to +-30 degrees (p .5) and Gaussian blur (p .3) of a training crop."""
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode
        if np.random.rand() < 0.5:
            ang = float(np.random.uniform(-30, 30))
            X = TF.rotate(X, ang, InterpolationMode.BILINEAR)
            Y = TF.rotate(Y, ang, InterpolationMode.NEAREST)
            M = TF.rotate(M, ang, InterpolationMode.NEAREST)          # fill 0 = no supervision outside the photo
        if np.random.rand() < 0.3:
            sigma = float(np.random.uniform(0.1, 1.5))
            X = TF.gaussian_blur(X, [2 * int(np.ceil(3 * sigma)) + 1] * 2, [sigma, sigma])
        return X, Y, M

    def copy_paste(self, X, Y, M):
        """I19: paste the rare-class pixels of another random support photo (same size, random flips)."""
        stem = self.stems[int(np.random.randint(len(self.stems)))]
        X2 = self.base_resizer(self.load_image(stem))
        Y2, M2 = self.load_label(stem)
        if Y2.shape[-2:] != X.shape[-2:] or X2.shape[-2:] != X.shape[-2:]:
            return X, Y, M
        if np.random.rand() < 0.5:
            X2, Y2, M2 = (t.flip(-1) for t in (X2, Y2, M2))
        if np.random.rand() < 0.5:
            X2, Y2, M2 = (t.flip(-2) for t in (X2, Y2, M2))
        m = ((Y2[list(RARE)] * M2[list(RARE)]).sum(0) > 0)[None]
        return torch.where(m, X2, X), torch.where(m, Y2, Y), torch.where(m, M2, M)


class FloodLearner(CustomLearner):
    """The starter's loaders build `self.BaseDataset`; `register_custom(dataset_cls=...)` only fills a
    registry nothing reads, so the dataset must be attached to the learner class itself.
    pos_weight (I12) / dice (I14): Batch 13 loss options; unset = the starter's masked BCE, unchanged."""
    BaseDataset = FloodDataset
    pos_weight = None                 # per-channel BCE weight of positive pixels (list of C floats)
    dice = 0.0                        # weight of the soft-Dice term added to the BCE

    def compute_loss(self, Y_pred, Y, M):
        if self.pos_weight is None and not self.dice:
            return super().compute_loss(Y_pred, Y, M)
        return rr_loss(Y_pred, Y, M, self.pos_weight, self.dice)


def rr_loss(Y_pred, Y, M, pos_weight=None, dice=0.0):
    """Masked BCE with optional per-channel positive weights plus dice x soft Dice (per channel over the batch,
    averaged over the channels). The channel axis is T of the trainer's (B, T, N, C, H, W) tensors (C = 1), or
    axis -3 of a (B, C, H, W) tensor. Same value as the starter's loss when pos_weight is None and dice is 0."""
    ax = 1 if Y_pred.ndim == 6 else Y_pred.ndim - 3
    if pos_weight is None:
        bce = (M * F.binary_cross_entropy_with_logits(Y_pred, Y, reduction='none')).mean()
    else:
        shape = [1] * Y_pred.ndim
        shape[ax] = -1
        pw = torch.as_tensor(pos_weight, dtype=Y_pred.dtype, device=Y_pred.device).reshape(shape)
        w = M * (1 + (pw - 1) * Y)
        bce = (w * F.binary_cross_entropy_with_logits(Y_pred, Y, reduction='none')).mean()
    loss, out = bce, {'loss_bce': bce.detach()}
    if dice:
        p = Y_pred.float().sigmoid() * M
        dims = [d for d in range(Y_pred.ndim) if d != ax]
        inter = (p * Y).sum(dims)
        den = p.sum(dims) + (Y * M).sum(dims)
        d = 1 - (2 * inter + 1) / (den + 1)
        dl = d.mean()
        loss = loss + dice * dl
        out['loss_dice'] = dl.detach()
    out['loss'] = loss.detach()
    return loss, out


# ----------------------------------------------------------------------------- run folders
def build_run_folder(run_root, data_root, support_ids, query_ids=(), split_support='train', split_query='val'):
    """PLAN §2.5: <run_root>/train/{images,labels}/<rank:03d>_<id>.* (the loader takes the first
    `shot` files by name, so supports are nested) and <run_root>/test/... for queries. Symlinks."""
    if os.path.isdir(run_root):
        shutil.rmtree(run_root)
    for sub, ids, split, ranked in (('train', support_ids, split_support, True),
                                    ('test', query_ids, split_query, False)):
        for d in ('images', 'labels'):
            os.makedirs(os.path.join(run_root, sub, d), exist_ok=True)
        for rank, i in enumerate(ids):
            name = f'{rank:03d}_{i}' if ranked else str(i)
            os.symlink(os.path.join(data_root, split, 'images', f'{i}.jpg'),
                       os.path.join(run_root, sub, 'images', f'{name}.jpg'))
            os.symlink(os.path.join(data_root, split, 'labels', f'{i}.png'),
                       os.path.join(run_root, sub, 'labels', f'{name}.png'))
    return run_root


def read_ids(path):
    return [l.strip() for l in open(path) if l.strip()]


def load_query(data_root, split, i):
    """(X (3,H,W) float in [0,1], label uint8 (H,W)) for one preprocessed image."""
    X = torch.from_numpy(np.array(Image.open(os.path.join(data_root, split, 'images', f'{i}.jpg')).convert('RGB')))
    lab = np.asarray(Image.open(os.path.join(data_root, split, 'labels', f'{i}.png')))
    return X.permute(2, 0, 1).float() / 255, lab
