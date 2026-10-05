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


def label_to_onehot(lab, n_ch=N_CH):
    """uint8 (H, W) channel-id map -> Y, M float tensors (C, H, W)."""
    lab = torch.from_numpy(np.array(lab, dtype=np.uint8)).long()
    Y = torch.stack([(lab == c) for c in range(n_ch)]).float()
    M = (lab != IGNORE).float()[None].expand(n_ch, -1, -1).clone()
    return Y, M


def best_crop_offset(Y, M, size, stride=64):
    """Top-left (y, x) of the size x size window covering the most channels, then the most
    weighted pixels. Deterministic; uses only the support image's own label."""
    _, H, W = Y.shape
    best, best_off = None, (0, 0)
    for y in sorted(set(list(range(0, H - size + 1, stride)) + [H - size])):
        for x in sorted(set(list(range(0, W - size + 1, stride)) + [W - size])):
            frac = (Y[:, y:y + size, x:x + size] * M[:, y:y + size, x:x + size]).mean((1, 2))
            score = (float((frac >= 0.01).sum()), float((frac * CROP_WEIGHTS).sum()))
            if best is None or score > best:
                best, best_off = score, (y, x)
    return best_off


class FloodDataset(CustomDenseDataset):
    scale_range = (320, 480)          # training crop side before resizing to img_size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2)   # replaces the starter's hue=0.2

    def load_label(self, stem):
        lab = Image.open(self._find(self.label_dir, stem))
        return label_to_onehot(lab, self.n_channels)

    def postprocess(self, X, Y, M):
        S = self.img_size[0]
        if not self.eval_mode:
            H, W = X.shape[-2:]
            L = int(np.random.randint(self.scale_range[0], min(self.scale_range[1], H, W) + 1))
            y, x = np.random.randint(0, H - L + 1), np.random.randint(0, W - L + 1)
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
        elif self.resize:
            y, x = best_crop_offset(Y, M, S)
            X, Y, M = (t[:, y:y + S, x:x + S] for t in (X, Y, M))
        return X.contiguous(), Y.contiguous(), M.contiguous()


class FloodLearner(CustomLearner):
    """The starter's loaders build `self.BaseDataset`; `register_custom(dataset_cls=...)` only fills a
    registry nothing reads, so the dataset must be attached to the learner class itself."""
    BaseDataset = FloodDataset


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
    X = torch.from_numpy(np.asarray(Image.open(os.path.join(data_root, split, 'images', f'{i}.jpg')).convert('RGB')))
    lab = np.asarray(Image.open(os.path.join(data_root, split, 'labels', f'{i}.png')))
    return X.permute(2, 0, 1).float() / 255, lab
