"""Shared colours and overlay helpers (PLAN Part A: green clear road, blue flooded road, red flooded
house, dark blue water; dry houses are amber, not grey, because grey vanished on grey roofs). Used by the figures and, later, the web-asset export.
CPU only (NumPy + Pillow)."""

import os

import numpy as np
from PIL import Image

# class id (rr.metrics: 0-4 channels, 5 other, 255 ignore) -> RGB
COLORS = {
    'floodnet': {0: (46, 204, 64), 1: (0, 116, 217), 2: (230, 40, 40), 3: (245, 180, 0), 4: (0, 31, 120)},
    # RescueNet uses the same five slots: clear road, blocked road, intact building, damaged building, water
    'rescuenet': {0: (46, 204, 64), 1: (200, 0, 160), 2: (245, 180, 0), 3: (230, 40, 40), 4: (0, 31, 120)},
}
NAMES = {'floodnet': ['road clear', 'road flooded', 'building flooded', 'building dry', 'water'],
         'rescuenet': ['road clear', 'road blocked', 'building intact', 'building damaged', 'water']}


def load_image(data, ds, split, i, res='r768'):
    return np.asarray(Image.open(os.path.join(data, f'{ds}_{res}', split, 'images', f'{i}.jpg')).convert('RGB'))


def load_label(data, ds, split, i, res='r768'):
    return np.asarray(Image.open(os.path.join(data, f'{ds}_{res}', split, 'labels', f'{i}.png')))


def colorize(cls, ds='floodnet'):
    """class map (H, W) -> RGBA uint8; 'other' and ignore are transparent."""
    out = np.zeros(cls.shape + (4,), np.uint8)
    for c, rgb in COLORS[ds].items():
        m = cls == c
        out[m, :3], out[m, 3] = rgb, 255
    return out


def blend(img, cls, ds='floodnet', alpha=0.55):
    """photo (H, W, 3) with the class colours on top (other pixels untouched)."""
    rgba = colorize(cls, ds)
    a = rgba[..., 3:4].astype(np.float32) / 255 * alpha
    return (img.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a).astype(np.uint8)


def legend_handles(ds='floodnet'):
    from matplotlib.patches import Patch
    return [Patch(color=np.array(COLORS[ds][c]) / 255, label=n) for c, n in enumerate(NAMES[ds])]
