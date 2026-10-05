"""Support encoding and tiled inference (PLAN §3.2, §4.3).

The starter's inference either squashes a query to img_size or (autocrop) tiles it with duplicate
tiles and a wide Gaussian. Here: 384 tiles, stride 192 (6 unique tiles at 768x576, 15 at 1024x768),
Gaussian weights with sigma = tile/4, probabilities blended and renormalised.
"""

import time

import torch


def tile_offsets(size, tile, stride):
    offs = list(range(0, max(size - tile, 0) + 1, stride))
    if offs[-1] != size - tile:
        offs.append(size - tile)
    return offs


def gaussian_window(tile, sigma=None, device='cpu'):
    sigma = sigma or tile / 4
    r = torch.arange(tile, device=device, dtype=torch.float32) - (tile - 1) / 2
    g = torch.exp(-r ** 2 / (2 * sigma ** 2))
    return g[:, None] * g[None, :]


def encode_support(model, verbose=False):
    """(Re-)build and encode the support set from the learner's current support files.
    Must be called after any parameter change (fine-tuning) because the encoded tokens are cached."""
    model.model.reset_support()
    model.support_data = model.prepare_support_data()
    task = model.config.task
    dummy = torch.zeros(1, 3, *model.config.img_size, device='cuda')
    t0 = time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float32):
        model.encode_support(dummy, task, verbose)
    torch.cuda.synchronize()
    return time.time() - t0


@torch.no_grad()
def tiled_predict(model, X, tile=384, stride=192, batch=6):
    """X: (3, H, W) float [0,1] on CPU -> probabilities (C, H, W) float on CPU.
    Bypasses the learner's postprocess_final so the sigmoid probabilities come back."""
    learner, task = model.learner, model.config.task
    # trainer.inference tests `X.shape[-2:] != config.img_size`; a list never equals torch.Size, which
    # would upsample every tile to base_size. A tuple compares correctly, so tiles stay 384x384.
    model.config.img_size = tuple(model.config.img_size)
    _, H, W = X.shape
    ys, xs = tile_offsets(H, tile, stride), tile_offsets(W, tile, stride)
    offs = [(y, x) for y in ys for x in xs]
    Xg = X.cuda()
    tiles = torch.stack([Xg[:, y:y + tile, x:x + tile] for y, x in offs])
    w = gaussian_window(tile, device='cuda')
    out, wsum = None, torch.zeros(1, H, W, device='cuda')
    final, learner.postprocess_final = learner.postprocess_final, (lambda y: y)
    try:
        preds = torch.cat([model.inference(tiles[i:i + batch], task).float()
                           for i in range(0, len(tiles), batch)])
    finally:
        learner.postprocess_final = final
    out = torch.zeros(preds.shape[1], H, W, device='cuda')
    for p, (y, x) in zip(preds, offs):
        out[:, y:y + tile, x:x + tile] += p * w
        wsum[:, y:y + tile, x:x + tile] += w
    return (out / wsum).cpu()
