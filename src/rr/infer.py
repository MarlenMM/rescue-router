"""Support encoding and tiled inference (PLAN §3.2, §4.3).

The starter's inference either squashes a query to img_size or (autocrop) tiles it with duplicate
tiles and a wide Gaussian. Here: 384 tiles, stride 192 (6 unique tiles at 768x576, 15 at 1024x768),
Gaussian weights with sigma = tile/4, probabilities blended and renormalised.

`amp`: 'none' keeps the starter's fp32 inference; 'fp16' runs encoder/matching/decoder under fp16
autocast (T4 tensor cores, half the activation memory). The starter pins `trainer.inference` to
fp32 with an `@torch.autocast(float32)` decorator, so the fp16 path calls the undecorated function.
"""

import time

import torch


def tile_offsets(size, tile, stride):
    offs = list(range(0, max(size - tile, 0) + 1, stride))
    if offs[-1] != size - tile:
        offs.append(size - tile)
    return offs


AMP_DTYPES = {'none': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}


def amp_context(amp):
    """Autocast for --amp; 'none' = fp32 autocast, which is what the starter's inference uses."""
    return torch.autocast('cuda', dtype=AMP_DTYPES[amp])


def _inference_fn(model, amp):
    if amp == 'none':
        return model.inference
    fn = type(model).inference
    fn = getattr(fn, '__wrapped__', fn)          # strip @torch.autocast(float32)
    return lambda X, task: fn(model, X, task)


def gaussian_window(tile, sigma=None, device='cpu'):
    sigma = sigma or tile / 4
    r = torch.arange(tile, device=device, dtype=torch.float32) - (tile - 1) / 2
    g = torch.exp(-r ** 2 / (2 * sigma ** 2))
    return g[:, None] * g[None, :]


def encode_support(model, verbose=False, amp='none'):
    """(Re-)build and encode the support set from the learner's current support files.
    Must be called after any parameter change (fine-tuning) because the encoded tokens are cached."""
    model.model.reset_support()
    model.support_data = model.prepare_support_data()
    task = model.config.task
    dummy = torch.zeros(1, 3, *model.config.img_size, device='cuda')
    t0 = time.time()
    with torch.no_grad(), amp_context(amp):
        model.encode_support(dummy, task, verbose)
    torch.cuda.synchronize()
    return time.time() - t0


@torch.no_grad()
def tiled_predict(model, X, tile=384, stride=192, batch=6, amp='none'):
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
    infer = _inference_fn(model, amp)
    final, learner.postprocess_final = learner.postprocess_final, (lambda y: y)
    try:
        with amp_context(amp):
            preds = torch.cat([infer(tiles[i:i + batch], task).float()
                               for i in range(0, len(tiles), batch)])
    finally:
        learner.postprocess_final = final
    out = torch.zeros(preds.shape[1], H, W, device='cuda')
    for p, (y, x) in zip(preds, offs):
        out[:, y:y + tile, x:x + tile] += p * w
        wsum[:, y:y + tile, x:x + tile] += w
    return (out / wsum).cpu()
