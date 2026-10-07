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


# ----------------------------------------------------------------------------- Batch 13 prediction options
def predict_multiscale(model, X, scales=(1.0,), stride=192, amp='none'):
    """I24/I25: tiled prediction at every scale (bilinear resize of the photo, probabilities resized back) and the
    mean. scales == (1.0,) is exactly tiled_predict(model, X, stride=stride)."""
    import torch.nn.functional as F
    if tuple(scales) == (1.0,):
        return tiled_predict(model, X, stride=stride, amp=amp)
    _, H, W = X.shape
    acc = None
    for s in scales:
        if s == 1.0:
            P = tiled_predict(model, X, stride=stride, amp=amp)
        else:
            hs, ws = int(round(H * s)), int(round(W * s))
            Xs = F.interpolate(X[None], (hs, ws), mode='bilinear', align_corners=False, antialias=s < 1)[0]
            P = tiled_predict(model, Xs, stride=stride, amp=amp)
            P = F.interpolate(P[None], (H, W), mode='bilinear', align_corners=False, antialias=s > 1)[0]
        acc = P if acc is None else acc + P
    return acc / len(scales)


def set_support_crop(view=0, weights=None):
    """I1/I2: which support crop the next encode_support() builds (rr.data.FloodDataset class options)."""
    from rr.data import FloodDataset
    FloodDataset.support_view = view
    FloodDataset.crop_weights = None if weights is None else torch.tensor(weights, dtype=torch.float32)


def compile_encoder(model):
    """I36: torch.compile every image-encoder block's forward (module objects and state-dict keys unchanged)."""
    n = 0
    for blk in model.model.image_encoder.backbone.blocks:
        blk.forward = torch.compile(blk.forward, dynamic=False)
        n += 1
    return n


def load_finetuned_weights(model, ckpt):
    """Swap in another fine-tuned checkpoint of the same architecture (snapshots, weight averages) without
    rebuilding the 3 GB model (I33). Returns the number of tensors loaded."""
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict']
    own = model.state_dict()
    bad = [k for k in sd if k not in own or own[k].shape != sd[k].shape]
    assert not bad, f'{ckpt}: {len(bad)} tensors do not fit, e.g. {bad[:3]}'
    model.load_state_dict(sd, strict=False)
    model.model.reset_support()
    return len(sd)


def predict_views(model, load, ids, views=1, crop_weights=None, scales=(1.0,), stride=192, amp='none',
                  flip_dims=(), chunk=60):
    """Probabilities (float32 (C, H, W) per id, in order) averaged over `views` support crops (I1), `scales` (I25)
    and the tile stride (I24). load(i) -> (X, label). Images go in chunks so every view's support set is
    encoded once per chunk. Returns (probs list, labels list, support-encode seconds)."""
    out, labs, t_sup = {}, {}, 0.0
    for c0 in range(0, len(ids), chunk):
        part = ids[c0:c0 + chunk]
        for v in range(views):
            set_support_crop(v, crop_weights)
            t_sup += encode_support(model, amp=amp)
            for i in part:
                X, lab = load(i)
                if flip_dims:
                    X = torch.flip(X, flip_dims)
                P = predict_multiscale(model, X, scales, stride, amp)
                if flip_dims:
                    P = torch.flip(P, flip_dims)
                out[i] = P if v == 0 else out[i] + P
                labs[i] = lab
        for i in part:
            out[i] = (out[i] / views).numpy()
    set_support_crop(0, None)
    return [out[i] for i in ids], [labs[i] for i in ids], t_sup
