"""Runtime compatibility shims for the vendored Chameleon code (applied before loading a model).

SciPy >= 1.14 removed `scipy.interpolate.interp2d`, which `model/transformers/helpers.py`
(`resize_rel_pos_bias`) calls to resize BEiT's relative-position tables whenever img_size != 224.
On the regular grids used there, SciPy's documented drop-in replacement is RectBivariateSpline
(s = 0, same cubic B-spline): interp2d(x, y, z)(xn, yn) == RectBivariateSpline(y, x, z)(yn, xn).

fp16: the label encoder's ViT attention computes (q @ k^T) * scale, i.e. scales after the matmul, and
the raw q.k overflows fp16 (NaN from block 0 on). `patch_vit_attention_fp16` computes those logits in
fp32 when q/k are half precision; otherwise the original code runs unchanged. The label features are
also large (~1.5e4), so the matching module's value projection overflows too: `patch_matching_fp32`
runs the whole (small) matching module in fp32 when fp16 autocast is active.
"""

import numpy as np

from rr import device as D


class _Interp2d:
    def __init__(self, x, y, z, kind='cubic'):
        from scipy.interpolate import RectBivariateSpline
        k = {'linear': 1, 'cubic': 3, 'quintic': 5}[kind]
        self._spline = RectBivariateSpline(np.asarray(y, float), np.asarray(x, float), np.asarray(z, float), kx=k, ky=k)

    def __call__(self, xn, yn):
        return self._spline(np.asarray(yn, float), np.asarray(xn, float))


def patch_scipy_interp2d():
    import scipy.interpolate as si
    try:
        si.interp2d([0, 1, 2, 3], [0, 1, 2, 3], np.zeros((4, 4)), kind='cubic')
    except (NotImplementedError, AttributeError):
        si.interp2d = _Interp2d
        return True
    return False


def patch_vit_attention_fp16():
    import torch
    from model.transformers import vision_transformer as vt
    if getattr(vt.Attention, '_rr_fp16', False):
        return False

    def forward(self, x, b_idx=None):
        B, N, C = x.shape
        qkv = self.qkv(x, b_idx=b_idx).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if q.dtype == torch.float16:        # under fp16 autocast (x itself may be fp32 after LayerNorm)
            with torch.autocast(D.DEVICE, enabled=False):
                attn = ((q.float() * self.scale) @ k.float().transpose(-2, -1)).softmax(dim=-1).to(v.dtype)
        else:                               # original code
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x, b_idx=b_idx))

    vt.Attention.forward = forward
    vt.Attention._rr_fp16 = True
    return True


def _fp16_autocast():
    import torch
    if D.DEVICE != 'cuda':
        return torch.is_autocast_enabled(D.DEVICE) and torch.get_autocast_dtype(D.DEVICE) == torch.float16
    get = getattr(torch, 'get_autocast_dtype', None)
    dtype = get('cuda') if get else torch.get_autocast_gpu_dtype()
    return torch.is_autocast_enabled() and dtype == torch.float16


def patch_matching_fp32():
    import torch
    from model.matching import MatchingModule
    if getattr(MatchingModule, '_rr_fp32', False):
        return False
    orig = MatchingModule.forward

    def up(x):
        if torch.is_tensor(x):
            return x.float() if x.is_floating_point() else x
        return type(x)(up(v) for v in x) if isinstance(x, (list, tuple)) else x

    def forward(self, *args, **kwargs):
        if not _fp16_autocast():
            return orig(self, *args, **kwargs)
        with torch.autocast(D.DEVICE, enabled=False):
            return orig(self, *up(args), **{k: up(v) for k, v in kwargs.items()})

    MatchingModule.forward = forward
    MatchingModule._rr_fp32 = True
    return True


def enable_grad_checkpointing(model, encoders=('image_encoder', 'label_encoder')):
    """Recompute each transformer block of the encoders (BEiT image encoder, ViT label encoder) in the
    backward pass instead of storing its activations (attention maps dominate: 24 layers x
    (channels x images) x 576^2 at 384 px). Gradients are unchanged (RNG state is preserved for
    drop-path); costs one extra encoder forward."""
    import torch
    from torch.utils.checkpoint import checkpoint

    blocks = [blk for e in encoders for blk in getattr(model.model, e).backbone.blocks]
    for blk in blocks:
        if getattr(blk, '_rr_ckpt', False):
            continue
        fwd = blk.forward

        def wrapped(x, *args, _fwd=fwd, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_fwd, x, *args, use_reentrant=False, **kwargs)
            return _fwd(x, *args, **kwargs)

        blk.forward = wrapped
        blk._rr_ckpt = True
    return len(blocks)
