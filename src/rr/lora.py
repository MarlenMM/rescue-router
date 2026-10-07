"""I9 (Batch 13): LoRA adapters on the qkv projections of Chameleon's image encoder (BEiT-v2-L blocks).

BEiT's attention reads `self.qkv.weight` directly (F.linear(x, self.qkv.weight) + qkv_bias), so each qkv Linear is
replaced by LoRAQKV, whose `weight` property returns W + (alpha / r) * B @ A: the frozen W stays, A (r x d) and B
(3d x r, zero-initialised: the model starts exactly as before) are trained. Injected after the model is loaded.
Checkpoints: the fine-tuned state dict holds the A/B tensors (keys ...attn.qkv.lora_A / lora_B) and the config has
rr_lora = (r, alpha); load_chameleon_lora() loads such a checkpoint for stage 2 (the starter's loader is strict).
"""

import os
import tempfile

import torch
import torch.nn as nn


class LoRAQKV(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.register_buffer('weight_base', base.weight.detach().clone(), persistent=False)
        out_f, in_f = base.weight.shape
        self.lora_A = nn.Parameter(torch.randn(r, in_f, device=base.weight.device) / in_f ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=base.weight.device))
        self.scale = alpha / r

    @property
    def weight(self):
        return self.weight_base + self.scale * (self.lora_B @ self.lora_A).to(self.weight_base.dtype)

    def forward(self, x):
        return nn.functional.linear(x, self.weight)


def inject_lora(model, r=8, alpha=16):
    """Replace every image-encoder block's attn.qkv; returns the new parameter names (state-dict keys)."""
    names = []
    for i, blk in enumerate(model.model.image_encoder.backbone.blocks):
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, r, alpha)
        names += [f'model.image_encoder.backbone.blocks.{i}.attn.qkv.lora_A', f'model.image_encoder.backbone.blocks.{i}.attn.qkv.lora_B']
    model.config.rr_lora = (r, alpha)
    return names


def is_lora_ckpt(path):
    return any('lora_A' in k for k in torch.load(path, map_location='cpu', weights_only=False)['state_dict'])


def load_chameleon_lora(su, cfg, mt_ckpt, ckpt):
    """Stage-2 load of a LoRA checkpoint: the starter loads everything but the adapters, then they are injected and
    filled. Returns (model, cfg) like su.load_chameleon."""
    from rr.infer import load_finetuned_weights
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    r, alpha = getattr(ck['hyper_parameters']['config'], 'rr_lora')
    plain = {k: v for k, v in ck['state_dict'].items() if 'lora_' not in k}
    with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
        torch.save(dict(ck, state_dict=plain), f.name)
    try:
        model, cfg = su.load_chameleon(cfg, mt_ckpt, f.name, verbose=False)
    finally:
        os.remove(f.name)
    inject_lora(model, r, alpha)
    load_finetuned_weights(model, ckpt)
    return model, cfg
