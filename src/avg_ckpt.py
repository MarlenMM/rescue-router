"""I23 (Batch 13): average the fine-tuned tensors of several snapshots of one run into one checkpoint. CPU.

    python src/avg_ckpt.py --out ckpt/floodnet_50shot_s0_stratified_wavg1000-2000.ckpt ckpt/*_step1000.ckpt ckpt/*_step1500.ckpt ...

Floating-point tensors are averaged (float32), others taken from the first checkpoint; the config (hyper_parameters)
is the first checkpoint's, so stage 2 rebuilds the model exactly as for the snapshots.
"""

import argparse

import torch


def average(paths):
    cks = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    keys = list(cks[0]['state_dict'])
    assert all(list(c['state_dict']) == keys for c in cks), 'snapshots hold different tensors'
    sd = {}
    for k in keys:
        ts = [c['state_dict'][k] for c in cks]
        sd[k] = (torch.stack([t.float() for t in ts]).mean(0).to(ts[0].dtype) if ts[0].is_floating_point() else ts[0])
    return {'state_dict': sd, 'hyper_parameters': cks[0]['hyper_parameters'], 'averaged_from': list(paths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('ckpts', nargs='+')
    a = ap.parse_args()
    torch.save(average(a.ckpts), a.out)
    print(f'averaged {len(a.ckpts)} checkpoints -> {a.out}')


if __name__ == '__main__':
    main()
