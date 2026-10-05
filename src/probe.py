"""GPU memory/timing probe for PLAN §4.4: one training config or one inference config per process.

    python src/probe.py train <channel_sampling> 0 [gc] [tf32]     # 6 steps, median s/step, peak GB
    python src/probe.py infer <chunk_size> <channel_chunk_size> [tf32]   # 20-shot support + 4 val images
Run each config in a fresh process (an OOM leaves the CUDA cache dirty).
"""
import json
import sys
import time

import torch

HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
sys.path[:0] = [HERE, '/content/AIC200-Project-StructuredPrediction/chameleon']
mode, a1, a2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
flags = set(sys.argv[4:])
OUT = '/content/drive/MyDrive/AIC200-my_work/rescue_router'
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = 'tf32' in flags

import finetune  # noqa: E402
from rr.data import build_run_folder, load_query, read_ids  # noqa: E402
from rr.infer import encode_support, tiled_predict  # noqa: E402

sup = read_ids(f'{OUT}/supports/floodnet_stratified_s0.txt')[:20]
run_root = build_run_folder('/content/runs_data/probe', '/content/rr_data/floodnet_r768', sup)
su = finetune.setup_chameleon('/content/AIC200-Project-StructuredPrediction/chameleon', run_root)
from dataset.utils import to_device  # noqa: E402
from train.optim import get_optimizer  # noqa: E402

cfg = su.build_config(stage=1, dataset='custom', n_tasks=5, loss_type='bce', shot=20, eval_shot=20, img_size=384,
                      base_size=[576, 768], global_batch_size=2, n_steps=10, lr=1e-3,
                      channel_sampling=(a1 if mode == 'train' else -1), seed=0,
                      chunk_size=(a1 if mode == 'infer' else 1), channel_chunk_size=(a2 if mode == 'infer' else 1))
model, cfg = su.load_chameleon(cfg, '/content/generalist.ckpt', verbose=False)
if 'gc' in flags:
    from rr.compat import enable_grad_checkpointing
    enable_grad_checkpointing(model)
res = dict(mode=mode, a1=a1, a2=a2, flags=sorted(flags), gpu=torch.cuda.get_device_name(0))
try:
    if mode == 'train':
        opt, _ = get_optimizer(cfg, model.model)
        model.train()
        it, ts = iter(model.learner.get_train_loader()), []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(6):
            b = to_device(next(it), 'cuda')
            torch.cuda.synchronize(); t0 = time.time()
            Yp, Yq, Mq, _ = model(b)
            loss, _ = model.learner.compute_loss(Yp, Yq, Mq)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize(); ts.append(time.time() - t0)
        res.update(peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), s_per_step=round(sorted(ts)[3], 3))
    else:
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        t_sup = encode_support(model)
        res.update(support_encode_s=round(t_sup, 1), support_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
        torch.cuda.reset_peak_memory_stats()
        val = read_ids(f'{OUT}/splits/floodnet_val_tune30.txt')[:4]
        t0 = time.time()
        for i in val:
            P = tiled_predict(model, load_query('/content/rr_data/floodnet_r768', 'val', i)[0])
        torch.cuda.synchronize()
        res.update(out_shape=list(P.shape), infer_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                   s_per_image=round((time.time() - t0) / len(val), 2))
except torch.OutOfMemoryError:
    res['result'] = 'OOM'
print('PROBE ' + json.dumps(res), flush=True)
