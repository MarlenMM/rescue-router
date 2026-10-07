"""Best-val checkpoint selection for a `finetune.py --defer_val` run (50 shots on a T4). GPU.

    python src/select_ckpt.py --dataset floodnet --shot 50 [--seed 0 --sel stratified] -- <predict.py args>

Evaluates every snapshot ckpt/<run_id>_step<N>.ckpt on val30 with predict.py, one fresh process per
snapshot (fp16, chunk 1 / channel_chunk 1: 12 GB, fits a T4), then writes what an in-process run
writes: ckpt/<run_id>.ckpt (the best snapshot), preds/<run_id>_<val>.npz, metrics/<run_id>.json
(best + final) and the val history in logs/<run_id>.json. Unknown arguments (--drive, --chameleon,
--data, --runs, --mt_ckpt) are passed on to predict.py.
Batch 13: --tag (the finetune.py variant name), --min_step N (skip earlier snapshots), --steps 750,1000 (only these);
EMA snapshots <run_id>_ema_step<N>.ckpt (finetune.py --ema) are validated too and recorded as log['val_ema'] /
metrics['ema'], but the selection stays among the plain snapshots (the EMA is judged on its own, PLAN rule).
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)   # else the shared --data arg is taken as --dataset
    ap.add_argument('--dataset', default='floodnet')
    ap.add_argument('--shot', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sel', default='stratified')
    ap.add_argument('--drive', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--min_step', type=int, default=0)
    ap.add_argument('--steps', default='', help='comma list: validate only these steps')
    a, rest = ap.parse_known_args()
    rest = [r for r in rest if r != '--']
    run_id = f'{a.dataset}_{a.shot}shot_s{a.seed}_{a.sel}' + (f'_{a.tag}' if a.tag else '')
    val_file = f'{a.dataset}_val_tune30'
    ck = os.path.join(a.drive, 'ckpt')
    step_of = lambda f: int(re.search(r'_step(\d+)\.ckpt$', f).group(1))
    keep = lambda f: step_of(f) >= a.min_step and (not a.steps or step_of(f) in {int(x) for x in a.steps.split(',')})
    snaps = sorted([f for f in glob.glob(os.path.join(ck, f'{run_id}_step*.ckpt')) if keep(f)], key=step_of)
    emas = sorted([f for f in glob.glob(os.path.join(ck, f'{run_id}_ema_step*.ckpt')) if keep(f)], key=step_of)
    assert snaps, f'no snapshots for {run_id}'
    hist, hist_ema = [], []
    for f in snaps + emas:
        step = int(re.search(r'_step(\d+)\.ckpt$', f).group(1))
        tag = f'{os.path.basename(f)[:-5]}_{val_file}_fp16'
        mf = os.path.join(a.drive, 'metrics', f'{tag}.json')
        if not os.path.exists(mf):
            cmd = [sys.executable, os.path.join(HERE, 'predict.py'), '--ckpt', f, '--dataset', a.dataset,
                   '--split_file', val_file, '--split', 'val', '--amp', 'fp16', '--chunk', '1', '--channel_chunk', '1',
                   '--drive', a.drive, *rest]
            rc = subprocess.run(cmd, cwd=HERE).returncode
            if rc != 0:
                print(f'[{run_id}] val of step {step} FAILED rc={rc}', flush=True)
                continue
        res = dict(json.load(open(mf)), step=step)
        is_ema = f in emas
        (hist_ema if is_ema else hist).append(res)
        print(f'[{run_id}] VAL {"EMA " if is_ema else ""}step {step}: miou={res["miou"]:.3f}', flush=True)
    assert hist, f'no snapshot of {run_id} could be validated'
    best = max(hist, key=lambda r: r['miou'])
    btag = f'{run_id}_step{best["step"]}'
    shutil.copy(os.path.join(ck, f'{btag}.ckpt'), os.path.join(ck, f'{run_id}.ckpt'))
    shutil.copy(os.path.join(a.drive, 'preds', f'{btag}_{val_file}_fp16.npz'),
                os.path.join(a.drive, 'preds', f'{run_id}_{val_file}.npz'))
    out = {k: v for k, v in best.items() if k != 'step'}      # predict.py's keys, then the run's own
    out.update(run_id=run_id, snapshot=best['run_id'], split=val_file, tau=0.5, best_step=best['step'],
               selection='deferred (select_ckpt.py)', final=hist[-1])
    if hist_ema:
        out['ema'] = hist_ema
    json.dump(out, open(os.path.join(a.drive, 'metrics', f'{run_id}.json'), 'w'), indent=1)
    lf = os.path.join(a.drive, 'logs', f'{run_id}.json')
    if os.path.exists(lf):
        log = json.load(open(lf))
        log['val'], log['best'] = hist, best
        if hist_ema:
            log['val_ema'] = hist_ema
        json.dump(log, open(lf, 'w'), indent=1)
    print(f'[{run_id}] best val mIoU {best["miou"]:.3f} at step {best["step"]} -> {run_id}.ckpt', flush=True)


if __name__ == '__main__':
    main()
