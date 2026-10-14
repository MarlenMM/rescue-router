"""Phase 2 GPU plans (experiments/phase2/PROTOCOL.md) as lane plans for kaggle/b13/make_kernel.py, and a local
pre-flight that runs the same job chains on this Mac (MPS/CPU) at toy size before any Kaggle quota is spent.

    python kaggle/phase2/make_plan.py plan <session> > plan.json        # sessions: var, curve
    python kaggle/phase2/make_plan.py cost                              # T4 minutes per session (measured costs)
    python kaggle/phase2/make_plan.py preflight <session> [--jobs 2]    # toy run of the first jobs, locally

Every training job = the submitted recipe: 1000 steps, cosine lr 1e-3, batch 2, fp16 + gradient checkpointing,
--freeze_rest, snapshots at 750 and 1000 (no in-process validation), then their weight average (avg_ckpt.py) saved
as ckpt/<run>.ckpt, then predictions (fp16, chunk 1) on val30 and the run's report split(s). Decode (guided filter ->
tau -> SAM), prior thresholds, routing: on the CPU after download (src/run_decode.py, src/phase2_report.py).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from rr.runs import run_name  # noqa: E402

# measured on Kaggle T4s (PLAN §16.5): training s/step with --freeze_rest; prediction s/photo by support count
TRAIN_S_PER_STEP = 2.6
PRED_S = {1: 2.7, 5: 3.7, 10: 4.1, 20: 5.9, 50: 17.0}
N_PHOTOS = {'floodnet_val_tune30': 30, 'floodnet_test_eval100': 100, 'floodnet_val_confirm100': 100,
            'floodnet_test_full': 448, 'rescuenet_val_tune30': 30, 'rescuenet_test_eval100': 100}


def run_jobs(ds, shot, seed=0, draw=None, eval_shot=0, splits=None, steps=1000, tag=''):
    """[finetune, avg_ckpt, predict...] for one run; returns (run id, job group, T4 minutes)"""
    rid = run_name(ds, shot, seed, tag=tag, support_seed=draw)
    snap = steps - 250
    ft = ['finetune', ['--dataset', ds, '--shot', shot, '--seed', seed, '--n_steps', steps, '--amp', 'fp16', '--grad_ckpt',
                       '--freeze_rest', '--defer_val', '--val_every', 250, '--snap_from', snap, '--chunk', 1,
                       '--channel_chunk', 1, '--gbs', 2] + (['--support_seed', draw] if draw is not None else [])
          + (['--eval_shot', eval_shot] if eval_shot else []) + (['--tag', tag] if tag else [])]
    avg = ['avg_ckpt', ['--out', f'{{OUT}}/ckpt/{rid}.ckpt', f'{{OUT}}/ckpt/{rid}_step{snap}.ckpt', f'{{OUT}}/ckpt/{rid}_step{steps}.ckpt']]
    splits = splits or [f'{ds}_val_tune30', f'{ds}_test_eval100']
    preds = [['predict', ['--ckpt', f'{{OUT}}/ckpt/{rid}.ckpt', '--dataset', ds, '--split_file', sf,
                          '--split', 'val' if '_val_' in sf else 'test', '--amp', 'fp16', '--chunk', 1, '--channel_chunk', 1]]
             for sf in splits]
    k = eval_shot or shot
    minutes = (steps * TRAIN_S_PER_STEP + sum(N_PHOTOS[sf] for sf in splits) * PRED_S[min(PRED_S, key=lambda s: abs(s - k))]) / 60
    return rid, [ft, avg] + preds, minutes


def predict_only(ckpt_run, ds, splits):
    jobs = [['predict', ['--ckpt', f'{{OUT}}/ckpt/{ckpt_run}.ckpt', '--dataset', ds, '--split_file', sf,
                         '--split', 'val' if '_val_' in sf else 'test', '--amp', 'fp16', '--chunk', 1, '--channel_chunk', 1]]
            for sf in splits]
    return ckpt_run, jobs, sum(N_PHOTOS[sf] for sf in splits) * PRED_S[50] / 60


def session(name):
    """list of (run id, job group, minutes), longest first is not needed: lanes are balanced greedily below"""
    if name == 'var':            # A3: 3 training seeds x 3 support draws at 50 shots; (0, 0) is the submitted model
        return [run_jobs('floodnet', 50, s, d) for d in (0, 1, 2) for s in (0, 1, 2) if (s, d) != (0, 0)]
    if name == 'curve':          # A4: one recipe for the curve and RescueNet (tag p2: the published runs keep their
        # names); 100 labels predicted with the first 50; + F2: the submitted model on the confirmation split
        return [run_jobs('floodnet', k, tag='p2') for k in (1, 5, 10, 20)] + \
               [run_jobs('floodnet', 100, eval_shot=50, tag='p2'),
                run_jobs('rescuenet', 20, tag='p2'), run_jobs('rescuenet', 50, tag='p2'),
                predict_only('floodnet_50shot_s0_stratified', 'floodnet', ['floodnet_val_confirm100'])]
    raise ValueError(name)


def lanes(items, n=2):
    """greedy longest-first assignment of job groups to the GPUs; returns ({gpu: [group, ...]}, minutes per gpu)"""
    plan, load = {g: [] for g in range(n)}, [0.0] * n
    for rid, jobs, m in sorted(items, key=lambda x: -x[2]):
        g = load.index(min(load))
        plan[g].append(jobs)
        load[g] += m
    return plan, load


def main():
    cmd = sys.argv[1]
    if cmd == 'plan':
        plan, _ = lanes(session(sys.argv[2]))
        print(json.dumps({str(g): v for g, v in plan.items()}, indent=1))
    elif cmd == 'cost':
        tot = 0
        for s in ('var', 'curve'):
            items = session(s)
            _, load = lanes(items)
            wall = max(load) + 15                  # + setup (clone, pip, data, model load)
            tot += wall
            print(f'{s:8s} {len(items)} runs, T4 minutes {sum(m for _, _, m in items):.0f}, '
                  f'2xT4 wall (= quota) {wall:.0f} min')
            for rid, _, m in items:
                print(f'    {rid:45s} {m:5.0f} min')
        print(f'total quota ~{tot / 60:.1f} h of 30 h/week')
    elif cmd == 'preflight':
        preflight(sys.argv[2], int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 1)


def preflight(name, n_groups):
    """the first n job groups of a session with toy sizes, through the same scripts, on this machine"""
    out = os.path.expanduser('~/rr_local/preflight')
    for d in ('ckpt', 'metrics', 'logs', 'preds'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    for sub in ('splits', 'supports'):
        if not os.path.exists(os.path.join(out, sub)):
            os.symlink(os.path.join(ROOT, sub), os.path.join(out, sub))
    py = os.path.expanduser('~/.venvs/rr-cham/bin/python')
    common = ['--drive', out, '--data', os.path.expanduser('~/rr_data'), '--runs', os.path.expanduser('~/rr_local/runs'),
              '--mt_ckpt', os.path.expanduser('~/rr_data/weights/generalist.ckpt'),
              '--chameleon', os.path.expanduser('~/rr_local/chameleon')]
    for rid, jobs, _ in session(name)[:n_groups]:
        for script, args in jobs:
            args = [str(x).replace('{OUT}', out) for x in args]
            if script == 'finetune':                      # toy size: 1 shot, 4 steps, snapshots 2 and 4
                args = toy(args, {'--shot': '1', '--n_steps': '4', '--val_every': '2', '--snap_from': '2'})
                if '--eval_shot' in args:
                    args = toy(args, {'--eval_shot': '1'})
            if script in ('avg_ckpt', 'predict'):
                args = [a.replace('_50shot_', '_1shot_').replace('_100shot_', '_1shot_').replace('_step750', '_step2')
                        .replace('_step1000', '_step4') for a in args]
                args = [a for a in args]
            if script == 'predict':
                args += ['--limit', '1', '--force']
            cmd = [py, os.path.join(ROOT, 'src', f'{script}.py'), *args,
                   *(common if script in ('finetune', 'predict', 'select_ckpt', 'infer_sweep') else [])]
            print('$', ' '.join(cmd[1:])[:200], flush=True)
            r = subprocess.run(cmd, env=dict(os.environ, RR_DEVICE=os.environ.get('RR_DEVICE', 'mps')),
                               capture_output=True, text=True)
            tail = [l for l in (r.stdout + r.stderr).splitlines() if 'Warning' not in l and 'warn' not in l][-3:]
            print(f'  rc={r.returncode}', *tail, sep='\n  ', flush=True)
            if r.returncode:
                sys.exit(f'PREFLIGHT FAILED: {script}')
    print('PREFLIGHT OK')


def toy(args, repl):
    args = list(args)
    for k, v in repl.items():
        if k in args:
            args[args.index(k) + 1] = v
    return args


if __name__ == '__main__':
    main()
