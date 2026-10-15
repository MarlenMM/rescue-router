"""Phase 2 GPU plans (experiments/phase2/PROTOCOL.md) as lane plans for kaggle/b13/make_kernel.py, and a local
pre-flight that runs the same job chains on this Mac (MPS/CPU) at toy size before any Kaggle quota is spent.

    python kaggle/phase2/make_plan.py plan <session> > plan.json        # sessions: curve, var (var_full)
    python kaggle/phase2/make_plan.py cost                              # T4 minutes per session (measured costs)
    python kaggle/phase2/make_plan.py preflight <session> [--jobs 2]    # toy run of the first jobs, locally
    python kaggle/phase2/make_plan.py kernel <session>                  # ~/rr_push/rr-p2-<session>/ ready to push

Every training job = the submitted recipe: 1000 steps, cosine lr 1e-3, batch 2, fp16 + gradient checkpointing,
--freeze_rest, snapshots at 750 and 1000 (no in-process validation), then their weight average (avg_ckpt.py) saved
as ckpt/<run>.ckpt, then predictions (fp16, chunk 1) on val30 and the run's report split(s). Decode (guided filter ->
tau -> SAM), prior thresholds, routing: on the CPU after download (src/run_decode.py, src/phase2_report.py).
"""
import json
import os
import re
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
            'floodnet_test_full': 448, 'rescuenet_val_tune30': 30, 'rescuenet_test_eval100': 100, 'rescuenet_test_full': 450}


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


# FloodNet points up to 20 shots predict the whole test set (448 photos) instead of eval100: prediction is cheap there,
# eval100 is cut from it on the CPU (src/subset_cache.py, bit-exact: photos are predicted independently), and the
# curve is then paired photo by photo with the baseline curve on the full test (amendment 1, PROTOCOL.md)
FULL = ['floodnet_val_tune30', 'floodnet_test_full']
# amendment 4: the same for RescueNet 5-20 shots (its 450 test photos; 50 shots stay on eval100, prediction is 17 s/photo)
RN_FULL = ['rescuenet_val_tune30', 'rescuenet_test_full']
# amendment 5 (for Marlen to keep or drop at the launch): Chameleon's only lead so far (RescueNet 20 photos, with its 30
# tuning photos) rests on one support draw; two more draws of the same 20-shot point (recipe p2, full RescueNet test)
# in `var`. Draws 3 and 4: RescueNet's draws 1 and 2 overlap its val30 (src/splits_local.py rn_draws). Drop: ().
VAR_RN_DRAWS = (3, 4)


def session(name):
    """list of (run id, job group, minutes), longest first is not needed: lanes are balanced greedily below"""
    if name == 'var':            # A3, lean L design (amendment 1): support draws 1, 2 with seed 0 and training seeds
        # 1, 2 with draw 0; (0, 0) is the submitted model. Pairs with the baseline's draws 0-2 by draw.
        return [run_jobs('floodnet', 50, s, d) for s, d in ((0, 1), (0, 2), (1, 0), (2, 0))] + \
               [run_jobs('rescuenet', 20, 0, d, tag='p2', splits=RN_FULL) for d in VAR_RN_DRAWS]
    if name == 'var_full':       # A3 as first registered: 3 training seeds x 3 support draws (two-way variance split)
        return [run_jobs('floodnet', 50, s, d) for d in (0, 1, 2) for s in (0, 1, 2) if (s, d) != (0, 0)]
    if name == 'curve':          # A4: one recipe for the curve and RescueNet (tag p2: the published runs keep their
        # names); 100 labels predicted with the first 50; + F2: the submitted model on the confirmation split;
        # RescueNet 5 and 10 (amendment 1): the low-label end on the second disaster
        return [run_jobs('floodnet', k, tag='p2', splits=FULL) for k in (1, 5, 10, 20)] + \
               [run_jobs('floodnet', 100, eval_shot=50, tag='p2')] + \
               [run_jobs('rescuenet', k, tag='p2', splits=RN_FULL) for k in (5, 10, 20)] + \
               [run_jobs('rescuenet', 50, tag='p2')] + \
               [predict_only('floodnet_50shot_s0_stratified', 'floodnet', ['floodnet_val_confirm100'])]
    raise ValueError(name)


def lanes(items, n=2):
    """job groups -> GPUs; returns ({gpu: [group, ...]}, minutes per gpu). Two GPUs: the exact best split (all 2^k
    assignments; k <= 16 here), else greedy longest-first. In each lane, prediction-only groups (the submitted model on
    confirm100: the model choice and F2 depend on it) go first, so a long session can never skip them, then the
    RescueNet runs (the open question: where Chameleon may still win), then the longest first. The order within a
    lane does not change its length; it decides what exists if a session is cut short."""
    items = list(items)
    if n == 2 and len(items) <= 16:
        best = min(range(2 ** len(items)), key=lambda mask: max(
            sum(m for k, (_, _, m) in enumerate(items) if mask >> k & 1),
            sum(m for k, (_, _, m) in enumerate(items) if not mask >> k & 1)))
        assign = [best >> k & 1 for k in range(len(items))]
    else:
        assign, load = [], [0.0] * n
        for _, _, m in items:                     # (greedy in the given order; callers sort longest first)
            g = load.index(min(load))
            assign.append(g)
            load[g] += m
    plan, load = {g: [] for g in range(n)}, [0.0] * n
    first = lambda x: any(sc == 'finetune' for sc, _ in x[1])                  # noqa: E731
    for (rid, jobs, m), g in sorted(zip(items, assign), key=lambda xg: (first(xg[0]), not xg[0][0].startswith('rescuenet'),
                                                                         -xg[0][2])):
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
        for s in ('curve', 'var', 'var_full'):
            items = session(s)
            _, load = lanes(items)
            wall = max(load) + 18                  # + setup (clone, pip, data, model load) and the environment check
            tot += wall if s != 'var_full' else 0
            print(f'{s:8s} {len(items)} runs, T4 minutes {sum(m for _, _, m in items):.0f}, '
                  f'2xT4 wall (= quota) {wall:.0f} min')
            for rid, _, m in items:
                print(f'    {rid:45s} {m:5.0f} min')
        print(f'total quota (curve + var) ~{tot / 60:.1f} h of 30 h/week; var_full is the alternative to var')
    elif cmd == 'kernel':
        kernel(sys.argv[2])
    elif cmd == 'preflight':
        preflight(sys.argv[2], int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 1)


SETUP = """
for tar in glob.glob('/kaggle/input/**/rescuenet_r768.tar', recursive=True)[:1]:   # RescueNet runs (curve session)
    with tarfile.open(tar) as t:
        t.extractall('/content/rr_data', filter='data')
for f in glob.glob('/kaggle/input/**/rr-final-ckpt/**/floodnet_50shot_s0_stratified.ckpt', recursive=True)[:1]:
    shutil.copy(f, f'{OUT}/ckpt/floodnet_50shot_s0_stratified.ckpt')     # the submitted model (confirm100, env check)
sh(f'ls /content/rr_data {OUT}/ckpt')
"""

# Environment check (PLAN §16.6), run on GPU 0 before the lanes start (~3 min): the submitted model predicts the first
# 3 photos of floodnet_test_full_part1 exactly as the published run did (fp16, chunk 1) and src/env_check.py compares
# them with the published T4 cache (4x4 block means, embedded below). On a failure the session stops at once: runs in
# a different environment would not be comparable with the published ones, and the rest of the quota is kept.
ENV_CACHE = os.path.join(ROOT, 'results', 'kaggle_v2', 'rr', 'preds',
                         'floodnet_50shot_s0_stratified_floodnet_test_full_part1_fp16.npz')
ENV_CHECK = """
import base64 as _b64
_E = '/tmp/envcheck'
for d in ('ckpt', 'metrics', 'logs', 'preds'):
    os.makedirs(f'{_E}/{d}', exist_ok=True)
for sub in ('splits', 'supports'):
    if not os.path.exists(f'{_E}/{sub}'):
        os.symlink(f'{OUT}/{sub}', f'{_E}/{sub}')
open(f'{_E}/ref.npz', 'wb').write(_b64.b64decode(ENV_REF_B64))
if not os.path.exists(f'{OUT}/ckpt/floodnet_50shot_s0_stratified.ckpt'):
    open(f'{OUT}/logs/envcheck.txt', 'w').write('ENVCHECK FAIL: no submitted checkpoint (rr-final-ckpt not mounted)')
    print('ENVIRONMENT CHECK FAILED: the dataset marlenmelis/rr-final-ckpt is not mounted', flush=True)
    sys.exit(1)
shutil.copy(f'{OUT}/ckpt/floodnet_50shot_s0_stratified.ckpt', f'{_E}/ckpt/')
sh(f'cd /content/rr/src && CUDA_VISIBLE_DEVICES=0 python predict.py --ckpt {_E}/ckpt/floodnet_50shot_s0_stratified.ckpt '
   f'--dataset floodnet --query_dataset floodnet --split_file floodnet_test_full_part1 --split test --amp fp16 --chunk 1 '
   f'--channel_chunk 1 --limit 3 --force --drive {_E} --data /content/rr_data --mt_ckpt /content/generalist.ckpt '
   f'--chameleon /content/starter0/chameleon --runs /content/runs_envcheck')
_c = sh(f'python /content/rr/src/env_check.py check --ref {_E}/ref.npz --cache '
        + (glob.glob(f'{_E}/preds/*.npz') or ['(no prediction)'])[0])
open(f'{OUT}/logs/envcheck.txt', 'w').write(_c.stdout + _c.stderr)
if _c.returncode:
    print('ENVIRONMENT CHECK FAILED: stopping before any run (see logs/envcheck.txt)', flush=True)
    sys.exit(1)
print(f'environment check passed after {(time.time() - T0) / 60:.1f} min', flush=True)
"""
# the image of every 5-8 Oct Kaggle run (kaggle kernels pull marlenmelis/rr-b13-final -m), pinned for comparability
DOCKER_IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'


def env_ref_b64():
    """the reference of the environment check as base64 (src/env_check.py ref, 3 photos)"""
    import base64
    import io

    import numpy as np
    from env_check import make_ref
    buf = io.BytesIO()
    np.savez_compressed(buf, **make_ref(ENV_CACHE, 3))
    return base64.b64encode(buf.getvalue()).decode()


def kernel(name):
    """write the session's plan and build the Kaggle kernel folder with kaggle/b13/make_kernel.py"""
    plan, load = lanes(session(name))
    path = os.path.join(HERE, f'plan_{name}.json')
    json.dump({str(g): v for g, v in plan.items()}, open(path, 'w'), indent=1)
    limit = max(load) / 60 + 1.0                    # a job group may still start up to 1 h past the planned end
    sources = 'marlenmelis/rr-final-ckpt'          # the submitted checkpoint: confirm100 (curve), env check (both)
    setup = SETUP + f"ENV_REF_B64 = '{env_ref_b64()}'\n" + ENV_CHECK
    subprocess.run([sys.executable, os.path.join(ROOT, 'kaggle', 'b13', 'make_kernel.py'), '--slug', f'rr-p2-{name}',
                    '--plan', path, '--start_limit', f'{limit:.1f}', '--setup', setup, '--datasets', sources,
                    '--docker_image', DOCKER_IMAGE], check=True)
    print(f'plan {path}; lanes {[round(x) for x in load]} min; start limit {limit:.1f} h; needs the private dataset '
          f'{sources} (the submitted checkpoint file); image pinned to the 8 Oct one; environment check first')


def preflight(name, n_groups):
    """the first n job groups of a session with toy sizes, through the same scripts, on this machine. One pre-flight at
    a time (a lock): they share the starter folder ~/rr_local/chameleon, whose data_paths.yaml every Chameleon job
    rewrites (16 Oct: two pre-flights started together both failed reading a half-written one)"""
    import fcntl
    lock = open(os.path.expanduser('~/rr_local/preflight.lock'), 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    out = os.path.expanduser('~/rr_local/preflight')
    import shutil
    for d in ('ckpt', 'metrics', 'logs', 'preds'):        # a clean slate: toy snapshots left by an earlier pre-flight
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)   # made finetune skip training (15 Oct: 13 groups)
        os.makedirs(os.path.join(out, d))
    for sub in ('splits', 'supports'):
        link = os.path.join(out, sub)
        if os.path.islink(link) and os.readlink(link) != os.path.join(ROOT, sub):
            os.remove(link)                                   # stale, e.g. after the project folder moved (14 Oct)
        if not os.path.lexists(link):
            os.symlink(os.path.join(ROOT, sub), link)
    py = os.path.expanduser('~/.venvs/rr-cham/bin/python')
    common = ['--drive', out, '--data', os.path.expanduser('~/rr_data'), '--runs', os.path.expanduser('~/rr_local/runs'),
              '--mt_ckpt', os.path.expanduser('~/rr_data/weights/generalist.ckpt'),
              '--chameleon', os.path.expanduser('~/rr_local/chameleon')]
    for rid, jobs, _ in session(name)[:n_groups]:
        trains = any(script == 'finetune' for script, _ in jobs)
        if not trains:                                    # predict-only group: the submitted checkpoint itself
            shutil.copy(os.path.join(ROOT, 'results', 'kaggle_v2', 'rr', 'ckpt', f'{rid}.ckpt'), os.path.join(out, 'ckpt'))
        for script, args in jobs:
            args = [str(x).replace('{OUT}', out) for x in args]
            if script == 'finetune':                      # toy size: 1 shot, 4 steps, snapshots 2 and 4
                args = toy(args, {'--shot': '1', '--n_steps': '4', '--val_every': '2', '--snap_from': '2'})
                if '--eval_shot' in args:
                    args = toy(args, {'--eval_shot': '1'})
            if script in ('avg_ckpt', 'predict') and trains:
                args = [re.sub(r'_\d+shot_', '_1shot_', a).replace('_step750', '_step2').replace('_step1000', '_step4')
                        for a in args]
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
            if script == 'finetune' and 'skipping' in r.stdout + r.stderr:
                sys.exit('PREFLIGHT FAILED: finetune skipped training (stale outputs?), so this code was not run')
    print('PREFLIGHT OK')
    print(f'SRC_SHA256 {src_hash()}')               # launch_check.py: the code this pre-flight ran (content, not time)


def src_hash(root=ROOT):
    """SHA-256 over the contents of src/**/*.py (tests excluded), in path order: what a pre-flight checks"""
    import glob
    import hashlib
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(root, 'src', '**', '*.py'), recursive=True)):
        rel = os.path.relpath(f, root)
        if rel.startswith(os.path.join('src', 'tests')):
            continue
        h.update(rel.encode() + b'\0' + open(f, 'rb').read() + b'\0')
    return h.hexdigest()


def toy(args, repl):
    args = list(args)
    for k, v in repl.items():
        if k in args:
            args[args.index(k) + 1] = v
    return args


if __name__ == '__main__':
    main()
