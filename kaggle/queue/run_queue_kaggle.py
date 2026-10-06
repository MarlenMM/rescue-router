"""Rescue Router GPU queue on Kaggle (2x T4): setup, then one run_queue.py per GPU (PLAN §15).
Inputs: dataset marlenmelis/rr-code (src + splits + supports), notebook output marlenmelis/rr-fetch-data
(data tars, meta-checkpoint, rescuenet_val_tune30.txt). Outputs: /kaggle/working/rr/{ckpt,metrics,logs,preds}.
QUEUE picks the job lists: 'core' = the first queue (notebook rr-gpu-queue, 6 Oct 02:40 KST);
'rerun50' = the two 50-shot runs again with deferred validation (notebook rr-gpu-queue50), because
in-process 50-shot val30 hit CUDA OOM at step 500 in the core queue; 'select50' = only the snapshot
selection + eval100 of those two runs (notebook rr-gpu-select50; the snapshots come from the
rr-gpu-queue50 output, whose select step failed on an argparse abbreviation bug); 'extra' = zero-label
transfer of FloodNet 50-shot to RescueNet eval100 + the full 448-image FloodNet test of FloodNet 50-shot,
split in two halves over the GPUs (notebook rr-gpu-extra; checkpoint from the rr-gpu-select50 output);
'flips' = Batch 7: test-time flips (h, v, hv) of FloodNet 50-shot on val30 (decode re-tuning) and the full test
(eval100 first, then the other 348 in two halves), plus the Gradio live check on GPU 1 before its flips
(notebook rr-gpu-flips; checkpoint from rr-gpu-select50)."""
import glob, json, os, shutil, subprocess, sys, tarfile, time
QUEUE = 'flips'

T0 = time.time()
def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f'$ {cmd[:100]} [rc={r.returncode}]\n' + (r.stdout + r.stderr)[-1500:], flush=True)
    return r
def find(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    assert hits, f'{name} not found under /kaggle/input'
    return hits[0]

sh('nvidia-smi --query-gpu=index,name,memory.total --format=csv; nproc; free -g | head -2')
OUT = '/kaggle/working/rr'
for d in ('ckpt', 'metrics', 'logs', 'preds', 'splits', 'supports'):
    os.makedirs(f'{OUT}/{d}', exist_ok=True)

# code + lists (the dataset may hold the tarball or its extracted contents)
try:
    src = os.path.dirname(find('src/finetune.py'))
    root = os.path.dirname(src)
except AssertionError:
    root = '/content/rr_code'
    with tarfile.open(find('rr_code.tar.gz')) as t:
        t.extractall(root, filter='data')
shutil.copytree(root + '/src', '/content/rr/src', dirs_exist_ok=True)
for sub in ('splits', 'supports'):
    for f in glob.glob(f'{root}/{sub}/*.txt'):
        shutil.copy(f, f'{OUT}/{sub}/')
shutil.copy(find('rescuenet_val_tune30.txt'), f'{OUT}/splits/')
sh(f'ls {OUT}/splits {OUT}/supports /content/rr/src')

# starter (one copy per GPU: the starter writes data_paths.yaml into its own folder)
sh('git clone -q --depth 1 https://github.com/GitGyun/AIC200-Project-StructuredPrediction.git /content/starter')
for g in (0, 1):
    shutil.copytree('/content/starter', f'/content/starter{g}', symlinks=True)
import numpy
sh(f'pip install -q pytorch_lightning==2.4.0 timm==0.6.13 easydict einops scikit-image pycocotools fastremap numba flow_vis "numpy=={numpy.__version__}"')

# data + checkpoint
os.makedirs('/content/rr_data', exist_ok=True)
for tarname in ('floodnet_r768.tar', 'rescuenet_r768.tar'):
    with tarfile.open(find(tarname)) as t:
        t.extractall('/content/rr_data', filter='data')
ck = find('generalist.ckpt')
assert os.path.getsize(ck) == 2967114775
os.symlink(ck, '/content/generalist.ckpt')
sh('ls /content/rr_data /content/rr_data/*; python -c "import torch; print(torch.__version__, torch.cuda.device_count())"')
print(f'setup done in {(time.time() - T0) / 60:.1f} min', flush=True)

# queues (recipe: fp16 + grad ckpt, cosine lr, best-val checkpoint; PLAN §15)
def ft(ds, shot, steps, val_every, chunk, defer=False):
    return ['finetune', ['--dataset', ds, '--shot', shot, '--seed', 0, '--n_steps', steps, '--amp', 'fp16', '--grad_ckpt',
                         '--val_every', val_every, '--chunk', chunk, '--channel_chunk', 1] + (['--defer_val'] if defer else [])]
def sel(ds, shot):
    return ['select_ckpt', ['--dataset', ds, '--shot', shot]]
def pr(ds, shot, chunk):
    return ['predict', ['--ckpt', f'{OUT}/ckpt/{ds}_{shot}shot_s0_stratified.ckpt', '--dataset', ds,
                        '--split_file', f'{ds}_test_eval100', '--split', 'test', '--amp', 'fp16', '--chunk', chunk, '--channel_chunk', 1]]
Q_CORE = {0: [ft('floodnet', 1, 1000, 250, 6), pr('floodnet', 1, 6),
         ft('floodnet', 50, 2000, 500, 1), pr('floodnet', 50, 1),
         ft('floodnet', 20, 2000, 250, 6), pr('floodnet', 20, 6),
         ft('floodnet', 10, 1000, 250, 6), pr('floodnet', 10, 6)],
     1: [ft('rescuenet', 20, 2000, 250, 6), pr('rescuenet', 20, 6),
         ft('floodnet', 5, 1000, 250, 6), pr('floodnet', 5, 6),
         ft('rescuenet', 50, 2000, 500, 1), pr('rescuenet', 50, 1)]}
Q_RERUN50 = {0: [ft('floodnet', 50, 2000, 500, 1, defer=True), sel('floodnet', 50), pr('floodnet', 50, 1)],
             1: [ft('rescuenet', 50, 2000, 500, 1, defer=True), sel('rescuenet', 50), pr('rescuenet', 50, 1)]}
Q_SELECT50 = {0: [sel('floodnet', 50), pr('floodnet', 50, 1)], 1: [sel('rescuenet', 50), pr('rescuenet', 50, 1)]}
FN50 = f'{OUT}/ckpt/floodnet_50shot_s0_stratified.ckpt'
def pr_fn50(split_file, query_dataset='floodnet'):
    return ['predict', ['--ckpt', FN50, '--dataset', 'floodnet', '--query_dataset', query_dataset,
                        '--split_file', split_file, '--split', 'test', '--amp', 'fp16', '--chunk', 1, '--channel_chunk', 1]]
def fl(split_file, flip, split='test'):
    return ['predict', ['--ckpt', FN50, '--dataset', 'floodnet', '--split_file', split_file, '--split', split,
                        '--amp', 'fp16', '--chunk', 1, '--channel_chunk', 1, '--flip', flip]]
Q_FLIPS = {0: [fl('floodnet_val_tune30', 'h', 'val'), fl('floodnet_test_eval100', 'h'), fl('floodnet_test_eval100', 'v'),
               fl('floodnet_test_rest348_a', 'h'), fl('floodnet_test_rest348_a', 'v'), fl('floodnet_test_rest348_b', 'h')],
           1: [fl('floodnet_val_tune30', 'v', 'val'), fl('floodnet_val_tune30', 'hv', 'val'), fl('floodnet_test_eval100', 'hv'),
               fl('floodnet_test_rest348_a', 'hv'), fl('floodnet_test_rest348_b', 'v'), fl('floodnet_test_rest348_b', 'hv')]}
Q_EXTRA = {0: [pr_fn50('floodnet_test_full_part1')],
           1: [pr_fn50('rescuenet_test_eval100', 'rescuenet'), pr_fn50('floodnet_test_full_part2')]}
Q = {'core': Q_CORE, 'rerun50': Q_RERUN50, 'select50': Q_SELECT50, 'extra': Q_EXTRA, 'flips': Q_FLIPS}[QUEUE]
if QUEUE in ('extra', 'flips'):         # the headline checkpoint from rr-gpu-select50
    hits = glob.glob('/kaggle/input/**/rr/ckpt/floodnet_50shot_s0_stratified.ckpt', recursive=True)
    assert hits, 'floodnet_50shot_s0_stratified.ckpt not found in the inputs'
    shutil.copy(hits[0], FN50)
    sh(f'ls -la {OUT}/ckpt; wc -l {OUT}/splits/floodnet_test_full_part*.txt')
if QUEUE == 'select50':                 # snapshots + training logs of the 50-shot runs from rr-gpu-queue50
    for f in glob.glob('/kaggle/input/**/rr/ckpt/*50shot*_step*.ckpt', recursive=True):
        shutil.copy(f, f'{OUT}/ckpt/')
    for f in glob.glob('/kaggle/input/**/rr/logs/*_50shot_s0_stratified.json', recursive=True):
        shutil.copy(f, f'{OUT}/logs/')
    sh(f'ls -la {OUT}/ckpt {OUT}/logs')
    assert len(glob.glob(f'{OUT}/ckpt/*_step*.ckpt')) == 8, 'expected 4 snapshots per 50-shot run'
procs = {}
for g, jobs in Q.items():
    json.dump(jobs, open(f'{OUT}/logs/queue_gpu{g}.json', 'w'), indent=1)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), RR_CODE_VERSION=open(f'{root}/VERSION').read().strip() if os.path.exists(f'{root}/VERSION') else 'unknown',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    cmd = [sys.executable, '/content/rr/src/run_queue.py', '--queue', f'{OUT}/logs/queue_gpu{g}.json', '--log', f'{OUT}/logs/queue_gpu{g}.log',
           '--budget_h', '9.5', '--', '--drive', OUT, '--chameleon', f'/content/starter{g}/chameleon',
           '--data', '/content/rr_data', '--runs', f'/content/runs_gpu{g}', '--mt_ckpt', '/content/generalist.ckpt']
    if QUEUE == 'flips' and g == 1:     # Gradio live check first (~10 min), then this GPU's flips
        cmd = ['bash', '-c', f'{sys.executable} {root}/kaggle/live/live_check.py > {OUT}/logs/live_check.out 2>&1; '
               + ' '.join(map(str, cmd))]
    procs[g] = subprocess.Popen(cmd, env=env, stdout=open(f'{OUT}/logs/queue_gpu{g}.out', 'w'), stderr=subprocess.STDOUT)

while any(p.poll() is None for p in procs.values()):
    time.sleep(600)
    print(f'===== {(time.time() - T0) / 3600:.2f} h', flush=True)
    for g in procs:
        lines = [l for l in open(f'{OUT}/logs/queue_gpu{g}.out').read().splitlines()
                 if any(k in l for k in ('START', 'OK', 'FAILED', 'VAL', 'best', 'step', 'Error', 'DONE', 'snapshot'))]
        print(f'--- gpu{g}:\n' + '\n'.join(lines[-4:]), flush=True)
    sh('nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader')
for g in procs:
    print(f'--- gpu{g} final:\n' + open(f'{OUT}/logs/queue_gpu{g}.log').read()[-3000:], flush=True)
for f in glob.glob(f'{OUT}/ckpt/*_last.pt'):      # resume states of unfinished runs stay (small)
    print('unfinished:', f)
print(f'ALL DONE {(time.time() - T0) / 3600:.2f} h', flush=True)
