"""Rescue Router GPU queue on Kaggle (2x T4): setup, then one run_queue.py per GPU (PLAN §15).
Inputs: dataset marlenmelis/rr-code (src + splits + supports), notebook output marlenmelis/rr-fetch-data
(data tars, meta-checkpoint, rescuenet_val_tune30.txt). Outputs: /kaggle/working/rr/{ckpt,metrics,logs,preds}."""
import glob, json, os, shutil, subprocess, sys, tarfile, time

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
def ft(ds, shot, steps, val_every, chunk):
    return ['finetune', ['--dataset', ds, '--shot', shot, '--seed', 0, '--n_steps', steps, '--amp', 'fp16', '--grad_ckpt',
                         '--val_every', val_every, '--chunk', chunk, '--channel_chunk', 1]]
def pr(ds, shot, chunk):
    return ['predict', ['--ckpt', f'{OUT}/ckpt/{ds}_{shot}shot_s0_stratified.ckpt', '--dataset', ds,
                        '--split_file', f'{ds}_test_eval100', '--split', 'test', '--amp', 'fp16', '--chunk', chunk, '--channel_chunk', 1]]
Q = {0: [ft('floodnet', 1, 1000, 250, 6), pr('floodnet', 1, 6),
         ft('floodnet', 50, 2000, 500, 1), pr('floodnet', 50, 1),
         ft('floodnet', 20, 2000, 250, 6), pr('floodnet', 20, 6),
         ft('floodnet', 10, 1000, 250, 6), pr('floodnet', 10, 6)],
     1: [ft('rescuenet', 20, 2000, 250, 6), pr('rescuenet', 20, 6),
         ft('floodnet', 5, 1000, 250, 6), pr('floodnet', 5, 6),
         ft('rescuenet', 50, 2000, 500, 1), pr('rescuenet', 50, 1)]}
procs = {}
for g, jobs in Q.items():
    json.dump(jobs, open(f'{OUT}/logs/queue_gpu{g}.json', 'w'), indent=1)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), RR_CODE_VERSION=open(f'{root}/VERSION').read().strip() if os.path.exists(f'{root}/VERSION') else 'unknown',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    cmd = [sys.executable, '/content/rr/src/run_queue.py', '--queue', f'{OUT}/logs/queue_gpu{g}.json', '--log', f'{OUT}/logs/queue_gpu{g}.log',
           '--budget_h', '9.5', '--', '--drive', OUT, '--chameleon', f'/content/starter{g}/chameleon',
           '--data', '/content/rr_data', '--runs', f'/content/runs_gpu{g}', '--mt_ckpt', '/content/generalist.ckpt']
    procs[g] = subprocess.Popen(cmd, env=env, stdout=open(f'{OUT}/logs/queue_gpu{g}.out', 'w'), stderr=subprocess.STDOUT)

while any(p.poll() is None for p in procs.values()):
    time.sleep(600)
    print(f'===== {(time.time() - T0) / 3600:.2f} h', flush=True)
    for g in procs:
        lines = [l for l in open(f'{OUT}/logs/queue_gpu{g}.out').read().splitlines()
                 if any(k in l for k in ('START', 'OK', 'FAILED', 'VAL', 'best', 'step', 'Error', 'DONE'))]
        print(f'--- gpu{g}:\n' + '\n'.join(lines[-4:]), flush=True)
    sh('nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader')
for g in procs:
    print(f'--- gpu{g} final:\n' + open(f'{OUT}/logs/queue_gpu{g}.log').read()[-3000:], flush=True)
for f in glob.glob(f'{OUT}/ckpt/*_last.pt'):      # resume states of unfinished runs stay (small)
    print('unfinished:', f)
print(f'ALL DONE {(time.time() - T0) / 3600:.2f} h', flush=True)
