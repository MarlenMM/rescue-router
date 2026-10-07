"""Batch 13 GPU rounds on Kaggle (2x T4 per session; PLAN §15 "Batch 13"). SESSION picks the lane plan.
Inputs: dataset marlenmelis/rr-code (src + splits + supports at the Batch 13 commit), notebook output
marlenmelis/rr-fetch-data (data tars, generalist.ckpt). Outputs: /kaggle/working/rr/{ckpt,metrics,logs,preds}.
Each GPU runs its (finetune, select_ckpt) pairs in order; a new pair starts only before START_LIMIT_H, and the
selection of a finished training always runs. Recipe of every run = the published one (fp16, grad ckpt, cosine,
50 shots, chunk 1) + I34 (1000 steps) + --freeze_rest (same updates, less memory) + I17 EMA (decay .998, scored
at the last step); snapshots at 750 and 1000 are validated on val30 (select_ckpt), plus the EMA at 1000."""
import glob, json, os, shutil, subprocess, sys, tarfile, threading, time
SESSION = 'r1b'
START_LIMIT_H = 3.0

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
try:
    src = os.path.dirname(find('src/finetune.py'))
    root = os.path.dirname(src)
except AssertionError:
    root = '/content/rr_code'
    with tarfile.open(find('rr_code.tar.gz')) as t:
        t.extractall(root, filter='data')
VERSION = open(f'{root}/VERSION').read().strip()
print('code version', VERSION, flush=True)
shutil.copytree(root + '/src', '/content/rr/src', dirs_exist_ok=True)
for sub in ('splits', 'supports'):
    for f in glob.glob(f'{root}/{sub}/*.txt'):
        shutil.copy(f, f'{OUT}/{sub}/')
sh(f'ls {OUT}/splits {OUT}/supports')
sh('git clone -q --depth 1 https://github.com/GitGyun/AIC200-Project-StructuredPrediction.git /content/starter')
for g in (0, 1):
    shutil.copytree('/content/starter', f'/content/starter{g}', symlinks=True)
import numpy
sh(f'pip install -q pytorch_lightning==2.4.0 timm==0.6.13 easydict einops scikit-image pycocotools fastremap numba flow_vis "numpy=={numpy.__version__}"')
os.makedirs('/content/rr_data', exist_ok=True)
with tarfile.open(find('floodnet_r768.tar')) as t:
    t.extractall('/content/rr_data', filter='data')
ck = find('generalist.ckpt')
assert os.path.getsize(ck) == 2967114775
os.symlink(ck, '/content/generalist.ckpt')
sh('ls /content/rr_data/*; python -c "import torch; print(torch.__version__, torch.cuda.device_count())"')
print(f'setup done in {(time.time() - T0) / 60:.1f} min', flush=True)


def ft(tag, extra=(), steps=1000, val_every=250, gbs=2, sel='stratified', ema=0.998):
    return ['finetune', ['--dataset', 'floodnet', '--shot', 50, '--seed', 0, '--sel', sel, '--n_steps', steps,
                         '--amp', 'fp16', '--grad_ckpt', '--freeze_rest', '--defer_val', '--val_every', val_every,
                         '--snap_from', steps - val_every, '--chunk', 1, '--channel_chunk', 1, '--gbs', gbs,
                         '--ema', ema, '--ema_from', steps, '--tag', tag, *extra]]
def sel(tag, s='stratified'):
    return ['select_ckpt', ['--dataset', 'floodnet', '--shot', 50, '--sel', s, '--tag', tag]]
def pair(tag, extra=(), **kw):
    return [ft(tag, extra, **kw), sel(tag, kw.get('sel', 'stratified'))]
FN50 = f'{OUT}/ckpt/floodnet_50shot_s0_stratified.ckpt'
def sweep(variants, split_file='floodnet_val_tune30', split='val'):
    return ['infer_sweep', ['--ckpt', FN50, '--split_file', split_file, '--split', split, '--variants', json.dumps(variants)]]
if SESSION == 'r1b':                    # the published checkpoint from the rr-gpu-select50 output
    hits = glob.glob('/kaggle/input/**/rr/ckpt/floodnet_50shot_s0_stratified.ckpt', recursive=True)
    assert hits, 'floodnet_50shot_s0_stratified.ckpt not found in the inputs'
    shutil.copy(hits[0], FN50)

PLANS = {
    # round 1, session a: the I34 control (plain recipe, 1000 steps), I5, the rare-class bundle, the regulariser bundle,
    # I14, I16
    'r1a': {0: [pair('b13ctl'), pair('b13rare', ['--pos_weight', '1,3,1,3,1', '--rare_crop', 0.5]),
                pair('b13dice', ['--dice', 0.5])],
            1: [pair('b13ld', ['--tune', 'label_decoder']),
                pair('b13reg', ['--rot_blur', '--copy_paste', 0.3, '--attn_dropout', 0.7]),
                pair('b13lr2', ['--lr', 2e-3])]},
    # round 1, session b: I6, I10 (authors' episodes: 2 support + 2 query per step, same images as 1000 x 2), I4, I8,
    # then the prediction-only I24 (stride 128) and I25 (scales 1 + 1.25) with the published checkpoint (I33 sweep)
    'r1b': {0: [pair('b13rp', ['--tune', 'relpos']), pair('b13bal', sel='balanced'), [sweep([dict(name='st128', stride=128)])]],
            1: [pair('b13ep4', steps=500, val_every=125, gbs=4, ema=0.996), pair('b13sa', ['--separate_alpha']),
                [sweep([dict(name='sc1-1.25', scales=[1, 1.25])])]]},
}
PLAN = PLANS[SESSION]
COMMON = ['--drive', OUT, '--data', '/content/rr_data', '--mt_ckpt', '/content/generalist.ckpt']


def lane(g, pairs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), RR_CODE_VERSION=VERSION, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    common = COMMON + ['--chameleon', f'/content/starter{g}/chameleon', '--runs', f'/content/runs_gpu{g}']
    log = open(f'{OUT}/logs/b13_gpu{g}.out', 'a')
    for jobs in pairs:
        if (time.time() - T0) / 3600 > START_LIMIT_H:
            log.write(f'START_LIMIT reached, skipping {jobs[0][0]} {jobs[0][1][-1][:60]}\n'); log.flush()
            continue
        for script, args in jobs:
            cmd = [sys.executable, f'/content/rr/src/{script}.py', *map(str, args), *common]
            log.write(f'{time.strftime("%H:%M:%S")} START {" ".join(map(str, args))}\n'); log.flush()
            t = time.time()
            rc = subprocess.run(cmd, cwd='/content/rr/src', env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            log.write(f'{time.strftime("%H:%M:%S")} {"OK" if rc == 0 else f"FAILED rc={rc}"} {script} after {(time.time() - t) / 60:.1f} min\n')
            log.flush()
            if rc != 0 and script == 'finetune':
                break


threads = [threading.Thread(target=lane, args=(g, p)) for g, p in PLAN.items()]
for t in threads:
    t.start()
while any(t.is_alive() for t in threads):
    time.sleep(300)
    print(f'===== {(time.time() - T0) / 3600:.2f} h', flush=True)
    for g in PLAN:
        lines = [l for l in open(f'{OUT}/logs/b13_gpu{g}.out').read().splitlines()
                 if any(k in l for k in ('START', 'OK', 'FAILED', 'VAL', 'best', ' step ', 'Error', 'DONE', 'snapshot', 'trainable'))]
        print(f'--- gpu{g}:\n' + '\n'.join(lines[-5:]), flush=True)
    sh('nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader')
for g in PLAN:
    print(f'--- gpu{g} final:\n' + '\n'.join(l for l in open(f'{OUT}/logs/b13_gpu{g}.out').read().splitlines()
                                            if any(k in l for k in ('START', 'OK', 'FAILED', 'VAL', 'best', 'Error'))), flush=True)
for f in glob.glob(f'{OUT}/ckpt/*_last.pt'):
    os.remove(f)
print(f'ALL DONE {(time.time() - T0) / 3600:.2f} h', flush=True)
