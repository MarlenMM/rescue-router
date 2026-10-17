"""The Mac queue's last DeepLabV3 baselines on a Kaggle 2xT4 session (16 Oct, PLAN §16.8; Marlen: "we can run
anything you want in kaggle"): the 24 GB Mac froze running a 7.5 GB MPS training next to the Phase 2 post-processing,
and a T4 trains one in ~25 min instead of ~75. Same script, recipe and seeds as experiments/phase2/mac_queue.txt
(rn35, rn40, rn80: equal total labels, amendment 4; rn20d3, rn20d4: amendment 5), on CUDA instead of MPS; plus a
device control: rn20 (draw 0, done on the Mac 16 Oct 01:41) again on the T4, written to a separate folder, so the
CUDA-vs-MPS difference is measured instead of assumed.

    python kaggle/phase2/baselines_kernel.py [--jobs mac_queue|rn20seeds]   # ~/rr_push/<slug>/ (code of git HEAD)
    cd ~/rr_push/<slug> && ~/.venvs/kaggle/bin/kaggle kernels push -p .
    ~/.venvs/kaggle/bin/python kaggle/fetch_output.py marlenmelis/<slug> results/phase2/download_<jobs>
Job sets: mac_queue (16 Oct, slug rr-p2-baselines, below) and rn20seeds (17 Oct, rr-p2-rn20seeds, PLAN §16.9: the
RescueNet fairness check, training seeds 1 and 2 on the 20 draw-0 photos, each then predicted again with flips).
Outputs: baselines/{preds,metrics,logs,ckpt}/ (-> results/baselines/), baselines_cuda_control/ (-> results/
baselines_cuda_control/), logs/<job>.log.
"""
import base64
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from make_plan import DOCKER_IMAGE  # noqa: E402

RN = '--dataset rescuenet --splits rescuenet_val_tune30,rescuenet_test_eval100,rescuenet_test_full'
JOBS = {
    'mac_queue': ('rr-p2-baselines', {
        0: [('rn80', 'baselines', f'{RN} --shot 80'), ('rn35', 'baselines', f'{RN} --shot 35'),
            ('rn20cuda', 'baselines_cuda_control', f'{RN} --shot 20')],
        1: [('rn40', 'baselines', f'{RN} --shot 40'), ('rn20d3', 'baselines', f'{RN} --shot 20 --seed 0 --support_seed 3'),
            ('rn20d4', 'baselines', f'{RN} --shot 20 --seed 0 --support_seed 4')]}),
    # the second run of each lane finds the checkpoint of the first and only predicts, with flips (cache name _tta)
    'rn20seeds': ('rr-p2-rn20seeds', {
        g: [(f'rn20s{s}', 'baselines', f'{RN} --shot 20 --seed {s} --support_seed 0'),
            (f'rn20s{s}tta', 'baselines', f'{RN} --shot 20 --seed {s} --support_seed 0 --tta')] for g, s in ((0, 1), (1, 2))}),
}

RUNNER = r'''"""rr-p2-baselines: DeepLabV3 baselines on 2xT4 (PLAN §16.8; built by kaggle/phase2/baselines_kernel.py)."""
import base64, glob, io, os, subprocess, sys, tarfile, threading, time
T0 = time.time()
CODE_B64 = '@CODE@'
LANES = @LANES@
W = '/kaggle/working'


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f'$ {cmd[:120]} [rc={r.returncode}]\n' + (r.stdout + r.stderr)[-1500:], flush=True)
    return r


def find(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    assert hits, f'{name} not found under /kaggle/input'
    return hits[0]


sh('nvidia-smi --query-gpu=index,name,memory.total --format=csv; nproc; free -g | head -2')
root = '/content/rr_code'
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64))) as t:
    t.extractall(root, filter='data')
print('code version', open(f'{root}/VERSION').read().strip(), flush=True)
os.makedirs('/content/rr_data', exist_ok=True)
with tarfile.open(find('rescuenet_r768.tar')) as t:
    t.extractall('/content/rr_data', filter='data')
sh('ls /content/rr_data/*; python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, '
   'torch.cuda.device_count())"')
# the COCO weights once, before two lanes race for the same cache file
sh('python -c "import torchvision.models.segmentation as S; '
   'S.deeplabv3_resnet50(weights=S.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)"')
os.makedirs(f'{W}/logs', exist_ok=True)
print(f'setup done in {(time.time() - T0) / 60:.1f} min', flush=True)


def lane(g, jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), RR_DATA='/content/rr_data')
    log = open(f'{W}/logs/lane{g}.out', 'a')
    for name, out, args in jobs:
        cmd = [sys.executable, f'{root}/src/baseline_seg.py', *args, '--device', 'cuda', '--out', f'{W}/{out}']
        log.write(f'{time.strftime("%H:%M:%S")} START {name} {" ".join(args)}\n')
        log.flush()
        t = time.time()
        with open(f'{W}/logs/{name}.log', 'w') as jl:
            rc = subprocess.run(cmd, cwd=root, env=env, stdout=jl, stderr=subprocess.STDOUT).returncode
        log.write(f'{time.strftime("%H:%M:%S")} {"OK" if rc == 0 else f"FAILED rc={rc}"} {name} after '
                  f'{(time.time() - t) / 60:.1f} min\n')
        log.flush()


threads = [threading.Thread(target=lane, args=(g, jobs)) for g, jobs in LANES.items()]
for t in threads:
    t.start()
while any(t.is_alive() for t in threads):
    time.sleep(300)
    print(f'===== {(time.time() - T0) / 3600:.2f} h', flush=True)
    for g in LANES:
        print(f'--- gpu{g}: ' + ' | '.join(open(f'{W}/logs/lane{g}.out').read().splitlines()[-2:]), flush=True)
    for f in sorted(glob.glob(f'{W}/logs/rn*.log')):
        tail = [x for x in open(f).read().splitlines() if x.strip()][-1:]
        print(f'    {os.path.basename(f)}: {tail[0][-160:] if tail else ""}', flush=True)
for g in LANES:
    print(f'--- gpu{g} final:\n' + open(f'{W}/logs/lane{g}.out').read(), flush=True)
for f in glob.glob(f'{W}/**/*.part', recursive=True):     # resume states: not needed once a run finished
    os.remove(f)
print(f'ALL DONE {(time.time() - T0) / 3600:.2f} h', flush=True)
'''


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default='mac_queue', choices=sorted(JOBS))
    SLUG, LANES = JOBS[ap.parse_args().jobs]
    subprocess.run(['zsh', os.path.join(ROOT, 'kaggle/build_rr_code.sh')], check=True, capture_output=True)
    b64 = base64.b64encode(open(os.path.expanduser('~/rr_push/ds/rr_code.tar.gz'), 'rb').read()).decode()
    lanes = {g: [(n, out, shlex.split(args)) for n, out, args in jobs] for g, jobs in LANES.items()}
    s = RUNNER.replace('@CODE@', b64).replace('@LANES@', json.dumps(lanes))
    compile(s, 'run_baselines_kaggle.py', 'exec')
    out = os.path.expanduser(f'~/rr_push/{SLUG}')
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'run_baselines_kaggle.py'), 'w').write(s)
    meta = dict(id=f'marlenmelis/{SLUG}', title=SLUG, code_file='run_baselines_kaggle.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=True, enable_internet=True,
                machine_shape='NvidiaTeslaT4', dataset_sources=[], competition_sources=[],
                kernel_sources=['marlenmelis/rr-fetch-data'], model_sources=[],
                docker_image_pinning_type='original', docker_image=DOCKER_IMAGE)
    json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', out, len(s), 'bytes; lanes', {g: [n for n, _, _ in j] for g, j in LANES.items()})


if __name__ == '__main__':
    main()
