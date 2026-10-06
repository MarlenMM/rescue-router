"""Rescue Router Gradio app with LIVE inference on a Kaggle T4, reachable through a public Gradio share link
(PLAN §15 0l: live hosting plan for the presentation). Not an evaluation: it only serves the app.

Push (needs Marlen's OK: GPU time):  cd kaggle/gradio_live && ~/.venvs/kaggle/bin/kaggle kernels push -p .
Setup takes ~10 min; then the log prints the share link (https://<id>.gradio.live) in a banner, and
/kaggle/working/share_url.txt holds it. The app keeps serving for HOURS, then the script ends (GPU time = setup +
HOURS). To stop early, cancel the run in the Kaggle web UI (the CLI has no cancel).
Inputs: dataset marlenmelis/rr-code (kaggle/build_rr_code.sh: src + decodes + tiers + viewer index), notebook
outputs marlenmelis/rr-fetch-data (data tars, generalist.ckpt), marlenmelis/rr-gpu-select50 (FloodNet 50-shot
checkpoint + eval100 cache), marlenmelis/rr-gpu-queue (RescueNet 20-shot eval100 cache)."""
import glob, os, shutil, subprocess, sys, tarfile, threading, time
HOURS = 3.0                                                            # presentation day; the Batch 7 test ran 0.75
GRADIO = 'gradio==6.29.1'                                          # the version the app was written and tested with
T0 = time.time()
def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f'$ {cmd[:100]} [rc={r.returncode}]\n' + (r.stdout + r.stderr)[-1500:], flush=True)
    return r
def find(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    assert hits, f'{name} not found under /kaggle/input'
    return hits[0]

sh('nvidia-smi --query-gpu=index,name,memory.total --format=csv')
try:
    root = os.path.dirname(os.path.dirname(find('src/gradio_app.py')))
except AssertionError:
    root = '/content/rr_code'
    with tarfile.open(find('rr_code.tar.gz')) as t:
        t.extractall(root, filter='data')
shutil.copytree(root, '/content/rr', dirs_exist_ok=True)
sh('git clone -q --depth 1 https://github.com/GitGyun/AIC200-Project-StructuredPrediction.git /content/starter')
import numpy
sh(f'pip install -q pytorch_lightning==2.4.0 timm==0.6.13 easydict einops scikit-image pycocotools fastremap numba flow_vis '
   f'"{GRADIO}" "numpy=={numpy.__version__}"')
os.makedirs('/content/rr_data', exist_ok=True)
for tar in ('floodnet_r768.tar', 'rescuenet_r768.tar'):                   # cached scenes of both datasets
    with tarfile.open(find(tar)) as t:
        t.extractall('/content/rr_data', filter='data')
ck = find('generalist.ckpt')
assert os.path.getsize(ck) == 2967114775
RES = '/content/rr/results/kaggle_v2/rr'
os.makedirs(f'{RES}/preds', exist_ok=True)
for run, ds in (('floodnet_50shot_s0_stratified', 'floodnet'), ('rescuenet_20shot_s0_stratified', 'rescuenet')):
    shutil.copy(find(f'preds/{run}_{ds}_test_eval100_fp16.npz'), f'{RES}/preds/')
    assert os.path.exists(f'{RES}/decode/{run}.json'), 'rr-code must ship results/kaggle_v2/rr/decode (build_rr_code.sh)'
shutil.copy(find('ckpt/floodnet_50shot_s0_stratified.ckpt'), '/content/ft.ckpt')
print(f'setup done in {(time.time() - T0) / 60:.1f} min', flush=True)

cmd = [sys.executable, '-u', '/content/rr/src/gradio_app.py', '--live', '--share', '--ckpt', '/content/ft.ckpt',
       '--mt_ckpt', ck, '--chameleon', '/content/starter/chameleon', '--data', '/content/rr_data',
       '--runs', '/content/runs_data', '--amp', 'fp16']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
end = time.time() + HOURS * 3600
found = {}

def pump():                                                             # echo the app's log for its whole life
    for line in p.stdout:                                               # (an undrained pipe would block the app)
        print(line, end='', flush=True)
        if 'url' not in found and 'gradio.live' in line:
            found['url'] = line[line.index('https://'):].split()[0]
            open('/kaggle/working/share_url.txt', 'w').write(found['url'] + '\n')
            print('\n' + '=' * 70 + f'\nSHARE LINK: {found["url"]}\nserving until '
                  f'{time.strftime("%H:%M UTC", time.gmtime(end))}\n' + '=' * 70, flush=True)
threading.Thread(target=pump, daemon=True).start()
while p.poll() is None and time.time() < end:                           # keep serving; heartbeat every 10 min
    time.sleep(min(600, max(1, end - time.time())))
    print(f'still serving {found.get("url", "(no share link yet: see the log)")} '
          f'({max(0, end - time.time()) / 60:.0f} min left)', flush=True)
p.terminate()
try:
    p.wait(timeout=30)
except subprocess.TimeoutExpired:
    p.kill()
print(f'stopped after {(time.time() - T0) / 3600:.2f} h (app exit code {p.wait()})', flush=True)
