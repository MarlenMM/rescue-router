"""Rescue Router Gradio app with LIVE inference on a Google Colab T4 behind a public Gradio share link: the
presentation-day link (Thu 8 Oct ~15:15 KST; PLAN §15 Batch 12). Colab, not Kaggle: the job is ~3 h (GPU rule, 7 Oct).
Kaggle fallback: run_gradio_live.py in this folder. Needs Marlen's OK each time (GPU time).

Paste the whole file into one Colab cell (runtime: T4 GPU) or exec it through colab-mcp. It needs:
  - My Drive: 20260876_MarlenMelis_data.zip and 20260876_MarlenMelis_app.zip anywhere (the submission's zips). With
    the app zip the app is src/gradio_viewer.py: the web viewer inside Gradio, uploads shown with every viewer visual
    (animated truck -> boat -> walk); it also adds the 98 RescueNet eval100 photos the data zip leaves out. Without
    it, the classic src/gradio_app.py;
  - the course folder shortcut "AIC200-StructuredPrediction" in My Drive (generalist.ckpt), as in the starter notebook.
Setup ~8 min (unzip, unpack caches, pip, model load on the first upload). The cell returns as soon as the share link
is printed; the app keeps serving in the background until HOURS have passed or the runtime is released. Free Colab
drops idle sessions: keep the tab open and touch it now and then. Release the runtime afterwards
(Runtime > Disconnect and delete runtime, or google.colab.runtime.unassign()).
"""
import glob, os, shutil, subprocess, sys, threading, time
HOURS = 3.0
GRADIO = 'gradio==6.29.1'                                          # the version the app was written and tested with
T0 = time.time()


def sh(cmd, tail=1500):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f'$ {cmd[:110]} [rc={r.returncode}]\n' + (r.stdout + r.stderr)[-tail:], flush=True)
    return r


def find(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]


from google.colab import drive                                         # noqa: E402
drive.mount('/content/drive')
MY = '/content/drive/MyDrive'
sh('nvidia-smi --query-gpu=name,memory.total --format=csv')
DATA_ZIP = find('/content/20260876_*_data.zip', f'{MY}/20260876_*_data.zip', f'{MY}/*/20260876_*_data.zip',
                f'{MY}/**/20260876_*_data.zip')
APP_ZIP = find('/content/20260876_*_app.zip', f'{MY}/20260876_*_app.zip', f'{MY}/*/20260876_*_app.zip',
               f'{MY}/**/20260876_*_app.zip')
META = find(f'{MY}/AIC200-StructuredPrediction/checkpoints/**/generalist*.ckpt', f'{MY}/*/checkpoints/**/generalist*.ckpt',
            '/content/generalist.ckpt')
assert DATA_ZIP, 'put 20260876_MarlenMelis_data.zip in My Drive'
assert META, 'generalist.ckpt not found: add the course folder shortcut to My Drive'
print('data zip:', DATA_ZIP, '| app zip:', APP_ZIP, '| meta ckpt:', META, flush=True)

RR = '/content/rr'
if not os.path.exists(f'{RR}/BUNDLE.json'):
    sh(f'unzip -q -o "{DATA_ZIP}" -d {RR}')
if APP_ZIP and not os.path.exists(f'{RR}/viewer/index.html'):      # viewer, gradio_viewer.py, RescueNet photos
    sh(f'unzip -q -o "{APP_ZIP}" "rr_data/*" "viewer/*" "src/gradio_viewer.py" -d {RR}')
APP = f'{RR}/src/gradio_viewer.py' if os.path.exists(f'{RR}/viewer/index.html') else f'{RR}/src/gradio_app.py'
if not os.path.exists(f'{RR}/results/kaggle_v2/rr/preds/floodnet_50shot_s0_stratified_floodnet_test_eval100_fp16.npz'):
    sh(f'cd {RR} && {sys.executable} src/cache_pack.py unpack --src results/kaggle_v2/rr_packed --workers {os.cpu_count()}',
       tail=300)
if not os.path.isdir('/content/starter'):
    sh('git clone -q --depth 1 https://github.com/GitGyun/AIC200-Project-StructuredPrediction.git /content/starter')
import numpy                                                           # noqa: E402
sh(f'{sys.executable} -m pip install -q pytorch_lightning==2.4.0 timm==0.6.13 easydict einops scikit-image pycocotools '
   f'fastremap numba flow_vis "{GRADIO}" "numpy=={numpy.__version__}"', tail=400)
print(f'setup done in {(time.time() - T0) / 60:.1f} min', flush=True)

cmd = [sys.executable, '-u', APP, '--live', '--share',
       '--ckpt', f'{RR}/results/kaggle_v2/rr/ckpt/floodnet_50shot_s0_stratified.ckpt', '--mt_ckpt', META,
       '--chameleon', '/content/starter/chameleon', '--data', f'{RR}/rr_data', '--runs', '/content/runs_data',
       '--amp', 'fp16']
LOG = '/content/gradio_live.log'
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=RR)
END = time.time() + HOURS * 3600
found = {}


def pump():                                       # drain the app's output for its whole life (else it blocks)
    with open(LOG, 'a') as log:
        for line in p.stdout:
            log.write(line); log.flush()
            if 'url' not in found and 'gradio.live' in line:
                found['url'] = line[line.index('https://'):].split()[0]
                open('/content/share_url.txt', 'w').write(found['url'] + '\n')


def stop_later():
    while p.poll() is None and time.time() < END:
        time.sleep(30)
    if p.poll() is None:
        p.terminate()


threading.Thread(target=pump, daemon=True).start()
threading.Thread(target=stop_later, daemon=True).start()
while 'url' not in found and p.poll() is None and time.time() - T0 < 1800:
    time.sleep(2)
if 'url' in found:
    print('=' * 70 + f'\nSHARE LINK: {found["url"]}  ({os.path.basename(APP)})\nserving until '
          f'{time.strftime("%H:%M", time.localtime(END))} (runtime clock; log {LOG})\n' + '=' * 70, flush=True)
else:
    print(f'no share link (app exit code {p.poll()}); last log lines:', flush=True)
    print(open(LOG).read()[-3000:])
