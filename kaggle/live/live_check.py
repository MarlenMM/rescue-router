"""Live-inference check of the Gradio app on Kaggle (one T4, ~15 min; PLAN §15 Track B step 3).
Loads FloodNet 50-shot exactly as `src/gradio_app.py --live` does (LiveModel), predicts 3 eval100 photos,
compares them with the eval100 cache (same checkpoint, same recipe: must agree up to fp16 noise), runs one
photo through the upload path (upscaled to 4000x3000 first, then prep_upload), and renders the app's view.
Inputs: dataset marlenmelis/rr-code (src incl. gradio_app.py), notebook outputs marlenmelis/rr-fetch-data
(data tars, generalist.ckpt) and marlenmelis/rr-gpu-select50 (checkpoint + eval100 cache).
Outputs: /kaggle/working/live/{live_check.json, view_*.jpg}."""
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

sh('nvidia-smi --query-gpu=index,name,memory.total --format=csv')
OUT = '/kaggle/working/live'
os.makedirs(OUT, exist_ok=True)
try:
    root = os.path.dirname(os.path.dirname(find('src/gradio_app.py')))
except AssertionError:
    root = '/content/rr_code'
    with tarfile.open(find('rr_code.tar.gz')) as t:
        t.extractall(root, filter='data')
shutil.copytree(root, '/content/rr', dirs_exist_ok=True)
sh('git clone -q --depth 1 https://github.com/GitGyun/AIC200-Project-StructuredPrediction.git /content/starter')
import numpy
sh(f'pip install -q pytorch_lightning==2.4.0 timm==0.6.13 easydict einops scikit-image pycocotools fastremap numba flow_vis "numpy=={numpy.__version__}"')
os.makedirs('/content/rr_data', exist_ok=True)
with tarfile.open(find('floodnet_r768.tar')) as t:
    t.extractall('/content/rr_data', filter='data')
ck = find('generalist.ckpt')
assert os.path.getsize(ck) == 2967114775
RUN = 'floodnet_50shot_s0_stratified'
RES = '/content/results'
os.makedirs(f'{RES}/preds', exist_ok=True); os.makedirs(f'{RES}/decode', exist_ok=True)
shutil.copy(find(f'ckpt/{RUN}.ckpt'), '/content/ft.ckpt')
shutil.copy(find(f'preds/{RUN}_floodnet_test_eval100_fp16.npz'), f'{RES}/preds/')
shutil.copy(f'/content/rr/results_decode/{RUN}.json', f'{RES}/decode/')      # shipped with the code tarball
print(f'setup done in {(time.time() - T0) / 60:.1f} min', flush=True)

sys.path.insert(0, '/content/rr/src')
import numpy as np
from PIL import Image
import types
import gradio_app as G
a = types.SimpleNamespace(ckpt='/content/ft.ckpt', mt_ckpt=ck, chameleon='/content/starter/chameleon', data='/content/rr_data',
                          runs='/content/runs_data', support_file='/content/rr/supports/floodnet_stratified_s0.txt',
                          amp='fp16', chunk=1, channel_chunk=1, results=RES)
t = time.time(); live = G.LiveModel(a); t_load = time.time() - t
z = np.load(f'{RES}/preds/{RUN}_floodnet_test_eval100_fp16.npz')
cache = dict(zip([str(i) for i in z['ids']], z['probs']))
dec = json.load(open(f'{RES}/decode/{RUN}.json'))['chosen']
from rr.metrics import decode
report = dict(load_s=round(t_load, 1), images={})
for i in ('7301', '7241', '9107'):
    photo = np.asarray(Image.open(f'/content/rr_data/floodnet_r768/test/images/{i}.jpg').convert('RGB'))
    t = time.time(); P = live(photo); dt = time.time() - t
    C = cache[i]
    agree = float((decode(P / 255., **dec) == decode(C / 255., **dec)).mean())
    report['images'][i] = dict(s=round(dt, 1), max_abs_diff=int(np.abs(P.astype(int) - C).max()),
                               mean_abs_diff=float(np.abs(P.astype(int) - C).mean()), decode_agreement=agree)
    print(i, report['images'][i], flush=True)
    s = G.build_scene('floodnet', photo, P, dec, gt=None)
    Image.fromarray(G.render(G.new_state(s, i))).save(f'{OUT}/view_{i}.jpg', quality=85)
big = np.asarray(Image.open('/content/rr_data/floodnet_r768/test/images/7301.jpg').resize((4000, 3000), Image.BICUBIC))
t = time.time(); P = live(G.prep_upload(big)); report['upload_4000x3000_s'] = round(time.time() - t, 1)
st = G.new_state(G.build_scene('floodnet', G.prep_upload(big), P, dec, gt=None), 'upload')
report['upload_counter'] = G.counter_md(st)
Image.fromarray(G.render(st)).save(f'{OUT}/view_upload.jpg', quality=85)
report['total_min'] = round((time.time() - T0) / 60, 1)
json.dump(report, open(f'{OUT}/live_check.json', 'w'), indent=1)
print(json.dumps(report, indent=1))
