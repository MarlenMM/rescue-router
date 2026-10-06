"""Generate the final notebook (PLAN §7) from the cell sources below. CPU only, no Jupyter needed.

    python notebook/build_notebook.py [--out notebook/rescue_router.ipynb]

The submitted notebook is this one after a top-to-bottom run on a fresh T4 (Batch 9, G3), with all outputs saved.
"""

import argparse
import json
import os

CELLS = []


def md(s):
    CELLS.append(('markdown', s.strip('\n')))


def code(s):
    CELLS.append(('code', s.strip('\n')))


md(r'''
# Rescue Router: few-shot flood mapping and rescue routing with Chameleon

**KAIST AIC200 mini-project · Track A + B · 20260876**

A drone photo after a flood is mapped into five classes (clear road, flooded road, flooded building, dry building,
water) by **Chameleon** (Kim et al., ECCV 2024) adapted from **50 labelled FloodNet photos** (Track A). A router then
sends a truck along dry roads, a boat across water and a team on foot to each house, and marks the houses a truck can
reach with confidence (Track B: a web viewer and a Gradio app around the model).

| section | what | runs |
|---|---|---|
| 1 Setup | environment, GPU, data bundle, flags | ~3 min |
| 2 Data | splits, hashes, class mapping, samples | CPU |
| 3 Formulation | channels, labels, tiling | CPU |
| 4 Fine-tuning | recipe, training curves of the 7 runs, a short live fine-tuning run | GPU ~7 min |
| 5 Results | **live recompute of FloodNet 50- and 20-shot vs the caches**, learning curve, CIs, full test, baselines | GPU ~6 min + CPU |
| 6 Failure analysis | where the model goes wrong | CPU |
| 7 Cross-disaster | RescueNet: zero-label transfer and re-adaptation | CPU |
| 8 Routing | router, routing metrics on predicted maps (live), tiers, ablations | CPU |
| 9 App | live upload through the app's model, Gradio, web viewer | GPU ~1 min |
| 10 Reproducibility | weights, data, versions, runtime log | |

With the default flags the notebook needs one T4 (16 GB) and about 30 minutes. Everything that took GPU-hours (the 7
fine-tuning runs and their prediction caches, made on Kaggle T4s with the code in `src/`) is loaded from the data
bundle, and the headline models are recomputed live on a test subset to show the caches are what this code produces.
''')

md('## 1. Setup')
code(r'''
# ---- flags -------------------------------------------------------------------------------------------------------
RUN_TRAINING      = False  # True: retrain all 7 models (FloodNet 1/5/10/20/50, RescueNet 20/50): ~12 GPU-h on a T4
RUN_DEMO_TRAINING = True   # a short fine-tuning run (60 steps, 5 shots) to show the training code works (~6 min)
RUN_LIVE          = True   # recompute FloodNet 50- and 20-shot on 10 eval100 photos, compare with the caches (~6 min)
RUN_FULL_REBUILD  = False  # recompute every report on CPU (src/rebuild_all.py; ~15 min on 8 cores, slower here)
RUN_APP           = False  # launch the Gradio app with a public share link at the end
N_LIVE            = 10     # test photos for the live check
''')
code(r'''
import os, sys, glob, json, time, shutil, subprocess, hashlib, platform
T0, TIMES = time.time(), {}
def tic(name):
    TIMES[name] = time.time()
def toc(name):
    TIMES[name] = round(time.time() - TIMES[name], 1); print(f'[{name}: {TIMES[name] / 60:.1f} min]')

def sh(cmd, cwd=None, tail=40, check=True, env=None):
    """run a shell command, print the last lines of its output"""
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, env=env)
    out = (r.stdout + r.stderr).strip().splitlines()
    if tail:
        print('\n'.join(out[-tail:]))
    if check and r.returncode:
        raise RuntimeError(f'command failed ({r.returncode}): {cmd}')
    return r

IN_COLAB = 'google.colab' in sys.modules
ON_KAGGLE = os.path.isdir('/kaggle/input')
BASE = '/tmp/rr_nb' if ON_KAGGLE else '/content' if IN_COLAB else os.path.abspath('nb_work')   # Kaggle: keep outputs small
RR = os.path.join(BASE, 'rr')                      # the unpacked data bundle = the repository layout
os.makedirs(BASE, exist_ok=True)
print('environment:', 'Kaggle' if ON_KAGGLE else 'Colab' if IN_COLAB else 'local', '| python', platform.python_version())
sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv', check=False)
''')
code(r'''
tic('setup')
# ---- the course's meta-trained checkpoint (Colab: the shared course Drive folder, as in the starter notebook) -----
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
def find(*patterns):
    """first existing file among glob patterns (no walk over a whole Drive: that can take minutes on Colab)"""
    for p in patterns:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
MYDRIVE = '/content/drive/MyDrive'
META_CKPT = os.environ.get('RR_META_CKPT') or find(
    '/kaggle/input/**/generalist.ckpt',                                               # Kaggle: notebook input
    f'{MYDRIVE}/AIC200-StructuredPrediction/checkpoints/**/generalist*.ckpt',          # Colab: course folder shortcut
    f'{MYDRIVE}/*/checkpoints/**/generalist*.ckpt', '/content/generalist.ckpt', os.path.expanduser('~/generalist.ckpt'))
assert META_CKPT, 'generalist.ckpt not found: add the course folder shortcut to your Drive (see the starter notebook)'
print('meta-trained checkpoint:', META_CKPT, f'({os.path.getsize(META_CKPT) / 1e9:.2f} GB)')

# ---- our data bundle (20260876_<NAME>_data.zip, or an already unpacked copy) --------------------------------------
if not os.path.exists(os.path.join(RR, 'BUNDLE.json')):
    unpacked = find('/kaggle/input/**/BUNDLE.json') if ON_KAGGLE else None
    if unpacked:
        shutil.copytree(os.path.dirname(unpacked), RR)
    else:
        z = os.environ.get('RR_DATA_ZIP') or find('/content/20260876_*_data.zip', f'{MYDRIVE}/20260876_*_data.zip',
                                                  f'{MYDRIVE}/AIC200-my_work/**/20260876_*_data.zip', '/kaggle/input/**/20260876_*_data.zip',
                                                  os.path.expanduser('~/20260876_*_data.zip'))
        assert z, ('data bundle not found: put 20260876_<NAME>_data.zip in /content (upload) or in My Drive, '
                   'or set RR_DATA_ZIP to its path')
        print('unzipping', z); sh(f'unzip -q -o "{z}" -d "{RR}"')
os.chdir(RR); sys.path.insert(0, os.path.join(RR, 'src'))
BUNDLE = json.load(open('BUNDLE.json')); COMMIT = open('VERSION').read().strip()
print('bundle:', BUNDLE['created'], '| code commit', COMMIT, f"| {BUNDLE['total_bytes'] / 1e6:.0f} MB")

# ---- prediction caches: unpack (lossless; every array is checked against its SHA-256) -----------------------------
if not os.path.exists('results/kaggle_v2/rr/preds/floodnet_50shot_s0_stratified_floodnet_test_full_fp16.npz'):
    sh(f'{sys.executable} src/cache_pack.py unpack --src results/kaggle_v2/rr_packed --workers {os.cpu_count()}', tail=3)

# ---- starter code (Chameleon) and its packages, as in the course notebook -------------------------------------------
STARTER = os.path.join(BASE, 'starter')
if not os.path.isdir(STARTER):
    sh(f'git clone -q --depth 1 https://github.com/GitGyun/AIC200-Project-StructuredPrediction.git {STARTER}')
CHAMELEON = os.path.join(STARTER, 'chameleon')
import numpy
sh(f'{sys.executable} -m pip install -q pytorch_lightning==2.4.0 timm==0.6.13 easydict einops scikit-image pycocotools '
   f'fastremap numba flow_vis "numpy=={numpy.__version__}"', tail=3)

# ---- working folder for GPU scripts: they read splits/ and supports/ next to their outputs ----------------------
WORK = os.path.join(RR, 'results/kaggle_v2/rr')
for d in ('splits', 'supports'):
    shutil.copytree(d, os.path.join(WORK, d), dirs_exist_ok=True)
DATA = os.path.join(RR, 'rr_data')
if not os.path.exists(os.path.expanduser('~/rr_data')):       # the scripts' default data folder
    os.symlink(DATA, os.path.expanduser('~/rr_data'))
GPU_ARGS = f'--data {DATA} --runs {BASE}/runs --chameleon {CHAMELEON} --mt_ckpt {META_CKPT}'
import torch
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 if torch.cuda.is_available() else 0
print(f'GPU: {GPU} ({GPU_GB:.1f} GB) | torch {torch.__version__}')
if GPU_GB and GPU_GB < 14:
    print('less than 14 GB: 50-shot inference (12 GB peak) may not fit; the live check uses 20 shots only')
toc('setup')
''')

md(r'''
## 2. Data

**FloodNet** (Hurricane Harvey, Texas 2017; 2,343 drone photos, 4000×3000; CDLA-Permissive-1.0) and **RescueNet**
(Hurricane Michael, Florida 2018; CC0 on figshare). Photos are area-downsampled to 768×576, labels nearest-neighbour.
All splits were fixed before any support selection; their SHA-256 sums are checked here.

| split | FloodNet | RescueNet | used for |
|---|---|---|---|
| supports (`supports/*_s0.txt`) | 50 of the 1,445 train photos, stratified, nested (first k = the k-shot set) | 50 of a 419-photo pool from val | fine-tuning |
| val30 | 30 val photos | 30 photos outside the pool | checkpoint selection, decode tuning, router gates |
| eval100 | 100 test photos, 50 % with flooding | 100 test photos | learning curve, viewer |
| full test | all 448 test photos | | headline numbers |
''')
code(r'''
tic('data')
bad = []
for line in open('splits/SHA256SUMS'):
    h, f = line.split()
    if os.path.exists(f) and hashlib.sha256(open(f, 'rb').read()).hexdigest() != h:
        bad.append(f)
print('split files checked against SHA256SUMS:', 'all match' if not bad else f'MISMATCH {bad}')
for f in ['supports/floodnet_stratified_s0.txt', 'supports/rescuenet_stratified_s0.txt', 'splits/floodnet_val_tune30.txt',
          'splits/floodnet_test_eval100.txt', 'splits/floodnet_test_full.txt', 'splits/rescuenet_val_tune30.txt',
          'splits/rescuenet_test_eval100.txt']:
    print(f'{f:42s} {sum(1 for l in open(f) if l.strip()):4d} photos')
''')
md(r'''
**Class mapping** (5 task channels + "other"; channel ids 0–4, 254 = other, 255 = ignore):

| ch | FloodNet (from its 10 classes) | RescueNet (same slots, from its 11 classes) |
|---|---|---|
| 0 | road clear (road-non-flooded) | road clear |
| 1 | road flooded | road blocked (water, sand or debris) |
| 2 | building flooded | building intact (no + minor damage) |
| 3 | building dry (building-non-flooded) | building damaged (major + total destruction) |
| 4 | water | water |
| other | background, tree, grass, vehicle, pool | the rest |
''')
code(r'''
import numpy as np, matplotlib.pyplot as plt
from PIL import Image
from viz import blend, load_image, load_label, COLORS, NAMES
from rr.metrics import gt_classes
ev = [l.strip() for l in open('splits/floodnet_test_eval100.txt') if l.strip()]
fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
for k, i in enumerate(['7301', '7241', '9107']):
    img, lab = load_image(DATA, 'floodnet', 'test', i), gt_classes(load_label(DATA, 'floodnet', 'test', i))
    ax[0, k].imshow(img); ax[0, k].set_title(f'FloodNet {i}')
    ax[1, k].imshow(blend(img, lab, 'floodnet', 0.6)); ax[1, k].set_title('labels')
for a in ax.flat: a.axis('off')
handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(COLORS['floodnet'][c]) / 255) for c in range(5)]
fig.legend(handles, NAMES['floodnet'], loc='lower center', ncol=5); plt.tight_layout(); plt.show()
toc('data')
''')

md(r'''
## 3. Task formulation

Chameleon predicts one binary map per task channel from a support set of (image, label) pairs. Our task has **C = 5
channels**: the label of a photo is `Y` (5, H, W), one-hot over the channels ("other" pixels are 0 in all five), and
the mask `M` is 1 except on ignored pixels. At inference a pixel gets the channel whose probability is highest among
those above a threshold τ (0.5, or per-channel values tuned on val30), else "other".

Training uses 384×384 crops (side 320–480 px rescaled, flips, rot90, brightness/contrast). Inference tiles the
768×576 photo with 384-px tiles at stride 192 (6 tiles) and blends them with Gaussian weights (σ = 96):
''')
code(r'''
from rr.infer import tile_offsets
fig, ax = plt.subplots(figsize=(7, 5.3))
ax.imshow(load_image(DATA, 'floodnet', 'test', '7301'))
for y in tile_offsets(576, 384, 192):
    for x in tile_offsets(768, 384, 192):
        ax.add_patch(plt.Rectangle((x, y), 384, 384, fill=False, lw=2, color=plt.cm.tab10((x // 192 + 3 * y // 192) % 10)))
ax.set_title('6 tiles of 384 px, stride 192'); ax.axis('off'); plt.show()
''')

md(r'''
## 4. Fine-tuning

The course's meta-trained Chameleon is fine-tuned per dataset and shot count with the same recipe (**recipe v2**,
chosen so a 50-shot run fits a 16 GB T4): 384-px crops, all 5 channels, global batch 2, lr 1e-3 with a cosine
schedule, fp16 autocast with a gradient scaler, gradient checkpointing in the image encoder; 1,000 steps up to 10
shots, 2,000 for 20 and 50; validation on val30 every 250 steps (500 at 50 shots, on saved snapshots, because 50-shot
inference does not fit next to the training state) and the checkpoint with the best val30 mIoU is kept. All 7 runs
ran on Kaggle T4s (`kaggle/queue/run_queue_kaggle.py` → `src/run_queue.py` → `src/finetune.py`, `src/select_ckpt.py`,
`src/predict.py`). Their logs:
''')
code(r'''
tic('finetune')
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
for f in sorted(glob.glob('results/kaggle_v2/rr/logs/*_s0_stratified.json')):
    g = json.load(open(f)); run = g['run_id'].replace('_s0_stratified', '')
    if g.get('losses'):
        ax[0].plot([l['step'] for l in g['losses']], [l['loss'] for l in g['losses']], label=run)
    if g.get('val'):
        ax[1].plot([v['step'] for v in g['val']], [v['miou'] for v in g['val']], 'o-', label=run)
    t = g.get('timing', {})
    print(f"{run:18s} GPU {g.get('gpu')}, {t.get('train_s_per_50_steps', 0) / 50:.2f} s/step, peak {t.get('train_peak_gb')} GB,"
          + (f" best val30 mIoU {g['best']['miou']:.3f} at step {g['best']['step']}" if g.get('best') else ''))
ax[0].set(title='training loss', xlabel='step', yscale='log'); ax[1].set(title='val30 mIoU (τ 0.5)', xlabel='step')
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8); plt.tight_layout(); plt.show()
''')
code(r'''
if RUN_TRAINING:   # the full matrix, as on Kaggle (one process per run; resumable): ~12 GPU-hours on a T4
    for shot, steps, val, chunk, extra in [(1, 1000, 250, 6, ''), (5, 1000, 250, 6, ''), (10, 1000, 250, 6, ''),
                                           (20, 2000, 250, 6, ''), (50, 2000, 500, 1, '--defer_val')]:
        sh(f'{sys.executable} src/finetune.py --dataset floodnet --shot {shot} --n_steps {steps} --val_every {val} --amp fp16 '
           f'--grad_ckpt --chunk {chunk} --channel_chunk 1 {extra} --drive {BASE}/retrain {GPU_ARGS}', tail=5)
    print('then src/select_ckpt.py for the 50-shot run and src/predict.py for the caches (see kaggle/queue)')
if RUN_DEMO_TRAINING and torch.cuda.is_available():   # a short real run of the same code: 60 steps at 5 shots, validated on val30 at the end
    DEMO = os.path.join(BASE, 'demo_run')
    for d in ('splits', 'supports'):
        shutil.copytree(d, os.path.join(DEMO, d), dirs_exist_ok=True)
    t = time.time()
    sh(f'{sys.executable} src/finetune.py --dataset floodnet --shot 5 --n_steps 60 --val_every 60 --ckpt_every 60 '
       f'--log_every 10 --amp fp16 --grad_ckpt --chunk 6 --channel_chunk 1 --drive {DEMO} {GPU_ARGS}', tail=6)
    g = json.load(open(os.path.join(DEMO, 'logs', 'floodnet_5shot_s0_stratified.json')))
    print(f'demo run: {(time.time() - t) / 60:.1f} min; losses', [round(l["loss"], 3) for l in g['losses']],
          f"| val30 mIoU after 60 steps {g['val'][-1]['miou']:.3f} (the real 5-shot run reached {json.load(open('results/kaggle_v2/rr/logs/floodnet_5shot_s0_stratified.json'))['best']['miou']:.3f} after 500 steps)")
toc('finetune')
''')

md(r'''
## 5. Results (Track A)

### 5.1 Live recompute of the headline models

The prediction caches were made on Kaggle T4s by `src/predict.py`. Here the same script runs the FloodNet 50-shot
(headline) and 20-shot models again on the first test photos of eval100, and the result is compared with the cache:
the probabilities should be identical (same code, same fp16 inference, same GPU type).
''')
code(r'''
tic('live')
from rr.metrics import decode, confusion, summarize, WET
DEC = {r: {k: v for k, v in json.load(open(f'results/kaggle_v2/rr/decode/{r}.json'))['chosen'].items() if k != 'variant'}
       for r in [os.path.basename(f)[:-5] for f in glob.glob('results/kaggle_v2/rr/decode/*.json')]}
def load_cache(name):
    z = np.load(f'results/kaggle_v2/rr/preds/{name}.npz'); return dict(zip([str(i) for i in z['ids']], z['probs']))
LIVE = {}
if RUN_LIVE and torch.cuda.is_available():
    for shot, chunk in ((50, 1), (20, 6)):
        if shot == 50 and GPU_GB < 14: continue
        run = f'floodnet_{shot}shot_s0_stratified'
        t = time.time()
        sh(f'{sys.executable} src/predict.py --ckpt {WORK}/ckpt/{run}.ckpt --dataset floodnet --split_file floodnet_test_eval100 '
           f'--split test --amp fp16 --chunk {chunk} --channel_chunk 1 --limit {N_LIVE} --force --drive {WORK} {GPU_ARGS}', tail=4)
        live = load_cache(f'{run}_floodnet_test_eval100_n{N_LIVE}_fp16'); cache = load_cache(f'{run}_floodnet_test_eval100_fp16')
        d = [np.abs(live[i].astype(int) - cache[i]).max() for i in live]
        agree = np.mean([(decode(live[i] / 255., **DEC[run]) == decode(cache[i] / 255., **DEC[run])).mean() for i in live])
        LIVE[shot] = dict(minutes=round((time.time() - t) / 60, 1), max_abs_diff=int(max(d)), decode_agreement=float(agree))
        print(f'FloodNet {shot}-shot, {len(live)} photos live: max |live - cache| = {max(d)} (uint8 units), '
              f'tuned-decode pixel agreement {agree:.4f}, {LIVE[shot]["minutes"]} min')
else:
    print('RUN_LIVE is off or no GPU: the results below come from the caches only')
toc('live')
''')
md(r'''
### 5.2 Learning curve (eval100, from the caches)

mIoU over the 5 channels, at τ = 0.5 and with each run's decode tuned on val30 (`src/decode_tune.py`: per-channel τ,
optionally a building-level flooded/dry rule). 95 % bootstrap CIs over images (B = 2000, `src/bootstrap_ci.py`).
''')
code(r'''
tic('cpu_results')
# FloodNet 50-shot decode re-tuned here on val30 (the others are loaded): must give the published decode
sh(f'{sys.executable} src/decode_tune.py --runs floodnet_50shot_s0_stratified', tail=1)
print('re-tuned decode equals the shipped one:',
      {k: v for k, v in json.load(open('results/kaggle_v2/rr/decode/floodnet_50shot_s0_stratified.json'))['chosen'].items() if k != 'variant'} == DEC['floodnet_50shot_s0_stratified'])
def miou_of(cache, ds, dec, split='test'):
    cm = np.zeros((6, 6), np.int64)
    for i, p in cache.items():
        cm += confusion(decode(p / 255., **dec), gt_classes(load_label(DATA, ds, split, i)))
    return summarize(cm, NAMES[ds], WET[ds])
rows = []
for ds, shots in (('floodnet', (1, 5, 10, 20, 50)), ('rescuenet', (20, 50))):
    for s in shots:
        run = f'{ds}_{s}shot_s0_stratified'; c = load_cache(f'{run}_{ds}_test_eval100_fp16')
        a, b = miou_of(c, ds, dict(tau=0.5)), miou_of(c, ds, DEC[run])
        rows.append((ds, s, a['miou'], b['miou'], b['drivable_precision']))
sh(f'{sys.executable} src/bootstrap_ci.py', tail=0)   # recomputes every CI from the caches (seeded)
CI = json.load(open('results/kaggle_v2/report/ci/ci.json'))
print(f"{'dataset':10s} {'shots':>5s} {'mIoU τ0.5':>22s} {'mIoU tuned':>22s} {'drv P (tuned)':>14s}")
for ds, s, m0, m1, p in rows:
    c0, c1 = CI['curve'][f'{ds}/{s}/{ds}_test_eval100/tau0.5']['miou'], CI['curve'][f'{ds}/{s}/{ds}_test_eval100/tuned']['miou']
    print(f"{ds:10s} {s:5d} {m0:.3f} [{c0['lo']:.3f}, {c0['hi']:.3f}]   {m1:.3f} [{c1['lo']:.3f}, {c1['hi']:.3f}]   {p:.3f}")
print('\npaired differences (eval100):')
for k, v in CI['paired'].items():
    print(f"  {k:16s} {v['value']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
''')
code(r'''
sh(f'{sys.executable} src/collect_results.py --results results/kaggle_v2/rr --out results/kaggle_v2/report', tail=0)
display_img = lambda f, w=None: (plt.figure(figsize=(w or 10, (w or 10) * Image.open(f).height / Image.open(f).width)),
                                 plt.imshow(Image.open(f)), plt.axis('off'), plt.show())
display_img('results/kaggle_v2/report/curve_floodnet.png', 9)
''')
md(r'''
### 5.3 Headline: FloodNet 50-shot on the full 448-photo test, and the published baselines

The supervised baselines are from the FloodNet paper (Table 4: trained on all 1,445 labelled train photos, 713×713);
we compare on the same five classes. They are far ahead: this is 50 labels against 1,445.
''')
code(r'''
full = load_cache('floodnet_50shot_s0_stratified_floodnet_test_full_fp16')
r0, r1 = miou_of(full, 'floodnet', dict(tau=0.5)), miou_of(full, 'floodnet', DEC['floodnet_50shot_s0_stratified'])
c0, c1 = CI['curve']['floodnet/50/floodnet_test_full/tau0.5']['miou'], CI['curve']['floodnet/50/floodnet_test_full/tuned']['miou']
print(f'FloodNet 50-shot, full test ({len(full)} photos): mIoU {r0["miou"]:.3f} [{c0["lo"]:.3f}, {c0["hi"]:.3f}] at τ 0.5, '
      f'{r1["miou"]:.3f} [{c1["lo"]:.3f}, {c1["hi"]:.3f}] tuned; drivable precision {r1["drivable_precision"]:.3f}')
print('per class (tuned):', {n: round(r1[f'iou_{n}'], 3) for n in NAMES['floodnet']})
sh(f'{sys.executable} src/baseline_table.py --results results/kaggle_v2/rr --out results/kaggle_v2/report/baselines.md', tail=0)
print(open('results/kaggle_v2/report/baselines.md').read())
print('Near-duplicates (src/dupes.py): test photos showing the same ground as a support photo:',
      {k: len(v) for k, v in json.load(open('results/kaggle_v2/report/dupes/dupes.json'))['runs']['floodnet']['overlapping'].items()},
      'of 448; without them:', {k: round(v['miou']['value'], 3) for k, v in CI['without_overlap'].items() if 'floodnet/50' in k})
print('\nTest-time flips (4 views, src/tta.py): not adopted. ' + open('results/kaggle_v2/report/tta/tta.md').read().split('## Decision')[1].strip())
toc('cpu_results')
''')

md(r'''
## 6. Failure analysis

Where the 50-shot model goes wrong (`src/failures.py`, `src/figures.py`, eval100): the dangerous error is shallow or
muddy water read as dry land (a truck would be sent into it); the ground truth itself is ambiguous at bridges and
under trees. The second figure shows good and bad cases side by side.
''')
code(r'''
w = json.load(open('results/kaggle_v2/report/failures/worst.json'))
print(json.dumps(w, indent=1)[:1500])
display_img('results/kaggle_v2/report/figures/failure_gallery.png', 14)
display_img('results/kaggle_v2/report/figures/qualitative.png', 14)
''')

md(r'''
## 7. Cross-disaster: RescueNet

Can the FloodNet model be used on another hurricane without labels? Zero-label transfer is scored on the classes both
datasets share (road, building, water); re-adaptation fine-tunes on 20 or 50 RescueNet labels (`src/transfer_eval.py`).
''')
code(r'''
tic('transfer')
sh(f'{sys.executable} src/transfer_eval.py', tail=0)
print(open('results/kaggle_v2/report/transfer.md').read())
toc('transfer')
''')

md(r'''
## 8. Routing (Track B)

`src/router.py` works on 4×4-pixel cells (192×144 per photo). States are (cell, mode) with modes **truck → boat → on
foot**, never back into a vehicle. A truck drives on clear road, or on dry land at 4× cost; on predicted maps only where
the water probability p_wet = p(flooded road) + p(water) is below a gate (0.3 FloodNet, 0.2 RescueNet, each tuned on
val30); the risk-aware cost adds 5·p_wet per cell. A boat needs water or a flooded road; launching costs 40. The last
stretch can be walked (8 per cell). Vehicles have a width (truck 2.5 m, boat 1.6 m; ground scale from the labelled
cars), parked cars do not block a road, a boat can pass under tree canopy, and no vehicle squeezes diagonally between
two obstacles. Each house gets a status: TRUCK, BOAT, FOOT or ISOLATED (no route: air team).

The **labelled map** is the same router on the human labels: the datasets label pixels, not reachability. On predicted
maps we score the houses that are **cut off by road** (status ≠ TRUCK) against the labelled map. Each truck-reachable
house also gets a **confidence tier**: routed again with a cautious gate (0.1 / 0.15), a house still reachable is
"confident", otherwise "check first".
''')
code(r'''
tic('routing')
def route(ds, split_file, preds, decode_json, out):
    sh(f'{sys.executable} src/route_eval.py --dataset {ds} --split_file {split_file} --preds {preds} --decode {decode_json} '
       f'--lams 5 --out {out}', tail=0)
    return json.load(open(out))['routing']['lam5']
os.makedirs(f'{BASE}/routing_live', exist_ok=True)
for name, ds, run, sf, cache in [('FloodNet 50, eval100', 'floodnet', 'floodnet_50shot_s0_stratified', 'floodnet_test_eval100', 'floodnet_test_eval100_fp16'),
                                 ('FloodNet 50, full test', 'floodnet', 'floodnet_50shot_s0_stratified', 'floodnet_test_full', 'floodnet_test_full_fp16'),
                                 ('RescueNet 20, eval100', 'rescuenet', 'rescuenet_20shot_s0_stratified', 'rescuenet_test_eval100', 'rescuenet_test_eval100_fp16')]:
    dj = f'{BASE}/routing_live/decode_{run}.json'; json.dump(DEC[run], open(dj, 'w'))
    r = route(ds, sf, f'results/kaggle_v2/rr/preds/{run}_{cache}.npz', dj, f'{BASE}/routing_live/{sf}.json')
    ci = CI['routing'].get({'FloodNet 50, eval100': 'FloodNet 50-shot eval100', 'FloodNet 50, full test': 'FloodNet 50-shot full test',
                            'RescueNet 20, eval100': 'RescueNet 20-shot eval100'}[name], {}).get('all')
    print(f"{name:24s} houses {r['n_buildings']:5d} | cut off by road: P {r['stranded_precision']:.3f} R {r['stranded_recall']:.3f} "
          f"F1 {r['stranded_f1']:.3f}" + (f" [{ci['lo']:.3f}, {ci['hi']:.3f}]" if ci else '')
          + f" | unsafe routes {r['unsafe_route_rate']:.3f}")
toc('routing')
''')
code(r'''
for f, title in [('results/routing/tiers/tiers.md', 'Confidence tiers'), ('results/routing/gate/gate.md', 'Off-road gate (FloodNet 50)'),
                 ('results/routing/physical/ablation.md', 'Physical rules'), ('results/routing/sensitivity/sensitivity.md', 'Sensitivity')]:
    print(f'===== {title} ({f})'); print(open(f).read()[:3500])
''')
code(r'''
# one scene end to end: RescueNet 12195, labelled map, house 8: truck -> boat across the canal -> on foot
import types, gradio_app as G
a = types.SimpleNamespace(results=os.path.join(RR, 'results/kaggle_v2/rr'), data=DATA)
for src in ('pred', 'gt'):
    st = G.cached_state('rescuenet:12195', a, src=src)
    st['sel'] = (8, *G.route_for(st, 8))
    print(f'--- {G.SIDE[src]}:', G.counter_md(st).replace('**', '').replace('  \n', ' | '))
    print(G.route_md(st, 8).replace('**', '').replace('  \n', '\n'))
plt.figure(figsize=(12, 9)); plt.imshow(G.render(st)); plt.axis('off'); plt.title('RescueNet 12195, labelled map, house 8'); plt.show()
''')

md(r'''
## 9. App

**Web viewer** (`docs/` in the app zip; plain HTML/JS + Leaflet, no server): all 200 eval100 photos, prediction vs
labelled map, house statuses and tiers, animated routes computed in the browser by a JavaScript port of the router
(bit-exact with Python on all 200 scenes), minutes and metres, water-probability layer, shareable links.

**Gradio app** (`src/gradio_app.py`): the same from Python, and **live inference on an uploaded photo**. Here an upload
goes through exactly what the app does: the FloodNet 50-shot model (`LiveModel`), then the router on the prediction
(an upload has no labels, so the houses and the entry point come from the prediction).
''')
code(r'''
tic('app')
if RUN_LIVE and torch.cuda.is_available() and GPU_GB >= 14:
    la = types.SimpleNamespace(ckpt=f'{WORK}/ckpt/floodnet_50shot_s0_stratified.ckpt', mt_ckpt=META_CKPT, chameleon=CHAMELEON,
                               data=DATA, runs=f'{BASE}/runs', support_file=os.path.join(RR, 'supports/floodnet_stratified_s0.txt'),
                               amp='fp16', chunk=1, channel_chunk=1, results=a.results)
    live = G.LiveModel(la)
    os.chdir(RR)                                           # loading Chameleon in-process changes the working directory
    photo_file = os.path.join(RR, 'demo/uploads/floodnet_7307.jpg')   # a FloodNet test photo outside eval100
    t = time.time(); notes = []
    photo = G.prep_upload(np.asarray(Image.open(photo_file).convert('RGB')), notes)
    s = G.build_scene('floodnet', photo, live(photo), DEC['floodnet_50shot_s0_stratified'], gt=None)
    s.update(tier_stats=G.tier_stats('floodnet'))
    st = G.new_state(s, 'upload')
    print(f'upload {photo_file}: {time.time() - t:.1f} s (inference + routing)', notes)
    print(G.counter_md(st).replace('**', '').replace('  \n', ' | '))
    plt.figure(figsize=(12, 9)); plt.imshow(G.render(st)); plt.axis('off'); plt.title('uploaded photo, live'); plt.show()
    if RUN_APP:
        sh(f'{sys.executable} -m pip install -q gradio==6.29.1', tail=1)
        G.build_ui(a, live).launch(share=True)
    else:
        del live; torch.cuda.empty_cache()
else:
    print('RUN_LIVE is off or the GPU is too small: see the app zip for the viewer and the Gradio app')
toc('app')
''')
code(r'''
if RUN_FULL_REBUILD:   # every report from the caches, then the 200 viewer scenes and tests (CPU)
    sh(f'{sys.executable} src/rebuild_all.py -j {os.cpu_count()} --skip export,viewer,tests', tail=25)
''')

md(r'''
## 10. Reproducibility

- **Pretrained weights:** Chameleon meta-trained generalist checkpoint from the course
  (<https://github.com/GitGyun/AIC200-Project-StructuredPrediction>, shared Drive folder; model: Kim et al., *Chameleon:
  A Data-Efficient Generalist for Dense Visual Prediction in the Wild*, ECCV 2024, <https://github.com/GitGyun/chameleon>;
  its image encoder is BEiT-v2-Large). Our 7 fine-tuned checkpoints (adapted parameters only, ~5 MB each) are in the
  data bundle (`results/kaggle_v2/rr/ckpt/`).
- **Data:** FloodNet <https://github.com/BinaLab/FloodNet-Supervised_v1.0> (CDLA-Permissive-1.0); RescueNet
  <https://springernature.figshare.com/collections/6647354> (CC0). Preprocessing: `src/prep_data.py`.
- **Everything else:** `src/rebuild_all.py` recomputes every reported number from the caches on CPU and checks the
  headline ones; `PLAN.md` §15 (in the repository) logs every run and decision.
''')
code(r'''
import importlib.metadata as im
print('code commit', COMMIT, '| bundle', BUNDLE['created'], '| GPU', GPU)
print({p: im.version(p) for p in ('torch', 'numpy', 'scipy', 'scikit-image', 'pytorch_lightning', 'timm') if p != 'torch'} | {'torch': torch.__version__})
print('live check:', LIVE)
TIMES['total'] = round(time.time() - T0, 1)
print('runtime (s):', {k: v for k, v in TIMES.items() if isinstance(v, float)})
print(f'total {TIMES["total"] / 60:.1f} min')
''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rescue_router.ipynb'))
    a = ap.parse_args()
    cells = []
    for kind, src in CELLS:
        lines = src.split('\n')
        c = dict(cell_type=kind, metadata={}, source=[l + '\n' for l in lines[:-1]] + [lines[-1]])
        if kind == 'code':
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
              metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                            language_info=dict(name='python'), accelerator='GPU', colab=dict(gpuType='T4')))
    json.dump(nb, open(a.out, 'w'), indent=1)
    print(f'{a.out}: {len(cells)} cells')


if __name__ == '__main__':
    main()
