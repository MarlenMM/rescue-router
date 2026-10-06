"""Build the data bundle the final notebook runs from (PLAN §2.6 data zip, §7 notebook). CPU only.

    python src/make_bundle.py --out <dir> [--zip <file.zip>] [--data ~/rr_data]

The bundle mirrors the repository, so every script runs from its root with the default paths:
  src/ splits/ supports/ demo/ kaggle/live/ README.md requirements.txt VERSION   code at git HEAD (+ commit hash)
  rr_data/floodnet_r768/train/{images,labels}            the 50 support photos (live inference, demo fine-tuning)
  rr_data/floodnet_r768/val/{images,labels,labels_raw}   val30 (decode tuning, demo validation)
  rr_data/floodnet_r768/test/{labels,labels_raw}         all 448 test labels; images: eval100 only
  rr_data/rescuenet_r768/{val,test}/{labels,labels_raw}  RescueNet val30 + eval100 labels (photos: the 2 showcase scenes)
  results/kaggle_v2/rr/{ckpt,decode,metrics,logs}        the 7 fine-tuned checkpoints, decodes, metrics, training logs
  results/kaggle_v2/rr_packed/preds/*.ptar               the 16 prediction caches, packed losslessly (src/cache_pack.py;
                                                         `python src/cache_pack.py unpack --src results/kaggle_v2/rr_packed`)
  results/kaggle_v2/report/, results/routing/            every report (JSON, Markdown, figures) as published
  BUNDLE.json, README_data.md, LICENSES.md
The meta-trained checkpoint (generalist.ckpt, 2.8 GB) is not included: it is the course's, on its shared Drive folder.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cache_pack  # noqa: E402

RUNS = ['floodnet_1shot', 'floodnet_5shot', 'floodnet_10shot', 'floodnet_20shot', 'floodnet_50shot',
        'rescuenet_20shot', 'rescuenet_50shot']


def ids(name):
    return [l.strip() for l in open(os.path.join(ROOT, 'splits' if not name.startswith('supports/') else '',
                                                  name if name.startswith('supports/') else f'{name}.txt')) if l.strip()]


def copy_split(data, out, ds, split, id_list, parts):
    for part in parts:
        os.makedirs(os.path.join(out, 'rr_data', f'{ds}_r768', split, part), exist_ok=True)
        ext = 'jpg' if part == 'images' else 'png'
        for i in id_list:
            shutil.copy2(os.path.join(data, f'{ds}_r768', split, part, f'{i}.{ext}'),
                         os.path.join(out, 'rr_data', f'{ds}_r768', split, part, f'{i}.{ext}'))


def readme_data(out, sizes):
    open(os.path.join(out, 'README_data.md'), 'w').write(f'''# Rescue Router: data bundle

Unzip, then from the unzipped folder: `python src/cache_pack.py unpack --src results/kaggle_v2/rr_packed`
(restores the prediction caches bit for bit), and run the notebook or `python src/rebuild_all.py`.

| part | content | size |
|---|---|---|
''' + '\n'.join(f'| `{k}` | {v[0]} | {v[1] / 1e6:.0f} MB |' for k, v in sizes.items()) + '''

Classes (5 channels + "other"): FloodNet road clear / road flooded / building flooded / building dry / water (from the
10 FloodNet classes: road-non-flooded, road-flooded, building-flooded, building-non-flooded, water; trees, grass,
vehicles, pools and background are "other"; `labels_raw` keeps the original ids). RescueNet uses the same slots for
road clear / road blocked / building intact (no + minor damage) / building damaged (major + total destruction) / water.
Photos are 768x576 (area-downsampled from 4000x3000, JPEG q95); labels nearest-neighbour. Splits and supports are
listed in `splits/` and `supports/` with SHA-256 sums. Sources: FloodNet https://github.com/BinaLab/FloodNet-Supervised_v1.0 ;
RescueNet https://springernature.figshare.com/collections/6647354 . See LICENSES.md.
''')
    open(os.path.join(out, 'LICENSES.md'), 'w').write('''# Licences

- **FloodNet** (Rahnemoonfar et al., IEEE Access 2021): Community Data License Agreement – Permissive 1.0
  (CDLA-Permissive-1.0), https://github.com/BinaLab/FloodNet-Supervised_v1.0 . The FloodNet photos and labels here are
  downsampled copies; attribution: "FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding".
- **RescueNet** (Rahnemoonfar, Chowdhury, Murphy, Scientific Data 2023): released on figshare under CC0
  (https://springernature.figshare.com/collections/6647354); the GitHub release states CC BY-NC-ND 4.0. Only labels are
  included here, unchanged except for downsampling, for non-commercial academic use with attribution.
- **Chameleon** (Kim et al., ECCV 2024): https://github.com/GitGyun/chameleon ; the meta-trained checkpoint is not
  included (course Drive). Our fine-tuned checkpoints contain only the adapted parameters.
- Our code and results: course project, 20260876.
''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--zip', default=None)
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--results', default=os.path.join(ROOT, 'results/kaggle_v2/rr'))
    a = ap.parse_args()
    out = a.out
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    # code at HEAD
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', 'src', 'splits', 'supports', 'demo'], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print('warning: uncommitted changes in code are NOT in the bundle (git HEAD):\n' + dirty)
    arc = subprocess.run(['git', 'archive', 'HEAD', 'src', 'splits', 'supports', 'demo', 'kaggle/live', 'README.md',
                          'requirements.txt'], cwd=ROOT, capture_output=True, check=True).stdout
    subprocess.run(['tar', '-x', '-C', out], input=arc, check=True)
    open(os.path.join(out, 'VERSION'), 'w').write(commit + '\n')
    # data
    sup = ids('supports/floodnet_stratified_s0.txt')[:50]
    copy_split(a.data, out, 'floodnet', 'train', sup, ['images', 'labels'])
    copy_split(a.data, out, 'floodnet', 'val', ids('floodnet_val_tune30'), ['images', 'labels', 'labels_raw'])
    copy_split(a.data, out, 'floodnet', 'test', ids('floodnet_test_full'), ['labels', 'labels_raw'])
    copy_split(a.data, out, 'floodnet', 'test', ids('floodnet_test_eval100'), ['images'])
    copy_split(a.data, out, 'rescuenet', 'val', ids('rescuenet_val_tune30'), ['labels', 'labels_raw'])
    copy_split(a.data, out, 'rescuenet', 'test', ids('rescuenet_test_eval100'), ['labels', 'labels_raw'])
    copy_split(a.data, out, 'rescuenet', 'test', ['12195', '15833'], ['images'])     # the two showcase scenes
    # results
    rr = os.path.join(out, 'results/kaggle_v2/rr')
    os.makedirs(os.path.join(rr, 'ckpt'))
    for r in RUNS:
        shutil.copy2(os.path.join(a.results, 'ckpt', f'{r}_s0_stratified.ckpt'), os.path.join(rr, 'ckpt'))
    for sub in ('decode', 'metrics', 'logs'):
        shutil.copytree(os.path.join(a.results, sub), os.path.join(rr, sub))
    for f in glob.glob(os.path.join(rr, 'decode', '*_tta4.json')):          # TTA not adopted: its report has it
        os.remove(f)
    shutil.copytree(os.path.join(ROOT, 'results/kaggle_v2/report'), os.path.join(out, 'results/kaggle_v2/report'))
    shutil.copytree(os.path.join(ROOT, 'results/routing'), os.path.join(out, 'results/routing'))
    os.makedirs(os.path.join(out, 'results/kaggle_v2/rr_packed/preds'))
    t = time.time()
    jobs = [(os.path.join(a.results, 'preds', f'{n}.npz'), os.path.join(out, 'results/kaggle_v2/rr_packed/preds', f'{n}.ptar'))
            for n in cache_pack.PACK]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(10) as ex:
        list(ex.map(cache_pack.pack_one, jobs))
    print(f'caches packed in {time.time() - t:.0f} s')

    def size(p):
        p = os.path.join(out, p)
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) \
            else os.path.getsize(p)
    sizes = {'results/kaggle_v2/rr_packed/': ('16 prediction caches (lossless)', size('results/kaggle_v2/rr_packed')),
             'results/kaggle_v2/rr/ckpt/': ('7 fine-tuned checkpoints', size('results/kaggle_v2/rr/ckpt')),
             'rr_data/': ('photos (supports, val30, FloodNet eval100) + labels', size('rr_data')),
             'results/ (reports)': ('report and routing JSON/Markdown/figures',
                                    size('results/kaggle_v2/report') + size('results/routing')),
             'code': ('src, splits, supports, demo photos, kaggle/live', sum(size(p) for p in ('src', 'splits', 'supports', 'demo', 'kaggle')))}
    readme_data(out, sizes)
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    json.dump(dict(name='rescue-router data bundle', commit=commit, created=time.strftime('%Y-%m-%d %H:%M %Z'),
                   total_bytes=total, parts={k: v[1] for k, v in sizes.items()}, packed_caches=cache_pack.PACK),
              open(os.path.join(out, 'BUNDLE.json'), 'w'), indent=1)
    print(f'bundle {out}: {total / 1e6:.0f} MB', {k: f'{v[1] / 1e6:.0f} MB' for k, v in sizes.items()})
    if a.zip:
        with zipfile.ZipFile(a.zip, 'w', zipfile.ZIP_STORED) as z:     # contents are compressed already
            for d, _, fs in os.walk(out):
                for f in sorted(fs):
                    p = os.path.join(d, f)
                    z.write(p, os.path.relpath(p, out))
        print(f'zip {a.zip}: {os.path.getsize(a.zip) / 1e6:.0f} MB')


if __name__ == '__main__':
    main()
