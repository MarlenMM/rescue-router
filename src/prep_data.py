"""Rescue Router data preparation (PLAN.md §2.2-2.4). CPU only, run once on Colab.

Phases (each is a function; `main` runs them in order):
  survey    per-pair sizes, EXIF orientation, raw-class pixel fractions -> stats_*.csv
  align     image/label alignment check (size mismatches, EXIF-rotated images) -> decisions + overlays
  splits    fixed evaluation splits + stratified, nested support orders (seeds 0-2)
  export    768x576 (all) and 1024x768 (FloodNet subset) images, channel labels, raw labels -> tar files

Label convention (PLAN §3): uint8 PNG, value = harmonised channel id 0-4, 254 = other, 255 = ignore.
Raw labels (original class ids) are kept next to them in labels_raw/ for statistics and ablations.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import tarfile
import time
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)

# ------------------------------------------------------------------------------------------- spec
FLOOD_CLASSES = ['background', 'building_flooded', 'building_non_flooded', 'road_flooded',
                 'road_non_flooded', 'water', 'tree', 'vehicle', 'pool', 'grass']
RESC_CLASSES = ['background', 'water', 'building_no_damage', 'building_minor_damage',
                'building_major_damage', 'building_total_destruction', 'vehicle', 'road_clear',
                'road_blocked', 'tree', 'pool']
FLOOD2CH = {4: 0, 3: 1, 1: 2, 2: 3, 5: 4}
RESC2CH = {7: 0, 8: 1, 2: 2, 3: 2, 4: 3, 5: 3, 1: 4}
CHANNELS = {'floodnet': ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water'],
            'rescuenet': ['road_clear', 'road_blocked', 'bldg_intact', 'bldg_damaged', 'water']}
OTHER, IGNORE = 254, 255
SIZES = {'r768': (768, 576), 'r1024': (1024, 768)}   # (W, H)
PRESENT = 1e-3            # a class is "present" in an image if it covers >= 0.1% of its pixels
SHOT_CHECKPOINTS = (5, 10, 20, 50, 100)
N_SUPPORT = 100


def lut_for(mapping):
    lut = np.full(256, OTHER, np.uint8)
    for src, ch in mapping.items():
        lut[src] = ch
    lut[IGNORE] = IGNORE
    return lut


def dataset_spec(name, raw_root):
    """Where each split's images/labels live, and the class/channel definitions."""
    if name == 'floodnet':
        splits = {s: (f'{raw_root}/{s}/{s}-org-img', f'{raw_root}/{s}/{s}-label-img')
                  for s in ('train', 'val', 'test')}
        return dict(splits=splits, n_classes=10, classes=FLOOD_CLASSES, lut=lut_for(FLOOD2CH))
    if name == 'rescuenet':
        splits = {s: (f'{raw_root}/{s}-org-img', f'{raw_root}/{s}-label-img') for s in ('val', 'test')}
        return dict(splits=splits, n_classes=11, classes=RESC_CLASSES, lut=lut_for(RESC2CH))
    raise KeyError(name)


def list_pairs(img_dir, lab_dir):
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith('.jpg'):
            continue
        stem = os.path.splitext(f)[0]
        lab = os.path.join(lab_dir, f'{stem}_lab.png')
        if not os.path.exists(lab):
            raise FileNotFoundError(f'no label for {f}: {lab}')
        pairs.append((stem, os.path.join(img_dir, f), lab))
    n_lab = len([f for f in os.listdir(lab_dir) if f.endswith('.png')])
    assert n_lab == len(pairs), f'{lab_dir}: {n_lab} labels vs {len(pairs)} images'
    return pairs


# ------------------------------------------------------------------------------------- 1. survey
def _survey_one(args):
    ds, split, stem, img_path, lab_path, n_classes = args
    im = Image.open(img_path)
    orient = im.getexif().get(274, 1) or 1
    lab = np.array(Image.open(lab_path))
    if lab.ndim != 2:
        raise ValueError(f'{lab_path}: label is not single-channel ({lab.shape})')
    counts = np.bincount(lab.ravel(), minlength=256)
    unknown = int(counts[n_classes:].sum())
    fr = counts[:n_classes] / lab.size
    row = dict(dataset=ds, split=split, id=stem, img_w=im.size[0], img_h=im.size[1],
               lab_w=lab.shape[1], lab_h=lab.shape[0], exif_orientation=orient,
               img_bytes=os.path.getsize(img_path), unknown_label_px=unknown)
    row.update({f'frac_{c}': round(float(v), 6) for c, v in enumerate(fr)})
    return row


def survey(ds, spec, workers=2):
    jobs = [(ds, split, stem, ip, lp, spec['n_classes'])
            for split, (idir, ldir) in spec['splits'].items() for stem, ip, lp in list_pairs(idir, ldir)]
    with Pool(workers) as pool:
        rows = pool.map(_survey_one, jobs, chunksize=8)
    return rows


# -------------------------------------------------------------------------------------- 2. align
def _edge_score(gray, lab):
    """Mean image gradient on label boundaries / mean gradient overall (higher = better aligned)."""
    g = gray.astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    b = np.zeros(lab.shape, bool)
    b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b[1:, :] |= lab[1:, :] != lab[:-1, :]
    return float(mag[b].mean() / (mag.mean() + 1e-6)) if b.any() else float('nan')


def _read_gray_small(path, scale=8):
    flag = {2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
            8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[scale]
    return cv2.imread(path, flag | cv2.IMREAD_IGNORE_ORIENTATION)


def _overlay(path_img, lab, out_path, rotate180=False, img_resize_to=None):
    img = cv2.imread(path_img, cv2.IMREAD_REDUCED_COLOR_4 | cv2.IMREAD_IGNORE_ORIENTATION)
    if rotate180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.resize(img, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_AREA)
    edges = np.zeros(lab.shape, bool)
    edges[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edges[1:, :] |= lab[1:, :] != lab[:-1, :]
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    img[edges] = (0, 0, 255)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _align_one(args):
    stem, img_path, lab_path, kind = args
    lab_full = np.array(Image.open(lab_path))
    gray = _read_gray_small(img_path, 8)
    h, w = gray.shape
    lab = cv2.resize(lab_full, (w, h), interpolation=cv2.INTER_NEAREST)
    out = dict(id=stem, kind=kind, score_raw=_edge_score(gray, lab),
               score_rot180=_edge_score(cv2.rotate(gray, cv2.ROTATE_180), lab))
    return out


def align(ds, spec, rows, meta_dir, workers=2, n_overlay=5, n_reference=30, seed=0):
    """Decide how EXIF-rotated and size-mismatched pairs are aligned, from edge agreement."""
    os.makedirs(os.path.join(meta_dir, 'alignment'), exist_ok=True)
    paths = {(r['split'], r['id']): None for r in rows}
    for split, (idir, ldir) in spec['splits'].items():
        for stem, ip, lp in list_pairs(idir, ldir):
            paths[(split, stem)] = (ip, lp)

    mismatch = [r for r in rows if (r['img_w'], r['img_h']) != (r['lab_w'], r['lab_h'])]
    rotated = [r for r in rows if r['exif_orientation'] != 1]
    normal = [r for r in rows if r['exif_orientation'] == 1 and r not in mismatch]
    rng = np.random.default_rng(seed)
    ref = [normal[i] for i in rng.choice(len(normal), min(n_reference, len(normal)), replace=False)]

    jobs = [(r['id'], *paths[(r['split'], r['id'])], 'exif') for r in rotated]
    jobs += [(r['id'], *paths[(r['split'], r['id'])], 'reference') for r in ref]
    with Pool(workers) as pool:
        res = pool.map(_align_one, jobs)
    ref_scores = [x['score_raw'] for x in res if x['kind'] == 'reference']
    exif = [x for x in res if x['kind'] == 'exif']
    n_raw_better = sum(x['score_raw'] > x['score_rot180'] for x in exif)
    decision = {'exif_rule': 'ignore_exif' if n_raw_better >= len(exif) / 2 else 'apply_exif_rot180',
                'exif_n': len(exif), 'exif_raw_better': n_raw_better,
                'reference_score_median': float(np.median(ref_scores)) if ref_scores else None,
                'size_mismatch_n': len(mismatch), 'size_rule': 'resize_image_to_label'}

    # size mismatches (none expected in this release): label is authoritative, image resized to it
    mm_rows = []
    for r in mismatch:
        mm_rows.append(dict(dataset=ds, split=r['split'], id=r['id'], img_w=r['img_w'], img_h=r['img_h'],
                            lab_w=r['lab_w'], lab_h=r['lab_h'], action='resize_image_to_label'))
    # overlays: 5 EXIF-rotated pairs (both readings) and up to 5 mismatched pairs
    for x in exif[:n_overlay]:
        split = next(r['split'] for r in rotated if r['id'] == x['id'])
        ip, lp = paths[(split, x['id'])]
        lab = np.array(Image.open(lp))
        lab = cv2.resize(lab, (lab.shape[1] // 4, lab.shape[0] // 4), interpolation=cv2.INTER_NEAREST)
        for rot in (False, True):
            _overlay(ip, lab, os.path.join(meta_dir, 'alignment',
                                           f'{ds}_{x["id"]}_exif_{"rot180" if rot else "raw"}.jpg'), rot)
    for r in mismatch[:n_overlay]:
        ip, lp = paths[(r['split'], r['id'])]
        lab = np.array(Image.open(lp))
        lab = cv2.resize(lab, (lab.shape[1] // 4, lab.shape[0] // 4), interpolation=cv2.INTER_NEAREST)
        _overlay(ip, lab, os.path.join(meta_dir, 'alignment', f'{ds}_{r["id"]}_mismatch_resize.jpg'))
    return decision, res, mm_rows


# ------------------------------------------------------------------------------------- 3. splits
def _present(r, c):
    return r[f'frac_{c}'] >= PRESENT


def flags(ds, r):
    """Per-image booleans used for stratification (raw class ids)."""
    if ds == 'floodnet':
        target = _present(r, 1) or _present(r, 3)                    # flooded building / flooded road
        water = _present(r, 5)
        one_shot = _present(r, 4) and _present(r, 3) and (_present(r, 1) or _present(r, 2))
    else:
        target = _present(r, 8) or _present(r, 4) or _present(r, 5)  # blocked road / damaged building
        water = _present(r, 1)
        one_shot = _present(r, 7) and _present(r, 8) and any(_present(r, c) for c in (2, 3, 4, 5))
    return target, water, one_shot


def stratified_subset(ds, rows, n, seed):
    """n images, half with a target class (flooded road/building), half without; sorted ids."""
    rng = np.random.default_rng(seed)
    pos = sorted(r['id'] for r in rows if flags(ds, r)[0])
    neg = sorted(r['id'] for r in rows if not flags(ds, r)[0])
    n_pos = min(n // 2, len(pos))
    pick = list(rng.choice(pos, n_pos, replace=False)) + list(rng.choice(neg, n - n_pos, replace=False))
    return sorted(pick, key=int)


def support_order(ds, rows, seed, n=N_SUPPORT):
    """Nested support order: position 0 satisfies the 1-shot rule; targets and non-targets
    alternate (T, N, T, N, ...), so every prefix of length k has ceil(k/2) target images; then a
    prefix water check is repaired by swapping in the nearest later water image of the same kind."""
    rng = np.random.default_rng(seed)
    info = {r['id']: flags(ds, r) for r in rows}
    eligible = sorted(i for i, f in info.items() if f[2])
    first = str(rng.choice(eligible))
    T = [i for i in rng.permutation(sorted(i for i, f in info.items() if f[0])) if i != first]
    N = list(rng.permutation(sorted(i for i, f in info.items() if not f[0])))
    order = [first]
    while len(order) < n:
        order.append((N if len(order) % 2 == 1 else T).pop(0))
    tail = T + N                                       # unused images, for repairs
    for k in SHOT_CHECKPOINTS:
        if k > n or any(info[i][1] for i in order[:k]):
            continue
        # replace the last image of the prefix with a water image of the same target status
        last = order[k - 1]
        cand = next(j for j in tail if info[j][1] and info[j][0] == info[last][0])
        tail.remove(cand)
        order[k - 1] = cand
    check_support(ds, info, order)
    return [str(i) for i in order]


def check_support(ds, info, order):
    assert len(set(order)) == len(order)
    assert info[order[0]][2], '1-shot image must contain road-clear, road-flooded/blocked and a building'
    for k in SHOT_CHECKPOINTS:
        if k > len(order):
            continue
        pre = order[:k]
        assert sum(info[i][0] for i in pre) >= math.ceil(k / 2), (k, 'targets')
        assert any(info[i][1] for i in pre), (k, 'water')


def make_splits(ds, rows, out_root):
    by = lambda s: [r for r in rows if r['split'] == s]
    sp, su = os.path.join(out_root, 'splits'), os.path.join(out_root, 'supports')
    os.makedirs(sp, exist_ok=True)
    os.makedirs(su, exist_ok=True)
    files = {}
    if ds == 'floodnet':
        test, val, train = by('test'), by('val'), by('train')
        files['splits/floodnet_test_full.txt'] = sorted((r['id'] for r in test), key=int)
        files['splits/floodnet_test_eval100.txt'] = stratified_subset(ds, test, 100, 0)
        files['splits/floodnet_val_tune30.txt'] = stratified_subset(ds, val, 30, 0)
        files['splits/floodnet_support_pool.txt'] = sorted((r['id'] for r in train), key=int)
        pool_rows = train
    else:
        test, val = by('test'), by('val')
        files['splits/rescuenet_test_full.txt'] = sorted((r['id'] for r in test), key=int)
        files['splits/rescuenet_test_eval100.txt'] = stratified_subset(ds, test, 100, 0)
        files['splits/rescuenet_support_pool.txt'] = sorted((r['id'] for r in val), key=int)
        pool_rows = val
    for seed in (0, 1, 2):
        files[f'supports/{ds}_stratified_s{seed}.txt'] = support_order(ds, pool_rows, seed)
    for rel, ids in files.items():
        with open(os.path.join(out_root, rel), 'w') as f:
            f.write('\n'.join(ids) + '\n')
    return files


def sha256sums(out_root, rels):
    lines = []
    for rel in sorted(rels):
        h = hashlib.sha256(open(os.path.join(out_root, rel), 'rb').read()).hexdigest()
        lines.append(f'{h}  {rel}')
    with open(os.path.join(out_root, 'splits', 'SHA256SUMS'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return lines


# ------------------------------------------------------------------------------------- 4. export
def _export_one(args):
    img_path, lab_path, outs, rot180, lut = args
    img = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_2 | cv2.IMREAD_IGNORE_ORIENTATION)
    lab = np.array(Image.open(lab_path))
    if rot180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    for (w, h), img_out, lab_out, raw_out in outs:
        # label is authoritative: the image is mapped onto the label's frame (direct resize = resize
        # to label size then downsample); area for images, nearest for labels
        im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        lb = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(img_out, im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(raw_out, lb)
        cv2.imwrite(lab_out, lut[lb])
    return img_path


def export(ds, spec, rows, decision, work_root, subset_1024=None, workers=2):
    """Write <work_root>/<ds>_<res>/<split>/{images,labels,labels_raw}/<id>.{jpg,png}."""
    rot = {r['id'] for r in rows if r['exif_orientation'] != 1} \
        if decision['exif_rule'] == 'apply_exif_rot180' else set()
    jobs = []
    for split, (idir, ldir) in spec['splits'].items():
        for stem, ip, lp in list_pairs(idir, ldir):
            outs = []
            for res, (w, h) in SIZES.items():
                if res == 'r1024' and (subset_1024 is None or stem not in subset_1024):
                    continue
                base = os.path.join(work_root, f'{ds}_{res}', split)
                for d in ('images', 'labels', 'labels_raw'):
                    os.makedirs(os.path.join(base, d), exist_ok=True)
                outs.append(((w, h), f'{base}/images/{stem}.jpg', f'{base}/labels/{stem}.png',
                             f'{base}/labels_raw/{stem}.png'))
            jobs.append((ip, lp, outs, stem in rot, spec['lut']))
    t0 = time.time()
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_export_one, jobs, chunksize=4)):
            if (i + 1) % 500 == 0:
                print(f'  {ds}: {i + 1}/{len(jobs)} exported ({time.time() - t0:.0f}s)', flush=True)
    return len(jobs)


def tar_dir(src_dir, tar_path):
    with tarfile.open(tar_path, 'w') as tf:
        tf.add(src_dir, arcname=os.path.basename(src_dir))
    return os.path.getsize(tar_path)


def write_csv(path, rows):
    if not rows:
        open(path, 'w').write('')
        return
    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--floodnet_raw', default='/content/floodnet_raw/FloodNet-Supervised_v1.0')
    ap.add_argument('--rescuenet_raw', default='/content/rescuenet_raw')
    ap.add_argument('--work', default='/content/prep')
    ap.add_argument('--out', required=True, help='Drive folder, e.g. MyDrive/AIC200-my_work/rescue_router')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    meta = os.path.join(args.out, 'meta')
    os.makedirs(meta, exist_ok=True)
    os.makedirs(os.path.join(args.out, 'data'), exist_ok=True)
    log = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'present_threshold': PRESENT,
           'sizes': SIZES, 'channels': CHANNELS, 'floodnet_map': FLOOD2CH, 'rescuenet_map': RESC2CH}
    all_mm, all_split_files = [], []
    for ds, raw in (('floodnet', args.floodnet_raw), ('rescuenet', args.rescuenet_raw)):
        spec = dataset_spec(ds, raw)
        t0 = time.time()
        rows = survey(ds, spec, args.workers)
        write_csv(os.path.join(meta, f'stats_{ds}.csv'), rows)
        print(f'[{ds}] survey: {len(rows)} pairs ({time.time() - t0:.0f}s)', flush=True)

        decision, align_res, mm = align(ds, spec, rows, meta, args.workers)
        write_csv(os.path.join(meta, f'alignment_{ds}.csv'), align_res)
        all_mm += mm
        print(f'[{ds}] alignment: {decision}', flush=True)

        files = make_splits(ds, rows, args.out)
        all_split_files += list(files)
        info = {r['id']: flags(ds, r) for r in rows}
        for rel, ids in files.items():
            n_t = sum(info[i][0] for i in ids)
            print(f'[{ds}] {rel}: {len(ids)} ids, {n_t} with target class, first 5 = {ids[:5]}')

        subset = None
        if ds == 'floodnet':
            subset = set(files['splits/floodnet_test_full.txt']) | set(files['splits/floodnet_val_tune30.txt'])
            for s in (0, 1, 2):
                subset |= set(files[f'supports/floodnet_stratified_s{s}.txt'])
        n = export(ds, spec, rows, decision, args.work, subset, args.workers)
        print(f'[{ds}] export: {n} pairs ({time.time() - t0:.0f}s total)', flush=True)
        for res in SIZES:
            src = os.path.join(args.work, f'{ds}_{res}')
            if os.path.isdir(src):
                name = f'{ds}_{res}' + ('_subset' if res == 'r1024' else '')
                size = tar_dir(src, os.path.join(args.out, 'data', f'{name}.tar'))
                print(f'[{ds}] data/{name}.tar: {size / 1e6:.0f} MB', flush=True)
        log[ds] = dict(decision=decision, n_pairs=len(rows),
                       r1024_subset=len(subset) if subset else 0)
    with open(os.path.join(meta, 'size_mismatch.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'split', 'id', 'img_w', 'img_h', 'lab_w', 'lab_h', 'action'])
        w.writeheader()
        w.writerows(all_mm)
    for line in sha256sums(args.out, all_split_files):
        print(line)
    log['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(log, open(os.path.join(meta, 'prep_log.json'), 'w'), indent=1, default=str)
    print('DONE')


if __name__ == '__main__':
    main()
