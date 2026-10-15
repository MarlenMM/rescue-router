"""Splits from the local labels: regenerate the published ones, and a second FloodNet validation split (PLAN §16).

  python src/splits_local.py regen     # rebuild every file in splits/SHA256SUMS that is missing locally (support
                                       # pools, RescueNet full test, support orders s1/s2: on Drive only until now);
                                       # each is written only if its SHA-256 matches the published sum
  python src/splits_local.py confirm   # splits/floodnet_val_confirm100.txt (fix F2, below)
  python src/splits_local.py rn_draws  # supports/rescuenet_stratified_s3/s4.txt: RescueNet draws without val30 (below)

rn_draws: RescueNet's supports and its val30 come from the same val pool; val30 was cut outside the draw-0 supports
only (5 Oct), so the published draws 1 and 2 hold 8 and 5 val30 photos (3 and 1 in their first 20). Draws 3 and 4 are
made by the same rule (prep_data.support_order, seeds 3 and 4) on the pool without val30; the published files stay
as they are (their hashes are in the submitted notebook). finetune.py / baseline_seg.py refuse overlapping supports.

confirm: a second FloodNet validation split, disjoint from val30, for confirming choices.

val30 chose the checkpoint, the decode, the router gates and all 36 Batch 13 ideas, so its numbers carry a winner's
curse. splits/floodnet_val_confirm100.txt is drawn from the other 420 val photos with the same stratified rule as
val30 (prep_data.stratified_subset: half with a flooded class where possible; FloodNet val has 54 such photos, 15 of
them in val30, so 39 + 61) and is used only to confirm a choice already made on val30, never to make one.

The class fractions come from the 768x576 raw labels (the published splits used full-resolution stats); `regen` and
the val30 check below show that they give exactly the published files.
"""
import hashlib
import os
import sys
import types

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
cv2 = types.ModuleType('cv2')                                  # prep_data imports cv2 for its export step only
cv2.setNumThreads = lambda n: None
sys.modules.setdefault('cv2', cv2)
import prep_data as P  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

N, SEED = 100, 1


def label_rows(ds, splits):
    rows = []
    for sp in splits:
        d = os.path.join(RR_DATA, f'{ds}_r768', sp, 'labels_raw')
        for f in sorted(os.listdir(d)):
            a = np.array(Image.open(os.path.join(d, f)))
            r = dict(id=f.split('.')[0], split=sp)
            r.update({f'frac_{c}': float((a == c).sum() / a.size) for c in range(12)})
            rows.append(r)
    return rows


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def regen():
    """make_splits on the local labels into a scratch folder; copy every missing file whose hash matches"""
    import tempfile
    want = dict(l.split()[::-1] for l in open(os.path.join(ROOT, 'splits', 'SHA256SUMS')).read().splitlines())
    tmp = tempfile.mkdtemp()
    made = {}
    for ds, splits in (('floodnet', ('train', 'val', 'test')), ('rescuenet', ('val', 'test'))):
        made.update(P.make_splits(ds, label_rows(ds, splits), tmp))
    for rel in sorted(want):
        dst = os.path.join(ROOT, rel)
        if rel not in made:
            print(f'not made by prep_data: {rel}')
            continue
        h = sha(os.path.join(tmp, rel))
        if os.path.exists(dst):
            print(f'{"same" if sha(dst) == h else "DIFFERS (local file kept)"}: {rel}')
        elif h == want[rel]:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, 'wb').write(open(os.path.join(tmp, rel), 'rb').read())
            print(f'restored (hash matches): {rel}')
        else:
            print(f'NOT restored, hash differs: {rel}')


def confirm():
    rows = label_rows('floodnet', ('val',))
    val30 = open(os.path.join(ROOT, 'splits', 'floodnet_val_tune30.txt')).read().split()
    assert P.stratified_subset('floodnet', rows, 30, 0) == val30, 'stats do not reproduce val30'
    rest = [r for r in rows if r['id'] not in set(val30)]
    ids = P.stratified_subset('floodnet', rest, N, SEED)
    assert not set(ids) & set(val30) and len(set(ids)) == N
    rel = 'splits/floodnet_val_confirm100.txt'
    open(os.path.join(ROOT, rel), 'w').write('\n'.join(ids) + '\n')
    h = hashlib.sha256(open(os.path.join(ROOT, rel), 'rb').read()).hexdigest()
    sums = os.path.join(ROOT, 'splits', 'SHA256SUMS')
    lines = [l for l in open(sums).read().splitlines() if not l.endswith(rel)] + [f'{h}  {rel}']
    open(sums, 'w').write('\n'.join(lines) + '\n')
    n_pos = sum(P.flags('floodnet', r)[0] for r in rest if r['id'] in set(ids))
    print(f'{rel}: {N} photos ({n_pos} with a flooded class), sha256 {h[:12]}')


def rn_draws(seeds=(3, 4)):
    val30 = set(open(os.path.join(ROOT, 'splits', 'rescuenet_val_tune30.txt')).read().split())
    pool = open(os.path.join(ROOT, 'splits', 'rescuenet_support_pool.txt')).read().split()
    rows = [r for r in label_rows('rescuenet', ('val',)) if r['id'] in set(pool) - val30]
    assert len(rows) == len(pool) - len(val30) == 419, len(rows)
    s0 = open(os.path.join(ROOT, 'supports', 'rescuenet_stratified_s0.txt')).read().split()
    assert P.support_order('rescuenet', label_rows('rescuenet', ('val',)), 0) == s0, 'stats do not reproduce draw 0'
    sums = os.path.join(ROOT, 'splits', 'SHA256SUMS')
    for seed in seeds:
        ids = P.support_order('rescuenet', rows, seed)
        assert not set(ids) & val30 and len(set(ids)) == P.N_SUPPORT
        rel = f'supports/rescuenet_stratified_s{seed}.txt'
        open(os.path.join(ROOT, rel), 'w').write('\n'.join(ids) + '\n')
        h = sha(os.path.join(ROOT, rel))
        lines = [x for x in open(sums).read().splitlines() if not x.endswith(rel)] + [f'{h}  {rel}']
        open(sums, 'w').write('\n'.join(lines) + '\n')
        print(f'{rel}: {len(ids)} photos, first 20 share {len(set(ids[:20]) & set(s0[:20]))} with draw 0, sha256 {h[:12]}')


if __name__ == '__main__':
    {'regen': regen, 'confirm': confirm, 'rn_draws': rn_draws}[sys.argv[1] if len(sys.argv) > 1 else 'confirm']()
