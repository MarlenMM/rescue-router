"""I4 (Batch 13): a class-balanced choice of the 50 FloodNet support photos (CPU, from the train labels only).

    python src/balanced_supports.py [--n 50] [--out supports/floodnet_balanced_s0.txt]

Greedy: start empty, repeatedly add the train photo that most increases sum_c sqrt(pixels of channel c in the set)
(concave, so a channel with few pixels gains most: flooded roads and dry buildings get more labelled pixels than in
the stratified list), ties by photo id. Same pool as the stratified list (the 1,445 train photos); val30 and test are
never touched. Prints the channel pixel shares of both lists.
"""

import argparse
import os

import numpy as np
from PIL import Image
from rr.paths import RR_DATA

NAMES = ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(RR_DATA, 'floodnet_r768'))
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--out', default='supports/floodnet_balanced_s0.txt')
    a = ap.parse_args()
    lab_dir = os.path.join(a.data, 'train', 'labels')
    ids = sorted((f[:-4] for f in os.listdir(lab_dir) if f.endswith('.png')), key=int)
    px = np.stack([np.bincount(np.asarray(Image.open(os.path.join(lab_dir, f'{i}.png'))).ravel(), minlength=256)[:5]
                   for i in ids]).astype(np.float64)
    chosen, tot = [], np.zeros(5)
    free = np.ones(len(ids), bool)
    for _ in range(a.n):
        gain = np.sqrt(tot + px).sum(1) - np.sqrt(tot).sum()
        gain[~free] = -1
        j = int(np.argmax(gain))
        chosen.append(ids[j])
        free[j] = False
        tot += px[j]
    open(a.out, 'w').write('\n'.join(chosen) + '\n')
    strat = [l.strip() for l in open('supports/floodnet_stratified_s0.txt') if l.strip()][:a.n]
    idx = {i: k for k, i in enumerate(ids)}
    st = px[[idx[i] for i in strat]].sum(0)
    print('channel share of labelled pixels   ' + ' '.join(f'{n:>13}' for n in NAMES))
    print('stratified (published)             ' + ' '.join(f'{v:13.3f}' for v in st / st.sum()))
    print('balanced (I4)                      ' + ' '.join(f'{v:13.3f}' for v in tot / tot.sum()))
    print('photos with the channel: strat     ' + ' '.join(f'{v:13d}' for v in (px[[idx[i] for i in strat]] > 0).sum(0)))
    print('photos with the channel: balanced  ' + ' '.join(f'{v:13d}' for v in (px[[idx[i] for i in chosen]] > 0).sum(0)))
    print(f'overlap with the stratified list: {len(set(chosen) & set(strat))} of {a.n}; wrote {a.out}')


if __name__ == '__main__':
    main()
