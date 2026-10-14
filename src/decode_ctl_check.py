"""Batch 13.3 follow-up (CPU): the new model's own decode: tuned tau, guided filter (none / r4 1e-3), then clean-up
(no two-step: it hurts this model). Usage: python src/decode_ctl_check.py <val30 cache> [out.json]"""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postproc_sweep as PS
from decode_tune import load
from rr.paths import RR_DATA

if __name__ == '__main__':
    c = sys.argv[1]
    root = os.path.join(RR_DATA, 'floodnet_r768')
    probs, gts = load(c, root, 'val')
    probs = [p.astype(np.float32) / 255 for p in probs]
    ids = [str(i) for i in np.load(c)['ids']]
    imgs = [np.asarray(Image.open(f'{root}/val/images/{i}.jpg').convert('L'), np.float32) / 255 for i in ids]
    out = []
    for gf in (None, (4, 1e-3), (2, 1e-3)):
        pp = [PS.guided_filter(I, P, *gf) for I, P in zip(imgs, probs)] if gf else probs
        S = PS.Scorer(pp, gts, 10)
        tau, r = PS.tune_tau(S)
        best = (0, 0, r)
        for k in (0, 200, 400, 800, 1600, 3200):
            for rr in (0, 1, 2, 4, 6):
                if k or rr:
                    x = S('clean', tau=tau, k=k, r=rr)
                    if x['drivable_precision'] >= PS.P_MIN - 1e-9 and x['miou'] > best[2]['miou']:
                        best = (k, rr, x)
        S.close()
        row = dict(gf=gf, tau=tau, miou_tau=r['miou'], dp_tau=r['drivable_precision'], k=best[0], r=best[1],
                   miou=best[2]['miou'], dp=best[2]['drivable_precision'], val30=best[2])
        out.append(row)
        print('gf', gf, 'tau', tau, 'tuned', round(r['miou'], 4), '+clean', best[:2], round(best[2]['miou'], 4),
              'dp', round(best[2]['drivable_precision'], 3), flush=True)
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], 'w'), indent=1)
