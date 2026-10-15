"""Same-budget baselines vs Chameleon (PLAN §16, A5). CPU only, from the caches of src/baseline_seg.py.

Every baseline run (results/baselines/preds/baseline_<arch>[_rescuenet]_<k>shot_s<seed>[_sup<d>][_<tag>][_tta]_<split>.npz)
is compared with Chameleon trained on the SAME photos (same dataset, k and support draw):
  - Chameleon from Phase 2 (--phase2: the one-recipe 'p2' curve, the 50-shot draws/seeds) when it exists, else the
    published run (first recipe for 1-20 shots, the submitted model at 50; draw 0 only);
  - on the largest photo set both predicted (FloodNet full test, else eval100; RescueNet eval100), with 95 % bootstrap
    CIs over photos and the paired difference (same resamples);
  - baseline decodes: its own argmax over the 5 classes + other (no tuning labels) and per-class thresholds tuned on
    val30 like Chameleon's (decode_tune rule; the same 80-label budget); Chameleon: its chosen decode;
  - routing (router v3, the published gate): stranded F1 and false-safe rate on eval100, paired with Chameleon's
    eval100 routing of the same k; FloodNet 50-shot draw-0 runs also on the full test, paired with the submitted model.
Also: mean ± sd over support draws / training seeds per k, and the 50-shot paired difference without the 14 test photos
that overlap a draw-0 support photo (report/dupes). Variant runs (self-training 'st', test-time flips 'tta', small
architectures) are listed as such; the product-model choice among them is made on val30 + confirm100 by
src/ensemble.py (PROTOCOL.md), not here.

    python src/baseline_report.py [--phase2 results/phase2/rr] [--out results/baselines/report]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import postproc_sweep as PS  # noqa: E402
from bootstrap_ci import boot, f1_from  # noqa: E402
from router import false_safe_from  # noqa: E402
from rr.final_decode import decode_photo, load_chosen, photo  # noqa: E402
from rr.metrics import WET, confusion, decode, gt_classes, summarize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402
from run_decode import GF  # noqa: E402

REPO = os.path.dirname(HERE)
NAMES = {'floodnet': PS.NAMES['floodnet'], 'rescuenet': PS.NAMES['rescuenet']}
SUBMITTED = 'floodnet_50shot_s0_stratified'
RUN_RE = re.compile(r'baseline_(?P<arch>deeplabv3_r50|deeplabv3_mbv3|lraspp_mbv3)(?:_(?P<ds>rescuenet))?_(?P<k>\d+|all)shot'
                    r'_s(?P<seed>\d+)(?:_sup(?P<draw>\d+))?(?:_(?P<tag>(?!tta$)[a-z0-9]+))?(?P<tta>_tta)?$')
EVAL = {'floodnet': ('floodnet_test_full', 'floodnet_test_eval100'), 'rescuenet': ('rescuenet_test_full', 'rescuenet_test_eval100')}


def parse(run):
    m = RUN_RE.match(run)
    if not m:
        return None
    d = m.groupdict()
    seed = int(d['seed'])
    return dict(run=run, arch=d['arch'], dataset=d['ds'] or 'floodnet', k=0 if d['k'] == 'all' else int(d['k']), seed=seed,
                draw=int(d['draw']) if d['draw'] else seed, tag=d['tag'] or '', tta=bool(d['tta']))


def miou_fn(ds):
    return lambda c: summarize(c, NAMES[ds], WET[ds])['miou']


class Labels:
    """ground-truth class maps, loaded once per (dataset, split file)"""
    def __init__(self):
        self.c = {}

    def __call__(self, ds, sf):
        if sf not in self.c:
            ids = open(os.path.join(REPO, 'splits', f'{sf}.txt')).read().split()
            root = os.path.join(RR_DATA, f'{ds}_r768', 'val' if '_val_' in sf else 'test', 'labels')
            self.c[sf] = {i: gt_classes(np.asarray(Image.open(os.path.join(root, f'{i}.png')))) for i in ids}
        return self.c[sf]


def cache(folder, run, sf):
    for s in ('', '_fp16'):
        p = os.path.join(folder, 'preds', f'{run}_{sf}{s}.npz')
        if os.path.exists(p):
            return p
    return None


def probs(path):
    z = np.load(path)
    return {str(i): P.astype(np.float32) / 255 for i, P in zip(z['ids'], z['probs'])}


def cham_cms(folder, run, ds, sf, gts):
    """per-photo confusions of a Chameleon run with its chosen decode (SAM maps reused from the full test for eval100)"""
    dec = load_chosen(folder, run)
    P = probs(cache(folder, run, sf))
    split = 'val' if '_val_' in sf else 'test'
    out = {}
    for i, g in gts.items():
        maps = [s for s in (sf, f'{ds}_test_full') if s in (dec.get('classmaps') or {})]
        out[i] = confusion(decode_photo(P[i], dec, img=photo(RR_DATA, ds, split, i), split_file=maps[0] if maps else sf,
                                        photo_id=i, root=REPO), g)
    return out


def label_free_decodes(results, out, shots=(1, 5, 10, 20, 50)):
    """run_decode.py (no SAM) on the published FloodNet runs into <out>/decode/, for their label-free thresholds"""
    os.makedirs(os.path.join(out, 'preds'), exist_ok=True)
    for k in shots:
        run = f'floodnet_{k}shot_s0_stratified'
        if os.path.exists(os.path.join(out, 'decode', f'{run}.json')):
            continue
        for sf in ('floodnet_val_tune30', 'floodnet_test_eval100'):
            src = cache(results, run, sf)
            dst = os.path.join(out, 'preds', os.path.basename(src))
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(src), dst)
        subprocess.run([sys.executable, os.path.join(HERE, 'run_decode.py'), '--results', out, '--run', run], check=True,
                       stdout=subprocess.DEVNULL)


def label_free_tau(folder, run, lf_dir, ds='floodnet', k=50, draw=0):
    """Chameleon's thresholds from its k training labels only. FloodNet: run_decode.py 'prior_tau' (guided filter ->
    class shares of the supports matched on the val30 photos, their labels unused), the run's own decode first (Phase
    2), else the published runs' (lf_dir). RescueNet (its chain has no guided filter): the same matching on the raw
    probabilities, computed here once and kept in <lf_dir>/decode/<run>.json"""
    for f in (os.path.join(folder, 'decode', f'{run}.json'), os.path.join(lf_dir, 'decode', f'{run}.json')):
        if os.path.exists(f):
            d = json.load(open(f))
            if 'prior_tau' in d:
                return d['prior_tau']
    vf = cache(folder, run, f'{ds}_val_tune30')
    if ds != 'rescuenet' or vf is None:
        return None
    from prior_decode import match, support_prior
    tau, _ = match([P[:, ::4, ::4] for P in probs(vf).values()], support_prior(k, draw, ds))
    f = os.path.join(lf_dir, 'decode', f'{run}.json')
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(dict(run=run, prior_tau=[float(t) for t in tau], rule='prior matching on the raw val30 probabilities '
                   f'(labels unused) to the class shares of the {k} support labels (baseline_report.label_free_tau)'),
              open(f, 'w'), indent=1)
    return [float(t) for t in tau]


def cham_label_free_cms(folder, run, ds, sf, gts, tau):
    """per-photo confusions of a Chameleon run decoded with its label-free thresholds (FloodNet: after the guided
    filter, as run_decode.py; no SAM; RescueNet: on the raw probabilities)"""
    P = probs(cache(folder, run, sf))
    split = 'val' if '_val_' in sf else 'test'
    if ds == 'rescuenet':
        return {i: confusion(decode(P[i], tau=tau), g) for i, g in gts.items()}
    return {i: confusion(decode(PS.guided_filter(photo(RR_DATA, ds, split, i)().mean(2).astype(np.float32) / 255, P[i], *GF),
                                tau=tau), g) for i, g in gts.items()}


def cham_for(b, a):
    """(folder, run, label) of the Chameleon run trained on the same photos, Phase 2 first; None if there is none"""
    ds, k, draw = b['dataset'], b['k'], b['draw']
    cands = []
    if a.phase2:
        if ds == 'floodnet' and k == 50:
            cands.append((a.phase2, f'{SUBMITTED}_sup{draw}' if draw else SUBMITTED, 'Phase 2 draw' if draw else 'submitted'))
        elif draw == 0:
            cands.append((a.phase2, f'{ds}_{k}shot_s0_stratified_p2', 'Phase 2 (one recipe)'))
    if draw == 0:
        cands.append((a.results, f'{ds}_{k}shot_s0_stratified', 'submitted' if (ds, k) == ('floodnet', 50) else 'published (first recipe)'))
    for folder, run, label in cands:
        if os.path.exists(os.path.join(folder, 'decode', f'{run}.json')) and cache(folder, run, EVAL[ds][-1]):
            return folder, run, label
    return None


def route(a, b, sf, preds):
    """router v3 on the baseline's argmax maps (cached JSON); returns the per-photo status confusions"""
    f = os.path.join(a.out, f'routing_{b["run"]}.json' if sf.endswith('eval100') else f'routing_full_{b["run"]}.json')
    if not os.path.exists(f):
        dj = os.path.join(a.out, 'decode_argmax.json')
        json.dump({'argmax': True}, open(dj, 'w'))
        subprocess.run([sys.executable, os.path.join(HERE, 'route_eval.py'), '--dataset', b['dataset'], '--split_file', sf,
                        '--preds', preds, '--decode', dj, '--lams', '5', '--out', f], check=True, stdout=subprocess.DEVNULL)
    return per_photo_status(f)


def per_photo_status(f):
    return {r['id']: np.array(r['status_confusion']) for r in json.load(open(f))['per_image'] if 'status_confusion' in r}


def paired(xb, xc, ids, fn, B):
    rb, db = boot(xb, ids, fn, B, np.random.default_rng(0))
    rc, dc = boot(xc, ids, fn, B, np.random.default_rng(0))
    d = db - dc
    return dict(baseline=rb, chameleon=rc, diff=dict(value=rb['value'] - rc['value'], lo=float(np.percentile(d, 2.5)),
                                                    hi=float(np.percentile(d, 97.5))))


def cham_routing_file(folder, run, sf):
    if run == SUBMITTED and sf == 'floodnet_test_full':
        return os.path.join(REPO, 'results', 'routing', 'full', 'pred_floodnet_50shot_full_tuned.json')
    if folder.endswith(os.path.join('kaggle_v2', 'rr')):
        return os.path.join(REPO, 'results', 'routing', f'pred_{run}_tuned.json')
    return os.path.join(os.path.dirname(folder), 'routing', f'{run}.json')        # phase2_report.py --routing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baselines', default='results/baselines')
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--phase2', default=None, help='Phase 2 results folder (p2 curve, 50-shot draws and seeds)')
    ap.add_argument('--out', default='results/baselines/report')
    ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--label_free', default='results/kaggle_v2/report/label_free',
                    help='run_decode.py outputs (decode/<run>.json with prior_tau) for the published 1-50-shot runs')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    label_free_decodes(a.results, a.label_free)
    a.results = os.path.abspath(a.results)
    a.phase2 = os.path.abspath(a.phase2) if a.phase2 and os.path.isdir(a.phase2) else None
    gts = Labels()
    dupes = set(json.load(open(os.path.join(REPO, 'results', 'kaggle_v2', 'report', 'dupes', 'dupes.json')))
                ['runs']['floodnet']['overlapping']['50shot'])
    runs = sorted({re.sub(r'_(floodnet|rescuenet)_(test_full|test_eval100|val_tune30|val_confirm100)$', '',
                          os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(a.baselines, 'preds', 'baseline_*.npz'))})
    bl = [b for b in map(parse, runs) if b and cache(a.baselines, b['run'], EVAL[b['dataset']][-1])]
    rows, cham_cache = [], {}
    for b in bl:
        ds = b['dataset']
        fn = miou_fn(ds)
        cmp = cham_for(b, a)
        sf = next((s for s in EVAL[ds] if cache(a.baselines, b['run'], s) and (cmp is None or cache(cmp[0], cmp[1], s))), None)
        if sf is None:
            continue
        g = gts(ds, sf)
        ids = sorted(g)
        Pb = probs(cache(a.baselines, b['run'], sf))
        row = dict(b, split=sf, n=len(ids), chameleon=cmp and dict(folder=os.path.relpath(cmp[0], REPO), run=cmp[1], label=cmp[2]))
        dec = {'argmax': {'argmax': True}}
        vf = cache(a.baselines, b['run'], f'{ds}_val_tune30')
        if vf:                                         # thresholds tuned on val30 (the 80-label budget)
            gv = gts(ds, f'{ds}_val_tune30')
            Pv = probs(vf)
            S = PS.Scorer([Pv[i] for i in gv], [gv[i] for i in gv], 6)
            tau, _ = PS.tune_tau(S)
            S.close()
            dec['tuned'] = {'tau': tau}
            row['tau'] = tau
        if cmp:
            key = (cmp[0], cmp[1], sf)
            if key not in cham_cache:
                cham_cache[key] = cham_cms(cmp[0], cmp[1], ds, sf, g)
            cc = cham_cache[key]
        for name, d in dec.items():
            cms = {i: confusion(decode_photo(Pb[i], d) if d.get('argmax') else decode(Pb[i], **d), g[i]) for i in ids}
            tot = sum(cms.values())
            r = dict(per_class={n: summarize(tot, NAMES[ds], WET[ds])[f'iou_{n}'] for n in NAMES[ds]})
            if cmp:
                r.update(paired(cms, cc, ids, fn, a.B))
                tau = label_free_tau(cmp[0], cmp[1], a.label_free, ds, b['k'], b['draw']) if name == 'argmax' else None
                if tau:                                # label-fair: both models see only the k training labels
                    if key + ('lf',) not in cham_cache:
                        cham_cache[key + ('lf',)] = cham_label_free_cms(cmp[0], cmp[1], ds, sf, g, tau)
                    r['label_free'] = paired(cms, cham_cache[key + ('lf',)], ids, fn, a.B)
                if name == 'argmax' and ds == 'floodnet' and b['draw'] == 0 and sf == 'floodnet_test_full':
                    keep = [i for i in ids if i not in dupes]
                    r['without_overlap'] = paired(cms, cc, keep, fn, a.B)['diff']
            else:
                r['baseline'], _ = boot(cms, ids, fn, a.B, np.random.default_rng(0))
            row[name] = r
        # routing: eval100 paired with Chameleon's eval100 routing; FloodNet 50-shot draw 0 also on the full test
        ev = f'{ds}_test_eval100'
        if cache(a.baselines, b['run'], ev):
            sb = route(a, b, ev, cache(a.baselines, b['run'], ev))
            row['routing_eval100'] = {}
            fc = cmp and cham_routing_file(cmp[0], cmp[1], ev)
            sc = per_photo_status(fc) if fc and os.path.exists(fc) else None
            for m, f in (('stranded_f1', f1_from), ('false_safe', false_safe_from)):
                common = sorted(set(sb) & set(sc)) if sc else sorted(sb)
                row['routing_eval100'][m] = paired(sb, sc, common, f, a.B) if sc else \
                    dict(baseline=boot(sb, common, f, a.B, np.random.default_rng(0))[0])
        if ds == 'floodnet' and b['k'] == 50 and b['draw'] == 0 and cache(a.baselines, b['run'], 'floodnet_test_full'):
            sb = route(a, b, 'floodnet_test_full', cache(a.baselines, b['run'], 'floodnet_test_full'))
            sc = per_photo_status(cham_routing_file(a.results, SUBMITTED, 'floodnet_test_full'))
            common = sorted(set(sb) & set(sc))
            row['routing_full'] = {m: paired(sb, sc, common, f, a.B) for m, f in (('stranded_f1', f1_from),
                                                                                  ('false_safe', false_safe_from))}
        rows.append(row)
        x = row['argmax']
        print(b['run'], sf, round(x['baseline']['value'], 4), 'vs', cmp and cmp[1],
              x.get('chameleon', {}).get('value'), flush=True)
    json.dump(rows, open(os.path.join(a.out, 'baselines.json'), 'w'), indent=1, default=float)
    open(os.path.join(a.out, 'baselines.md'), 'w').write('\n'.join(report(rows)) + '\n')
    print('\n'.join(report(rows)))


def f3(d):
    return f"{d['value']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def fd(d):
    return f"{d['value']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def label(r):
    v = [r['arch'] if r['arch'] != 'deeplabv3_r50' else '',
         {'st': 'self-trained', 'ens3': 'seed ensemble (3)'}.get(r['tag'], r['tag']),
         'flips' if r['tta'] else '']
    return ', '.join(x for x in v if x) or 'plain'


def report(rows):
    md = ['# Same-label-budget baselines vs Chameleon on the same photos', '',
          'Baseline: DeepLabV3-ResNet50 (COCO weights) fine-tuned on the k photos Chameleon used (`src/baseline_seg.py`; '
          'fixed recipe, no validation labels); decoded by its own argmax (k labels) or with thresholds tuned on val30 '
          '(k + 30 labels, like Chameleon). Chameleon: its chosen decode (val30-tuned). Paired differences: same photos, '
          'same bootstrap resamples (2,000). Variants: self-trained = + pseudo-labelled unlabelled photos '
          '(`src/baseline_pseudo.py`), flips = test-time flip average.', '',
          '## Learning curve (draw 0, seed 0)', '',
          '| dataset | k | variant | photos | baseline argmax | baseline val30-tuned | Chameleon (which) | baseline − Chameleon |',
          '|---|---|---|---|---|---|---|---|']
    key = lambda r: (r['dataset'], r['k'] or 10 ** 4, r['arch'], r['tag'], r['tta'])            # noqa: E731
    for r in sorted((r for r in rows if r['draw'] == 0 and r['seed'] == 0), key=key):
        c = r['argmax']
        md.append(f"| {r['dataset']} | {r['k'] or 'all'} | {label(r)} | {r['split'].split('_', 1)[1]} ({r['n']}) | "
                  f"{f3(c['baseline'])} | {f3(r['tuned']['baseline']) if 'tuned' in r else '–'} | "
                  + (f"{c['chameleon']['value']:.3f} ({r['chameleon']['label']}) | **{fd(c['diff'])}** |" if 'diff' in c else '– | – |'))
    over = [r for r in rows if 'without_overlap' in r['argmax']]
    if over:
        md += ['', 'Without the 14 full-test photos that overlap a draw-0 support photo: ' + '; '.join(
            f"{r['k']}-shot {label(r)} {fd(r['argmax']['without_overlap'])}" for r in over) + '.']
    lf = [r for r in rows if 'label_free' in r['argmax']]
    if lf:
        md += ['', '## Same labels for both: k training labels each, no tuning labels', '',
               "Chameleon decoded with its label-free thresholds (`run_decode.py` prior_tau: guided filter, then thresholds "
               "that give the 30 validation photos, without their labels, the class shares of the k support labels; no "
               "SAM; RescueNet: the same matching on the raw probabilities, as its chain has no filter) vs the baseline's "
               "argmax. The table above gives Chameleon its 30 tuning labels as well.", '',
               '| dataset | k | variant | photos | baseline | Chameleon label-free | baseline − Chameleon |',
               '|---|---|---|---|---|---|---|']
        for r in sorted(lf, key=key):
            x = r['argmax']['label_free']
            md.append(f"| {r['dataset']} | {r['k'] or 'all'} | {label(r)} | {r['split'].split('_', 1)[1]} ({r['n']}) | "
                      f"{f3(x['baseline'])} | {f3(x['chameleon'])} | **{fd(x['diff'])}** |")
    md += ['', '## Per class (full test or eval100, baseline argmax)', '',
           '| run | photos | ' + ' | '.join(NAMES['floodnet']) + ' |', '|---|---|' + '---|' * 5]
    for r in sorted(rows, key=key):
        md.append(f"| {r['run']} | {r['n']} | " + ' | '.join(f'{v:.3f}' for v in r['argmax']['per_class'].values()) + ' |')
    rt = [r for r in rows if 'routing_eval100' in r]
    md += ['', '## Routing (router v3, published gate), eval100: baseline argmax maps vs Chameleon of the same k', '',
           '| run | stranded F1 baseline | Chameleon | difference | false safe baseline | Chameleon | difference |',
           '|---|---|---|---|---|---|---|']
    for r in sorted(rt, key=key):
        f, s = r['routing_eval100']['stranded_f1'], r['routing_eval100']['false_safe']
        cell = lambda x: (f"{f3(x['baseline'])} | {x['chameleon']['value']:.3f} | {fd(x['diff'])}" if 'diff' in x   # noqa: E731
                          else f"{f3(x['baseline'])} | – | –")
        md.append(f"| {r['run']} | {cell(f)} | {cell(s)} |")
    full = [r for r in rows if 'routing_full' in r]
    if full:
        md += ['', '## Routing on the full FloodNet test (448 photos), 50 shots, vs the submitted Chameleon model', '',
               '| run | metric | baseline [95 % CI] | Chameleon | baseline − Chameleon [95 % CI] |', '|---|---|---|---|---|']
        for r in full:
            for m, x in r['routing_full'].items():
                md.append(f"| {r['run']} | {m.replace('_', ' ')} | {f3(x['baseline'])} | {x['chameleon']['value']:.3f} | "
                          f"{fd(x['diff'])} |")
    groups = defaultdict(list)
    for r in rows:
        if not r['tag'] and not r['tta']:
            groups[(r['dataset'], r['arch'], r['k'])].append(r)
    spread = [(g, v) for g, v in groups.items() if len(v) > 1]
    if spread:
        md += ['', '## Spread over support draws and training seeds (baseline argmax, its own evaluation photos)', '',
               '| dataset | k | runs (seed, draw) | mIoU each | mean ± sd | paired vs Chameleon of the same draw |',
               '|---|---|---|---|---|---|']
        for (ds, arch, k), v in sorted(spread):
            vals = [r['argmax']['baseline']['value'] for r in v]
            md.append(f"| {ds} | {k} | {', '.join(f'({r['seed']}, {r['draw']})' for r in v)} | "
                      f"{', '.join(f'{x:.3f}' for x in vals)} | {np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f} | "
                      + ', '.join(fd(r['argmax']['diff']) for r in v if 'diff' in r['argmax']) + ' |')
    return md


if __name__ == '__main__':
    main()
