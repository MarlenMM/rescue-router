"""One-command CPU rebuild of every result, report, scene and the web viewer from the prediction caches (Batch 8).

    python src/rebuild_all.py              # everything (~30 min on 8 cores), then the tests and the headline check
    python src/rebuild_all.py -j 4         # parallel steps (default: 6)
    python src/rebuild_all.py --only tiers,export,viewer      # a few steps (their dependencies must exist)
    python src/rebuild_all.py --check      # only compare the headline numbers with the published ones
    python src/rebuild_all.py --list       # the steps and what each depends on

Inputs (not in git): the prediction caches and decodes in results/kaggle_v2/rr/ (preds/, metrics/) and the data in
~/rr_data (PLAN §15 "Key files"); the test-time-flip caches in results/kaggle_v2/flips/ (optional, step 'tta').
GPU work (fine-tuning, prediction caches) is not part of this: see kaggle/ and PLAN §15.
Every step is one of the documented scripts; logs go to results/rebuild_logs/<step>.log. All randomness is seeded
(bootstrap, RANSAC), so a rebuild reproduces the published numbers exactly; --check verifies the headline ones.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
R, REP, RT = 'results/kaggle_v2/rr', 'results/kaggle_v2/report', 'results/routing'
FN50, RN20 = 'floodnet_50shot_s0_stratified', 'rescuenet_20shot_s0_stratified'


def py(script, *args):
    return [PY, os.path.join('src', script), *map(str, args)]


def route(ds, split_file, out, preds=None, decode=None, *extra):
    cmd = py('route_eval.py', '--dataset', ds, '--split_file', split_file, '--out', out, *extra)
    if preds:
        cmd += ['--preds', preds]
    if decode:
        cmd += ['--decode', decode]
    return cmd


FULL = f'{R}/preds/{FN50}_floodnet_test_full_fp16.npz'
TTA_FULL = f'{R}/preds/{FN50}_tta4_floodnet_test_full_fp16.npz'
# name: (commands run in order, dependencies, what it writes)
STEPS = {
    'collect': ([py('collect_results.py', '--results', R, '--out', REP)], [], 'report/results.{md,json}, learning curve'),
    'decode': ([py('decode_tune.py', '--results', R)], [], 'rr/decode/<run>.json: val30-tuned decodes'),
    'decode_table': ([py('decode_tune.py', '--table', f'{REP}/decode.md')], ['decode'], 'report/decode.md'),
    'baselines': ([py('baseline_table.py', '--results', R, '--out', f'{REP}/baselines.md')], ['decode'],
                  'report/baselines.md (PSPNet etc.), full_test_floodnet50.json'),
    'transfer': ([py('transfer_eval.py')], ['decode'], 'report/transfer.md: cross-disaster'),
    'gt_routing': ([route(ds, f'{ds}_test_eval100', f'{RT}/gt_{ds}_eval100_off{o}.json', None, None, '--offroad', o)
                    for ds in ('floodnet', 'rescuenet') for o in (0, 4)], [], 'routing/gt_*: labelled-map statistics'),
    'route_report': ([py('route_report.py', '--results', R, '--out', RT)], ['decode'], 'routing/routing.md (eval100, gated)'),
    'route_nogate': ([py('route_report.py', '--results', R, '--out', f'{RT}/nogate', '--table', '../routing_nogate.md',
                         '--extra', '--offroad_maxwet -1')], ['decode'], 'routing/routing_nogate.md'),
    'route_full': ([route('floodnet', 'floodnet_test_full', f'{RT}/full/pred_floodnet_50shot_full_default.json', FULL),
                    route('floodnet', 'floodnet_test_full', f'{RT}/full/pred_floodnet_50shot_full_tuned.json', FULL,
                          f'{RT}/decode_{FN50}.json')], ['route_report'], 'routing/full/: FloodNet 50 on the 448-image test'),
    'gate': ([py('gate_sweep.py')], ['decode'], 'routing/gate*/gate.md: off-road gate sweeps'),
    'physical': ([py('physical_ablation.py')], ['decode'], 'routing/physical/ablation.md'),
    'tiers': ([py('tiers.py')], ['decode'], 'routing/tiers/: confidence tiers, cautious gate, p_wet calibration'),
    'calibration': ([py('calibrate.py', '--results', R, '--out', f'{REP}/calibration')], ['decode'],
                    'report/calibration/: per-class Platt scaling, ECE, plain-0.5 decode (PLAN §16 B3)'),
    'prior_decode': ([py('prior_decode.py', '--results', R, '--out', f'{REP}/prior_decode')], ['decode'],
                     'report/prior_decode/: thresholds from the 50 training labels only (PLAN §16 F1)'),
    'false_safe': ([py('false_safe.py', '--results', R, '--out', f'{RT}/false_safe')], ['decode'],
                   'routing/false_safe/: why houses are falsely truck-reachable (PLAN §16 A2)'),
    'router_sweep': ([py('router_sweep.py', '--results', R, '--out', f'{RT}/router_sweep')], ['decode', 'calibration'],
                     'routing/router_sweep/: gate x calibrated wetness x off-road reach (information)'),
    'same_budget': ([py('baseline_report.py'), py('ensemble.py')], ['decode', 'route_full'],
                  'results/baselines/report/: same-budget baselines vs Chameleon, ensemble (needs results/baselines)'),
    'label_audit': ([py('label_audit.py'), py('label_audit.py', '--impact')], ['route_full', 'same_budget'],
                    'report/label_audit/: FloodNet label errors (photo + label only) and their impact on both models'),
    'consensus': ([py('consensus.py')], ['decode', 'same_budget'],
                  'routing/consensus/: two-model agreement as a dispatch check (exploratory, not adopted)'),
    'sensitivity': ([py('sensitivity.py')], ['decode'], 'routing/sensitivity/: cost independence, sensitivity, detection'),
    'dupes': ([py('dupes.py')], [], 'report/dupes/: near-duplicate test/support photos'),
    'bootstrap': ([py('bootstrap_ci.py')], ['decode', 'dupes', 'route_report', 'route_full'], 'report/ci/: bootstrap CIs'),
    'failures': ([py('failures.py')], ['route_report'], 'report/failures/'),
    'figures': ([py('figures.py')], ['failures', 'tiers'], 'report/figures/'),
    'tta': ([py('tta.py', 'build'), py('decode_tune.py', '--runs', f'{FN50}_tta4'),
             [PY, '-c', 'import json; d = json.load(open("%s/decode/%s_tta4.json"))["chosen"]; json.dump(d, open('
              '"%s/tta/decode_tta4.json", "w"))' % (R, FN50, RT)],
             route('floodnet', 'floodnet_test_full', f'{RT}/tta/pred_floodnet_50shot_tta4_full_tuned.json', TTA_FULL,
                   f'{RT}/tta/decode_tta4.json'),
             route('floodnet', 'floodnet_test_full', f'{RT}/tta/pred_floodnet_50shot_tta4_full_default.json', TTA_FULL),
             py('tta.py', 'report')], ['decode', 'route_full'], 'report/tta/: test-time flips (not adopted)'),
    'b13': ([py('b13_report.py')], ['decode'], 'report/b13/: Batch 13 headline switch recomputed (val30 chain, paired test)'),
    'export': ([py('export_web_assets.py', '--out', 'app/scenes')], ['tiers'], 'app/scenes/: 200 viewer scenes'),
    'viewer': ([py('build_viewer.py')], ['export'], 'docs/: the built web viewer'),
    'tests': ([[PY, '-m', 'pytest', 'src/tests', '-q'], ['node', '--test', 'app/test/router.test.mjs']], ['export'],
              'Python tests + JS router parity (bit-exact on all scenes)'),
}

# headline numbers as published (PLAN §15 Quick summary); --check compares the rebuilt files with these (3 decimals).
# Updated 8 Oct (Batch 13.6): FloodNet 50-shot = the Batch 13 model (the published 7 Oct values: results/published_7oct.tar)
EXPECTED = {
    'FloodNet 50 full test mIoU, tau 0.5': (f'{REP}/ci/ci.json', ['curve', 'floodnet/50/floodnet_test_full/tau0.5', 'miou', 'value'], 0.278496),
    'FloodNet 50 full test mIoU, tuned': (f'{REP}/ci/ci.json', ['curve', 'floodnet/50/floodnet_test_full/tuned', 'miou', 'value'], 0.5068),
    'FloodNet 50 eval100 mIoU, tau 0.5': (f'{REP}/ci/ci.json', ['curve', 'floodnet/50/floodnet_test_eval100/tau0.5', 'miou', 'value'], 0.3526),
    'FloodNet 50 eval100 mIoU, tuned': (f'{REP}/ci/ci.json', ['curve', 'floodnet/50/floodnet_test_eval100/tuned', 'miou', 'value'], 0.518506),
    'FloodNet 20 eval100 mIoU, tuned': (f'{REP}/ci/ci.json', ['curve', 'floodnet/20/floodnet_test_eval100/tuned', 'miou', 'value'], 0.4225),
    'FloodNet 5 eval100 mIoU, tuned': (f'{REP}/ci/ci.json', ['curve', 'floodnet/5/floodnet_test_eval100/tuned', 'miou', 'value'], 0.353),
    'RescueNet 20 eval100 mIoU, tuned': (f'{REP}/ci/ci.json', ['curve', 'rescuenet/20/rescuenet_test_eval100/tuned', 'miou', 'value'], 0.4206),
    'stranded F1, FloodNet 50 eval100': (f'{REP}/ci/ci.json', ['routing', 'FloodNet 50-shot eval100', 'all', 'value'], 0.8194),
    'stranded F1, FloodNet 50 full test': (f'{REP}/ci/ci.json', ['routing', 'FloodNet 50-shot full test', 'all', 'value'], 0.7524),
    'stranded F1, RescueNet 20 eval100': (f'{REP}/ci/ci.json', ['routing', 'RescueNet 20-shot eval100', 'all', 'value'], 0.8182),
    'confident-truck houses, FloodNet eval100': (f'{RT}/tiers/tiers.json', ['tiers', 'floodnet_eval100', 'CONFIDENT_TRUCK', 'n'], 208),
    'confident-truck share truly truck-reachable': (f'{RT}/tiers/tiers.json', ['tiers', 'floodnet_eval100', 'CONFIDENT_TRUCK', 'gt_truck_share'], 1.0),
    'building detection F1, FloodNet (IoU .5)': (f'{RT}/sensitivity/sensitivity.json', ['detection', 'floodnet', 0, 'f1'], 0.8063),
    'TTA full test mIoU, own tuned decode (not adopted)': (f'{REP}/tta/tta.json', ['splits', 'full448', 'tuned', 'tta', 'value'], 0.4548),
}


def check():
    ok = True
    for name, (f, keys, want) in EXPECTED.items():
        try:
            v = json.load(open(os.path.join(ROOT, f)))
            for k in keys:
                v = v[k]
            good = round(float(v), 3) == round(want, 3)
        except (OSError, KeyError, IndexError, TypeError) as e:
            v, good = f'missing ({type(e).__name__})', False
        ok &= good
        print(f'  {"OK  " if good else "FAIL"} {name}: {v if isinstance(v, str) else round(float(v), 4)} (published {want})')
    print('headline check:', 'all as published' if ok else 'DIFFERENCES above')
    return ok


def run_step(name, log_dir):
    t = time.time()
    with open(os.path.join(log_dir, f'{name}.log'), 'w') as log:
        for cmd in STEPS[name][0]:
            log.write('$ ' + ' '.join(cmd) + '\n')
            log.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc:
                return name, rc, time.time() - t
    return name, 0, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-j', type=int, default=6)
    ap.add_argument('--only', default=None)
    ap.add_argument('--skip', default='')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        for n, (cmds, deps, what) in STEPS.items():
            print(f'{n:13s} <- {",".join(deps) or "-":40s} {what}')
        return
    if a.check:
        sys.exit(0 if check() else 1)
    todo = list(STEPS) if a.only is None else a.only.split(',')
    todo = [s for s in todo if s not in a.skip.split(',')]
    if 'tta' in todo and not os.path.isdir(os.path.join(ROOT, 'results/kaggle_v2/flips/rr/preds')):
        print('no flip caches (results/kaggle_v2/flips/): skipping tta')
        todo.remove('tta')
    for d in ('results/rebuild_logs', f'{RT}/full', f'{RT}/tta'):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    log_dir = os.path.join(ROOT, 'results/rebuild_logs')
    done, failed, running, t0 = set(), {}, {}, time.time()
    pending = list(todo)
    with ThreadPoolExecutor(a.j) as pool:
        while pending or running:
            for s in list(pending):                                    # start every step whose deps are done
                deps = [d for d in STEPS[s][1] if d in todo]
                if any(d in failed for d in deps):
                    pending.remove(s)
                    failed[s] = 'dependency failed'
                elif all(d in done for d in deps) and len(running) < a.j:
                    pending.remove(s)
                    running[pool.submit(run_step, s, log_dir)] = s
                    print(f'{time.time() - t0:7.0f}s start {s}', flush=True)
            if not running:
                break
            fin, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in fin:
                s, rc, dt = f.result()
                del running[f]
                if rc:
                    failed[s] = f'exit {rc} (see results/rebuild_logs/{s}.log)'
                else:
                    done.add(s)
                print(f'{time.time() - t0:7.0f}s {"done" if not rc else "FAILED"} {s} ({dt / 60:.1f} min)', flush=True)
    print(f'\n{len(done)} steps done in {(time.time() - t0) / 60:.1f} min' + (f'; failed: {failed}' if failed else ''))
    ok = check()
    sys.exit(0 if ok and not failed else 1)


if __name__ == '__main__':
    main()
