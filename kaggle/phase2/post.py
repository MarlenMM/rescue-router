"""Every CPU step after a Phase 2 session, in order (experiments/phase2/PROTOCOL.md), so none is forgotten.

    python kaggle/phase2/post.py <curve|var|var_full> [--results results/phase2/rr] [--dry] [--no_sam]

Before: ~/.venvs/kaggle/bin/python kaggle/fetch_output.py marlenmelis/rr-p2-<session> results/phase2
(the kernel writes /kaggle/working/rr/..., so the files land in results/phase2/rr/).
Steps (each skipped when its output exists; --dry prints them); first it prints the kernel's environment check:
  1. eval100 caches cut from the full-test caches of the FloodNet 1-20-shot and RescueNet 5-20-shot runs
     (src/subset_cache.py; amendments 1 and 4)
  2. the submitted model's own prediction caches (val30, eval100, full test) linked into the Phase 2 folder,
     so it is decoded and reported with the new runs (seed 0, draw 0); its confirm100 cache (curve session) copied to
     the published folder, where router_sweep / tiers / ensemble read it
  3. run_decode.py (--sam) for every FloodNet run; decode_tune.py for the RescueNet runs
  4. phase2_report.py --routing
  5. curve only: router_sweep (gate) and tiers (cautious gate) on val30 + confirm100 (into results/phase2/report/,
     never over the published reports), ensemble.py model choice on
     val30 + confirm100 (the pre-registered rules); baseline_report.py pairs the baselines with the Phase 2 runs and
     curve_figure.py draws the curves with the one-recipe Phase 2 points
Then refresh the off-site backup (src/backup.py pack; release/backup/upload.sh) and send the notification.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, HERE)
from make_plan import session  # noqa: E402

PUB = os.path.join(ROOT, 'results', 'kaggle_v2', 'rr')
SUBMITTED = 'floodnet_50shot_s0_stratified'


# every registered product-model candidate among the baselines (PROTOCOL amendment 1 item 4, amendment 3); the choice
# is made only when all of them have their confirm100 predictions (the Mac queue makes them; a choice among a subset
# would not be the registered rule)
REGISTERED = ['baseline_deeplabv3_r50_50shot_s0', 'baseline_deeplabv3_r50_50shot_s0_tta', 'baseline_deeplabv3_r50_50shot_s0_st',
              'baseline_deeplabv3_r50_50shot_s0_st_tta', 'baseline_deeplabv3_mbv3_50shot_s0_st', 'baseline_lraspp_mbv3_50shot_s0_st',
              'baseline_deeplabv3_r50_50shot_s0_ens3', 'baseline_deeplabv3_r50_50shot_s0_ens3_tta']


def candidate_order(run):
    """the order of the baseline candidates = the "simpler first" tie-break of the model choice (ensemble.choose_best):
    one ResNet-50 model before the small models' self-training, plain before self-trained, no flips before flips,
    and the seed ensembles (amendment 3) after every single model"""
    return ('_ens3' in run, 'deeplabv3_r50' not in run, '_st' in run, run.endswith('_tta'), run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', choices=['curve', 'var', 'var_full'])
    ap.add_argument('--results', default=os.path.join(ROOT, 'results', 'phase2', 'rr'))
    ap.add_argument('--pub', default=PUB, help='the published results folder (submitted model)')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--no_sam', action='store_true', help='skip the SAM step of the decode (fast check)')
    ap.add_argument('--baseline_out', default=os.path.join(ROOT, 'results', 'baselines', 'report'))
    a = ap.parse_args()
    py, R = sys.executable, a.results
    if not a.dry:
        sys.path.insert(0, SRC)
        import memguard
        memguard.ensure_running()                              # pauses the youngest heavy job when memory runs out
    runs = [rid for rid, _, _ in session(a.session)]

    def run(cmd, out=None):
        if out and os.path.exists(out):
            print(f'= {os.path.relpath(out, ROOT)}')
            return
        print('$', ' '.join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in map(str, cmd)), flush=True)
        if not a.dry:
            subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)

    env = next((f for f in (os.path.join(R, 'logs', a.session, 'envcheck.txt'), os.path.join(R, 'logs', 'envcheck.txt'))
                if os.path.exists(f)), '')                      # the kernel's environment check (PLAN §16.6; finish.py)
    lines = open(env).read().splitlines() if env else []
    print(next((x for x in lines if x.startswith('ENVCHECK')), 'WARNING no ENVCHECK line in logs/envcheck.txt: the '
               'session ran without the environment check, or it crashed'))
    missing = [r for r in runs if not glob.glob(os.path.join(R, 'preds', f'{r}_*'))]
    if missing:
        print('WARNING no prediction cache for', missing)
    # 1. eval100 from the full test (FloodNet 1-20 shots, RescueNet 5-20 shots)
    for ds in ('floodnet', 'rescuenet'):
        for f in sorted(glob.glob(os.path.join(R, 'preds', f'{ds}_*_{ds}_test_full*.npz'))):
            out = f.replace(f'{ds}_test_full', f'{ds}_test_eval100')
            run([py, os.path.join(SRC, 'subset_cache.py'), '--src', f, '--split_file', f'{ds}_test_eval100',
                 '--dataset', ds], out)
    # 2. the submitted model in both folders
    for sub in ('preds', 'decode', 'classmaps', 'ckpt'):
        os.makedirs(os.path.join(R, sub), exist_ok=True)
    # prediction caches only (read-only for every step); never the class maps: run_decode writes its own, and
    # writing through a link would overwrite the published ones
    for f in glob.glob(os.path.join(a.pub, 'preds', f'{SUBMITTED}_floodnet_*.npz')):
        dst = os.path.join(R, 'preds', os.path.basename(f))
        if not re.search(r'_(val_tune30|test_eval100|test_full)(_fp16)?\.npz$', f) or 'part' in f or os.path.exists(dst):
            continue
        print(f'link {os.path.relpath(dst, ROOT)}')
        if not a.dry:
            os.symlink(f, dst)
    conf = os.path.join(R, 'preds', f'{SUBMITTED}_floodnet_val_confirm100_fp16.npz')
    if os.path.exists(conf) and not os.path.islink(conf):
        dst = os.path.join(a.pub, 'preds', os.path.basename(conf))
        if not os.path.exists(dst):
            print(f'copy {os.path.relpath(dst, ROOT)}')
            if not a.dry:
                shutil.copy(conf, dst)
    # 3. decodes
    fn = sorted({re.sub(r'_floodnet_val_tune30(_fp16)?\.npz$', '', os.path.basename(f))
                 for f in glob.glob(os.path.join(R, 'preds', 'floodnet_*_floodnet_val_tune30*.npz'))})
    for r in fn:
        run([py, os.path.join(SRC, 'run_decode.py'), '--results', R, '--run', r] + ([] if a.no_sam else ['--sam']),
            os.path.join(R, 'decode', f'{r}.json'))
    rn = sorted({re.sub(r'_rescuenet_val_tune30(_fp16)?\.npz$', '', os.path.basename(f))
                 for f in glob.glob(os.path.join(R, 'preds', 'rescuenet_*_rescuenet_val_tune30*.npz'))})
    todo = [r for r in rn if not os.path.exists(os.path.join(R, 'decode', f'{r}.json'))]
    if todo:
        run([py, os.path.join(SRC, 'decode_tune.py'), '--results', R, '--runs', ','.join(todo)])
    # 4. report
    rep = os.path.join(os.path.dirname(R), 'report')
    run([py, os.path.join(SRC, 'phase2_report.py'), '--results', R, '--out', rep, '--routing'])
    # 5. the pre-registered choices (need the submitted model's confirm100 cache)
    if a.session == 'curve':
        if not os.path.exists(os.path.join(a.pub, 'preds', os.path.basename(conf))) and not a.dry:
            sys.exit('no confirm100 cache of the submitted model: the router / tier / model choices cannot run')
        # the re-chosen gates go to the Phase 2 report folder: the defaults are the published reports (tiers.json feeds
        # the viewer's route card and rebuild_all --check; calibration_pwet.png is a published figure)
        run([py, os.path.join(SRC, 'router_sweep.py'), '--splits', 'val30,confirm100,eval100',
             '--out', os.path.join(rep, 'router_sweep')])
        run([py, os.path.join(SRC, 'tiers.py'), '--choose_on', 'val30,confirm100', '--out', os.path.join(rep, 'tiers'),
             '--fig', 'none'])
        # every registered baseline candidate (amendment 1 item 4, amendment 3: the seed ensemble 'ens3'): 50 photos,
        # draw 0, seed 0 (or the ensemble of seeds 0-2), with confirm100 predictions
        bl = sorted((re.sub(r'_floodnet_val_confirm100\.npz$', '', os.path.basename(f)) for f in
                     glob.glob(os.path.join(ROOT, 'results', 'baselines', 'preds', 'baseline_*_50shot_s0*_floodnet_val_confirm100.npz'))
                     if re.search(r'_50shot_s0(_st|_ens3)?(_tta)?_floodnet_val_confirm100\.npz$', f)),
                    key=candidate_order)
        missing = [r for r in REGISTERED if r not in bl]
        if missing and not a.dry:
            print(f'NOT YET: the model choice waits for the confirm100 predictions of {", ".join(missing)} (Mac queue: '
                  'python src/job_queue.py experiments/phase2/mac_queue.txt --status); rerun this script when they exist')
        else:
            run([py, os.path.join(SRC, 'ensemble.py'), '--choose_on', 'val30,confirm100'] + (['--baseline', ','.join(bl)] if bl else []))
            # the choice on the full test with the registered paired bootstrap (vs Chameleon, vs the plain baseline)
            run([py, os.path.join(SRC, 'ensemble.py'), '--paired', '--baseline', ','.join(bl)],
                os.path.join(a.baseline_out, f'ensemble_{bl[0]}_choice_paired.json'))
    run([py, os.path.join(SRC, 'baseline_report.py'), '--phase2', R, '--out', a.baseline_out])
    run([py, os.path.join(SRC, 'curve_figure.py'), '--phase2', R, '--out', os.path.join(a.baseline_out, 'curve')])
    print('\nnext: src/backup.py pack && bash release/backup/upload.sh; PushNotification; PLAN §16.4; PROTOCOL deviation log')


if __name__ == '__main__':
    main()
