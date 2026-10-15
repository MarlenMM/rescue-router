"""Phase 2 tooling (PLAN §16): run-name parsing, the seed x draw variance split, prior matching."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase2_report as P2  # noqa: E402


def test_parse_run_names():
    assert P2.parse('floodnet_50shot_s0_stratified') == dict(dataset='floodnet', shot=50, seed=0, draw=0, sel='stratified', tag='')
    assert P2.parse('floodnet_50shot_s1_stratified_sup2') == dict(dataset='floodnet', shot=50, seed=1, draw=2, sel='stratified', tag='')
    assert P2.parse('floodnet_100shot_s0_stratified_es50') == dict(dataset='floodnet', shot=100, seed=0, draw=0, sel='stratified', tag='es50')


def test_parse_rescuenet():
    assert P2.parse('rescuenet_20shot_s0_stratified_p2')['dataset'] == 'rescuenet'


def test_anova_recovers_row_and_column_effects():
    draw = np.array([0.0, 0.03, -0.03])[:, None]          # strong support-draw effect
    seed = np.array([0.0, 0.0, 0.0])[None, :]             # no training-seed effect
    v = P2.anova2(0.5 + draw + seed)
    assert abs(v['sd_draw'] - 0.03) < 1e-9 and v['sd_seed'] == 0.0 and v['sd_rest'] < 1e-12
    v = P2.anova2(0.5 + np.zeros((3, 1)) + np.array([[0.0, 0.02, -0.02]]))    # seed effect only
    assert v['sd_draw'] == 0.0 and abs(v['sd_seed'] - 0.02) < 1e-9


def test_prior_match_hits_the_target_shares():
    from prior_decode import match, shares
    rng = np.random.default_rng(0)
    probs = [rng.random((5, 24, 32)).astype(np.float32) for _ in range(4)]
    prior = np.array([0.1, 0.05, 0.05, 0.05, 0.15, 0.6])
    tau, loss = match(probs, prior)
    got = shares(probs, tau)
    assert np.abs(got - prior).max() < 0.03, (got, prior)


def test_argmax_decode_for_softmax_baselines():
    from rr.final_decode import decode_photo
    P = np.zeros((5, 2, 2), np.float32)
    P[0, 0, 0], P[4, 0, 1] = 0.45, 0.6          # clear road .45 < other .55 -> other; water .6 > other .4 -> water
    P[1, 1, 0], P[2, 1, 0] = 0.3, 0.4           # flooded building .4 > other .3 > flooded road .3 -> flooded building
    cls = decode_photo(P, {'argmax': True})
    assert cls[0, 0] == 5 and cls[0, 1] == 4 and cls[1, 0] == 2 and cls[1, 1] == 5


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, 'results', 'kaggle_v2', 'rr')


def test_run_decode_reproduces_the_submitted_thresholds(tmp_path):
    """regression: the Phase 2 decode chain on the submitted model's val30 cache gives its published thresholds"""
    import subprocess
    import pytest
    from rr.paths import RR_DATA
    src = os.path.join(RESULTS, 'preds', 'floodnet_50shot_s0_stratified_floodnet_val_tune30.npz')
    if not (os.path.exists(src) and os.path.isdir(RR_DATA)):
        pytest.skip('needs the prediction caches and the data')
    (tmp_path / 'preds').mkdir()
    os.symlink(src, tmp_path / 'preds' / os.path.basename(src))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, os.path.join(here, 'run_decode.py'), '--results', str(tmp_path), '--run',
                    'floodnet_50shot_s0_stratified', '--workers', '4'], check=True, capture_output=True)
    import json
    d = json.load(open(tmp_path / 'decode' / 'floodnet_50shot_s0_stratified.json'))
    assert d['chosen']['tau'] == [0.25, 0.25, 0.45, 0.1, 0.15]
    assert abs(d['val30']['chosen_no_sam']['miou'] - 0.5163) < 5e-4


def test_baseline_run_names_round_trip():
    """baseline_seg.run_id and baseline_report.parse agree on every variant (draws, seeds, tags, flips, datasets)"""
    from argparse import Namespace
    import baseline_report as BR
    import baseline_seg as BS
    for kw in (dict(), dict(seed=1, support_seed=0), dict(dataset='rescuenet', shot=20), dict(pseudo='x'),
               dict(arch='lraspp_mbv3', pseudo='x'), dict(shot=0), dict(support_seed=2)):
        a = Namespace(**{**dict(arch='deeplabv3_r50', dataset='floodnet', shot=50, seed=0, support_seed=-1, tag='',
                                pseudo=''), **kw})
        run = BS.run_id(a)
        for tta in ('', '_tta'):
            p = BR.parse(run + tta)
            assert p, run + tta
            assert (p['arch'], p['dataset'], p['k'], p['seed']) == (a.arch, a.dataset, a.shot, a.seed)
            assert p['draw'] == (a.seed if a.support_seed < 0 else a.support_seed)
            assert p['tag'] == ('st' if a.pseudo else '') and p['tta'] == bool(tta)
    assert BS.run_id(Namespace(arch='deeplabv3_r50', dataset='floodnet', shot=50, seed=0, support_seed=-1, tag='',
                               pseudo='')) == 'baseline_deeplabv3_r50_50shot_s0'      # the published 14 Oct name


def test_subset_cache_keeps_photos_and_order(tmp_path):
    """eval100 cut from a full-test cache: the split file's order, the same bytes per photo"""
    import pytest
    from rr.paths import RR_DATA
    import subset_cache as SC
    ids_full = open(os.path.join(REPO, 'splits', 'floodnet_test_full.txt')).read().split()
    if not os.path.isdir(os.path.join(RR_DATA, 'floodnet_r768', 'test', 'labels')):
        pytest.skip('needs the data')
    rng = np.random.default_rng(0)
    probs = rng.integers(0, 256, (len(ids_full), 5, 4, 4), dtype=np.uint8)
    (tmp_path / 'preds').mkdir()
    src = tmp_path / 'preds' / 'floodnet_5shot_s0_stratified_p2_floodnet_test_full_fp16.npz'
    np.savez_compressed(src, probs=probs, ids=np.array(ids_full), cm=np.zeros((6, 6), np.int64))
    import PIL.Image
    orig = PIL.Image.open
    PIL.Image.open = lambda f, *a, **k: orig(f, *a, **k).resize((4, 4), PIL.Image.NEAREST)    # labels at the toy size
    try:
        out, res = SC.subset(str(src), 'floodnet_test_eval100')
    finally:
        PIL.Image.open = orig
    assert os.path.basename(out) == 'floodnet_5shot_s0_stratified_p2_floodnet_test_eval100_fp16.npz'
    z = np.load(out)
    ev = open(os.path.join(REPO, 'splits', 'floodnet_test_eval100.txt')).read().split()
    assert list(z['ids']) == ev and res['n_images'] == 100
    pos = {i: k for k, i in enumerate(ids_full)}
    assert all(np.array_equal(z['probs'][k], probs[pos[i]]) for k, i in enumerate(ev))


def test_lean_variance_plan_and_full_test_curve():
    sys.path.insert(0, os.path.join(REPO, 'kaggle', 'phase2'))
    import make_plan as MP
    var = [rid for rid, _, _ in MP.session('var')]
    assert var[:4] == ['floodnet_50shot_s0_stratified_sup1', 'floodnet_50shot_s0_stratified_sup2',
                       'floodnet_50shot_s1_stratified_sup0', 'floodnet_50shot_s2_stratified_sup0']
    assert var[4:] == [f'rescuenet_20shot_s0_stratified_sup{d}_p2' for d in MP.VAR_RN_DRAWS]     # amendment 5
    assert len(MP.session('var_full')) == 8
    curve = {rid: jobs for rid, jobs, _ in MP.session('curve')}
    for k in (1, 5, 10, 20):
        splits = [j[1][j[1].index('--split_file') + 1] for j in curve[f'floodnet_{k}shot_s0_stratified_p2'] if j[0] == 'predict']
        assert splits == ['floodnet_val_tune30', 'floodnet_test_full']
    assert {f'rescuenet_{k}shot_s0_stratified_p2' for k in (5, 10, 20, 50)} <= set(curve)
    for k, want in ((5, 'rescuenet_test_full'), (10, 'rescuenet_test_full'), (20, 'rescuenet_test_full'),
                    (50, 'rescuenet_test_eval100')):                  # amendment 4: RescueNet 5-20 on its full test
        splits = [j[1][j[1].index('--split_file') + 1] for j in curve[f'rescuenet_{k}shot_s0_stratified_p2'] if j[0] == 'predict']
        assert splits == ['rescuenet_val_tune30', want]


def test_self_training_batches_mix_labelled_and_pseudo():
    import torch
    import baseline_seg as BS
    lab = [(torch.zeros(3, 40, 40, dtype=torch.uint8), torch.zeros(40, 40, dtype=torch.uint8))]
    pse = [(torch.full((3, 40, 40), 255, dtype=torch.uint8), torch.full((40, 40), 5, dtype=torch.uint8))]
    x, y = next(BS.mixed(lab, pse, 32, 8, 4, np.random.default_rng(0)))
    assert x.shape == (8, 3, 32, 32) and y.dtype == torch.int64
    assert (y[:4] == 0).all() and (y[4:] == 5).all()


def test_decode_floor_lets_few_shot_runs_tune():
    """run_decode's floor = min(P_MIN, own tau-0.5 drivable precision): with the fixed .567 floor a few-shot run whose
    drivable precision is .3 everywhere could never leave tau .5 (amendment 1, item 5)"""
    import postproc_sweep as PS

    def scorer(drv):
        def S(kind, tau):
            return dict(miou=-sum((t - 0.2) ** 2 for t in tau), drivable_precision=drv)   # best at tau = .2 everywhere
        return S
    low = scorer(0.3)
    assert PS.tune_tau(low)[0] == [0.5] * 5                                   # the fixed floor freezes it
    tau, _ = PS.tune_tau(low, floor=min(PS.P_MIN, low('tau', [0.5] * 5)['drivable_precision']))
    assert tau == [0.2] * 5
    high = scorer(0.9)                                                         # 50-shot-like: both rules agree
    assert PS.tune_tau(high)[0] == PS.tune_tau(high, floor=min(PS.P_MIN, 0.9))[0] == [0.2] * 5


def test_model_choice_rule():
    """ensemble.choose_best: best pooled score; ties -> fewer labels, then the simpler (earlier) candidate"""
    from ensemble import choose_best
    order = ['chameleon', 'b argmax', 'w=0.0 b', 'w=0.5 b']
    labels = {'chameleon': 80, 'b argmax': 50, 'w=0.0 b': 80, 'w=0.5 b': 80}
    assert choose_best(order, {'chameleon': .52, 'b argmax': .50, 'w=0.0 b': .51, 'w=0.5 b': .515}, labels) == 'chameleon'
    assert choose_best(order, {'chameleon': .52, 'b argmax': .52, 'w=0.0 b': .52, 'w=0.5 b': .5}, labels) == 'b argmax'
    assert choose_best(order, {'chameleon': .52, 'b argmax': .5, 'w=0.0 b': .5, 'w=0.5 b': .52}, labels) == 'chameleon'
    assert choose_best(order, {'chameleon': .5, 'b argmax': .5, 'w=0.0 b': .5200001, 'w=0.5 b': .52}, labels) == 'w=0.0 b'


def test_env_check_passes_the_same_predictions_and_stops_on_drift(tmp_path):
    """src/env_check.py and the kernel's environment-check block (kaggle/phase2/make_plan.py ENV_CHECK)"""
    import base64
    import io
    import env_check as E
    rng = np.random.default_rng(0)
    probs = rng.integers(0, 256, (4, 5, 16, 24), dtype=np.uint8)
    ids = np.array(['7001', '7002', '7003', '7004'])
    ref_cache = tmp_path / 'published.npz'
    np.savez_compressed(ref_cache, probs=probs, ids=ids)
    ref = E.make_ref(str(ref_cache), 3)
    assert ref['blocks'].shape == (3, 5, 4, 6) and list(ref['ids']) == ['7001', '7002', '7003']
    same = tmp_path / 'same.npz'
    np.savez_compressed(same, probs=np.clip(probs[:3].astype(int) + rng.integers(-2, 3, probs[:3].shape), 0, 255).astype(np.uint8), ids=ids[:3])
    assert E.compare(ref, str(same))[0]                   # +-2/255 noise: the same environment
    drift = tmp_path / 'drift.npz'
    np.savez_compressed(drift, probs=np.clip(probs[:3].astype(int) + 12, 0, 255).astype(np.uint8), ids=ids[:3])
    assert not E.compare(ref, str(drift))[0]
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'kaggle', 'phase2'))
    import make_plan as MP
    buf = io.BytesIO()
    np.savez_compressed(buf, **ref)
    E_dir = tmp_path / 'envcheck'
    block = (MP.ENV_CHECK.replace("'/tmp/envcheck'", repr(str(E_dir))).replace('/content/rr/src/env_check.py', f'"{E.__file__}"')
             .replace("sh(f'python ", f"sh(f'{sys.executable} "))
    import shutil
    import subprocess
    import time
    out = tmp_path / 'out'
    for d in ('ckpt', 'logs', 'splits', 'supports'):
        (out / d).mkdir(parents=True)
    (out / 'ckpt' / 'floodnet_50shot_s0_stratified.ckpt').write_bytes(b'x')
    for new, ok in ((same, True), (drift, False)):
        def sh(cmd, new=new):                             # stand-in for the kernel's sh(): 'predict' writes the cache
            if 'predict.py' in cmd:
                shutil.copy(new, E_dir / 'preds' / 'floodnet_50shot_s0_stratified_floodnet_test_full_part1_n3_fp16.npz')
                return subprocess.CompletedProcess(cmd, 0, '', '')
            return subprocess.run(cmd, shell=True, capture_output=True, text=True)
        g = dict(os=os, glob=__import__('glob'), shutil=shutil, sys=sys, time=time, T0=time.time(), OUT=str(out), sh=sh,
                 ENV_REF_B64=base64.b64encode(buf.getvalue()).decode())
        if E_dir.exists():
            shutil.rmtree(E_dir)
        if ok:
            exec(block, g)
            assert 'PASS' in (out / 'logs' / 'envcheck.txt').read_text(), (out / 'logs' / 'envcheck.txt').read_text()
        else:
            with pytest.raises(SystemExit):
                exec(block, g)
            assert 'FAIL' in (out / 'logs' / 'envcheck.txt').read_text()
    (out / 'ckpt' / 'floodnet_50shot_s0_stratified.ckpt').unlink()     # the dataset did not mount: a clear stop
    shutil.rmtree(E_dir)
    with pytest.raises(SystemExit):
        exec(block, g)
    assert 'not mounted' in (out / 'logs' / 'envcheck.txt').read_text()


def test_model_choice_candidates_are_ordered_simpler_first():
    """post.py hands the baselines to ensemble.py in the tie-break order: single before ensemble (amendment 3)"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'kaggle', 'phase2'))
    import post
    runs = ['baseline_deeplabv3_r50_50shot_s0_ens3_tta', 'baseline_lraspp_mbv3_50shot_s0_st', 'baseline_deeplabv3_r50_50shot_s0_tta',
            'baseline_deeplabv3_r50_50shot_s0_ens3', 'baseline_deeplabv3_r50_50shot_s0', 'baseline_deeplabv3_r50_50shot_s0_st_tta',
            'baseline_deeplabv3_mbv3_50shot_s0_st', 'baseline_deeplabv3_r50_50shot_s0_st']
    assert sorted(runs, key=post.candidate_order) == [
        'baseline_deeplabv3_r50_50shot_s0', 'baseline_deeplabv3_r50_50shot_s0_tta', 'baseline_deeplabv3_r50_50shot_s0_st',
        'baseline_deeplabv3_r50_50shot_s0_st_tta', 'baseline_deeplabv3_mbv3_50shot_s0_st', 'baseline_lraspp_mbv3_50shot_s0_st',
        'baseline_deeplabv3_r50_50shot_s0_ens3', 'baseline_deeplabv3_r50_50shot_s0_ens3_tta']


def test_src_hash_follows_content_not_time(tmp_path):
    """make_plan.src_hash (pre-flight freshness): a touch changes nothing, an edit changes the hash, tests do not count"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'kaggle', 'phase2'))
    import make_plan as MP
    (tmp_path / 'src' / 'tests').mkdir(parents=True)
    (tmp_path / 'src' / 'a.py').write_text('x = 1\n')
    (tmp_path / 'src' / 'tests' / 't.py').write_text('y = 1\n')
    h0 = MP.src_hash(str(tmp_path))
    os.utime(tmp_path / 'src' / 'a.py', (1, 1))
    (tmp_path / 'src' / 'tests' / 't.py').write_text('y = 2\n')
    assert MP.src_hash(str(tmp_path)) == h0
    (tmp_path / 'src' / 'a.py').write_text('x = 2\n')
    assert MP.src_hash(str(tmp_path)) != h0


def test_lanes_put_rescuenet_first_without_changing_lengths():
    """lanes(): the exact 2-GPU split is unchanged; in each lane the prediction-only group, then the RescueNet runs,
    then the longest first (what exists if a session is cut short)"""
    sys.path.insert(0, os.path.join(REPO, 'kaggle', 'phase2'))
    import make_plan as MP
    for name in ('curve', 'var'):
        items = MP.session(name)
        plan, load = MP.lanes(items)
        assert abs(sum(load) - sum(m for _, _, m in items)) < 1e-9
        for groups in plan.values():
            kinds = []
            for g in groups:
                train = [a for sc, a in g if sc == 'finetune']
                kinds.append(2 if not train else 0 if train[0][train[0].index('--dataset') + 1] == 'rescuenet' else 1)
            pred_only = [k for k in kinds if k == 2]
            assert kinds[:len(pred_only)] == pred_only and kinds[len(pred_only):] == sorted(kinds[len(pred_only):])


def test_supports_never_overlap_tuning_photos():
    """rr.runs.check_supports refuses RescueNet draws 1-2 (they hold val30 photos); every draw a run uses passes"""
    from rr.runs import check_supports
    splits = os.path.join(REPO, 'splits')

    def draw(ds, d, k):
        return open(os.path.join(REPO, 'supports', f'{ds}_stratified_s{d}.txt')).read().split()[:k]
    for ds, d in (('floodnet', 0), ('floodnet', 1), ('floodnet', 2), ('rescuenet', 0), ('rescuenet', 3), ('rescuenet', 4)):
        check_supports(ds, draw(ds, d, 100), splits)
    for d in (1, 2):
        with pytest.raises(SystemExit):
            check_supports('rescuenet', draw('rescuenet', d, 20), splits)


def test_registered_candidates_match_the_mac_queue():
    """post.py makes the model choice only with every registered candidate (amendments 1, 3); the list is the one the
    Mac queue's choice_bl job uses, already in the tie-break order"""
    sys.path.insert(0, os.path.join(REPO, 'kaggle', 'phase2'))
    import post
    line = next(x for x in open(os.path.join(REPO, 'experiments', 'phase2', 'mac_queue.txt')) if x.startswith('choice_bl'))
    queue = line.split('--baseline ')[1].split()[0].split(',')
    assert queue == post.REGISTERED == sorted(post.REGISTERED, key=post.candidate_order)


def test_finish_merges_two_sessions_without_losing_logs(tmp_path):
    """finish.merge: both sessions' runs land in one results folder; the files every session writes under one name
    (environment check, lane logs) go to logs/<session>/, so the second download does not overwrite the first"""
    sys.path.insert(0, os.path.join(REPO, 'kaggle', 'phase2'))
    import finish
    dest = tmp_path / 'phase2' / 'rr'
    for session, run in (('curve', 'rescuenet_5shot_s0_stratified_p2'), ('var', 'floodnet_50shot_s0_stratified_sup1')):
        stage = tmp_path / 'phase2' / f'download_{session}'
        for rel, text in ((f'rr/preds/{run}_x.npz', run), (f'rr/logs/{run}.json', run), ('rr/logs/envcheck.txt', f'ENVCHECK OK {session}'),
                          ('rr/logs/b13_gpu0.out', session), ('rr/ckpt/floodnet_50shot_s0_stratified.ckpt', 'same'),
                          (f'rr-p2-{session}.log', session)):
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            (stage / rel).write_text(text)
        finish.merge(str(stage), str(dest), session)
        assert not stage.exists()
    for session in ('curve', 'var'):
        assert (dest / 'logs' / session / 'envcheck.txt').read_text() == f'ENVCHECK OK {session}'
        assert (dest / 'logs' / session / 'b13_gpu0.out').read_text() == session
        assert (tmp_path / 'phase2' / f'rr-p2-{session}.log').read_text() == session
    assert (dest / 'preds' / 'rescuenet_5shot_s0_stratified_p2_x.npz').exists()
    assert (dest / 'logs' / 'floodnet_50shot_s0_stratified_sup1.json').exists() and not (dest / 'logs' / 'envcheck.txt').exists()


def test_post_never_writes_over_the_published_reports(tmp_path):
    """post.py curve: the gate re-choices (router_sweep, tiers) write into the Phase 2 report folder, not the published
    results/routing/{router_sweep,tiers} or the published calibration figure"""
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(REPO, 'kaggle', 'phase2', 'post.py'), 'curve', '--dry',
                        '--results', str(tmp_path / 'rr')], capture_output=True, text=True, cwd=REPO)
    cmds = [x for x in r.stdout.splitlines() if x.startswith('$ ') and ('tiers.py' in x or 'router_sweep.py' in x)]
    assert len(cmds) == 2, r.stdout[-2000:]
    for c in cmds:
        assert f'--out {os.path.relpath(str(tmp_path), REPO)}' in c or f'--out {tmp_path}' in c, c
    assert '--fig none' in next(c for c in cmds if 'tiers.py' in c)
