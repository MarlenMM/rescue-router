"""Phase 2 tooling (PLAN §16): run-name parsing, the seed x draw variance split, prior matching."""

import os
import sys

import numpy as np

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
    assert var == ['floodnet_50shot_s0_stratified_sup1', 'floodnet_50shot_s0_stratified_sup2',
                   'floodnet_50shot_s1_stratified_sup0', 'floodnet_50shot_s2_stratified_sup0']
    assert len(MP.session('var_full')) == 8
    curve = {rid: jobs for rid, jobs, _ in MP.session('curve')}
    for k in (1, 5, 10, 20):
        splits = [j[1][j[1].index('--split_file') + 1] for j in curve[f'floodnet_{k}shot_s0_stratified_p2'] if j[0] == 'predict']
        assert splits == ['floodnet_val_tune30', 'floodnet_test_full']
    assert {f'rescuenet_{k}shot_s0_stratified_p2' for k in (5, 10, 20, 50)} <= set(curve)


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
