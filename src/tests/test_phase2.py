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


RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'kaggle_v2', 'rr')


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
