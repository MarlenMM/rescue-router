"""Tests for the decode rules and the segmentation metrics (src/rr/metrics.py), the bootstrap helper and the
test-time-flip averaging. Small hand-made arrays with hand-computed answers; no data needed.

    ~/.venvs/rr/bin/python -m pytest src/tests -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rr.metrics import IGNORE, OTHER_IDX, WET, building_vote, confusion, decode, gt_classes, summarize  # noqa: E402

NAMES = ['road_clear', 'road_flooded', 'bldg_flooded', 'bldg_dry', 'water']


def probs_of(*pixels):
    """(C, 1, W) probability map from per-pixel 5-vectors"""
    return np.array(pixels, np.float32).T[:, None, :]


# ---------- decode ----------
def test_scalar_tau_is_argmax_above_threshold_else_other():
    P = probs_of([.6, .2, 0, 0, 0], [.4, .45, 0, 0, 0], [.1, .1, .1, .1, .9], [.5, 0, 0, 0, 0])
    assert decode(P, tau=0.5)[0].tolist() == [0, OTHER_IDX, 4, 0]           # p = tau exactly counts


def test_per_channel_tau_equal_values_match_scalar():
    rng = np.random.default_rng(0)
    P = rng.random((5, 7, 9)).astype(np.float32)
    assert np.array_equal(decode(P, tau=0.4), decode(P, tau=[0.4] * 5))


def test_lower_tau_admits_and_favours_a_channel():
    P = probs_of([.4, .6, 0, 0, 0])
    assert decode(P, tau=0.5)[0, 0] == 1                                     # plain rule: the larger p wins
    assert decode(P, tau=[0.3, 0.5, .5, .5, .5])[0, 0] == 0                  # .4/.3 = 1.33 > .6/.5 = 1.2
    assert decode(probs_of([.35, .2, 0, 0, 0]), tau=[0.3, .5, .5, .5, .5])[0, 0] == 0   # admitted below 0.5


def test_building_vote_gives_one_label_per_building():
    cls = np.array([[2, 3, 3, 5, 3, 3]], np.uint8)                          # building A = cols 0-2, B = cols 4-5
    P = np.zeros((5, 1, 6), np.float32)
    P[2, 0, :3], P[3, 0, :3] = [.9, .3, .3], [.1, .5, .5]                   # A: p2 sum 1.5 vs p3 1.1 -> .58 flooded
    P[2, 0, 4:], P[3, 0, 4:] = [.2, .2], [.8, .8]                           # B: .2 -> dry
    out = building_vote(cls, P, 0.5)
    assert out[0].tolist() == [2, 2, 2, 5, 3, 3]
    assert building_vote(cls, P, 0.6)[0, :3].tolist() == [3, 3, 3]          # .58 < .6 -> dry


def test_wet_context_vote_uses_the_ring_around_each_building():
    from rr.metrics import wet_context_vote
    cls = np.full((7, 15), OTHER_IDX, np.uint8)
    cls[2:5, 2:5] = 3                                                        # building A ...
    cls[:, 0:2] = 4                                                          # ... with water on its left
    cls[2:5, 10:13] = 3                                                      # building B, dry land all around
    out = wet_context_vote(cls, 3, 0.3)
    assert (out[2:5, 2:5] == 2).all()                                        # A: flooded
    assert (out[2:5, 10:13] == 3).all()                                      # B: stays dry
    assert (out[:, 0:2] == 4).all() and (out[cls == OTHER_IDX] == OTHER_IDX).all()   # nothing else changes
    assert (wet_context_vote(cls, 3, 0.99)[2:5, 2:5] == 3).all()             # threshold above A's wet share


def test_decode_ctx_overrides_vote():
    P = np.zeros((5, 5, 5), np.float32)
    P[3, 1:4, 1:4] = .9                                                      # dry building, nothing wet around
    assert (decode(P, tau=.5, bldg_t=0.0, ctx=(2, .5))[1:4, 1:4] == 3).all()  # bldg_t=0 alone would say flooded
    assert (decode(P, tau=.5, bldg_t=0.0)[1:4, 1:4] == 2).all()


# ---------- labels and metrics ----------
def test_gt_classes_maps_other_and_keeps_ignore():
    lab = np.array([[0, 1, 2, 3, 4, 254, 255, 7]], np.uint8)
    assert gt_classes(lab)[0].tolist() == [0, 1, 2, 3, 4, OTHER_IDX, IGNORE, OTHER_IDX]


def test_confusion_rows_gt_cols_pred_ignore_skipped():
    gt = np.array([0, 0, 1, 255, 5], np.uint8)
    pred = np.array([0, 1, 1, 0, 5], np.uint8)
    cm = confusion(pred, gt)
    assert cm.sum() == 4 and cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 1 and cm[5, 5] == 1


def test_summarize_hand_example():
    gt = np.array([0, 0, 1, 1, 4, 5], np.uint8)
    pred = np.array([0, 1, 1, 1, 4, 0], np.uint8)
    r = summarize(confusion(pred, gt), NAMES, WET['floodnet'])
    assert r['iou_road_clear'] == pytest.approx(1 / 3)                        # tp 1, fp 1 (other), fn 1
    assert r['iou_road_flooded'] == pytest.approx(2 / 3)
    assert r['iou_water'] == 1.0
    assert r['iou_bldg_flooded'] == 0.0 and r['iou_bldg_dry'] == 0.0         # absent classes count as 0 in the mean
    assert r['miou'] == pytest.approx((1 / 3 + 2 / 3 + 1) / 5)
    assert r['drivable_precision'] == pytest.approx(1 / 2) and r['drivable_recall'] == pytest.approx(1 / 2)
    assert r['wet_iou'] == pytest.approx(3 / 4)                               # wet = {1, 2, 4}: tp 3, fp 1 (pixel 1)
    assert r['pixel_acc'] == pytest.approx(4 / 6)


def test_summarize_perfect():
    gt = np.array([0, 1, 2, 3, 4, 5], np.uint8)
    r = summarize(confusion(gt, gt), NAMES, WET['floodnet'])
    assert r['miou'] == 1.0 and r['drivable_precision'] == 1.0 and r['wet_iou'] == 1.0


# ---------- bootstrap and test-time flips ----------
def test_bootstrap_point_is_the_pooled_metric_and_ci_brackets_it():
    from bootstrap_ci import boot
    rng = np.random.default_rng(1)
    cms = {str(i): confusion(rng.integers(0, 6, 50).astype(np.uint8), rng.integers(0, 6, 50).astype(np.uint8))
           for i in range(30)}
    fn = lambda c: summarize(c, NAMES, WET['floodnet'])['miou']               # noqa: E731
    r, draws = boot(cms, list(cms), fn, 200, np.random.default_rng(0))
    assert r['value'] == pytest.approx(fn(sum(cms.values())))
    assert r['lo'] <= r['value'] <= r['hi'] and len(draws) == 200
    r2, _ = boot(cms, list(cms), fn, 200, np.random.default_rng(0))
    assert r2 == r                                                            # seeded: reproducible


def test_tta_average_rounds_mean_of_uint8_views():
    from tta import average
    views = [np.array([[0, 255, 10]], np.uint8), np.array([[1, 255, 11]], np.uint8)]
    assert average(views).dtype == np.uint8
    assert average(views).tolist() == [[0, 255, 10]]                           # .5 rounds to even, as np.round
    assert average(views + [np.array([[2, 0, 12]], np.uint8)]).tolist() == [[1, 170, 11]]
