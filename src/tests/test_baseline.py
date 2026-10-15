"""Same-budget baseline (baseline_seg.py): a run stopped part-way and rerun ends with the same weights (CPU)."""

import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
from PIL import Image  # noqa: E402

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(SRC)
W = os.path.join(torch.hub.get_dir(), 'checkpoints', 'lraspp_mobilenet_v3_large-d234d4ea.pth')


@pytest.mark.skipif(not os.path.exists(W), reason='torchvision LR-ASPP COCO weights not cached (no download in tests)')
def test_resume_after_a_stop_gives_the_same_weights(tmp_path):
    ids = open(os.path.join(REPO, 'supports', 'floodnet_stratified_s0.txt')).read().split()[:2]
    root = tmp_path / 'data' / 'floodnet_r768' / 'train'
    for d in ('images', 'labels'):
        (root / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in ids:                                         # tiny stand-in photos and labels (classes 0-4, 254 other)
        Image.fromarray(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)).save(root / 'images' / f'{i}.jpg')
        Image.fromarray(rng.choice([0, 1, 2, 3, 4, 254], (96, 128)).astype(np.uint8)).save(root / 'labels' / f'{i}.png')
    env = dict(os.environ, RR_DATA=str(tmp_path / 'data'), OMP_NUM_THREADS='1')
    base = [sys.executable, os.path.join(SRC, 'baseline_seg.py'), '--arch', 'lraspp_mbv3', '--shot', '2', '--iters', '6',
            '--bs', '2', '--crop', '64', '--save_every', '2', '--device', 'cpu', '--splits', '']

    def run(out, *extra):
        r = subprocess.run(base + ['--out', str(tmp_path / out), *extra], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        return r.stdout

    run('straight')
    assert 'stopped after step 3' in run('resumed', '--stop_after', '3')
    f = tmp_path / 'resumed' / 'ckpt' / 'baseline_lraspp_mbv3_2shot_s0.pt'
    assert not f.exists() and os.path.exists(str(f) + '.part')
    assert 'resumed from' in run('resumed')
    assert not os.path.exists(str(f) + '.part')
    a = torch.load(tmp_path / 'straight' / 'ckpt' / f.name)
    b = torch.load(f)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_job_queue_runs_pending_jobs_once_and_records_failures(tmp_path):
    """src/job_queue.py: done jobs are skipped on a restart, a failed one is not retried, appended jobs are picked up"""
    q, state = tmp_path / 'q.txt', tmp_path / 'state'
    q.write_text('# comment\na  true\nb  false\n\nc  true   # trailing comment\n')
    cmd = [sys.executable, os.path.join(SRC, 'job_queue.py'), str(q), '--state', str(state)]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    assert (state / 'done').read_text().split() == ['a', 'c'] and (state / 'failed').read_text().split() == ['b']
    q.write_text(q.read_text() + 'd  true\n')
    subprocess.run(cmd, capture_output=True)
    assert (state / 'done').read_text().split() == ['a', 'c', 'd']   # a, c not run again; b not retried
    out = subprocess.run(cmd + ['--status'], capture_output=True, text=True).stdout
    assert 'FAILED   b' in out and not (state / 'lock').exists()


def test_baseline_avg_writes_the_rounded_mean_as_a_run(tmp_path):
    """src/baseline_avg.py: the seed ensemble cache = rounded mean of the members, with its argmax confusion"""
    rng = np.random.default_rng(1)
    out, data = tmp_path / 'baselines', tmp_path / 'data'
    for d in ('preds', 'metrics', 'logs'):
        (out / d).mkdir(parents=True)
    (data / 'floodnet_r768' / 'val' / 'labels').mkdir(parents=True)
    ids = np.array(['7001', '7002'])
    for i in ids:
        Image.fromarray(rng.choice([0, 1, 2, 3, 4, 254], (8, 12)).astype(np.uint8)).save(
            data / 'floodnet_r768' / 'val' / 'labels' / f'{i}.png')
    members = []
    for r in ('m0', 'm1', 'm2'):
        p = rng.dirichlet(np.ones(6), (2, 8, 12)).transpose(0, 3, 1, 2)[:, :5]           # 5 class channels of a softmax
        P = np.round(p * 255).astype(np.uint8)
        members.append(P)
        np.savez_compressed(out / 'preds' / f'{r}_floodnet_val_tune30.npz', probs=P, ids=ids, cm=np.zeros((6, 6)))
    np.savez_compressed(out / 'preds' / 'm0_floodnet_test_full.npz', probs=members[0], ids=ids)   # only one member: skipped
    r = subprocess.run([sys.executable, os.path.join(SRC, 'baseline_avg.py'), '--runs', 'm0,m1,m2', '--out_run', 'ens',
                        '--baselines', str(out)], env=dict(os.environ, RR_DATA=str(data)), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    z = np.load(out / 'preds' / 'ens_floodnet_val_tune30.npz')
    assert np.array_equal(z['probs'], np.round(np.mean([m.astype(np.float32) for m in members], 0)).astype(np.uint8))
    assert list(z['ids']) == list(ids) and z['cm'].sum() == 2 * 8 * 12
    assert not (out / 'preds' / 'ens_floodnet_test_full.npz').exists() and 'skip floodnet_test_full' in r.stdout
