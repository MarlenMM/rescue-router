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
