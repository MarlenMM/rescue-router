"""src/cache_pack.py: a packed cache unpacks to exactly the same arrays (synthetic, no data needed)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cache_pack as C  # noqa: E402


def test_pack_unpack_is_bit_exact(tmp_path):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:12, :16]
    smooth = (np.sin(yy / 3)[None] * np.cos(xx / 4)[None] * 127 + 128).astype(np.uint8)    # like a probability map
    probs = np.stack([np.clip(smooth + rng.integers(-3, 4, (5, 12, 16)), 0, 255).astype(np.uint8) for _ in range(3)])
    probs[1, 2] = 0                                                                         # flat and extreme values
    probs[2, 4] = 255
    src = tmp_path / 'x.npz'
    np.savez_compressed(src, probs=probs, ids=np.array(['7', '10', '123']), cm=np.arange(36).reshape(6, 6))
    C.pack_one((str(src), str(tmp_path / 'x.ptar')))
    C.unpack_one((str(tmp_path / 'x.ptar'), str(tmp_path / 'y.npz')))
    a, b = np.load(src), np.load(tmp_path / 'y.npz')
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        assert np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype, k
