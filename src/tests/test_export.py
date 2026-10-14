"""scene.rrb packing (src/export_web_assets.pack_scene): the arrays read back bit-exact, header and buildings intact."""

import gzip
import json
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import export_web_assets as E  # noqa: E402

from rr.paths import RR_DATA as DATA  # noqa: E402
RESULTS = 'results/kaggle_v2/rr'


def unpack(blob):
    raw = gzip.decompress(blob)
    n = struct.unpack('<I', raw[:4])[0]
    head = json.loads(raw[4:4 + n])
    assert (4 + n) % 4 == 0
    arrays = {a['name']: np.frombuffer(raw, '<f4' if a['dtype'] == 'f32' else np.uint8, a['length'], 4 + n + a['offset'])
              for a in head['arrays']}
    return head, arrays


@pytest.mark.skipif(not os.path.exists(os.path.join(RESULTS, 'preds')) or not os.path.exists(DATA), reason='needs caches')
def test_pack_roundtrip():
    s = E.scene('floodnet', '7241', RESULTS, DATA)
    head, arr = unpack(E.pack_scene(s))
    g5 = s['grids'][5.0]
    for name in ('pred', 'gt'):
        assert np.array_equal(arr[f'{name}.cls'], g5[name].cls.ravel())
        assert np.array_equal(arr[f'{name}.risk'], g5[name].risk.astype(np.float32).ravel())
        assert np.array_equal(arr[f'{name}.flags'], g5[name].flags().ravel())
    assert np.array_equal(arr['pred_cautious.flags'], g5['pred_cautious'].flags().ravel())
    assert head['grid']['h'] * head['grid']['w'] == arr['pred.cls'].size
    assert [b['id'] for b in head['buildings']] == [r['id'] for r in s['buildings']]
    assert head['buildings'][0]['goals'] == s['buildings'][0]['goals']
    assert E.pack_scene(s) == E.pack_scene(s)                      # deterministic (gzip mtime 0)
