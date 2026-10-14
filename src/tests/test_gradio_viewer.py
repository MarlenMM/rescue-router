"""Uploaded photo -> viewer scene (src/gradio_viewer.export_upload): files and meta for the viewer's no-labels mode, the
'labelled map' slot equal to the prediction, and the JS router (app/js/router.js) giving the Python statuses."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import export_web_assets as E  # noqa: E402
import gradio_viewer as V  # noqa: E402
from test_export import unpack  # noqa: E402

from rr.paths import RR_DATA as DATA  # noqa: E402
RESULTS = os.path.join(REPO, 'results/kaggle_v2/rr')
JS = '''
import { readFileSync } from 'node:fs';
import { sceneGrids, parseScene, STATUS } from '%s';
const { grid, buildings } = await parseScene(readFileSync(process.argv[2]));
const sol = sceneGrids(grid, 5).pred.solve(grid.base);
console.log(JSON.stringify(buildings.map(b => STATUS[sol.status(b.goals)])));
'''


@pytest.mark.skipif(not os.path.exists(os.path.join(RESULTS, 'preds')) or not os.path.exists(DATA), reason='needs caches')
def test_export_upload(tmp_path):
    c = E.load_cached('floodnet', '7301', RESULTS, DATA)
    s = E.build_scene('floodnet', c['img'], c['probs'], c['decode'], gt=None)
    d = tmp_path / 'up' / 'abc'
    status = V.export_upload(s, str(d), 'test', c['run'])
    assert sorted(os.listdir(d)) == ['gt.png', 'meta.json', 'photo.jpg', 'pred.png', 'scene.rrb', 'thumb.jpg']
    meta = json.load(open(d / 'meta.json'))
    assert meta['upload'] is True and meta['iou'] == {} and meta['size'] == [768, 576]
    assert meta['n_buildings'] == len(s['buildings']) == len(status) > 0
    assert sum(meta['status_pred'].values()) == len(status)
    head, arr = unpack(open(d / 'scene.rrb', 'rb').read())
    for k in ('cls', 'risk', 'flags'):                              # no labels: the gt slot is the prediction
        assert np.array_equal(arr[f'gt.{k}'], arr[f'pred.{k}'])
    if shutil.which('node'):
        js = tmp_path / 'check.mjs'
        js.write_text(JS % os.path.join(REPO, 'app', 'js', 'router.js'))
        out = subprocess.run(['node', str(js), str(d / 'scene.rrb')], capture_output=True, text=True, check=True).stdout
        assert json.loads(out) == status


CKPT = os.path.join(REPO, 'results', 'baselines', 'ckpt', 'baseline_deeplabv3_r50_50shot_s0.pt')


@pytest.mark.skipif(not os.path.exists(CKPT), reason='needs the baseline checkpoint')
def test_cpu_live_baseline_upload(tmp_path):
    """--live_baseline: an uploaded photo -> CPU baseline -> its argmax decode -> a viewer scene the JS router agrees with"""
    from PIL import Image
    import gradio_app as G
    live = G.BaselineLive(CKPT)
    photo = G.prep_upload(np.asarray(Image.open(os.path.join(REPO, 'demo', 'uploads', 'floodnet_7307.jpg')).convert('RGB')))
    probs = live(photo)
    assert probs.shape == (5, 576, 768) and probs.dtype == np.uint8
    s = E.build_scene('floodnet', photo, probs, live.decode, gt=None)
    status = V.export_upload(s, str(tmp_path / 'up' / 'cpu'), 'test', 'baseline')
    assert len(status) == len(s['buildings']) > 0
    js = tmp_path / 'check.mjs'
    js.write_text(JS % os.path.join(REPO, 'app', 'js', 'router.js'))
    if shutil.which('node'):
        out = subprocess.run(['node', str(js), str(tmp_path / 'up' / 'cpu' / 'scene.rrb')], capture_output=True, text=True,
                             check=True).stdout
        assert json.loads(out) == status
