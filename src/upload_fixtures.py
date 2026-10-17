"""Fixtures for the parity test of the in-browser upload (app/js/scenebuild.js, app/test/scenebuild.test.mjs; PLAN §16.10).

For a few FloodNet test photos: the in-browser model's probabilities (app/models/floodnet_lraspp.onnx with
onnxruntime, uint8 as the viewer quantizes them) and the scene Python builds from them for an upload (no labels: the
houses and the entry come from the prediction; src/gradio_viewer.export_upload: decode, cell grids, gates, buildings,
goal rings, scene.rrb, statuses). The JS builder must reproduce the arrays, buildings, goals and statuses exactly.

    python src/upload_fixtures.py [--ids 7301,7611] [--out app/test/fixtures]
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from export_web_assets import build_scene  # noqa: E402
from gradio_viewer import export_upload  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', default='7301,7611')
    ap.add_argument('--models', default=os.path.join(REPO, 'app', 'models'))
    ap.add_argument('--out', default=os.path.join(REPO, 'app', 'test', 'fixtures'))
    a = ap.parse_args()
    import onnxruntime as ort
    meta = json.load(open(os.path.join(a.models, 'floodnet_lraspp.json')))
    sess = ort.InferenceSession(os.path.join(a.models, meta['file']), providers=['CPUExecutionProvider'])
    os.makedirs(a.out, exist_ok=True)
    index = []
    for i in a.ids.split(','):
        img = np.asarray(Image.open(os.path.join(RR_DATA, 'floodnet_r768', 'test', 'images', f'{i}.jpg')).convert('RGB'))
        p = sess.run(None, {'image': img.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        u8 = np.round(p * 255).astype(np.uint8)                     # as gradio_app.BaselineLive and the viewer
        s = build_scene('floodnet', img, u8, meta['decode'], gt=None)
        tmp = tempfile.mkdtemp()
        try:
            status = export_upload(s, tmp, '', meta['run'], 'b', meta['run'])
            shutil.copy(os.path.join(tmp, 'scene.rrb'), os.path.join(a.out, f'upload_{i}.rrb'))
            up = json.load(open(os.path.join(tmp, 'meta.json')))
        finally:
            shutil.rmtree(tmp)
        open(os.path.join(a.out, f'upload_{i}.probs.gz'), 'wb').write(gzip.compress(u8.tobytes(), 9, mtime=0))
        index.append(dict(id=i, shape=list(u8.shape), decode=meta['decode'], status=status,
                          status_pred=up['status_pred'], n_buildings=up['n_buildings']))
        print(i, up['n_buildings'], 'houses', up['status_pred'], flush=True)
    json.dump(index, open(os.path.join(a.out, 'uploads.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
