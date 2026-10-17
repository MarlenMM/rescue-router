"""The viewer's in-browser model (PLAN §16.10): a small baseline exported to ONNX for onnxruntime-web. CPU only.

The self-trained LR-ASPP-MobileNetV3 (src/baseline_seg.py, PROTOCOL amendment 1 item 4: the 50 FloodNet photos +
pseudo-labels of the unlabelled train photos; full test mIoU .609 argmax / .601 tuned, 13 MB) with its normalization
and softmax inside the graph, so the browser passes the RGB photo as float32 0-255 (1, 3, 576, 768) and gets the five
class probabilities (1, 5, 576, 768), as baseline_seg.predict. Its decode: the registered val30 + confirm100 rule
among its two decodes (argmax .5244, val30-tuned thresholds .5253: tuned), results/baselines/decode/<run>.json.
Checked against PyTorch with onnxruntime on test photos (probabilities and the uint8 maps the viewer decodes).

    python src/export_onnx.py [--out app/models]   # -> floodnet_lraspp.onnx + floodnet_lraspp.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from baseline_seg import MEAN, STD, build, predict  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402
from router import PROFILES  # noqa: E402
from viz import COLORS, NAMES  # noqa: E402

RUN = 'baseline_lraspp_mbv3_50shot_s0_st'
H, W = 576, 768


class Browser(torch.nn.Module):
    """RGB 0-255 float (1, 3, H, W) -> the 5 class probabilities (other = 1 - their sum), as baseline_seg.predict"""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('mean', MEAN.clone())
        self.register_buffer('std', STD.clone())

    def forward(self, x):
        z = (x / 255 - self.mean) / self.std
        return F.softmax(self.net(z)['out'], 1)[:, :5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'app', 'models'))
    ap.add_argument('--check', default='7301,7611,9107', help='FloodNet test photos for the PyTorch / onnxruntime check')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    net = build('lraspp_mbv3', 'cpu', pretrained=False)
    ckpt = os.path.join(REPO, 'results', 'baselines', 'ckpt', f'{RUN}.pt')
    net.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    net.eval()
    model = Browser(net).eval()
    f = os.path.join(a.out, 'floodnet_lraspp.onnx')
    torch.onnx.export(model, torch.zeros(1, 3, H, W), f, input_names=['image'], output_names=['probs'], opset_version=17,
                      dynamo=False)
    import onnxruntime as ort
    from PIL import Image
    sess = ort.InferenceSession(f, providers=['CPUExecutionProvider'])
    worst, same = 0.0, []
    for i in a.check.split(','):
        img = np.asarray(Image.open(os.path.join(RR_DATA, 'floodnet_r768', 'test', 'images', f'{i}.jpg')).convert('RGB'))
        p_torch = predict(net, torch.from_numpy(img.copy()).permute(2, 0, 1), 'cpu')
        p_onnx = sess.run(None, {'image': img.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        worst = max(worst, float(np.abs(p_torch - p_onnx).max()))
        same.append(float((np.round(p_torch * 255).astype(np.uint8) == np.round(p_onnx * 255).astype(np.uint8)).mean()))
    dec = json.load(open(os.path.join(REPO, 'results', 'baselines', 'decode', f'{RUN}.json')))['chosen']
    meta = dict(run=RUN, file=os.path.basename(f), bytes=os.path.getsize(f), input=dict(name='image', shape=[1, 3, H, W],
                range='RGB 0-255 float32, normalized inside the graph'), output=dict(name='probs', shape=[1, 5, H, W]),
                decode=dec, dataset='floodnet', class_names=NAMES['floodnet'],
                class_colors=[list(COLORS['floodnet'][c]) for c in range(5)], m_per_cell=PROFILES['floodnet']['cell_m'],
                labels='50 FloodNet photos (+ pseudo-labels of the unlabelled train '
                'photos) and the same 30 tuning photos as Chameleon (thresholds)',
                full_test_miou=dict(argmax=0.609, tuned=0.601),
                check=dict(photos=a.check.split(','), max_abs_prob_diff=round(worst, 7), uint8_equal=[round(x, 6) for x in same]))
    json.dump(meta, open(os.path.join(a.out, 'floodnet_lraspp.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
