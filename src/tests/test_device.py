"""rr.device on a machine without CUDA: the starter's hard-coded .cuda() calls must land on the chosen device."""

import os
import subprocess
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE = """
import torch
from rr import device as D
assert D.DEVICE == 'cpu', D.DEVICE
D.use_device()
m = torch.nn.Linear(2, 2).cuda()
t = torch.zeros(3).cuda()
assert next(m.parameters()).device.type == 'cpu' and t.device.type == 'cpu'
D.sync(); D.reset_peak(); D.empty_cache()
assert D.peak_gb() == 0.0 and D.name() == 'CPU'
print('ok')
"""


def test_cuda_calls_redirected_to_cpu():
    r = subprocess.run([sys.executable, '-c', CODE], cwd=SRC, env=dict(os.environ, RR_DEVICE='cpu'),
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr[-800:]
