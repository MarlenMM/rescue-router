"""Local data locations, overridable by environment variables (PLAN §16, fix F5).

  RR_DATA         preprocessed photos/labels: {floodnet_r768, rescuenet_r768, floodnet_r1024} (default ~/rr_data)
  RR_SAM_WEIGHTS  SAM ViT-B weights (default $RR_DATA/weights/sam_vit_b_01ec64.pth)
The GPU scripts (finetune, predict, infer_sweep, probe) keep their Colab/Kaggle defaults (/content/rr_data).
"""
import os

RR_DATA = os.environ.get('RR_DATA', os.path.expanduser('~/rr_data'))
SAM_WEIGHTS = os.environ.get('RR_SAM_WEIGHTS', os.path.join(RR_DATA, 'weights', 'sam_vit_b_01ec64.pth'))
