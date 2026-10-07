"""Batch 13 decode chain for any consumer of the prediction caches (route_eval, tiers, viewer export, Gradio, ...).

A run's decode JSON (results/.../decode/<run>.json, key "chosen") may now carry, besides tau / bldg_t / ctx:
  gf: [r, eps]  guided filter of the probabilities, guided by the grey photo (I30)
  clean: [k, r] clean-up of the class map: components < k px filled from the nearest larger one, closing r (I29)
  sam: {"mode": "grow", "min_area": 50}  SAM ViT-B outline refinement of the predicted buildings (I31)
  classmaps: {"<split_file>": "<npz with ids + class maps>"}  precomputed final maps (the SAM step is expensive)
Without these keys `decode_photo` is exactly rr.metrics.decode(probs, **chosen): every published number unchanged.

    dec = load_chosen(results, run)
    cls = decode_photo(P, dec, img=<RGB uint8 or None>, split_file='floodnet_test_full', photo_id='7301')
"""

import json
import os

import numpy as np

from rr.metrics import decode

EXTRA = ('gf', 'clean', 'sam', 'classmaps')
_MAPS = {}
_SAM = {}


def load_chosen(results, run):
    return json.load(open(os.path.join(results, 'decode', f'{run}.json')))['chosen']


def plain_kwargs(dec):
    return {k: v for k, v in dec.items() if k not in EXTRA and k != 'variant'}


def needs_photo(dec):
    return bool(dec.get('gf') or dec.get('sam'))


def _classmap(dec, split_file, photo_id, root):
    path = (dec.get('classmaps') or {}).get(split_file)
    if not path or photo_id is None:
        return None
    path = path if os.path.isabs(path) else os.path.join(root, path)
    if path not in _MAPS:
        if not os.path.exists(path):
            _MAPS[path] = None
        else:
            z = np.load(path)
            _MAPS[path] = ({str(i): k for k, i in enumerate(z['ids'])}, z['maps'])
    m = _MAPS[path]
    if m is None or str(photo_id) not in m[0]:
        return None
    return m[1][m[0][str(photo_id)]]


def sam_predictor(weights=None):
    if 'p' not in _SAM:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        weights = weights or os.environ.get('RR_SAM_WEIGHTS', os.path.expanduser('~/rr_data/weights/sam_vit_b_01ec64.pth'))
        dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        _SAM['p'] = (SamPredictor(sam_model_registry['vit_b'](checkpoint=weights).to(dev).eval()), dev)
    return _SAM['p']


def decode_photo(probs, dec, img=None, split_file=None, photo_id=None, root='.'):
    """probs float (C, H, W) in [0, 1] -> class map. img: the photo (H, W, 3) uint8, needed for gf / sam unless a
    precomputed class map exists for (split_file, photo_id)."""
    pre = _classmap(dec, split_file, photo_id, root)
    if pre is not None:
        return pre
    if not any(dec.get(k) for k in EXTRA):
        return decode(probs, **plain_kwargs(dec))
    from postproc_sweep import cleanup, guided_filter
    P = np.asarray(probs, np.float32)
    if dec.get('gf'):
        assert img is not None, 'decode with a guided filter needs the photo'
        P = guided_filter(img.mean(2).astype(np.float32) / 255, P, *dec['gf'])
    cls = decode(P, **plain_kwargs(dec))
    if dec.get('clean'):
        cls = cleanup(cls, *dec['clean'])
    if dec.get('sam'):
        assert img is not None, 'SAM refinement needs the photo'
        import torch
        from sam_refine import refine
        pred, dev = sam_predictor()
        with torch.no_grad():
            pred.set_image(img)

            def masks_for(boxes):
                tb = pred.transform.apply_boxes_torch(torch.as_tensor(boxes, device=dev), img.shape[:2])
                m, _, _ = pred.predict_torch(point_coords=None, point_labels=None, boxes=tb, multimask_output=False)
                return list(m[:, 0].cpu().numpy())
            cls = refine(cls, np.asarray(probs, np.float32), masks_for, dec['sam'].get('mode', 'grow'),
                         dec['sam'].get('min_area', 50), dec.get('tau'))
    return cls
