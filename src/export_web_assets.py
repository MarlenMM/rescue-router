"""Web-viewer assets for the showcase scenes (PLAN §6.4, §15 "Next" 5.1). CPU only, from the caches.

    python src/export_web_assets.py --out app/scenes             # all 200 eval100 scenes, the 8 showcase first
    python src/export_web_assets.py --out app/scenes --showcase  # only the 8 showcase scenes (quick)
    python src/export_web_assets.py --sheet out.png --scenes floodnet:7301,rescuenet:12143 ...

Per scene (folder <out>/<dataset>_<id>/):
  photo.jpg       the photo: 1024x768 for FloodNet (r1024 subset tar, framing checked against the 768 image),
                  768x576 for RescueNet; overlays and routing stay on the 768x576 grid
  thumb.jpg       192x144 picker thumbnail
  pred.png/gt.png RGBA class overlays (src/viz.py colours; 'other' transparent)
  scene.rrb       gzip of [uint32 header length][header JSON][pad to 4][arrays]: the header holds the grid meta
                  (router params, profile, base cells, ETA speeds), the buildings (id, sub-class, centroid, polygon,
                  goal rings per mode) and the array layout; the arrays are the cell classes (uint8), risk = p1 + p4
                  (float32, exactly the values router.Grid uses) and passability flags (uint8) of both maps, plus
                  the cautious-gate flags of the prediction. Read by app/js/router.js (parseScene).
  meta.json       ids, run, decode, per-image IoU, status and tier counts, caption, ground scale
  ref.json        Python reference per building (status, cost, ETA on pred / gt / pred_cautious, lam 5 and 0;
                  tier): only for the JS parity test (app/test/), not copied to docs/
plus <out>/scenes.json (index: showcase scenes first, then every eval100 photo of both datasets). The predicted map uses the run's tuned decode and the off-road gate
(route_eval.PRED_OFFROAD_MAXWET, per dataset); the GT map has no gate, as in route_eval.py.

Model switch (PLAN §16.6, §16.9): model b, a standard segmenter on the same photos (MODEL_B, per dataset: FloodNet
the Phase 2 product choice among the baselines, the seed ensemble of three DeepLabV3-ResNet50 with test-time flips and
thresholds tuned on the same 30 photos as Chameleon's; RescueNet the single DeepLabV3-ResNet50 on the same 20 photos,
argmax), gets its own layers per scene in <out>/<dataset>_<id>/b/: pred.png, scene.rrb, meta.json and ref.json as
above (photo, thumbnail and labelled map are shared), where its eval100 cache exists. Same router and gates as
Chameleon (the ensemble's own registered cautious gate is the same 0.05); the route card's tier shares come from each
model's own tiers report. The viewer opens each dataset on its better model (DEFAULT_MODEL); the switch shows the other.

    python src/export_web_assets.py --out app/scenes --models b    # only the baseline layers (index rebuilt)
"""

import argparse
import glob
import gzip
import json
import os
import shutil
import struct
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import router as R  # noqa: E402
from route_eval import PRED_CAUTIOUS_MAXWET, PRED_OFFROAD_MAXWET  # noqa: E402
from rr.metrics import gt_classes  # noqa: E402
from rr.final_decode import decode_photo  # noqa: E402
from viz import COLORS, NAMES, blend, colorize, load_image, load_label  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

RUNS = {'floodnet': 'floodnet_50shot_s0_stratified', 'rescuenet': 'rescuenet_20shot_s0_stratified'}
ROOT = os.path.dirname(HERE)
# the viewer's second model ('b', scenes/<key>/b/): on both datasets the seed ensemble of three DeepLabV3-ResNet50 on
# Chameleon's training photos, flips averaged, thresholds tuned on the same 30 photos (decodes in
# results/baselines/decode/<run>.json: src/ensemble.py for FloodNet, experiments/rescuenet_ensemble.py for RescueNet).
# FloodNet: the registered baseline-only product choice (PLAN §16.8: full test .617 vs Chameleon .507, false safe .068
# vs .111); RescueNet: the fairness check of PLAN §16.9. Tier shares from each run's `src/tiers.py ... --run <run>`.
ENS = 'baseline_deeplabv3_r50_50shot_s0_ens3_tta'
ENS_RN = 'baseline_deeplabv3_r50_rescuenet_20shot_s0_ens3_tta'
MODEL_B = dict(name='DeepLabV3', results=os.path.join(ROOT, 'results/baselines'),
               names={'floodnet': 'DeepLabV3 ensemble', 'rescuenet': 'DeepLabV3 ensemble'},
               tuned={'floodnet': True, 'rescuenet': True},             # thresholds tuned on the 30 tuning photos
               runs={'floodnet': ENS, 'rescuenet': ENS_RN},
               decode={ds: os.path.join(ROOT, f'results/baselines/decode/{r}.json') for ds, r in
                       (('floodnet', ENS), ('rescuenet', ENS_RN))},
               tiers={'floodnet': os.path.join(ROOT, 'results/baselines/report/tiers_ens3_tta/tiers.json'),
                      'rescuenet': os.path.join(ROOT, 'results/baselines/report/tiers_rescuenet_ens3_tta/tiers.json')},
               # routing of the tuned ensembles: ensemble.py --paired --baselines_only, experiments/rescuenet_ensemble.py
               routing={'floodnet': ['results/baselines/report/routing_full_product_choice_baselines.json',
                                     'results/baselines/report/routing_product_choice_baselines.json'],
                        'rescuenet': [f'results/baselines/report/routing_full_{ENS_RN}_tuned.json',
                                      f'results/baselines/report/routing_{ENS_RN}_tuned.json']})
# the model a dataset opens on: the better one there. FloodNet: the ensemble. RescueNet at 20 photos: Chameleon, also
# against the ensemble (registered fairness check, PLAN §16.9: eval100 .421 vs .373, paired −.048 [−.083, −.014]); the
# switch shows the other
DEFAULT_MODEL = {'floodnet': 'b', 'rescuenet': 'a'}
# where each model's measured false-safe rate (houses shown truck-reachable that are not, router v3, lam 5) comes from,
# per dataset: the full test where it was routed, else eval100. The printed briefing quotes the rate of the model and
# dataset on screen (one FloodNet number on RescueNet photos would overstate Chameleon's .026 there by 4x).
ROUTING = {'a': {'floodnet': ['results/routing/full/pred_floodnet_50shot_full_tuned.json',
                              'results/routing/pred_floodnet_50shot_s0_stratified_tuned.json'],
                 'rescuenet': ['results/routing/pred_rescuenet_20shot_s0_stratified_tuned.json']},
           'b': MODEL_B['routing']}
LAMS = (5.0, 0.0)
STATUS_RGB = {R.TRUCK: (46, 204, 64), R.BOAT: (255, 140, 0), R.FOOT: (167, 139, 250), R.ISOLATED: (230, 30, 60)}

# showcase scenes (dataset, id, caption); picked by Marlen from the contact sheets on 6 Oct
SCENES = [
    ('floodnet', '9107', 'Dry neighbourhood: every house reachable by truck'),
    ('floodnet', '7320', 'Flooded street: trucks reach most houses, three need a boat and a walk'),
    ('floodnet', '7301', 'Truck, boat and on-foot rescues in one scene'),
    ('floodnet', '7423', 'Boat rescues across flooded roads'),
    ('floodnet', '7611', 'Cut off by water: most houses need a boat, then a walk'),
    ('floodnet', '7241', 'Honest failure: shallow water read as dry land'),
    ('rescuenet', '15833', 'Hurricane Michael: clear road, intact and damaged houses'),
    ('rescuenet', '12195', 'Hurricane Michael: canal row reachable only on foot (truck, boat, then a walk)'),
]


def building_masks(gt, dataset, min_area=150):
    """The same components as router.buildings_from_gt (same opening, labels and area filter, same order)."""
    pr = R.PROFILES[dataset]
    b = ndimage.binary_opening(np.isin(gt, pr['bldg']), structure=np.ones((3, 3)))
    lab, k = ndimage.label(b)
    return [lab == i for i in range(1, k + 1) if (lab == i).sum() >= min_area]


def polygon(mask, tol=1.5):
    """Largest outer contour of a mask as [[x, y], ...] in pixel coordinates (pixel centres)."""
    c = measure.find_contours(np.pad(mask, 1).astype(np.float32), 0.5)
    c = max(c, key=len)
    c = measure.approximate_polygon(c, tol) - 1
    return [[round(float(x), 1), round(float(y), 1)] for y, x in c]


def b_cache(dataset):
    """the baseline's eval100 prediction cache (no _fp16 suffix), or None before it is predicted"""
    f = os.path.join(MODEL_B['results'], 'preds', f"{MODEL_B['runs'][dataset]}_{dataset}_test_eval100.npz")
    return f if os.path.exists(f) else None


def b_decode(dataset):
    d = MODEL_B['decode'][dataset]
    return json.load(open(d))['chosen'] if isinstance(d, str) else d


def load_cached(dataset, i, results, data, model='a'):
    """photo, GT class map, cached uint8 probabilities and the decode of one eval100 image (model 'a' = Chameleon,
    its tuned decode; 'b' = MODEL_B with its decode per dataset)."""
    if model == 'b':
        run, z, dec = MODEL_B['runs'][dataset], np.load(b_cache(dataset)), b_decode(dataset)
    else:
        run = RUNS[dataset]
        z = np.load(os.path.join(results, 'preds', f'{run}_{dataset}_test_eval100_fp16.npz'))
        dec = json.load(open(os.path.join(results, 'decode', f'{run}.json')))['chosen']
    ids = [str(x) for x in z['ids']]
    return dict(img=load_image(data, dataset, 'test', i), gt=gt_classes(load_label(data, dataset, 'test', i)),
                probs=z['probs'][ids.index(i)], decode=dec, run=run)


def build_scene(dataset, img, probs, dec, gt=None, photo_id=None, split_file=None):
    """Decoded maps, cell grids, base and buildings of one photo. probs: uint8 (C, H, W). Without GT (an
    uploaded photo) the base and the buildings come from the prediction itself."""
    pred = decode_photo(probs.astype(np.float32) / 255, dec, img=img, split_file=split_file, photo_id=photo_id)
    ref = pred if gt is None else gt
    rcls = R.cell_classes(ref)
    blds = R.buildings_from_gt(ref, dataset)
    masks = building_masks(ref, dataset)
    assert len(masks) == len(blds)
    recs = []
    for b, m in zip(blds, masks):
        cy, cx = b['centroid']
        recs.append(dict(id=b['id'], cls=b['cls'], area_px=b['area_px'], centroid=[round(cx, 1), round(cy, 1)],
                         polygon=polygon(m), ring=b['ring'].tolist(), py={}))
    return dict(dataset=dataset, img=img, gt=gt, pred=pred, decode=dec, base=R.base_cells(rcls, dataset, offroad=True),
                bld=blds, buildings=recs, cells=dict(pred=(R.cell_classes(pred), R.cell_probs(probs)),
                                                     gt=None if gt is None else (rcls, R.onehot_probs(gt))))


def make_grids(s, lam=R.DEFAULTS['lam'], boat_cost=R.DEFAULTS['boat_cost']):
    """Routing grids as route_eval.py builds them: the prediction with the off-road gate, GT without; plus the
    prediction with the cautious gate for the confidence tiers (Batch 2)."""
    kw = dict(lam=lam, boat_cost=boat_cost)
    ds = s['dataset']
    out = dict(pred=R.Grid(*s['cells']['pred'], ds, offroad_maxwet=PRED_OFFROAD_MAXWET[ds], **kw),
               pred_cautious=R.Grid(*s['cells']['pred'], ds, offroad_maxwet=PRED_CAUTIOUS_MAXWET[ds], **kw))
    if s['cells']['gt'] is not None:
        out['gt'] = R.Grid(*s['cells']['gt'], s['dataset'], **kw)
    return out


def scene(dataset, i, results, data, model='a'):
    """Everything about one cached eval100 scene, with the Python reference statuses and costs (lam 5, 0)."""
    c = load_cached(dataset, i, results, data, model)
    s = build_scene(dataset, c['img'], c['probs'], c['decode'], c['gt'], photo_id=i, split_file=f'{dataset}_test_eval100')
    s.update(id=i, run=c['run'], model=model, grids={lam: make_grids(s, lam) for lam in LAMS})
    g = s['grids'][LAMS[0]]['gt']
    for r, b in zip(s['buildings'], s['bld']):                      # goal rings per mode (same on both maps)
        r['goals'] = [g.goal(b, layer).tolist() for layer in (R.TRUCK, R.BOAT, R.FOOT)]
    if s['base'] is not None:
        for lam in LAMS:
            for name, g in s['grids'][lam].items():
                sol = g.solve(s['base'])
                for r, b in zip(s['buildings'], s['bld']):
                    rt = sol.route(b)
                    r['py'].setdefault(f'lam{lam:g}', {})[name] = dict(
                        status=R.STATUS[sol.status(b)], cost=None if rt is None else rt['cost'],
                        eta_min=None if rt is None else R.eta(rt, g)['minutes'])
    for r in s['buildings']:
        st = r['py'].get('lam5', {})
        r['status_pred'] = st.get('pred', {}).get('status', 'ISOLATED')
        r['status_gt'] = st.get('gt', {}).get('status', 'ISOLATED')
        sc = st.get('pred_cautious', {}).get('status', 'ISOLATED')
        r['tier_pred'] = R.tier(R.STATUS.index(r['status_pred']), R.STATUS.index(sc))
    return s


def iou_per_class(pred, gt):
    """per-channel IoU of one image (None where the class is absent from GT)"""
    out = []
    for c in range(R.N_CH):
        p, g = pred == c, gt == c
        out.append(round(float((p & g).sum() / (p | g).sum()), 3) if g.any() else None)
    return out


def grid_header(s):
    """grid meta for the JS router (everything but the arrays)"""
    g5 = s['grids'][5.0]
    p = g5['pred'].par
    return dict(h=g5['pred'].h, w=g5['pred'].w, cell=R.CELL, dataset=s['dataset'],
                params=dict(lam=p['lam'], boat_cost=p['boat_cost'], launch=p['launch'], boat_p=p['boat_p'],
                            offroad=p['offroad'], foot_cost=p['foot_cost'], alight=p['alight'],
                            canopy_cost=p['canopy_cost'], corner=bool(p['corner']), cell_m=g5['pred'].cell_m,
                            truck_halfwidth=p['truck_halfwidth'], boat_halfwidth=p['boat_halfwidth'],
                            car_gap=p['car_gap'], canopy_gap=p['canopy_gap'],
                            offroad_maxwet_pred=PRED_OFFROAD_MAXWET[s['dataset']],
                            offroad_maxwet_cautious=PRED_CAUTIOUS_MAXWET[s['dataset']], offroad_maxwet_gt=None),
                profile={k: list(v) if isinstance(v, tuple) else v for k, v in R.PROFILES[s['dataset']].items()},
                base=None if s['base'] is None else dict(cells=s['base'][0].tolist(), mode=R.STATUS[s['base'][1]]),
                eta=R.ETA)


def pack_scene(s):
    """scene.rrb bytes (see the module docstring); arrays in C order, little-endian"""
    g5 = s['grids'][5.0]
    arrays = {}
    for name in ('pred', 'gt'):
        g = g5[name]
        arrays[f'{name}.cls'] = g.cls.astype(np.uint8)
        arrays[f'{name}.risk'] = g.risk.astype('<f4')
        arrays[f'{name}.flags'] = g.flags()
    arrays['pred_cautious.flags'] = g5['pred_cautious'].flags()
    layout, off = [], 0
    for k, a in arrays.items():
        layout.append(dict(name=k, dtype='f32' if a.dtype.kind == 'f' else 'u8', offset=off, length=int(a.size)))
        off += a.nbytes
        off += -off % 4                                             # keep every array 4-byte aligned
    blds = [dict(id=r['id'], cls=r['cls'], centroid=r['centroid'], polygon=r['polygon'], goals=r['goals'])
            for r in s['buildings']]
    head = json.dumps(dict(version=1, grid=grid_header(s), buildings=blds, arrays=layout),
                      separators=(',', ':')).encode()
    head += b' ' * (-(4 + len(head)) % 4)
    data = bytearray(off)
    for l, a in zip(layout, arrays.values()):
        b = np.ascontiguousarray(a).tobytes()
        data[l['offset']:l['offset'] + len(b)] = b
    return gzip.compress(struct.pack('<I', len(head)) + head + bytes(data), 9, mtime=0)


def tier_stats(ds, path=os.path.join(ROOT, 'results/routing/tiers/tiers.json')):
    """share of truly truck-reachable houses per tier on eval100 (src/tiers.py), for the route card; None without
    the report, or when its gates are not the ones the viewer routes with"""
    if not os.path.exists(path):
        return None
    t = json.load(open(path))
    if f'{ds}_eval100' not in t['tiers']:
        return None
    if (t['gates'][ds]['tuned'], t['gates'][ds]['cautious']) != (PRED_OFFROAD_MAXWET[ds], PRED_CAUTIOUS_MAXWET[ds]):
        print(f'  {path}: {ds} gates {t["gates"][ds]["tuned"]} / {t["gates"][ds]["cautious"]} are not the router\'s '
              f'{PRED_OFFROAD_MAXWET[ds]} / {PRED_CAUTIOUS_MAXWET[ds]}: no tier shares on the route card')
        return None
    tt = t['tiers'][f'{ds}_eval100']
    return dict(cautious_gate=t['gates'][ds]['cautious'], tuned_gate=t['gates'][ds]['tuned'],
                confident=tt['CONFIDENT_TRUCK']['gt_truck_share'], probable=tt['PROBABLE_TRUCK']['gt_truck_share'])


def false_safe(model, ds, root=ROOT):
    """{rate, n_houses, split} of the first routing report that exists (ROUTING), or None"""
    for f in ROUTING[model][ds]:
        f = os.path.join(root, f)
        if os.path.exists(f):
            j = json.load(open(f))
            r = j['routing']['lam5']
            return dict(rate=round(r['false_safe_rate'], 4), n_false_safe=r['n_false_safe'],
                        split=j['split_file'].replace(f'{ds}_', ''), n_photos=j['n_images'])
    return None


def scene_scale(s, data=RR_DATA):
    """ground size of a routing cell for the distances and minutes shown: from this photo's labelled cars (median
    car 1.8 m wide) when it has >= 2, else the dataset value the router uses (router.PROFILES cell_m)"""
    vid = {'floodnet': 7, 'rescuenet': 6}[s['dataset']]
    raw = np.asarray(Image.open(os.path.join(data, f"{s['dataset']}_r768", 'test', 'labels_raw', f"{s['id']}.png")))
    w = [r.axis_minor_length for r in measure.regionprops(measure.label(raw == vid))
         if 40 <= r.area <= 3000 and r.eccentricity > 0.8]
    if len(w) >= 2:
        return dict(m_per_cell=round(float(R.CELL * 1.8 / np.median(w)), 3), scale_source=f'{len(w)} labelled cars')
    return dict(m_per_cell=R.PROFILES[s['dataset']]['cell_m'], scale_source='dataset value')


def counts(recs, key):
    return {k: sum(r[key] == k for r in recs) for k in R.STATUS}


def photo_hi(s, data, max_diff=6.0):
    """the 1024x768 photo of a FloodNet scene when the r1024 subset has it and it shows exactly the 768 frame
    (mean abs difference after downscaling <= max_diff grey levels, which also catches the EXIF-180 fix); else None"""
    f = os.path.join(data, f"{s['dataset']}_r1024", 'test', 'images', f"{s['id']}.jpg")
    if not os.path.exists(f):
        return None
    hi = Image.open(f).convert('RGB')
    lo = np.asarray(hi.resize(s['img'].shape[1::-1], Image.BOX), np.float32)
    d = float(np.abs(lo - s['img'].astype(np.float32)).mean())
    if d > max_diff:
        print(f"  {s['dataset']} {s['id']}: 1024 photo differs from the 768 one (mean |diff| {d:.1f}); using 768")
        return None
    return hi


def export(s, out, caption='', data=RR_DATA):
    """the scene's files; for model 'b' only its own layers, in the b/ subfolder"""
    b = s.get('model') == 'b'
    d = os.path.join(out, f"{s['dataset']}_{s['id']}", *(['b'] if b else []))
    os.makedirs(d, exist_ok=True)
    for old in ('grid.json', 'buildings.json'):                     # pre-Batch-4 format
        if os.path.exists(os.path.join(d, old)):
            os.remove(os.path.join(d, old))
    hi = photo_hi(s, data)
    if not b:
        (hi or Image.fromarray(s['img'])).save(os.path.join(d, 'photo.jpg'), quality=86 if hi else 90)
        Image.fromarray(s['img']).resize((192, 144), Image.LANCZOS).save(os.path.join(d, 'thumb.jpg'), quality=82)
    for k in ('pred',) if b else ('pred', 'gt'):
        Image.fromarray(colorize(s[k], s['dataset'])).save(os.path.join(d, f'{k}.png'), optimize=True)
    open(os.path.join(d, 'scene.rrb'), 'wb').write(pack_scene(s))
    json.dump([dict(id=r['id'], py=r['py'], tier_pred=r['tier_pred']) for r in s['buildings']],
              open(os.path.join(d, 'ref.json'), 'w'), separators=(',', ':'))
    ious = iou_per_class(s['pred'], s['gt'])
    meta = dict(id=s['id'], dataset=s['dataset'], run=s['run'], decode=s['decode'], caption=caption,
                size=[int(s['img'].shape[1]), int(s['img'].shape[0])],
                photo_px=list(hi.size) if hi else [int(s['img'].shape[1]), int(s['img'].shape[0])],
                class_names=NAMES[s['dataset']],
                class_colors=[COLORS[s['dataset']][c] for c in range(5)],
                iou=dict(zip(NAMES[s['dataset']], ious)),
                miou=round(float(np.mean([v for v in ious if v is not None])), 3) if any(v is not None for v in ious) else None,
                n_buildings=len(s['buildings']), status_pred=counts(s['buildings'], 'status_pred'),
                status_gt=counts(s['buildings'], 'status_gt'),
                n_disagree=sum(r['status_pred'] != r['status_gt'] for r in s['buildings']),
                tiers_pred={t: sum(r['tier_pred'] == t for r in s['buildings']) for t in R.TIERS},
                tier_stats=tier_stats(s['dataset'], *([MODEL_B['tiers'][s['dataset']]] if b else [])), **scene_scale(s))
    if b:
        meta.update(model='b', model_name=MODEL_B['names'][s['dataset']])
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
    return meta


def eval100_ids(dataset, results):
    z = np.load(os.path.join(results, 'preds', f'{RUNS[dataset]}_{dataset}_test_eval100_fp16.npz'))
    return [str(x) for x in z['ids']]


def _export_one(job):
    ds, i, cap, results, data, out, model = job
    m = export(scene(ds, i, results, data, model), out, cap, data)
    print(ds, i, model, m['status_gt'], m['status_pred'], flush=True)


def summary(m):
    """what the browse list shows of one exported model (badges, sorting)"""
    return dict(status_pred=m['status_pred'], n_disagree=m['n_disagree'], n_check=m['tiers_pred']['PROBABLE_TRUCK'],
                miou=m['miou'])


def index_entry(out, ds, i, cap):
    """scenes.json entry, read back from the exported meta.json files (Chameleon's and, if exported, b/)"""
    d = os.path.join(out, f'{ds}_{i}')
    m = json.load(open(os.path.join(d, 'meta.json')))
    e = dict(key=f'{ds}_{i}', dataset=ds, id=i, caption=cap, showcase=bool(cap), n_buildings=m['n_buildings'],
             status_pred=m['status_pred'], status_gt=m['status_gt'], n_disagree=m['n_disagree'],
             n_check=m['tiers_pred']['PROBABLE_TRUCK'], miou=m['miou'])
    if os.path.exists(os.path.join(d, 'b', 'meta.json')):
        e['b'] = summary(json.load(open(os.path.join(d, 'b', 'meta.json'))))
        refs = [os.path.join(d, *sub, 'ref.json') for sub in ((), ('b',))]
        if all(os.path.exists(f) for f in refs):              # houses the two models route differently (default view)
            ra, rb = (json.load(open(f)) for f in refs)
            if [r['id'] for r in ra] == [r['id'] for r in rb]:
                e['n_models'] = sum(x['py']['lam5']['pred']['status'] != y['py']['lam5']['pred']['status'] for x, y in zip(ra, rb))
    return e


def sheet_row(s, caption):
    """photo | GT overlay + GT statuses | prediction overlay + predicted statuses (one 3-panel row)."""
    panels = [s['img'].copy()]
    for k in ('gt', 'pred'):
        im = Image.fromarray(blend(s['img'], s[k], s['dataset'], 0.6))
        dr = ImageDraw.Draw(im)
        if s['base'] is not None:
            w = s['grids'][5.0]['gt'].w
            for c in s['base'][0]:
                y, x = divmod(int(c), w)
                dr.rectangle([x * 4, y * 4, x * 4 + 3, y * 4 + 3], fill=(255, 255, 255))
        for r in s['buildings']:
            st = r[f'status_{k}']
            dr.polygon([tuple(p) for p in r['polygon']], outline=STATUS_RGB[R.STATUS.index(st)], width=4)
        panels.append(np.asarray(im))
    row = np.concatenate([np.pad(p, ((0, 0), (4, 4), (0, 0)), constant_values=255) for p in panels], 1)
    cap = Image.new('RGB', (row.shape[1], 34), 'white')
    dr = ImageDraw.Draw(cap)
    cg, cp = counts(s['buildings'], 'status_gt'), counts(s['buildings'], 'status_pred')
    fmt = lambda c: f"{c['TRUCK']} truck / {c['BOAT']} boat / {c['FOOT']} foot / {c['ISOLATED']} no route"  # noqa: E731
    dr.text((8, 4), f"{s['dataset']} {s['id']} ({s['run'].split('_')[1]}): {caption}", fill='black')
    dr.text((8, 18), f"GT: {fmt(cg)}    prediction: {fmt(cp)}    "
                     f"(photo | labelled map + status | prediction + status)", fill=(60, 60, 60))
    return np.concatenate([np.asarray(cap), row], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/kaggle_v2/rr')
    ap.add_argument('--data', default=RR_DATA)
    ap.add_argument('--out', default='app/scenes')
    ap.add_argument('--scenes', default=None, help='dataset:id,... (default: the showcase, then all of eval100)')
    ap.add_argument('--showcase', action='store_true', help='only the 8 showcase scenes')
    ap.add_argument('--sheet', default=None, help='write a contact sheet PNG instead of the assets')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--models', default='a,b', help='a = Chameleon (the scene folder), b = MODEL_B (b/, where its '
                                                     'eval100 cache exists); the index is rebuilt from the files')
    a = ap.parse_args()
    models = a.models.split(',')
    caps = {(d, i): c for d, i, c in SCENES}
    if a.scenes:
        todo = [tuple(x.split(':')) for x in a.scenes.split(',')]
    else:
        todo = [(d, i) for d, i, _ in SCENES]
        if not a.showcase:
            todo += [(d, i) for d in RUNS for i in eval100_ids(d, a.results) if (d, i) not in caps]
    if a.sheet:
        rows = []
        for ds, i in todo:
            rows.append(sheet_row(scene(ds, i, a.results, a.data), caps.get((ds, i), '')))
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        Image.fromarray(np.concatenate(rows, 0)).save(a.sheet)
        return
    keep = {f'{d}_{i}' for d, i in todo}
    if not a.scenes:                                                # drop scene folders no longer exported
        for d in glob.glob(os.path.join(a.out, '*_*/')):
            if os.path.basename(d.rstrip('/')) not in keep:
                shutil.rmtree(d)
    jobs = [(ds, i, caps.get((ds, i), ''), a.results, a.data, a.out, 'a') for ds, i in todo] if 'a' in models else []
    if 'b' in models:
        have = {ds for ds in RUNS if b_cache(ds)}
        for ds in sorted(set(RUNS) - have):
            print(f"no {MODEL_B['runs'][ds]} eval100 cache in {MODEL_B['results']}/preds: {ds} without model b")
        jobs += [(ds, i, caps.get((ds, i), ''), a.results, a.data, a.out, 'b') for ds, i in todo if ds in have]
    from rr import mem
    with Pool(mem.workers(a.workers, 1.5, 'export workers')) as pool:    # ~1.5 GB per scene worker (PLAN §16.8)
        pool.map(_export_one, jobs)
    index = [index_entry(a.out, ds, i, caps.get((ds, i), '')) for ds, i in todo]
    have_b = {e['dataset'] for e in index if 'b' in e}
    json.dump(dict(scenes=index, runs=RUNS, offroad_maxwet_pred=PRED_OFFROAD_MAXWET,
                   offroad_maxwet_cautious=PRED_CAUTIOUS_MAXWET,
                   models=dict(a=dict(name='Chameleon', runs=RUNS, false_safe={d: false_safe('a', d) for d in RUNS}),
                               b=dict(name=MODEL_B['name'], names={d: MODEL_B['names'][d] for d in have_b},
                                      tuned={d: MODEL_B['tuned'][d] for d in have_b},
                                      runs={d: r for d, r in MODEL_B['runs'].items() if d in have_b},
                                      false_safe={d: false_safe('b', d) for d in MODEL_B['runs'] if d in have_b}),
                               default={d: DEFAULT_MODEL[d] if d in have_b else 'a' for d in RUNS})),
              open(os.path.join(a.out, 'scenes.json'), 'w'), separators=(',', ':'))
    print(f"{len(index)} scenes -> {a.out} ({sum('b' in e for e in index)} with model b)")


if __name__ == '__main__':
    main()
