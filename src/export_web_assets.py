"""Web-viewer assets for the showcase scenes (PLAN §6.4, §15 "Next" 5.1). CPU only, from the caches.

    python src/export_web_assets.py --out app/scenes                       # the 8 showcase scenes
    python src/export_web_assets.py --sheet out.png --scenes floodnet:7301,rescuenet:12143 ...

Per scene (folder <out>/<dataset>_<id>/):
  photo.jpg       the 768x576 photo
  pred.png/gt.png RGBA class overlays (src/viz.py colours; 'other' transparent)
  grid.json       cell grid of both maps for the JS router: classes (uint8), risk = p1 + p4 and
                  p_boat (float32, exactly the values router.Grid uses), base cells, router params
  buildings.json  GT buildings: polygon (px), centroid, goal ring (flat cell ids), GT sub-class,
                  status + route cost on the predicted and the GT map (Python reference, lam 5 and 0)
  meta.json       ids, run, decode, per-image IoU, status counts, caption
plus <out>/scenes.json (index). The predicted map uses the run's tuned decode and the off-road gate
(route_eval.PRED_OFFROAD_MAXWET, per dataset); the GT map has no gate, as in route_eval.py.
"""

import argparse
import base64
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import router as R  # noqa: E402
from route_eval import PRED_CAUTIOUS_MAXWET, PRED_OFFROAD_MAXWET  # noqa: E402
from rr.metrics import decode, gt_classes  # noqa: E402
from viz import COLORS, NAMES, blend, colorize, load_image, load_label  # noqa: E402

RUNS = {'floodnet': 'floodnet_50shot_s0_stratified', 'rescuenet': 'rescuenet_20shot_s0_stratified'}
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


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')


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


def load_cached(dataset, i, results, data):
    """photo, GT class map, cached uint8 probabilities and the tuned decode of one eval100 image."""
    run = RUNS[dataset]
    z = np.load(os.path.join(results, 'preds', f'{run}_{dataset}_test_eval100_fp16.npz'))
    ids = [str(x) for x in z['ids']]
    dec = json.load(open(os.path.join(results, 'decode', f'{run}.json')))['chosen']
    return dict(img=load_image(data, dataset, 'test', i), gt=gt_classes(load_label(data, dataset, 'test', i)),
                probs=z['probs'][ids.index(i)], decode=dec, run=run)


def build_scene(dataset, img, probs, dec, gt=None):
    """Decoded maps, cell grids, base and buildings of one photo. probs: uint8 (C, H, W). Without GT (an
    uploaded photo) the base and the buildings come from the prediction itself."""
    pred = decode(probs.astype(np.float32) / 255, **dec)
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


def scene(dataset, i, results, data):
    """Everything about one cached eval100 scene, with the Python reference statuses and costs (lam 5, 0)."""
    c = load_cached(dataset, i, results, data)
    s = build_scene(dataset, c['img'], c['probs'], c['decode'], c['gt'])
    s.update(id=i, run=c['run'], grids={lam: make_grids(s, lam) for lam in LAMS})
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


def grid_json(s):
    g5 = s['grids'][5.0]

    def one(g):
        return dict(cls=b64(g.cls.astype(np.uint8)), risk=b64(g.risk.astype(np.float32)),
                    pboat=b64(g.probs[list(g.prof['boat'])].sum(0).astype(np.float32)), flags=b64(g.flags()))
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
                pred=one(g5['pred']), gt=one(g5['gt']), pred_cautious=dict(flags=b64(g5['pred_cautious'].flags())),
                eta=R.ETA)


def tier_stats(ds, path=os.path.join(os.path.dirname(HERE), 'results/routing/tiers/tiers.json')):
    """share of truly truck-reachable houses per tier on eval100 (src/tiers.py), for the route card"""
    if not os.path.exists(path):
        return None
    t = json.load(open(path))
    tt = t['tiers'][f'{ds}_eval100']
    return dict(cautious_gate=t['gates'][ds]['cautious'], tuned_gate=t['gates'][ds]['tuned'],
                confident=tt['CONFIDENT_TRUCK']['gt_truck_share'], probable=tt['PROBABLE_TRUCK']['gt_truck_share'])


def scene_scale(s, data=os.path.expanduser('~/rr_data')):
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


def export(s, out, caption=''):
    d = os.path.join(out, f"{s['dataset']}_{s['id']}")
    os.makedirs(d, exist_ok=True)
    Image.fromarray(s['img']).save(os.path.join(d, 'photo.jpg'), quality=92)
    Image.fromarray(s['img']).resize((192, 144), Image.LANCZOS).save(os.path.join(d, 'thumb.jpg'), quality=85)
    for k in ('pred', 'gt'):
        Image.fromarray(colorize(s[k], s['dataset'])).save(os.path.join(d, f'{k}.png'), optimize=True)
    json.dump(grid_json(s), open(os.path.join(d, 'grid.json'), 'w'), separators=(',', ':'))
    json.dump(s['buildings'], open(os.path.join(d, 'buildings.json'), 'w'), separators=(',', ':'))
    meta = dict(id=s['id'], dataset=s['dataset'], run=s['run'], decode=s['decode'], caption=caption,
                size=[int(s['img'].shape[1]), int(s['img'].shape[0])], class_names=NAMES[s['dataset']],
                class_colors=[COLORS[s['dataset']][c] for c in range(5)],
                iou=dict(zip(NAMES[s['dataset']], iou_per_class(s['pred'], s['gt']))),
                n_buildings=len(s['buildings']), status_pred=counts(s['buildings'], 'status_pred'),
                status_gt=counts(s['buildings'], 'status_gt'),
                tiers_pred={t: sum(r['tier_pred'] == t for r in s['buildings']) for t in R.TIERS},
                tier_stats=tier_stats(s['dataset']), **scene_scale(s))
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
    return meta


def sheet_row(s, caption):
    """photo | GT overlay + GT statuses | prediction overlay + predicted statuses (one 3-panel row)."""
    panels = [s['img'].copy()]
    for k in ('gt', 'pred'):
        im = Image.fromarray(blend(s['img'], s[k], s['dataset'], 0.6))
        dr = ImageDraw.Draw(im)
        if s['base'] is not None:
            h, w = s['grids'][5.0]['gt'].h, s['grids'][5.0]['gt'].w
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
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--out', default='app/scenes')
    ap.add_argument('--scenes', default=None, help='dataset:id,... (default: SCENES)')
    ap.add_argument('--sheet', default=None, help='write a contact sheet PNG instead of the assets')
    a = ap.parse_args()
    caps = {(d, i): c for d, i, c in SCENES}
    todo = [tuple(x.split(':')) for x in a.scenes.split(',')] if a.scenes else [(d, i) for d, i, _ in SCENES]
    rows, index = [], []
    for ds, i in todo:
        s = scene(ds, i, a.results, a.data)
        cap = caps.get((ds, i), '')
        if a.sheet:
            rows.append(sheet_row(s, cap))
        else:
            m = export(s, a.out, cap)
            index.append(dict(key=f'{ds}_{i}', dataset=ds, id=i, caption=cap, n_buildings=m['n_buildings'],
                              status_pred=m['status_pred'], status_gt=m['status_gt']))
        print(ds, i, counts(s['buildings'], 'status_gt'), counts(s['buildings'], 'status_pred'), flush=True)
    if a.sheet:
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        Image.fromarray(np.concatenate(rows, 0)).save(a.sheet)
    else:
        json.dump(dict(scenes=index, runs=RUNS, offroad_maxwet_pred=PRED_OFFROAD_MAXWET),
                  open(os.path.join(a.out, 'scenes.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
