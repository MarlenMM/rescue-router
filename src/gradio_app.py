"""Rescue Router Gradio app (PLAN §6.3).

CPU, from the prediction caches (any eval100 photo, the 8 showcase scenes first):
    python src/gradio_app.py [--port 7860] [--share]
Live inference on an uploaded photo (GPU; FloodNet 50-shot checkpoint, same recipe as predict.py):
    python src/gradio_app.py --live --ckpt <ckpt> --mt_ckpt <generalist.ckpt> --chameleon <starter>/chameleon \
        --data <rr_data> --support_file supports/floodnet_stratified_s0.txt [--amp fp16] --share

Controls: map (prediction / ground truth), risk-aware truck cost (lam 5 / 0), boat cost per cell.
Click a house: its route (truck leg white, boat leg cyan), a summary, and the "unreachable by road" count.
Routing is router.py with the same settings as route_eval.py (off-road gate on the predicted map only).
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import router as R  # noqa: E402
from export_web_assets import RUNS, SCENES, build_scene, load_cached, make_grids  # noqa: E402
from viz import COLORS, NAMES, blend  # noqa: E402

K = 2                                                   # display scale (768x576 -> 1536x1152)
STATUS_RGB = {R.TRUCK: (34, 197, 94), R.BOAT: (249, 115, 22), R.ISOLATED: (239, 68, 68)}
CYAN, WHITE = (34, 211, 238), (255, 255, 255)
CLASS_WORD = {'floodnet': {2: 'flooded building', 3: 'dry building'},
              'rescuenet': {2: 'intact building', 3: 'damaged building'}}


# ---------- scene state ----------
def solve(st):
    """(Re)build the grids for the current costs and solve both maps from the base."""
    s = st['scene']
    st['grids'] = make_grids(s, lam=5.0 if st['risk'] else 0.0, boat_cost=st['boat'])
    st['sol'] = {k: g.solve(s['base']) for k, g in st['grids'].items()} if s['base'] is not None else {}
    st['status'] = {k: [sol.status(b) for b in s['bld']] for k, sol in st['sol'].items()}
    return st


def new_state(s, title, risk=True, boat=3.0, src='pred'):
    return solve(dict(scene=s, title=title, risk=risk, boat=boat, src=src if s['gt'] is not None else 'pred', sel=None))


def cached_state(key, args, **kw):
    ds, i = key.split(':')
    c = load_cached(ds, i, args.results, args.data)
    cap = {(d, j): t for d, j, t in SCENES}.get((ds, i), 'eval100 photo')
    return new_state(build_scene(ds, c['img'], c['probs'], c['decode'], c['gt']), f'{ds} {i}: {cap}', **kw)


# ---------- drawing ----------
def render(st):
    s, side = st['scene'], st['src']
    big = np.asarray(Image.fromarray(s['img']).resize((s['img'].shape[1] * K, s['img'].shape[0] * K), Image.LANCZOS))
    cls = s[side].repeat(K, 0).repeat(K, 1)
    im = Image.fromarray(blend(big, cls, s['dataset'], 0.5))
    dr = ImageDraw.Draw(im)
    if s['base'] is not None:
        w = s['cells']['pred'][0].shape[1]
        col = WHITE if s['base'][1] == R.TRUCK else CYAN
        for c in s['base'][0]:
            y, x = divmod(int(c), w)
            dr.rectangle([x * R.CELL * K, y * R.CELL * K, (x + 1) * R.CELL * K - 1, (y + 1) * R.CELL * K - 1], fill=col)
    status = st['status'].get(side, [R.ISOLATED] * len(s['bld']))
    for j, (r, stt) in enumerate(zip(s['buildings'], status)):
        poly = [(x * K, y * K) for x, y in r['polygon']]
        sel = st['sel'] is not None and st['sel'][0] == j
        dr.polygon(poly, outline=CYAN if sel else STATUS_RGB[stt], width=8 if sel else 4)
    route = st['sel'][1] if st['sel'] is not None else None
    if route is not None:
        pts = [(x * R.CELL * K + R.CELL * K / 2, y * R.CELL * K + R.CELL * K / 2) for y, x in route['cells']]
        k = route['modes'].index(R.BOAT) if R.BOAT in route['modes'] else len(pts)
        legs = [(pts[:max(k, 1)], WHITE), (pts[max(k - 1, 0):] if k < len(pts) else [], CYAN)]
        for leg, _ in legs:
            if len(leg) > 1:
                dr.line(leg, fill=(0, 0, 0), width=12, joint='curve')
        for leg, col in legs:
            if len(leg) > 1:
                dr.line(leg, fill=col, width=6, joint='curve')
        x0, y0 = pts[0]
        dr.ellipse([x0 - 9, y0 - 9, x0 + 9, y0 + 9], fill=WHITE, outline=(0, 0, 0), width=3)
    for r, stt in zip(s['buildings'], status):                     # badges on top of the route
        x, y = r['centroid'][0] * K, r['centroid'][1] * K
        dr.ellipse([x - 18, y - 18, x + 18, y + 18], fill=STATUS_RGB[stt], outline=WHITE, width=5)
    return np.asarray(im)


def counter_md(st):
    s = st['scene']
    if s['base'] is None:
        return '**No entry point** at the image border on this map.'
    M = len(s['bld'])

    def line(side, label):
        v = st['status'][side]
        road, boat = sum(x != R.TRUCK for x in v), sum(x == R.BOAT for x in v)
        return f'**{label}: unreachable by road {road} / {M}** · {boat} by boat · {road - boat} isolated'
    out = [line(st['src'], 'Prediction' if st['src'] == 'pred' else 'Ground truth')]
    if st['src'] == 'pred' and 'gt' in st['status']:
        out.append(line('gt', 'Ground truth'))
    return '  \n'.join(out)


def legend_md(ds):
    sw = ' · '.join(f'<span style="color:rgb{COLORS[ds][c]}">■</span> {n}' for c, n in enumerate(NAMES[ds]))
    return (f'{sw}<br><span style="color:#22c55e">●</span> truck · <span style="color:#f97316">●</span> truck, then boat · '
            '<span style="color:#ef4444">●</span> isolated · white line = truck leg, cyan = boat leg, squares = entry points')


def pick(st, x, y):
    """Building index under the click (inside its polygon, else the nearest badge within 20 px)."""
    from matplotlib.path import Path
    s = st['scene']
    for j, r in enumerate(s['buildings']):
        if Path(r['polygon']).contains_point((x, y)):
            return j
    d = [np.hypot(r['centroid'][0] - x, r['centroid'][1] - y) for r in s['buildings']]
    return int(np.argmin(d)) if d and min(d) * K <= 24 else None


def route_md(st, j):
    s, side = st['scene'], st['src']
    other = 'gt' if side == 'pred' else 'pred'
    stt = st['status'][side][j]
    route = st['sel'][1]
    title = {R.TRUCK: 'Truck can reach it', R.BOAT: 'Truck, then boat', R.ISOLATED: 'Isolated: send a foot or air team'}[stt]
    out = [f'### House {j}: {title}']
    if route is not None:
        k = route['modes'].index(R.BOAT) if R.BOAT in route['modes'] else None
        n_t = len(route['cells']) - 1 if k is None else max(k - 1, 0)
        n_b = 0 if k is None else len(route['cells']) - k
        legs = (f'truck {n_t} cells' if n_t else 'boat launched at the image edge') + (f', boat {n_b} cells' if n_b else '')
        out.append(f'{legs} · route cost **{route["cost"]:.0f}** (truck on road = 1 per cell, '
                   f'yards ×{R.DEFAULTS["offroad"]:g}, boat {st["boat"]:g} per cell + launch {R.DEFAULTS["launch"]:g})')
    else:
        out.append('No truck or boat route from the image edge on this map.')
    if other in st['status']:
        so = st['status'][other][j]
        name = 'Ground truth' if other == 'gt' else 'Prediction'
        out.append(f'{name}: **{R.STATUS[so].lower()}**' + (' ✓' if so == stt else ' ✗ (disagrees)'))
    out.append(f'Labelled {CLASS_WORD[s["dataset"]][s["bld"][j]["cls"]]}' if s['gt'] is not None
               else f'Predicted {CLASS_WORD[s["dataset"]][s["bld"][j]["cls"]]}')
    return '  \n'.join(out)


# ---------- live inference (GPU) ----------
class LiveModel:
    """FloodNet checkpoint loaded once, supports encoded once; __call__(photo) -> uint8 probs (5, 576, 768)."""

    def __init__(self, a):
        import torch
        sys.path.insert(0, a.chameleon)
        from finetune import SUPPORT_SPLIT, setup_chameleon
        from rr.data import build_run_folder, read_ids
        from rr.infer import encode_support
        self.torch, self.amp = torch, a.amp
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)['hyper_parameters']['config']
        supports = read_ids(a.support_file)[:ck.shot]
        run_root = build_run_folder(os.path.join(a.runs, 'gradio_live'), os.path.join(a.data, 'floodnet_r768'),
                                    supports, split_support=SUPPORT_SPLIT['floodnet'])
        su = setup_chameleon(a.chameleon, run_root)
        ov = su.derive_stage2_overrides(a.ckpt, verbose=False)
        ov.update(shot=ck.shot, eval_shot=ck.shot, chunk_size=a.chunk, channel_chunk_size=a.channel_chunk, n_tasks=5,
                  loss_type='bce', img_size=384, base_size=list(ck.base_size))
        self.model, _ = su.load_chameleon(su.build_config(stage=2, dataset='custom', **ov), a.mt_ckpt, a.ckpt, verbose=False)
        self.model.eval()
        print(f'support encoding {encode_support(self.model, amp=a.amp):.1f} s', flush=True)

    def __call__(self, photo):
        from rr.infer import tiled_predict
        X = self.torch.from_numpy(photo).permute(2, 0, 1).float() / 255
        return np.round(tiled_predict(self.model, X, amp=self.amp).numpy() * 255).astype(np.uint8)


def prep_upload(img):
    """any RGB photo -> 768x576 (landscape; portrait photos are rotated), as prep_data.py sizes the datasets"""
    im = Image.fromarray(img).convert('RGB')
    if im.height > im.width:
        im = im.rotate(90, expand=True)
    return np.asarray(im.resize((768, 576), Image.BOX if im.width >= 768 else Image.BICUBIC))


# ---------- UI ----------
def build_ui(args, live=None):
    import gradio as gr
    import json
    dec50 = json.load(open(os.path.join(args.results, 'decode', f'{RUNS["floodnet"]}.json')))['chosen']
    show = [(f'{"FloodNet" if d == "floodnet" else "RescueNet"} {i}: {c}', f'{d}:{i}') for d, i, c in SCENES]
    more = []
    for ds in ('floodnet', 'rescuenet'):
        ids = [l.strip() for l in open(os.path.join(REPO, 'splits', f'{ds}_test_eval100.txt')) if l.strip()]
        more += [(f'{ds} {i}', f'{ds}:{i}') for i in ids if (ds, i) not in {(d, j) for d, j, _ in SCENES}]

    def show_state(st):
        return st, render(st), counter_md(st), legend_md(st['scene']['dataset']), \
            f'**{st["title"]}** · click a house to send a rescue team'

    def on_scene(key, risk, boat, src):
        return show_state(cached_state(key, args, risk=risk, boat=boat, src='gt' if src == 'Ground truth' else 'pred'))

    def on_param(st, risk, boat, src):
        st.update(risk=risk, boat=boat, src='gt' if src == 'Ground truth' and st['scene']['gt'] is not None else 'pred')
        st = solve(st)
        if st['sel'] is not None and st['sol']:
            j = st['sel'][0]
            st['sel'] = (j, route_for(st, j))
            return st, render(st), counter_md(st), route_md(st, j)
        return st, render(st), counter_md(st), ''

    def on_click(st, evt: gr.SelectData):
        x, y = evt.index[0] / K, evt.index[1] / K
        j = pick(st, x, y)
        if j is None or not st['sol']:
            return st, render(st), ''
        st['sel'] = (j, route_for(st, j))
        return st, render(st), route_md(st, j)

    def on_upload(img, risk, boat):
        if img is None:
            return [gr.skip()] * 5
        if live is None:
            raise gr.Error('Live inference needs a GPU: start the app with --live in the notebook (Kaggle/Colab). '
                           'Locally, pick a scene from the cache.')
        photo = prep_upload(img)
        s = build_scene('floodnet', photo, live(photo), dec50, gt=None)
        return show_state(new_state(s, 'your photo (FloodNet 50-shot, no ground truth)', risk=risk, boat=boat))

    with gr.Blocks(title='Rescue Router') as demo:
        gr.Markdown('# Rescue Router\nFlood map of a drone photo by **Chameleon** adapted from **50 labelled photos** '
                    '(FloodNet; RescueNet from 20), then truck/boat rescue routes on the predicted map.')
        st = gr.State()
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                scene = gr.Dropdown(show + more, value=show[0][1], label='Scene (8 showcase scenes first, then all of eval100)')
                upload = gr.Image(type='numpy', sources=['upload'], label='…or upload a drone photo (live inference, GPU)')
                src = gr.Radio(['Prediction', 'Ground truth'], value='Prediction', label='Map')
                risk = gr.Checkbox(True, label='Risk-aware truck (avoid likely water)')
                boat = gr.Slider(1, 6, value=R.DEFAULTS['boat_cost'], step=0.5, label='Boat cost per cell (truck on road = 1)')
                counter = gr.Markdown()
            with gr.Column(scale=3):
                head = gr.Markdown()
                view = gr.Image(label='Click a house', interactive=False, format='jpeg')
                info = gr.Markdown()
                legend = gr.Markdown()
        outs = [st, view, counter, legend, head]
        demo.load(on_scene, [scene, risk, boat, src], outs)
        scene.change(on_scene, [scene, risk, boat, src], outs).then(lambda: '', None, info)
        for c in (risk, boat, src):
            c.change(on_param, [st, risk, boat, src], [st, view, counter, info])
        view.select(on_click, [st], [st, view, info])
        upload.upload(on_upload, [upload, risk, boat], outs)
    return demo


def route_for(st, j):
    """route shown for house j on the current map: the road route for a TRUCK house, else the cheapest"""
    sol, b = st['sol'][st['src']], st['scene']['bld'][j]
    return road_route(sol, b) if st['status'][st['src']][j] == R.TRUCK else sol.route(b)


def road_route(sol, b):
    """cheapest truck-only route to a TRUCK house (the web viewer shows the same, see app/js/router.js)"""
    n, r = sol.grid.n, b['ring']
    j = r[np.argmin(sol.dist[r])]
    nodes = [j]
    while sol.pred[nodes[-1]] >= 0:
        nodes.append(sol.pred[nodes[-1]])
    cells = np.array(nodes[::-1]) % n
    return dict(cells=list(zip(*np.divmod(cells, sol.grid.w))), modes=[R.TRUCK] * len(nodes), cost=float(sol.dist[j]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(REPO, 'results/kaggle_v2/rr'))
    ap.add_argument('--data', default=os.path.expanduser('~/rr_data'))
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--mt_ckpt', default='/content/generalist.ckpt')
    ap.add_argument('--chameleon', default='/content/AIC200-Project-StructuredPrediction/chameleon')
    ap.add_argument('--runs', default='/content/runs_data')
    ap.add_argument('--support_file', default=os.path.join(REPO, 'supports/floodnet_stratified_s0.txt'))
    ap.add_argument('--amp', default='fp16', choices=['none', 'fp16'])
    ap.add_argument('--chunk', type=int, default=1)
    ap.add_argument('--channel_chunk', type=int, default=1)
    a = ap.parse_args()
    a.results, a.data = os.path.abspath(a.results), os.path.abspath(a.data)
    live = LiveModel(a) if a.live else None
    build_ui(a, live).launch(server_name='127.0.0.1', server_port=a.port, share=a.share)


if __name__ == '__main__':
    main()
