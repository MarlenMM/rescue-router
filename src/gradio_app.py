"""Rescue Router Gradio app (PLAN §6.3).

CPU, from the prediction caches (any eval100 photo, the 8 showcase scenes first):
    python src/gradio_app.py [--port 7860] [--share]
Live inference on an uploaded photo (GPU; FloodNet 50-shot checkpoint, same recipe as predict.py):
    python src/gradio_app.py --live --ckpt <ckpt> --mt_ckpt <generalist.ckpt> --chameleon <starter>/chameleon \
        --data <rr_data> --support_file supports/floodnet_stratified_s0.txt [--amp fp16] --share

Controls: map (prediction / labelled map = the router on the human labels), risk-aware truck cost (lam 5 / 0),
boat cost per cell, and the entry point (auto, or click to choose where the rescue team enters).
Click a house: its route (truck white, boat cyan, walk violet dots), a summary, the "unreachable by road" count.
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
from export_web_assets import RUNS, SCENES, build_scene, load_cached, make_grids, scene_scale, tier_stats  # noqa: E402
from viz import COLORS, NAMES, blend  # noqa: E402

K = 2                                                   # display scale (768x576 -> 1536x1152)
STATUS_RGB = {R.TRUCK: (34, 197, 94), R.BOAT: (249, 115, 22), R.FOOT: (167, 139, 250), R.ISOLATED: (239, 68, 68)}
CHECK_RGB = (250, 204, 21)                              # 'probably by truck: check first' (confidence tiers)
CYAN, WHITE, VIOLET = (34, 211, 238), (255, 255, 255), (167, 139, 250)
LEG = {R.TRUCK: ('truck', WHITE), R.BOAT: ('boat', CYAN), R.FOOT: ('on foot', VIOLET)}
TITLE = {R.TRUCK: 'Truck can reach it', R.BOAT: 'Boat needed', R.FOOT: 'Last stretch on foot',
         R.ISOLATED: 'No route: send an air team'}
SIDE = {'pred': 'Prediction', 'gt': 'Labelled map'}
CLASS_WORD = {'floodnet': {2: 'flooded building', 3: 'dry building'},
              'rescuenet': {2: 'intact building', 3: 'damaged building'}}


# ---------- scene state ----------
def solve(st):
    """(Re)build the grids for the current costs and solve both maps from the base."""
    s = st['scene']
    st['grids'] = make_grids(s, lam=5.0 if st['risk'] else 0.0, boat_cost=st['boat'])
    base = st.get('entry') or s['base']
    st['sol'] = {k: g.solve(base) for k, g in st['grids'].items()} if base is not None else {}
    st['status'] = {k: [sol.status(b) for b in s['bld']] for k, sol in st['sol'].items()}
    st['tier'] = [R.tier(a, b) for a, b in zip(st['status']['pred'], st['status']['pred_cautious'])] \
        if st['sol'] else ['ISOLATED'] * len(s['bld'])
    return st


def route_for(st, j):
    """route shown for house j: a confident-truck house gets its route on the cautious map (it avoids land that
    may be wet), every other house its route on the current map"""
    sol = st['sol']['pred_cautious'] if st['src'] == 'pred' and st['tier'][j] == 'CONFIDENT_TRUCK' else st['sol'][st['src']]
    return sol.route(st['scene']['bld'][j]), sol.grid


def new_state(s, title, risk=True, boat=3.0, src='pred'):
    return solve(dict(scene=s, title=title, risk=risk, boat=boat, src=src if s['gt'] is not None else 'pred', sel=None,
                      entry=None, msg=None))


def set_entry(st, x, y):
    """Entry chosen by the user at image pixel (x, y): a 3x3-cell staging area, truck on dry ground, boat on water.
    The labelled map decides what the ground is (the prediction when there is none), as for the auto entry."""
    g = st['grids'].get('gt', st['grids']['pred'])
    cy, cx = min(g.h - 1, max(0, int(y // R.CELL))), min(g.w - 1, max(0, int(x // R.CELL)))
    c = cy * g.w + cx
    if g.wall.flat[c] and not g.boat_ok.flat[c]:
        st['msg'] = 'That is a building: click a road, a yard or water.'
        return st
    mode = R.TRUCK if g.truck_ok.flat[c] else R.BOAT
    ok = g.truck_ok if mode == R.TRUCK else g.boat_ok
    cells = [yy * g.w + xx for yy in range(cy - 1, cy + 2) for xx in range(cx - 1, cx + 2)
             if 0 <= yy < g.h and 0 <= xx < g.w and (ok[yy, xx] or yy * g.w + xx == c)]
    st.update(entry=(np.array(cells), mode), msg=None, sel=None)
    return solve(st)


def cached_state(key, args, **kw):
    ds, i = key.split(':')
    c = load_cached(ds, i, args.results, args.data)
    cap = {(d, j): t for d, j, t in SCENES}.get((ds, i), 'eval100 photo')
    s = build_scene(ds, c['img'], c['probs'], c['decode'], c['gt'])
    s.update(id=i, **scene_scale(dict(dataset=ds, id=i), args.data), tier_stats=tier_stats(ds))
    return new_state(s, f'{ds} {i}: {cap}', **kw)


# ---------- drawing ----------
def render(st):
    s, side = st['scene'], st['src']
    big = np.asarray(Image.fromarray(s['img']).resize((s['img'].shape[1] * K, s['img'].shape[0] * K), Image.LANCZOS))
    cls = s[side].repeat(K, 0).repeat(K, 1)
    im = Image.fromarray(blend(big, cls, s['dataset'], 0.5))
    dr = ImageDraw.Draw(im)
    base = st.get('entry') or s['base']
    if base is not None:
        w = s['cells']['pred'][0].shape[1]
        col = WHITE if base[1] == R.TRUCK else CYAN
        for c in base[0]:
            y, x = divmod(int(c), w)
            dr.rectangle([x * R.CELL * K, y * R.CELL * K, (x + 1) * R.CELL * K - 1, (y + 1) * R.CELL * K - 1], fill=col)
    status = st['status'].get(side, [R.ISOLATED] * len(s['bld']))
    for j, (r, stt) in enumerate(zip(s['buildings'], status)):
        poly = [(x * K, y * K) for x, y in r['polygon']]
        sel = st['sel'] is not None and st['sel'][0] == j
        dr.polygon(poly, outline=CYAN if sel else colour(st, j, stt), width=8 if sel else 4)
    route = st['sel'][1] if st['sel'] is not None else None
    if route is not None and len(route['cells']) > 1:
        pts = [(x * R.CELL * K + R.CELL * K / 2, y * R.CELL * K + R.CELL * K / 2) for y, x in route['cells']]
        modes = route['modes']
        legs = []
        for m in (R.TRUCK, R.BOAT, R.FOOT):                        # one leg per mode, joined at the switch cell
            if m in modes:
                first, last = modes.index(m), len(modes) - 1 - modes[::-1].index(m)
                legs.append((m, pts[max(first - 1, 0):last + 1]))
        for m, leg in legs:
            if len(leg) > 1:
                dr.line(leg, fill=(0, 0, 0), width=12, joint='curve')
        for m, leg in legs:
            if len(leg) > 1 and m != R.FOOT:
                dr.line(leg, fill=LEG[m][1], width=6, joint='curve')
            elif len(leg) > 1:
                for px, py in leg[::2]:                            # walking leg: dotted
                    dr.ellipse([px - 4, py - 4, px + 4, py + 4], fill=VIOLET)
        x0, y0 = pts[0]
        dr.ellipse([x0 - 9, y0 - 9, x0 + 9, y0 + 9], fill=WHITE, outline=(0, 0, 0), width=3)
    for j, (r, stt) in enumerate(zip(s['buildings'], status)):     # badges on top of the route
        x, y = r['centroid'][0] * K, r['centroid'][1] * K
        dr.ellipse([x - 18, y - 18, x + 18, y + 18], fill=colour(st, j, stt), outline=WHITE, width=5)
    return np.asarray(im)


def colour(st, j, stt):
    return CHECK_RGB if st['src'] == 'pred' and st['tier'][j] == 'PROBABLE_TRUCK' else STATUS_RGB[stt]


def counter_md(st):
    s = st['scene']
    msg = f"  \n*{st['msg']}*" if st.get('msg') else ''
    entry = 'chosen' if st.get('entry') else 'auto (largest road or dry land at the edge)'
    if not st['sol']:
        return '**No entry point** at the image border: choose one.' + msg
    M = len(s['bld'])

    def line(side):
        v = st['status'][side]
        n = {k: sum(x == k for x in v) for k in range(len(R.STATUS))}
        none = f" · {n[R.ISOLATED]} no route" if n[R.ISOLATED] else ''
        check = sum(t == 'PROBABLE_TRUCK' for t in st['tier']) if side == 'pred' else 0
        return (f"**{SIDE[side]}: unreachable by road {M - n[R.TRUCK]} / {M}** · {n[R.BOAT]} by boat · {n[R.FOOT]} on foot{none}"
                + (f" · **{check} more look truck-reachable: check first**" if check else ''))
    out = [line(st['src'])]
    other = 'gt' if st['src'] == 'pred' else 'pred'
    if other in st['status']:
        out.append(line(other))
    return '  \n'.join(out) + f'  \nEntry point: {entry}' + msg


def legend_md(ds):
    sw = ' · '.join(f'<span style="color:rgb{COLORS[ds][c]}">■</span> {n}' for c, n in enumerate(NAMES[ds]))
    return (f'{sw}<br><span style="color:#22c55e">●</span> truck · <span style="color:#f97316">●</span> boat needed · '
            '<span style="color:#a78bfa">●</span> last stretch on foot · <span style="color:#ef4444">●</span> no route (air team) · '
            'lines: white truck, cyan boat, violet dots on foot · squares = entry point')


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
    route, grid = st['sel'][1], st['sel'][2]
    tr = st['tier'][j] if side == 'pred' else None
    ts, pct = s.get('tier_stats'), (lambda v: f'{100 * v:.0f} %')
    title = {'CONFIDENT_TRUCK': 'Truck can reach it: confident',
             'PROBABLE_TRUCK': 'Probably reachable by truck: check first'}.get(tr, TITLE[stt])
    out = [f'### House {j}: {title}']
    if tr == 'CONFIDENT_TRUCK' and ts:
        out.append(f'Safe route: avoids land that may be wet. On the test photos {pct(ts["confident"])} of such houses '
                   f'were truly truck-reachable.')
    elif tr == 'PROBABLE_TRUCK' and ts:
        out.append(f'**The only truck route crosses land that may be wet.** On the test photos only {pct(ts["probable"])} '
                   f'of such houses were truly truck-reachable (confident tier: {pct(ts["confident"])}). Confirm first.')
    if route is not None:
        e = R.eta(route, grid, s.get('m_per_cell'))
        km = (lambda v: f'{v / 1000:.1f} km' if v >= 1000 else f'{round(v / 10) * 10:.0f} m')
        out.append('≈ **%d min** · %s' % (max(1, round(e['minutes'])),
                   ' → '.join(f'{LEG[m][0]} {km(v)}' for m, v in enumerate(e['metres']) if v > 0) or 'at the entry point'))
        steps = np.bincount(route['modes'][1:], minlength=3)
        legs = ' → '.join(f'{LEG[m][0]} {n}' for m, n in enumerate(steps) if n) or 'the house is at the entry point'
        out.append(f'{legs} cells · route cost **{route["cost"]:.0f}** (truck on road 1 per cell, yards ×{R.DEFAULTS["offroad"]:g}; '
                   f'boat {st["boat"]:g} + launch {R.DEFAULTS["launch"]:g}; on foot {R.DEFAULTS["foot_cost"]:g} + '
                   f'{R.DEFAULTS["alight"]:g} to get out)')
    else:
        out.append('No truck, boat or walking route from the entry point on this map.')
    word = {R.TRUCK: 'truck', R.BOAT: 'boat', R.FOOT: 'on foot', R.ISOLATED: 'no route'}
    if other in st['status']:
        so = st['status'][other][j]
        out.append(f'{SIDE[other]}: **{word[so]}**' + (' ✓' if so == stt else ' ✗ (disagrees)'))
    out.append(f'Label: {CLASS_WORD[s["dataset"]][s["bld"][j]["cls"]]}' if s['gt'] is not None
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
        return show_state(cached_state(key, args, risk=risk, boat=boat, src='gt' if src == 'Labelled map' else 'pred'))

    def on_param(st, risk, boat, src):
        st.update(risk=risk, boat=boat, src='gt' if src == 'Labelled map' and st['scene']['gt'] is not None else 'pred')
        st = solve(st)
        if st['sel'] is not None and st['sol']:
            j = st['sel'][0]
            st['sel'] = (j, *route_for(st, j))
            return st, render(st), counter_md(st), route_md(st, j)
        return st, render(st), counter_md(st), ''

    def on_click(st, action, evt: gr.SelectData):
        x, y = evt.index[0] / K, evt.index[1] / K
        if action == 'Set the entry point':
            st = set_entry(st, x, y)
            return st, render(st), counter_md(st), ''
        j = pick(st, x, y)
        if j is None or not st['sol']:
            return st, render(st), counter_md(st), ''
        st['sel'] = (j, *route_for(st, j))
        return st, render(st), counter_md(st), route_md(st, j)

    def on_auto(st):
        st.update(entry=None, msg=None, sel=None)
        st = solve(st)
        return st, render(st), counter_md(st), ''

    def on_upload(img, risk, boat):
        if img is None:
            return [gr.skip()] * 5
        if live is None:
            raise gr.Error('Live inference needs a GPU: start the app with --live in the notebook (Kaggle/Colab). '
                           'Locally, pick a scene from the cache.')
        photo = prep_upload(img)
        s = build_scene('floodnet', photo, live(photo), dec50, gt=None)
        s.update(tier_stats=tier_stats('floodnet'))
        return show_state(new_state(s, 'your photo (FloodNet 50-shot, no ground truth)', risk=risk, boat=boat))

    with gr.Blocks(title='Rescue Router') as demo:
        gr.Markdown('# Rescue Router\nFlood map of a drone photo by **Chameleon** adapted from **50 labelled photos** '
                    '(FloodNet; RescueNet from 20), then truck/boat rescue routes on the predicted map.')
        st = gr.State()
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                scene = gr.Dropdown(show + more, value=show[0][1], label='Scene (8 showcase scenes first, then all of eval100)')
                upload = gr.Image(type='numpy', sources=['upload'], label='…or upload a drone photo (live inference, GPU)')
                src = gr.Radio(['Prediction', 'Labelled map'], value='Prediction',
                               label='Map (labelled map = the same router on the human labels)')
                action = gr.Radio(['Route to a house', 'Set the entry point'], value='Route to a house',
                                  label='Clicking the photo will…')
                auto = gr.Button('Auto entry point (largest road or dry land at the edge)', size='sm')
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
        view.select(on_click, [st, action], [st, view, counter, info])
        auto.click(on_auto, [st], [st, view, counter, info])
        upload.upload(on_upload, [upload, risk, boat], outs)
    return demo


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
