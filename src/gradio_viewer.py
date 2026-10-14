"""Rescue Router Gradio app with the web viewer inside it: every viewer visual (zoom, swipe, animated truck -> boat ->
walk, water-probability layer, tiers, house list) for the 200 eval100 photos AND for an uploaded photo run live through
the model. The upload is predicted as in gradio_app.py (same LiveModel, decode, router), written as a viewer scene
(scene.rrb + overlays, src/export_web_assets.py format) and opened in the embedded viewer. The classic Gradio view
(gradio_app.py) is the second tab.

CPU, the cached scenes only (uploads need --live or --fake_live):
    python src/gradio_viewer.py [--port 7860] [--share]
Live inference on uploads (GPU), arguments as gradio_app.py:
    python src/gradio_viewer.py --live --ckpt <ckpt> --mt_ckpt <generalist.ckpt> --chameleon <starter>/chameleon \
        --data <rr_data> [--amp fp16] --share
TEST MODE without a GPU: --fake_live answers an uploaded eval100 FloodNet photo with its cached prediction.

The viewer files come from --viewer (default: docs/ in the repository, or viewer/ in the app zip) and are copied to a
work folder, which Gradio serves (allowed_paths); uploads go to <work>/scenes/up/<token>/, opened as #s=up/<token>.
An upload has no labels: the scene's "labelled map" slot repeats the prediction and the viewer hides everything that
compares with it (meta.upload).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gradio_app as G  # noqa: E402
import router as R  # noqa: E402
from export_web_assets import RUNS, build_scene, make_grids, pack_scene, tier_stats  # noqa: E402
from viz import COLORS, NAMES, colorize  # noqa: E402
from rr.paths import RR_DATA  # noqa: E402

START = 'rescuenet_12195&h=8&m=gt'           # canal row: truck, boat across the canal, then on foot
KEEP = 30                                    # uploaded scenes kept in the work folder


def export_upload(s, d, caption, run):
    """viewer scene folder for an uploaded photo (no labels); returns the Python statuses (TRUCK/BOAT/FOOT/ISOLATED)"""
    os.makedirs(d, exist_ok=True)
    ds = s['dataset']
    g = make_grids(s)                                                # pred (tuned gate) and pred_cautious
    g['gt'] = g['pred']                                              # no labels: the slot repeats the prediction
    s['grids'] = {5.0: g}
    for r, b in zip(s['buildings'], s['bld']):
        r['goals'] = [g['pred'].goal(b, layer).tolist() for layer in (R.TRUCK, R.BOAT, R.FOOT)]
    img = Image.fromarray(s['img'])
    img.save(os.path.join(d, 'photo.jpg'), quality=90)
    img.resize((192, 144), Image.LANCZOS).save(os.path.join(d, 'thumb.jpg'), quality=82)
    over = Image.fromarray(colorize(s['pred'], ds))
    for k in ('pred', 'gt'):
        over.save(os.path.join(d, f'{k}.png'), optimize=True)
    open(os.path.join(d, 'scene.rrb'), 'wb').write(pack_scene(s))
    status = []
    if s['base'] is not None:
        sol = g['pred'].solve(s['base'])
        status = [R.STATUS[sol.status(b)] for b in s['bld']]
    meta = dict(id='upload', dataset=ds, run=run, decode=s['decode'], caption=caption, upload=True,
                size=[int(s['img'].shape[1]), int(s['img'].shape[0])], photo_px=[int(s['img'].shape[1]), int(s['img'].shape[0])],
                class_names=NAMES[ds], class_colors=[COLORS[ds][c] for c in range(5)], iou={}, miou=None,
                n_buildings=len(s['buildings']), status_pred={k: status.count(k) for k in R.STATUS},
                tier_stats=tier_stats(ds), m_per_cell=R.PROFILES[ds]['cell_m'], scale_source='dataset value')
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
    return status


def prepare_viewer(src, work):
    """copy the built viewer to a folder Gradio may serve; uploads are written below it"""
    if os.path.exists(work):
        shutil.rmtree(work)
    shutil.copytree(src, work, ignore=shutil.ignore_patterns('test', 'ref.json', '.DS_Store'))
    os.makedirs(os.path.join(work, 'scenes', 'up'), exist_ok=True)
    return work


def iframe(work, hash_, height=820):
    return (f'<iframe src="/gradio_api/file={work}/index.html#s={hash_}" title="Rescue Router viewer" '
            f'style="width:100%;height:{height}px;border:0;border-radius:12px;background:#0b1220" '
            'allow="clipboard-write"></iframe>')


def build_app(args, live, work):
    import gradio as gr
    dec50 = json.load(open(os.path.join(args.results, 'decode', f'{RUNS["floodnet"]}.json')))['chosen']
    ex = args.examples
    examples = sorted(os.path.join(ex, f) for f in os.listdir(ex) if f.endswith('.jpg')) if os.path.isdir(ex) else []

    def on_upload(img):
        if img is None:
            return gr.skip(), gr.skip()
        if live is None:
            raise gr.Error('Live inference needs a GPU: start the app with --live (Colab/Kaggle T4). '
                           'The 200 test photos in the viewer work without it.')
        notes = []
        photo = G.prep_upload(img, notes)
        try:
            probs = live(photo)
        except ValueError as e:
            raise gr.Error(str(e))
        s = build_scene('floodnet', photo, probs, dec50, gt=None)
        token = uuid.uuid4().hex[:10]
        caption = ('TEST MODE: the cached prediction of eval100 photo %s, not the model' % live.last
                   if isinstance(live, G.FakeLive) else 'Your photo: FloodNet 50-shot model, live (no labels)')
        up = os.path.join(work, 'scenes', 'up')
        status = export_upload(s, os.path.join(up, token), caption, RUNS['floodnet'])
        old = sorted((os.path.getmtime(os.path.join(up, t)), t) for t in os.listdir(up))
        for _, t in old[:-KEEP]:
            shutil.rmtree(os.path.join(up, t), ignore_errors=True)
        n = len(s['buildings'])
        cut = sum(v != 'TRUCK' for v in status)
        msg = (f'**{caption}.** {n} houses found on the predicted map'
               + (f', {cut} not reachable by truck. Click a house in the viewer for its route.' if status else
                  ': no entry point found; choose one in the viewer (Entry point › Choose).')
               + (f'  \nPhoto: {"; ".join(notes)}.' if notes else ''))
        gr.Warning('No labels for an uploaded photo: houses and the entry point come from the prediction.')
        return iframe(work, f'up/{token}'), msg

    with gr.Blocks(title='Rescue Router') as demo:
        gr.Markdown('# Rescue Router\nFlood map of a drone photo by **Chameleon** adapted from **50 labelled photos** '
                    '(FloodNet; RescueNet from 20), then truck → boat → on-foot rescue routes on the predicted map.')
        with gr.Tabs():
            with gr.Tab('Viewer'):
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        upload = gr.Image(type='numpy', sources=['upload'], height=220,
                                          label='Upload a drone photo (live inference, GPU)')
                        if examples:
                            gr.Examples(examples, inputs=upload, label='Test photos the model never saw')
                        status = gr.Markdown(f'<small>{G.UPLOAD_NOTE}</small>')
                    with gr.Column(scale=4):
                        view = gr.HTML(iframe(work, START))
                upload.change(on_upload, [upload], [view, status])
            with gr.Tab('Classic view'):
                G.build_ui(args, live).render()
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(REPO, 'results/kaggle_v2/rr'))
    ap.add_argument('--data', default=RR_DATA)
    ap.add_argument('--viewer', default=next((p for p in (os.path.join(REPO, 'docs'), os.path.join(REPO, 'viewer'))
                                              if os.path.exists(os.path.join(p, 'index.html'))), None))
    ap.add_argument('--work', default=os.path.join(tempfile.gettempdir(), 'rr_viewer'))
    ap.add_argument('--examples', default=os.path.join(REPO, 'demo', 'uploads'),
                    help='folder of example photos (with --fake_live: eval100 FloodNet photos)')
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--fake_live', action='store_true', help='TEST MODE: uploads of eval100 photos get their cache')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--mt_ckpt', default='/content/generalist.ckpt')
    ap.add_argument('--chameleon', default='/content/AIC200-Project-StructuredPrediction/chameleon')
    ap.add_argument('--runs', default='/content/runs_data')
    ap.add_argument('--support_file', default=os.path.join(REPO, 'supports/floodnet_stratified_s0.txt'))
    ap.add_argument('--amp', default='fp16', choices=['none', 'fp16'])
    ap.add_argument('--chunk', type=int, default=1)
    ap.add_argument('--channel_chunk', type=int, default=1)
    a = ap.parse_args()
    assert a.viewer, 'built viewer not found: run src/build_viewer.py (docs/) or pass --viewer'
    a.results, a.data = os.path.abspath(a.results), os.path.abspath(a.data)
    work = prepare_viewer(a.viewer, os.path.abspath(a.work))
    live = G.LiveModel(a) if a.live else G.FakeLive(a) if a.fake_live else None
    build_app(a, live, work).launch(server_name='127.0.0.1', server_port=a.port, share=a.share, allowed_paths=[work])


if __name__ == '__main__':
    main()
