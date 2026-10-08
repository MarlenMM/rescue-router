"""Talk deck with the demo video (8 Oct): slides/deck_base_b13.pptx -> slides/20260876_MarlenMelis_slides.pptx.

    <python with python-pptx, qrcode, pillow> slides/make_final_deck.py

Same 7 slides (title + 6), same design. Added: the 90-s demo video on the Track B slide (its numbers become stat
cards, plus a QR code to the public demo), a QR code on the title, a photo -> flood map -> routes strip on the problem
slide, prediction next to the label on the Track A slide and on the failure case. Speaker notes rewritten for 5 min
with the video (about 760 words). Needs ffmpeg for the video stills. The base file is not modified.
"""
import io
import os
import subprocess

import qrcode
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(REPO, 'slides', 'assets')
VIDEO = os.path.join(REPO, 'slides', 'rescue_router_demo.mp4')
DEMO = 'https://marlenmm.github.io/rescue-router-demo/'
BLUE, ORANGE, INK, MUTED, LINE = '1F6FB2', 'C2410C', '1A202C', '4A5568', 'E3E1DA'
HEAD, BODY = 'Rubik', 'IBM Plex Sans'


# ---------- assets ----------
def frame(t, path, crop=None):
    """A still of the demo video at t seconds (optionally cropped: x, y, w, h in 1920x1080 pixels)."""
    vf = [f'crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}'] if crop else []
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(t), '-i', VIDEO, '-frames:v', '1', '-q:v', '2']
                   + (['-vf', ','.join(vf)] if vf else []) + [path], check=True)
    return path


def overlay(key, layer, path, alpha=0.6):
    """The viewer's look: the class overlay (pred.png / gt.png) at 60 % over the photo."""
    d = os.path.join(REPO, 'docs', 'scenes', key)
    photo = Image.open(os.path.join(d, 'photo.jpg')).convert('RGB').resize((768, 576), Image.LANCZOS)
    ov = Image.open(os.path.join(d, layer)).convert('RGBA')
    a = ov.getchannel('A').point(lambda v: int(v * alpha))
    photo.paste(ov.convert('RGB'), (0, 0), a)
    photo.save(path, quality=90)
    return path


def qr(path):
    q = qrcode.QRCode(border=2, box_size=12, error_correction=qrcode.constants.ERROR_CORRECT_M)
    q.add_data(DEMO)
    q.make(fit=True)
    q.make_image(fill_color='#0b1220', back_color='white').save(path)
    return path


def make_assets():
    os.makedirs(A, exist_ok=True)
    m = (494, 239, 872, 654)                                             # 7577's map area in the video (4:3)
    return dict(
        strip_photo=frame(0.3, os.path.join(A, 'strip_7577_photo.jpg'), m),
        strip_map=frame(3.3, os.path.join(A, 'strip_7577_map.jpg'), m),
        strip_route=frame(14.0, os.path.join(A, 'strip_7577_route.jpg'), m),
        poster=frame(13.6, os.path.join(A, 'video_poster.jpg')),
        pa_pred=overlay('floodnet_7320', 'pred.png', os.path.join(A, 'pair_7320_pred.jpg')),
        pa_gt=overlay('floodnet_7320', 'gt.png', os.path.join(A, 'pair_7320_gt.jpg')),
        qr=qr(os.path.join(A, 'qr_demo.png')),
    )


# ---------- drawing helpers ----------
def text(slide, x, y, w, h, runs, size=14, color=MUTED, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP):
    """runs: str or [(text, {bold, color, size})]."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    for t, o in ([(runs, {})] if isinstance(runs, str) else runs):
        r = p.add_run()
        r.text = t
        f = r.font
        f.name = o.get('font', font)
        f.size = Pt(o.get('size', size))
        f.bold = o.get('bold', bold)
        f.color.rgb = RGBColor.from_string(o.get('color', color))
    return tb


def card(slide, x, y, w, h, fill='FFFFFF', line=LINE):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.08
    s.fill.solid()
    s.fill.fore_color.rgb = RGBColor.from_string(fill)
    s.line.color.rgb = RGBColor.from_string(line)
    s.line.width = Pt(0.5)
    s.shadow.inherit = False
    return s


def stat(slide, x, y, w, h, big, small, color=BLUE):
    card(slide, x, y, w, h)
    text(slide, x + 0.2, y + 0.13, w - 0.4, 0.5, big, size=28, color=color, font=HEAD, bold=True)
    text(slide, x + 0.2, y + 0.66, w - 0.4, h - 0.72, small, size=12, color=MUTED)


def picture(slide, path, x, y, w, h=None, label=None, label_color='FFFFFF'):
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h) if h else None)
    if label:                                                            # a dark pill in the top-left corner
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + 0.07), Inches(y + 0.07),
                                      Inches(0.2 + 0.082 * len(label)), Inches(0.26))
        pill.adjustments[0] = 0.5
        pill.fill.solid()
        pill.fill.fore_color.rgb = RGBColor.from_string('0B1220')
        pill.fill.fore_color.brightness = 0
        pill.line.fill.background()
        pill.shadow.inherit = False
        tf = pill.text_frame
        tf.word_wrap = False
        tf.margin_left = tf.margin_right = Inches(0.06)
        tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        r = tf.paragraphs[0].add_run()
        r.text = label
        r.font.size, r.font.bold, r.font.name = Pt(10), True, BODY
        r.font.color.rgb = RGBColor.from_string(label_color)
    return pic


def arrow(slide, x, y, w=0.26, h=0.26, color=BLUE):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = RGBColor.from_string(color)
    s.line.fill.background()
    s.shadow.inherit = False


def remove(slide, names):
    for sh in list(slide.shapes):
        if sh.name in names:
            sh._element.getparent().remove(sh._element)


def shape(slide, name):
    return next(sh for sh in slide.shapes if sh.name == name)


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


# ---------- speaker notes (about 760 words: 5 min with the 90-s video) ----------
NOTES = [
    "Hello, I'm Marlen. My project is Rescue Router. I adapted Chameleon to map flooded roads and buildings in drone "
    "photos from only 50 labelled photos — that's Track A — and built a system that plans rescue routes by truck, boat "
    "and on foot on top of that map — that's Track B. The QR code opens the live demo, if you want to follow along.",

    "August 2022: Seoul had 141.5 millimetres of rain in one hour, the most since 1907. July 2023: the Osong underpass "
    "flooded in minutes and 14 people died. In the first hours, coordinators need one answer per house: can a truck get "
    "there, does it need a boat, or is it cut off? Drones deliver photos within the hour, but labelling one photo takes "
    "about an hour, and a standard model wants 1,445 labelled photos. So my question: how far do 50 labelled photos get "
    "us, and can that map route rescue teams? The strip at the bottom is the whole system: photo, flood map, route. The "
    "answers on the right: .507 mIoU on all 448 test photos — PSPNet, with 29 times more labels, gets .848 — cut-off "
    "houses found with F1 .752, and all 208 houses marked “confident: truck” were truly reachable.",

    "How I formulated it. Every pixel gets one of five channels: clear road, flooded road, flooded building, dry building "
    "and water; trees, grass and cars are “other”. Chameleon predicts one probability map per channel, decoded with a "
    "threshold per channel tuned on 30 validation photos, on six overlapping 384-pixel tiles. I chose the channels for "
    "the router, which works on 4-by-4-pixel cells: truck on clear road, boat on water, the last stretch on foot. I "
    "rejected all ten FloodNet classes, which spend the few labels on trees, a single “passable” channel, which can't "
    "plan a boat, and a cost map, which has no labels.",

    "Data: FloodNet, from Hurricane Harvey — I adapt on 1 to 50 labelled photos, tune on 30 and test on all 448. "
    "RescueNet, from Hurricane Michael, is a second disaster. Every label is the datasets' own. Weights: the course's "
    "Chameleon checkpoint; fine-tuning touches only the biases, the temperature and one LayerNorm, five megabytes per "
    "model, plus SAM to refine building outlines. 28 runs, about 29 GPU-hours on free Kaggle T4s, and the notebook "
    "reruns everything in about 28 minutes and reproduces the stored predictions exactly.",

    "Track A. The curve is mIoU against the number of labelled photos, with 95 percent intervals. One to five photos: "
    "plus .14, a real gain. Five to ten: nothing measurable. Twenty to fifty: plus .10 again. The final 50-shot model "
    "gets .507 on the full test, up from .472 after 36 tested ideas. On the right, a test photo: the prediction next to "
    "the human label. The gap to PSPNet is mostly flooded roads, .31 against .82: brown water and a dry road look alike "
    "from above. And a new hurricane needs its own labels: zero RescueNet labels give .32, twenty give .63.",

    "[Click the video: 90 seconds, no sound.] Track B: this is the real system. The drone photo becomes the five-class "
    "map, and every house gets a status badge. I click a house: the truck drives while the roads are dry, a boat takes "
    "over, the last stretch is on foot, with a time estimate. Now the human labels: here the same house is reachable on "
    "a dry road — my model was too cautious; the swipe compares both maps, and the dashed line is the other map's route. "
    "The team can choose where it enters, and the risk-aware truck avoids roads that are probably wet. A second "
    "hurricane: truck, boat across the canal, then on foot. Confidence tiers: houses marked “confident: truck” were all "
    "truly reachable, 208 of 208. The honest failure: shallow water read as land — but those houses are flagged “check "
    "first”, and the water-probability layer shows why. You can browse all 200 test photos of both hurricanes. And the "
    "Gradio app takes an upload — shown in test mode; live on a T4 it takes 17 seconds. Overall, cut-off houses are "
    "found with F1 .752.",

    "What I learned. The dangerous error is shallow water read as dry land: the map invites a truck into water. Left the "
    "prediction, right the label: the model saw this flooded street as land. But all six houses it wrongly calls "
    "truck-reachable are flagged “check first”, because every map is routed twice, once cautiously. A rule from this "
    "analysis — no off-road driving where water is likely — raised cut-off F1 from .74 to .84. So saying how sure we "
    "are matters more than a few IoU points. Of 36 ideas, four survived: a shorter schedule, weight averaging, an "
    "edge-aware decode and SAM. Limits: US hurricanes, not Korean city floods; single photos without GPS; and the "
    "labelled map is my router on human labels, not field truth. Thank you.",
]


def build(src, out):
    a = make_assets()
    prs = Presentation(src)
    s = prs.slides

    # title: QR code to the public demo
    t = s[0]
    card(t, 0.89, 5.86, 1.18, 1.18, fill='FFFFFF', line='FFFFFF')
    t.shapes.add_picture(a['qr'], Inches(0.93), Inches(5.9), Inches(1.1), Inches(1.1))
    text(t, 2.2, 6.06, 4.4, 0.8, [('Try it live on your phone\n', {'bold': True, 'color': 'FFFFFF', 'size': 14}),
                                 ('marlenmm.github.io/rescue-router-demo', {'color': '7DD3FC', 'size': 12})])

    # 1/6 problem: photo -> flood map -> routes
    p = s[1]
    w, h, y, g = 1.93, 1.45, 5.18, 0.3
    for k, (key, lab) in enumerate([('strip_photo', 'drone photo'), ('strip_map', 'flood map (50 labels)'),
                                    ('strip_route', 'truck → boat → on foot')]):
        x = 0.89 + k * (w + g)
        picture(p, a[key], x, y, w, h, label=lab)
        if k < 2:
            arrow(p, x + w + 0.03, y + h / 2 - 0.12, 0.24, 0.24)

    # 4/6 Track A: prediction next to the label (a good test photo; the failure is on 6/6)
    ta = s[4]
    picture(ta, a['pa_pred'], 7.94, 5.14, 2.0, 1.5, label='prediction, 50 labels')
    picture(ta, a['pa_gt'], 10.1, 5.14, 2.0, 1.5, label='human label')
    text(ta, 7.94, 6.66, 4.2, 0.2, 'test photo 7320 · mean IoU .84 on this photo', size=9)

    # 5/6 Track B: the video, numbers as stat cards, QR code
    tb = s[5]
    remove(tb, {'Image 0', 'Image 1', 'Image 2', 'Text 1', 'Text 2', 'Text 3', 'Shape 4', 'Shape 6', 'Shape 8',
                'Text 5', 'Text 7', 'Text 9'})
    vw = 7.15
    tb.shapes.add_movie(VIDEO, Inches(0.89), Inches(1.63), Inches(vw), Inches(vw * 9 / 16), poster_frame_image=a['poster'],
                        mime_type='video/mp4')
    text(tb, 0.89, 1.63 + vw * 9 / 16 + 0.08, vw, 0.3,
         [('The real system, 90 s: ', {'bold': True, 'color': INK}),
          ('web map of all 200 test photos with live routing, then the Gradio app (upload shown in CPU test mode)', {})],
         size=11)
    x0, cw = 8.3, 4.15
    stat(tb, x0, 1.63, cw, 1.12, '.752 F1', 'cut-off houses found on 1,232 test houses (RescueNet .818)')
    stat(tb, x0, 2.87, cw, 1.12, '208 / 208', '“confident: truck” houses truly reachable; “check first”: 79 %',
         color=ORANGE)
    stat(tb, x0, 4.11, cw, 1.12, '17 s', 'photo upload → flood map + routes, live on a free T4 GPU')
    card(tb, x0, 5.35, cw, 1.3)
    tb.shapes.add_picture(a['qr'], Inches(x0 + 0.12), Inches(5.43), Inches(1.14), Inches(1.14))
    text(tb, x0 + 1.4, 5.55, cw - 1.55, 1.0, [('Try it now\n', {'bold': True, 'color': INK, 'size': 14}),
                                             ('marlenmm.github.io/\nrescue-router-demo', {'color': BLUE, 'size': 12})])

    # 6/6 learned: the failure, prediction next to the label
    le = s[6]
    img = shape(le, 'Image 0')
    img._element.getparent().remove(img._element)
    picture(le, os.path.join(A, 'fail_7241_pred.jpg'), 0.89, 1.68, 2.2, 1.65, label='prediction')
    picture(le, os.path.join(A, 'fail_7241_gt.jpg'), 3.13, 1.68, 2.2, 1.65, label='human label')
    cap = shape(le, 'Text 1')
    cap.top, cap.width = Inches(3.43), Inches(4.44)

    for sl, n in zip(s, NOTES):
        notes(sl, n)
    prs.save(out)
    print('wrote', out, f'{os.path.getsize(out) / 1e6:.1f} MB')


if __name__ == '__main__':
    build(os.path.join(REPO, 'slides', 'deck_base_b13.pptx'), os.path.join(REPO, 'slides', '20260876_MarlenMelis_slides.pptx'))
