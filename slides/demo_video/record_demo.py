"""Record the 90-s demo video (PLAN §15 0z): slides/rescue_router_demo.mp4.

Scripted headless Chromium (Playwright) drives the real web viewer (docs/, served on :8766) and the Gradio app in
CPU test mode (src/gradio_app.py --fake_live on :7861). Frames come from the Chrome DevTools screencast (JPEG, with
timestamps), so the video keeps the real timing; ffmpeg turns them into 30 fps H.264. The caption bar and the cursor
are injected into the page (DOM), nothing else on the page is changed: the map area simply ends above the caption bar.

    python3 -m http.server 8766 --bind 127.0.0.1 --directory docs &
    ~/.venvs/rr/bin/python src/gradio_app.py --port 7861 --fake_live &
    <python with playwright> slides/demo_video/record_demo.py --out slides/rescue_router_demo.mp4

--only viewer|gradio records one part (frames are kept in --work, so the other part is reused); --shots writes a
still per scene into --work/shots to check that each scene shows what its caption says.
"""
import argparse
import asyncio
import base64
import json
import math
import os
import shutil
import subprocess
import tempfile
import time

from playwright.async_api import async_playwright

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIEWER = 'http://127.0.0.1:8766/'
GRADIO = 'http://127.0.0.1:7861/'
W, H, DSF = 1280, 720, 1.5            # CSS px; 1.5x = 1920x1080 frames with readable UI text
GW, GH = 1120, 630                    # Gradio: narrower, so its result map fits on screen (x 1.714 = 1920x1080)
CAP_H = 60                            # caption bar height (CSS px)
UPLOAD = os.path.expanduser('~/rr_data/floodnet_r1024/test/images/7476.jpg')   # eval100 photo (test mode knows it)

# scene: (start s, caption). Viewer part 0-74 s, Gradio part 74-90 s.
CAPTIONS = [
    (0, 'Drone photo → 5-class flood map (Chameleon, 50 labelled photos)'),
    (7, 'Click a house: truck while roads are dry, then boat, then on foot'),
    (19, 'Human labels find a dry road: the prediction was too cautious here'),
    (30, 'Choose where the team enters; avoid probably-wet roads'),
    (37, 'Second hurricane: truck → boat → on foot'),
    (46, 'Confidence tiers: “confident truck” houses were all truly reachable'),
    (53, 'Shallow water read as land, but those houses are flagged “check first”'),
    (64, 'All 200 test photos from two hurricanes'),
    (74, 'Gradio app: upload a photo, get map and routes (test mode shown)'),
]
VIEWER_LEN, GRADIO_LEN = 74.0, 16.0

OVERLAY = r"""
(() => {
  const CAP_H = %d;
  const css = `
  #__cap { position: fixed; left: 0; right: 0; bottom: 0; height: ${CAP_H}px; z-index: 2147483600; display: flex;
    align-items: center; justify-content: center; background: #05080f; border-top: 1px solid rgba(148,163,184,.25);
    color: #f8fafc; font: 600 25px/1.2 system-ui, -apple-system, "Segoe UI", sans-serif; letter-spacing: .1px;
    transition: opacity .18s; }
  #__cur { position: fixed; left: 0; top: 0; z-index: 2147483647; pointer-events: none; width: 26px; height: 26px;
    margin: -2px 0 0 -3px; filter: drop-shadow(0 1px 2px rgba(0,0,0,.7)); }
  .__rip { position: fixed; z-index: 2147483646; pointer-events: none; width: 36px; height: 36px; margin: -18px 0 0 -18px;
    border-radius: 50%%; border: 3px solid #22d3ee; background: rgba(34,211,238,.25); animation: __rip .5s ease-out forwards; }
  @keyframes __rip { from { transform: scale(.3); opacity: 1; } to { transform: scale(1.4); opacity: 0; } }
  /* viewer only: everything ends above the caption bar */
  .rrv #map { bottom: ${CAP_H}px !important; }
  .rrv .panel { bottom: ${CAP_H + 10}px !important; }
  .rrv .dock { bottom: ${CAP_H + 26}px !important; }
  .rrv .swipe { bottom: ${CAP_H}px !important; }
  /* gradio only: leave room for the caption bar at the end of the page */
  .rrg body, .rrg gradio-app { padding-bottom: ${CAP_H}px; }
  `;
  function setup() {
    if (document.getElementById('__cap')) return;
    const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
    document.documentElement.classList.add(document.getElementById('map') ? 'rrv' : 'rrg');
    const cap = document.createElement('div'); cap.id = '__cap'; document.body.appendChild(cap);
    const cur = document.createElement('div'); cur.id = '__cur';
    cur.innerHTML = '<svg width="26" height="26" viewBox="0 0 26 26"><path d="M3 2 L3 21.5 L8.3 16.6 L11.8 24.5 L15.6 22.9 L12.1 15.2 L19.2 15.2 Z" fill="#fff" stroke="#000" stroke-width="1.7" stroke-linejoin="round"/></svg>';
    document.body.appendChild(cur);
    const at = window.__curAt || [W0 / 2, H0 / 2];
    cur.style.transform = `translate(${at[0]}px, ${at[1]}px)`;
    addEventListener('mousemove', (e) => { cur.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`; window.__curAt = [e.clientX, e.clientY]; }, true);
    addEventListener('mousedown', (e) => {
      const r = document.createElement('div'); r.className = '__rip';
      r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
      document.body.appendChild(r); setTimeout(() => r.remove(), 600);
    }, true);
    window.__cap = (t) => { cap.textContent = t; };
  }
  const W0 = %d, H0 = %d;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', setup); else setup();
})();
""" % (CAP_H, W, H)

# viewer helpers: screen point of a house badge / an image pixel
HELPERS = r"""
window.__imgPt = (x, y) => { const s = rr.state.s, p = rr.map.latLngToContainerPoint(L.latLng(s.H - y, x)),
  r = document.getElementById('map').getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; };
window.__housePt = (id) => { const b = rr.state.s.blds.find(b => b.id === id); return __imgPt(...b.centroid); };
// framing for the video: fit the photo into the area left free by the title, the route card and the panel
(() => { const m = rr.map, fit = m.fitBounds.bind(m);
  m.options.zoomSnap = 0.05;
  m.fitBounds = (b) => fit(b, { paddingTopLeft: [324, document.querySelector('.title').getBoundingClientRect().bottom + 10],
                               paddingBottomRight: [364, 14] });
  m.fitBounds([[0, 0], [rr.state.s.H, rr.state.s.W]]); })();
0;
"""


class Recorder:
    """Chrome DevTools screencast -> numbered JPEGs + wall-clock timestamps."""

    def __init__(self, page, outdir):
        self.page, self.dir, self.frames = page, outdir, []
        shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir)

    async def start(self):
        self.cdp = await self.page.context.new_cdp_session(self.page)
        self.cdp.on('Page.screencastFrame', self._frame)
        await self.cdp.send('Page.startScreencast', {'format': 'jpeg', 'quality': 94, 'maxWidth': 1920,
                                                     'maxHeight': 1080, 'everyNthFrame': 1})

    def _frame(self, p):
        path = os.path.join(self.dir, f'{len(self.frames):06d}.jpg')
        with open(path, 'wb') as f:
            f.write(base64.b64decode(p['data']))
        self.frames.append((p['metadata']['timestamp'], path))
        asyncio.ensure_future(self.cdp.send('Page.screencastFrameAck', {'sessionId': p['sessionId']}))

    async def stop(self, t0, length):
        await asyncio.sleep(0.3)
        await self.cdp.send('Page.stopScreencast')
        with open(os.path.join(self.dir, 'frames.json'), 'w') as f:
            json.dump({'t0': t0, 'length': length, 'frames': self.frames}, f)
        print(f'  {len(self.frames)} frames, {len(self.frames) / length:.1f} fps on average')


class Driver:
    """Timeline + a visible cursor: every action is scheduled at an absolute time from the part's start."""

    def __init__(self, page, t0, base):
        self.page, self.t0, self.base, self.pos = page, t0, base, [W / 2, H / 2]

    def now(self):
        return time.time() - self.t0 + self.base

    async def at(self, t):
        d = t - self.now()
        if d < -0.15:
            print(f'  late by {-d:.2f} s at t={t}')
        if d > 0:
            await asyncio.sleep(d)

    async def caption(self, t):
        await self.page.evaluate('(t) => window.__cap(t)', t)

    async def glide(self, x, y, dur=0.55):
        x0, y0 = self.pos
        n = max(2, int(dur * 40))
        t_start = time.time()
        for k in range(1, n + 1):
            f = k / n
            e = f * f * (3 - 2 * f)                                      # smoothstep
            await self.page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e)
            d = t_start + dur * f - time.time()
            if d > 0:
                await asyncio.sleep(d)
        self.pos = [x, y]

    async def click(self, x=None, y=None, dur=0.55):
        if x is not None:
            await self.glide(x, y, dur)
        await self.page.mouse.down()
        await asyncio.sleep(0.07)
        await self.page.mouse.up()

    async def drag(self, x, y, dur=1.0):
        await self.page.mouse.down()
        await self.glide(x, y, dur)
        await self.page.mouse.up()

    async def box(self, sel):
        b = await self.page.locator(sel).first.bounding_box()
        return b['x'] + b['width'] / 2, b['y'] + b['height'] / 2

    async def click_el(self, sel, dur=0.55, dx=0.0):
        x, y = await self.box(sel)
        await self.click(x + dx, y, dur)

    async def wheel(self, dy, dur=0.5):
        n = max(2, int(dur * 30))
        for _ in range(n):
            await self.page.mouse.wheel(0, dy / n)
            await asyncio.sleep(dur / n)

    async def scroll_panel_to(self, sel, top=0, dur=0.5):
        """Scroll the control panel (eased, like a wheel) until `sel` sits `top` px below the panel's top. The cursor
        first moves onto the panel's edge, so the scroll reads as the user's."""
        r = await self.page.evaluate("(() => { const r = document.getElementById('panelBody').getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; })()")
        await self.glide(r[0] + r[2] - 14, min(max(self.pos[1], r[1] + 40), r[1] + r[3] - 40), 0.25)
        await self.page.evaluate("""([sel, top, ms]) => new Promise(res => {
          const b = document.getElementById('panelBody'), el = document.querySelector(sel);
          const y0 = b.scrollTop, y1 = Math.max(0, Math.min(b.scrollHeight - b.clientHeight,
            y0 + el.getBoundingClientRect().top - b.getBoundingClientRect().top - top));
          const t0 = performance.now();
          const f = (now) => { const k = Math.min(1, (now - t0) / ms), e = k * k * (3 - 2 * k);
            b.scrollTop = y0 + (y1 - y0) * e; if (k < 1) requestAnimationFrame(f); else res(); };
          requestAnimationFrame(f); })""", [sel, top, dur * 1000])

    async def house(self, hid):
        return await self.page.evaluate('(i) => __housePt(i)', hid)

    async def img(self, x, y):
        return await self.page.evaluate('([x, y]) => __imgPt(x, y)', [x, y])


async def shot(page, work, name, on):
    if on:
        os.makedirs(os.path.join(work, 'shots'), exist_ok=True)
        await page.screenshot(path=os.path.join(work, 'shots', f'{name}.png'))


async def viewer_part(ctx, work, shots):
    page = await ctx.new_page()
    await page.goto(VIEWER + '#s=floodnet_7577&m=pred')
    await page.wait_for_function('window.rr && rr.state.s && rr.state.s.blds.length > 0')
    await page.evaluate(HELPERS)
    await page.wait_for_timeout(1200)
    # start state: the bare drone photo (class overlay and house badges off), panel at the Map section
    await page.evaluate("""() => { for (const id of ['tClasses', 'tStatus']) { const c = document.getElementById(id);
      c.checked = false; c.dispatchEvent(new Event('change')); } }""")
    await page.evaluate("""() => { const b = document.getElementById('panelBody'), m = [...b.querySelectorAll('section')][1];
      b.scrollTop = m.offsetTop - b.offsetTop - 4; }""")
    await page.mouse.move(W * 0.45, H * 0.45)
    await page.wait_for_timeout(600)
    rec = Recorder(page, os.path.join(work, 'viewer'))
    await rec.start()
    t0 = time.time()
    d = Driver(page, t0, 0.0)
    d.pos = [W * 0.45, H * 0.45]
    cap = iter(CAPTIONS)

    # 1. 0-7 s: photo -> flood map -> house badges, zoom in and out
    await d.caption(next(cap)[1])
    await d.at(0.7); await d.click_el('#tClasses', 0.6)
    await d.at(2.2); await d.click_el('#tStatus', 0.5)
    hx, hy = await d.house(5)
    await d.at(3.4); await d.glide(hx, hy, 0.6)
    await d.at(4.2); await d.wheel(-120, 0.25)
    await shot(page, work, '1_flood_map', shots)
    await d.at(5.9); await d.wheel(120, 0.25)

    # 2. 7-19 s: route to house 1 on the prediction (auto entry point)
    await d.at(7.0); await d.caption(next(cap)[1])
    await page.wait_for_timeout(300)
    x, y = await d.house(1)
    await d.at(7.4); await d.click(x, y, 0.8)
    await d.at(13.6)
    await shot(page, work, '2_route_pred', shots)
    x, y = await d.box('#routeCard .eta')
    await d.glide(x - 40, y + 4, 0.7)
    await d.at(16.2); x, y = await d.box('#routeCard .legs'); await d.glide(x + 40, y + 2, 0.6)

    # 3. 19-30 s: labelled map, swipe, ghost route
    await d.at(19.0); await d.caption(next(cap)[1])
    await d.at(19.2); await d.click_el('.seg button[data-src="gt"]', 0.7)
    await d.at(23.0)
    await shot(page, work, '3_route_gt', shots)
    await d.at(23.4); await d.click_el('.seg button[data-src="swipe"]', 0.6)
    await d.at(24.6); await d.glide(*(await d.box('#swipeHandle')), 0.6)
    sx, sy = d.pos
    await d.at(25.3); await page.mouse.down()
    await d.glide(sx - 230, sy, 1.1)
    await d.glide(sx + 170, sy, 1.4)
    await page.mouse.up()
    await shot(page, work, '3_swipe', shots)

    # 4. 30-37 s: back to the prediction, choose an entry point, toggle the risk-aware truck
    await d.at(30.0); await d.caption(next(cap)[1])
    await d.at(30.1); await d.click_el('.seg button[data-src="pred"]', 0.6)
    await d.at(30.9); await d.scroll_panel_to('#entryPick', 120, 0.45)
    await d.at(31.6); await d.click_el('#entryPick', 0.45)
    ex, ey = await d.img(*ENTRY)
    await d.at(32.3); await d.click(ex, ey, 0.6)
    await shot(page, work, '4_entry', shots)
    await d.at(33.4); await d.scroll_panel_to('#tRisk', 140, 0.45)
    await d.at(34.2); await d.click_el('#tRisk', 0.5, dx=0)
    await shot(page, work, '4_risk_off', shots)
    await d.at(35.6); await d.click(*d.pos, 0.0)
    await shot(page, work, '4_risk_on', shots)

    # 5. 37-46 s: RescueNet 12195, labelled map, house 8
    await d.at(37.0); await d.caption(next(cap)[1])
    await d.at(37.1); await d.scroll_panel_to('#scenes', 4, 0.4)
    await d.at(37.7); await d.click_el('#scenes button[data-key="rescuenet_12195"]', 0.5)
    await d.at(38.8); await d.click_el('.seg button[data-src="gt"]', 0.6)
    await page.wait_for_timeout(200)
    x, y = await d.house(8)
    await d.at(40.4); await d.click(x, y, 0.8)
    await d.at(45.2)
    await shot(page, work, '5_rescuenet', shots)

    # 6. 46-53 s: FloodNet 9107, every house a confident truck
    await d.at(46.0); await d.caption(next(cap)[1])
    await d.at(46.1); await d.click_el('#scenes button[data-key="floodnet_9107"]', 0.6)
    await d.at(47.2); await d.click_el('.seg button[data-src="pred"]', 0.6)
    await page.wait_for_timeout(150)
    x, y = await d.house(GREEN_HOUSE)
    await d.at(48.5); await d.click(x, y, 0.7)
    await d.at(51.5); await d.glide(*(await d.box('#routeCard .tierline')), 0.7)
    await shot(page, work, '6_confident', shots)

    # 7. 53-64 s: FloodNet 7241, the honest failure: check-first badges, water probability, labelled map
    await d.at(53.0); await d.caption(next(cap)[1])
    await d.at(53.1); await d.click_el('#scenes button[data-key="floodnet_7241"]', 0.6)
    await page.wait_for_timeout(200)
    x, y = await d.house(CHECK_HOUSE)
    await d.at(54.8); await d.click(x, y, 0.8)
    await d.at(57.4); await d.scroll_panel_to('.seg button[data-src]', 200, 0.4)
    await d.at(58.2); await d.click_el('.seg button[data-layer="pwet"]', 0.5)
    await shot(page, work, '7_pwet', shots)
    await d.at(60.4); await d.click_el('.seg button[data-src="swipe"]', 0.6)
    await d.at(61.6); await d.glide(*(await d.box('#swipeHandle')), 0.4)
    sx, sy = d.pos
    await page.mouse.down(); await d.glide(sx - 130, sy, 0.9); await page.mouse.up()
    await shot(page, work, '7_swipe', shots)

    # 8. 64-74 s: browse all 200 test photos
    await d.at(64.0); await d.caption(next(cap)[1])
    await d.at(64.1); await d.click_el('.seg button[data-src="pred"]', 0.5)
    await d.at(64.8); await d.click_el('.seg button[data-layer="classes"]', 0.4)
    await d.at(65.4); await d.scroll_panel_to('#allScenes', 150, 0.4)
    await d.at(66.2); await d.click_el('.seg.ds button[data-ds="rescuenet"]', 0.5)
    await d.at(67.4); await d.click_el('.seg.ds button[data-ds="floodnet"]', 0.5)
    await d.at(68.5); await d.click_el('#sortScenes', 0.5)
    await page.select_option('#sortScenes', 'cut')
    x, y = await d.box('#allScenes')                                    # the list keeps the current photo in view:
    await d.glide(x, y, 0.3)                                            # scroll it back to the top first
    await page.evaluate("""() => new Promise(res => { const b = document.getElementById('allScenes'), y0 = b.scrollTop, t0 = performance.now();
      const f = (now) => { const k = Math.min(1, (now - t0) / 350); b.scrollTop = y0 * (1 - k * k * (3 - 2 * k));
        if (k < 1) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); })""")
    await d.at(69.6); await d.click_el('#allScenes button:nth-child(1)', 0.5)
    await d.at(71.0); await d.click_el('#allScenes button:nth-child(2)', 0.5)
    await d.at(72.2); await d.glide(*(await d.box('#counter .big')), 0.6)
    await shot(page, work, '8_browse', shots)
    await d.at(VIEWER_LEN)
    await rec.stop(t0, VIEWER_LEN)
    await page.close()


async def gradio_part(ctx, work, shots):
    page = await ctx.new_page()
    await page.goto(GRADIO)
    await page.wait_for_selector('text=Click to Upload', timeout=60000)
    await page.wait_for_timeout(2500)
    await page.mouse.move(GW * 0.6, GH * 0.5)
    rec = Recorder(page, os.path.join(work, 'gradio'))
    await rec.start()
    t0 = time.time()
    base = VIEWER_LEN
    d = Driver(page, t0, base)
    d.pos = [GW * 0.6, GH * 0.5]
    await d.caption(CAPTIONS[-1][1])
    await d.at(base + 0.4); await d.glide(*(await d.box('text=Click to Upload')), 0.8)
    async with page.expect_file_chooser() as fc:
        await d.click()
    await (await fc.value).set_files(UPLOAD)
    await page.wait_for_selector('text=TEST MODE: cached', timeout=60000)      # the result map of the upload
    await shot(page, work, '9_upload', shots)
    view = """() => { const im = [...document.querySelectorAll('img')].filter(i => i.naturalWidth > 500)
      .sort((a, b) => b.getBoundingClientRect().width - a.getBoundingClientRect().width)[0];
      const r = im.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }"""
    await d.at(base + 5.0); await d.glide(GW * 0.62, GH * 0.55, 0.4)
    await d.wheel((await page.evaluate(view))[1] - 8, 0.6)                    # scroll the map into view
    await page.wait_for_timeout(300)
    r = await page.evaluate(view)
    for k, (fx, fy) in enumerate(GRADIO_CLICKS):
        await d.at(base + 7.0 + 4.0 * k)
        await d.click(r[0] + fx * r[2], r[1] + fy * r[3], 0.8)
    await d.at(base + 14.8)
    await shot(page, work, '9_route', shots)
    await d.at(base + GRADIO_LEN)
    await rec.stop(t0, GRADIO_LEN)
    await page.close()


# scene choices (checked with --shots; PLAN §15 0z)
ENTRY = (640, 460)            # 7577: entry point in the yards at the lower right (image px); the risk-aware
                              # toggle then redraws house 1's route clearly (the search is in PLAN §15 0z)
GREEN_HOUSE = 3               # 9107
CHECK_HOUSE = 6               # 7241: a yellow "check first" house
GRADIO_CLICKS = [(0.074, 0.93), (0.358, 0.723)]   # 7476: a boat house, then a check-first house (fractions of the map)


def encode(work, out):
    """Frames of both parts -> 30 fps CFR H.264 (ffmpeg concat demuxer with per-frame durations)."""
    lst = os.path.join(work, 'concat.txt')
    with open(lst, 'w') as f:
        for part in ('viewer', 'gradio'):
            m = json.load(open(os.path.join(work, part, 'frames.json')))
            t0, t1 = m['t0'], m['t0'] + m['length']
            fr = [(ts, p) for ts, p in m['frames']]
            start = max([k for k, (ts, _) in enumerate(fr) if ts <= t0] or [0])
            fr = fr[start:]
            fr = [(t0 if k == 0 else ts, p) for k, (ts, p) in enumerate(fr) if ts < t1]
            for k, (ts, p) in enumerate(fr):
                nxt = fr[k + 1][0] if k + 1 < len(fr) else t1
                f.write(f"file '{p}'\nduration {max(nxt - ts, 0.001):.4f}\n")
        f.write(f"file '{p}'\n")
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
           '-vf', 'fps=30,scale=1920:1080:flags=lanczos:out_range=tv,format=yuv420p', '-fps_mode', 'cfr', '-r', '30',
           '-color_range', 'tv', '-t', str(VIEWER_LEN + GRADIO_LEN),
           '-c:v', 'libx264', '-preset', 'slow', '-crf', os.environ.get('CRF', '18'), '-tune', 'animation',
           '-movflags', '+faststart', '-an', out]
    subprocess.run(cmd, check=True)
    print('wrote', out, f'{os.path.getsize(out) / 1e6:.1f} MB')


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'slides', 'rescue_router_demo.mp4'))
    ap.add_argument('--work', default=os.path.join(tempfile.gettempdir(), 'rr_demo_video'))
    ap.add_argument('--only', choices=['viewer', 'gradio', 'encode'])
    ap.add_argument('--shots', action='store_true')
    a = ap.parse_args()
    if a.only != 'encode':
        async with async_playwright() as p:
            b = await p.chromium.launch()
            ctx = await b.new_context(viewport={'width': W, 'height': H}, device_scale_factor=DSF)
            await ctx.add_init_script(OVERLAY)
            if a.only in (None, 'viewer'):
                print('viewer part'); await viewer_part(ctx, a.work, a.shots)
            if a.only in (None, 'gradio'):
                gctx = await b.new_context(viewport={'width': GW, 'height': GH}, device_scale_factor=1920 / GW)
                await gctx.add_init_script(OVERLAY)
                print('gradio part'); await gradio_part(gctx, a.work, a.shots)
            await b.close()
    encode(a.work, a.out)


if __name__ == '__main__':
    asyncio.run(main())
