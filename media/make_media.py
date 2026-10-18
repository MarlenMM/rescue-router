"""Make the README media and the viewer's link-preview image from the built viewer (docs/). CPU only, ~1 min.

    python3 -m http.server 8766 --bind 127.0.0.1 --directory docs &
    uv run --no-project --python 3.12 --with playwright --with pillow python media/make_media.py

Drives the installed Google Chrome through Playwright (channel "chrome", no browser download). Writes:
  media/hero.webp            route animation: truck, boat, then on foot; the labelled map; the swipe comparison
  media/upload.webp          "Try your own drone photo" on photos outside the viewer's 200 (the one with most boat houses)
  media/social-preview.jpg   1280x640, for the repository's social preview
  app/og.jpg                 the same image, the viewer's link preview (run src/build_viewer.py afterwards)
Frames come from the Chrome DevTools screencast, resampled to a constant frame rate; the cursor and its click
ripple are drawn into the page (DOM), nothing else on the page is changed.
"""
import argparse
import asyncio
import base64
import io
import os
import re
import time

from PIL import Image
from playwright.async_api import async_playwright

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# photos outside the viewer's 200: a FloodNet test photo and FloodNet val photos no part of the project used
UPLOADS = [os.path.join(REPO, 'demo', 'uploads', 'floodnet_7307.jpg')] + sorted(
    os.path.join(REPO, 'demo', 'uploads', 'floodnet_unused', f)
    for f in os.listdir(os.path.join(REPO, 'demo', 'uploads', 'floodnet_unused')) if f.endswith('.jpg'))

CURSOR = r"""
(() => {
  if (document.getElementById('__cur')) return;
  const st = document.createElement('style');
  st.textContent = `#__cur { position: fixed; left: 0; top: 0; z-index: 2147483647; pointer-events: none; width: 26px;
      height: 26px; margin: -2px 0 0 -3px; filter: drop-shadow(0 1px 2px rgba(0,0,0,.7)); }
    .__rip { position: fixed; z-index: 2147483646; pointer-events: none; width: 36px; height: 36px; margin: -18px 0 0 -18px;
      border-radius: 50%; border: 3px solid #22d3ee; background: rgba(34,211,238,.25); animation: __rip .5s ease-out forwards; }
    @keyframes __rip { from { transform: scale(.3); opacity: 1; } to { transform: scale(1.4); opacity: 0; } }`;
  document.head.appendChild(st);
  const cur = document.createElement('div'); cur.id = '__cur';
  cur.innerHTML = '<svg width="26" height="26" viewBox="0 0 26 26"><path d="M3 2 L3 21.5 L8.3 16.6 L11.8 24.5 L15.6 22.9 L12.1 15.2 L19.2 15.2 Z" fill="#fff" stroke="#000" stroke-width="1.7" stroke-linejoin="round"/></svg>';
  document.body.appendChild(cur);
  addEventListener('mousemove', (e) => { cur.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`; }, true);
  addEventListener('mousedown', (e) => {
    const r = document.createElement('div'); r.className = '__rip';
    r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
    document.body.appendChild(r); setTimeout(() => r.remove(), 600);
  }, true);
})();
"""

# screen point of a house badge (image pixels -> map container -> page)
HOUSE_PT = """(id) => { const s = rr.state.s, b = s.blds.find(b => b.id === id),
  p = rr.map.latLngToContainerPoint(L.latLng(s.H - b.centroid[1], b.centroid[0])),
  r = document.getElementById('map').getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; }"""


# fit the photo into the area the title card, the panel and the counter leave free; zoomSnap so it fills it
FRAME = """() => { const m = rr.map, q = (s) => document.querySelector(s).getBoundingClientRect();
  if (!m.__fit) {
    m.__fit = m.fitBounds.bind(m); m.options.zoomSnap = 0.05;
    m.fitBounds = (b) => m.__fit(b, { paddingTopLeft: [q('.dock').right - 4, q('.title').bottom + 8],
                                     paddingBottomRight: [innerWidth - q('#panel').left + 10, 14] });
  }
  m.fitBounds([[0, 0], [rr.state.s.H, rr.state.s.W]]); }"""

# houses whose badge is on screen and clear of the cards, with one of the given statuses
# (0 truck, 1 boat, 2 boat then on foot / on foot, 3 no route)
CANDIDATES = """(want) => { const s = rr.state.s, q = (sel) => document.querySelector(sel).getBoundingClientRect();
  const free = { l: q('.dock').right + 10, t: q('.title').bottom + 30, r: q('#panel').left - 30, b: innerHeight - 40 };
  const pt = (b) => { const p = rr.map.latLngToContainerPoint(L.latLng(s.H - b.centroid[1], b.centroid[0])),
    r = document.getElementById('map').getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; };
  return s.blds.filter((b, i) => want.includes(s.status.pred[i])).filter((b) => { const [x, y] = pt(b);
    return x > free.l && x < free.r && y > free.t && y < free.b; }).map((b) => b.id); }"""


async def pick_house(page, want=(2, 1)):
    """the visible house whose route (on the prediction) has the most legs, then the most metres: truck, boat, on foot"""
    best = None
    for hid in await page.evaluate(CANDIDATES, list(want)):
        await page.evaluate(FRAME)
        await page.mouse.click(*(await page.evaluate(HOUSE_PT, hid)))
        await page.wait_for_timeout(250)
        text = await page.locator('#routeCard').inner_text()
        legs = [int(m) for m in re.findall(r'(?:truck|boat|on foot) (\d+) m', text.split('\n')[1] if '\n' in text else text)]
        score = (sum(m > 0 for m in legs), sum(legs))
        if best is None or score > best[0]:
            best = (score, hid)
        await page.keyboard.press('Escape')
    await page.evaluate(FRAME)
    await page.wait_for_timeout(400)
    return best[1]


class Screencast:
    """Chrome DevTools screencast -> (timestamp, JPEG bytes) in memory."""

    def __init__(self, page):
        self.page, self.frames = page, []

    async def start(self, width):
        self.cdp = await self.page.context.new_cdp_session(self.page)
        self.cdp.on('Page.screencastFrame', self._frame)
        await self.cdp.send('Page.startScreencast', {'format': 'jpeg', 'quality': 92, 'maxWidth': width,
                                                     'maxHeight': 4000, 'everyNthFrame': 1})

    def _frame(self, p):
        self.frames.append((p['metadata']['timestamp'], base64.b64decode(p['data'])))
        asyncio.ensure_future(self.cdp.send('Page.screencastFrameAck', {'sessionId': p['sessionId']}))

    async def stop(self):
        await asyncio.sleep(0.3)
        await self.cdp.send('Page.stopScreencast')

    def resample(self, t0, length, fps, size):
        """constant-rate frames: at each tick the latest screencast frame at or before it"""
        out, k = [], 0
        for i in range(int(length * fps)):
            t = t0 + i / fps
            while k + 1 < len(self.frames) and self.frames[k + 1][0] <= t:
                k += 1
            out.append(Image.open(io.BytesIO(self.frames[k][1])).convert('RGB').resize(size, Image.LANCZOS))
        return out


class Cursor:
    def __init__(self, page, x, y):
        self.page, self.pos = page, [x, y]

    async def glide(self, x, y, dur=0.55):
        x0, y0 = self.pos
        n, t_start = max(2, int(dur * 40)), time.time()
        for k in range(1, n + 1):
            f = k / n
            e = f * f * (3 - 2 * f)                                      # smoothstep
            await self.page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e)
            d = t_start + dur * f - time.time()
            if d > 0:
                await asyncio.sleep(d)
        self.pos = [x, y]

    async def click(self, x, y, dur=0.55):
        await self.glide(x, y, dur)
        await self.page.mouse.down(); await asyncio.sleep(0.07); await self.page.mouse.up()

    async def click_el(self, sel, dur=0.55):
        b = await self.page.locator(sel).first.bounding_box()
        await self.click(b['x'] + b['width'] / 2, b['y'] + b['height'] / 2, dur)


async def open_scene(ctx, url, w, h, dsf):
    page = await ctx.new_page()
    await page.set_viewport_size({'width': w, 'height': h})
    await page.goto(url)
    await page.wait_for_function('window.rr && rr.state.s && rr.state.s.blds.length > 0', timeout=60000)
    await page.wait_for_timeout(1500)
    return page


async def hero(browser, base, out):
    W, H, DSF, FPS, OUT = 1600, 900, 1.25, 12, (1280, 720)
    ctx = await browser.new_context(device_scale_factor=DSF)
    page = await open_scene(ctx, base + '#s=floodnet_7301', W, H, DSF)
    await page.evaluate("""() => { const b = document.getElementById('panelBody'), m = [...b.querySelectorAll('section')][1];
      b.scrollTop = m.offsetTop - b.offsetTop - 4; }""")                         # panel at its Map section
    await page.evaluate(FRAME)
    await page.wait_for_timeout(600)
    hid = await pick_house(page)
    await page.evaluate(CURSOR)
    c = Cursor(page, W * 0.42, H * 0.5)
    await page.mouse.move(*c.pos)
    await page.wait_for_timeout(400)
    rec = Screencast(page)
    await rec.start(int(W * DSF))
    t0 = time.time()

    async def at(t):
        d = t0 + t - time.time()
        if d > 0:
            await asyncio.sleep(d)

    await at(0.6); await c.click(*(await page.evaluate(HOUSE_PT, hid)), 0.8)   # truck -> boat -> on foot
    await at(6.4); await c.click_el('.seg button[data-src="gt"]', 0.8)          # the same router on the labels
    await at(8.6); await c.click_el('.seg button[data-src="swipe"]', 0.6)
    b = await page.locator('#swipeHandle').bounding_box()
    sx, sy = b['x'] + b['width'] / 2, b['y'] + b['height'] / 2
    await at(9.6); await c.glide(sx, sy, 0.5)
    await page.mouse.down(); await c.glide(sx - 260, sy, 1.2); await c.glide(sx + 120, sy, 1.2); await page.mouse.up()
    await at(14.0)
    await rec.stop()
    frames = rec.resample(t0, 14.0, FPS, OUT)                                  # screencast clock = wall clock
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, quality=72,
                   method=6, minimize_size=True)
    print(f'{out}: {len(frames)} frames, {os.path.getsize(out) / 1e6:.1f} MB')
    await ctx.close()


async def upload(browser, base, out):
    """map each candidate photo in the browser, keep the one with the most houses that need the boat"""
    W, H, DSF = 1280, 760, 1.5
    ctx = await browser.new_context(device_scale_factor=DSF)
    page = await open_scene(ctx, base + '#s=floodnet_9107', W, H, DSF)
    best = None
    for k, photo in enumerate(UPLOADS, 1):
        async with page.expect_file_chooser() as fc:
            await page.click('#uploadBtn')
        await (await fc.value).set_files(photo)
        await page.wait_for_function(f"rr.state.key === 'local/{k}' && rr.state.s && rr.state.s.meta && rr.state.s.meta.upload",
                                     timeout=120000)
        st = await page.evaluate('rr.state.s.status.pred')
        n = sum(v in (1, 2) for v in st)
        print(f'  {os.path.basename(photo)}: {len(st)} houses, {n} by boat or on foot')
        if best is None or n > best[0]:
            best = (n, k, photo)
    await page.evaluate(f"location.hash = 's=local/{best[1]}'")
    await page.wait_for_function(f"rr.state.key === 'local/{best[1]}'")
    await page.evaluate(FRAME)
    await page.wait_for_timeout(800)
    hid = await pick_house(page)
    await page.mouse.click(*(await page.evaluate(HOUSE_PT, hid)))
    await page.wait_for_timeout(6000)
    png = await page.screenshot()
    Image.open(io.BytesIO(png)).convert('RGB').resize((1600, 950), Image.LANCZOS).save(out, quality=82, method=6)
    print(f'{out}: {os.path.basename(best[2])}, house {hid}, {os.path.getsize(out) / 1e6:.2f} MB')
    await ctx.close()


async def social(browser, base, outs):
    W, H = 1280, 640
    ctx = await browser.new_context(device_scale_factor=1)
    page = await open_scene(ctx, base + '#s=floodnet_7301', W, H, 1)
    await page.add_style_tag(content='#panel { display: none !important; }')       # the map and its cards only
    await page.evaluate("document.getElementById('panel').getBoundingClientRect = () => ({ left: innerWidth - 14 })")
    await page.evaluate(FRAME)
    await page.wait_for_timeout(600)
    hid = await pick_house(page)
    await page.mouse.click(*(await page.evaluate(HOUSE_PT, hid)))
    await page.wait_for_timeout(6000)
    await page.mouse.move(W - 2, H - 2)
    im = Image.open(io.BytesIO(await page.screenshot())).convert('RGB')
    for o in outs:
        im.save(o, quality=88, optimize=True, progressive=True)
        print(f'{o}: {os.path.getsize(o) / 1e6:.2f} MB')
    await ctx.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8766/', help='URL of the built viewer')
    ap.add_argument('--only', choices=['hero', 'upload', 'social'])
    a = ap.parse_args()
    media = os.path.join(REPO, 'media')
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel='chrome')
        if a.only in (None, 'hero'):
            await hero(browser, a.base, os.path.join(media, 'hero.webp'))
        if a.only in (None, 'upload'):
            await upload(browser, a.base, os.path.join(media, 'upload.webp'))
        if a.only in (None, 'social'):
            await social(browser, a.base, [os.path.join(media, 'social-preview.jpg'), os.path.join(REPO, 'app', 'og.jpg')])
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
