"""Build the static web viewer: app/ -> docs/ (no build step, just a clean copy without tests). CPU only.

    python src/build_viewer.py [--out docs]

Run src/export_web_assets.py first (scene assets in app/scenes/), and `node --test app/test/router.test.mjs`.
The per-scene ref.json (Python reference for the parity test) is not copied. The built page loads its own scripts and
stylesheet with ?v=<hash of their contents> (index.html -> js/app.js, css/style.css; app.js -> its modules), so a
browser never runs files of two builds together (GitHub Pages lets them be cached for 10 minutes; 16 Oct a cached
i18n.js ran with a new app.js). app/ itself stays unstamped for local development.
GitHub Pages serves docs/ from this repository (https://marlenmm.github.io/rescue-router/); src/publish_demo.sh copies
the same folder to the separate demo repo MarlenMM/rescue-router-demo.
"""

import argparse
import hashlib
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = shutil.ignore_patterns('test', 'ref.json', '.DS_Store', '*.pyc', '__pycache__')   # ref.json: JS parity test only


def stamped(app):
    """{relative path: text} of the files the build rewrites: index.html and js/app.js with ?v=<content hash> on every
    reference to our own scripts and stylesheet (vendored Leaflet keeps its integrity-pinned URLs)"""
    h = hashlib.sha256()
    for rel in ('js/app.js', 'js/i18n.js', 'js/router.js', 'js/scenebuild.js', 'css/style.css'):
        h.update(rel.encode() + b'\0' + open(os.path.join(app, rel), 'rb').read())
    v = h.hexdigest()[:10]
    html = open(os.path.join(app, 'index.html'), encoding='utf-8').read()
    html = re.sub(r'(src|href)="(js/app\.js|css/style\.css)"', rf'\1="\2?v={v}"', html)
    js = open(os.path.join(app, 'js', 'app.js'), encoding='utf-8').read()
    js = re.sub(r"from '\./(router|i18n|scenebuild)\.js'", rf"from './\1.js?v={v}'", js)
    assert html.count(f'?v={v}') == 2 and js.count(f'?v={v}') == 3, 'cache-busting: a reference was not found'
    return {'index.html': html, os.path.join('js', 'app.js'): js}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs'))
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(os.path.join(ROOT, 'app'), a.out, ignore=SKIP)
    for rel, text in stamped(os.path.join(ROOT, 'app')).items():
        open(os.path.join(a.out, rel), 'w', encoding='utf-8').write(text)
    open(os.path.join(a.out, '.nojekyll'), 'w').close()           # serve files as they are on GitHub Pages
    n = sum(len(f) for _, _, f in os.walk(a.out))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, f in os.walk(a.out) for x in f)
    print(f'{a.out}: {n} files, {size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
