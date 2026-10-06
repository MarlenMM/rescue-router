"""Build the static web viewer: app/ -> docs/ (no build step, just a clean copy without tests). CPU only.

    python src/build_viewer.py [--out docs]

Run src/export_web_assets.py first (scene assets in app/scenes/), and `node --test app/test/router.test.mjs`.
The per-scene ref.json (Python reference for the parity test) is not copied.
The public demo repo (MarlenMM/rescue-router-demo) receives exactly this folder.
"""

import argparse
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = shutil.ignore_patterns('test', 'ref.json', '.DS_Store', '*.pyc', '__pycache__')   # ref.json: JS parity test only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs'))
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(os.path.join(ROOT, 'app'), a.out, ignore=SKIP)
    open(os.path.join(a.out, '.nojekyll'), 'w').close()           # serve files as they are on GitHub Pages
    n = sum(len(f) for _, _, f in os.walk(a.out))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, f in os.walk(a.out) for x in f)
    print(f'{a.out}: {n} files, {size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
