"""docs/ (the published viewer) is exactly what src/build_viewer.py makes from app/: a change to app/ without a rebuild,
or an edit made in docs/ directly, fails here (and in CI). index.html and js/app.js carry the build's cache-busting
?v=<hash> (build_viewer.stamped)."""
import filecmp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'src'))
import build_viewer  # noqa: E402
SKIP = {'test', 'ref.json', '.DS_Store', '__pycache__'}          # build_viewer.SKIP (+ the .nojekyll marker it adds)


def files(root):
    out = set()
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP]
        out |= {os.path.relpath(os.path.join(d, f), root) for f in fs if f not in SKIP and not f.endswith('.pyc')}
    return out


def test_docs_is_the_built_app():
    app, docs = os.path.join(REPO, 'app'), os.path.join(REPO, 'docs')
    a, d = files(app), files(docs) - {'.nojekyll'}
    assert a == d, f'only in app/: {sorted(a - d)[:5]}; only in docs/: {sorted(d - a)[:5]} -> python src/build_viewer.py'
    stamped = build_viewer.stamped(app)
    differ = [f for f in sorted(a) if (open(os.path.join(docs, f), encoding='utf-8').read() != stamped[f] if f in stamped
                                       else not filecmp.cmp(os.path.join(app, f), os.path.join(docs, f), shallow=False))]
    assert not differ, f'{len(differ)} files differ, e.g. {differ[:5]} -> python src/build_viewer.py'


def test_cache_busting_changes_with_the_code(tmp_path):
    """the ?v= hash follows the scripts' contents: an edit to i18n.js gives the page and app.js new module URLs"""
    import shutil
    app = tmp_path / 'app'
    shutil.copytree(os.path.join(REPO, 'app', 'js'), app / 'js')
    shutil.copytree(os.path.join(REPO, 'app', 'css'), app / 'css')
    shutil.copy(os.path.join(REPO, 'app', 'index.html'), app / 'index.html')
    before = build_viewer.stamped(str(app))
    (app / 'js' / 'i18n.js').write_text((app / 'js' / 'i18n.js').read_text() + '\n// changed\n')
    after = build_viewer.stamped(str(app))
    assert before['index.html'] != after['index.html'] and before[os.path.join('js', 'app.js')] != after[os.path.join('js', 'app.js')]
