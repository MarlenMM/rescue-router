"""docs/ (the published viewer) is exactly what src/build_viewer.py makes from app/: a change to app/ without a rebuild,
or an edit made in docs/ directly, fails here (and in CI)."""
import filecmp
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    differ = [f for f in sorted(a) if not filecmp.cmp(os.path.join(app, f), os.path.join(docs, f), shallow=False)]
    assert not differ, f'{len(differ)} files differ, e.g. {differ[:5]} -> python src/build_viewer.py'
