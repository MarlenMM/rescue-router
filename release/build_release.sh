#!/usr/bin/env bash
# Build the KLMS submission (PLAN §12, Batch 12). CPU only, ~2 min. Output (outside iCloud):
#   $OUT/20260876_MarlenMelis_project.zip = code.ipynb + writeup.pdf + slides.pptx + data.zip + app.zip
#
#   bash release/build_release.sh [OUT]          # default OUT=~/rr_push/submission
#
# data.zip = the bundle the G3 run used (src/make_bundle.py at 1b5d176, Kaggle dataset marlenmelis/rr-bundle-b13; Batch 13), as is:
# every file of its code (src, splits, supports, demo, kaggle/live, requirements) is unchanged since (checked below;
# files added later, such as src/gradio_viewer.py, ship in the app zip only; its README.md is the one of that commit).

# code.ipynb = notebook/rescue_router_executed.ipynb (executed on a fresh T4; never regenerate without re-running G3).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$HOME/rr_push/submission}"
P=20260876_MarlenMelis
BUNDLE_ZIP="$HOME/rr_push/bundle_b13/20260876_MarlenMelis_data.zip"
BUNDLE_COMMIT=1b5d176
DATA="$HOME/rr_data"
cd "$ROOT"

# the data zip's code must still be the code at HEAD (modified, deleted or renamed files; additions are fine)
if [[ -n "$(git diff --stat --diff-filter=MDRT "$BUNDLE_COMMIT" HEAD -- src splits supports demo kaggle/live requirements.txt)" ]]; then
  echo "code changed since the bundle ($BUNDLE_COMMIT): rebuild it with src/make_bundle.py and re-run G3" >&2; exit 1
fi
[[ -z "$(git status --porcelain src)" ]] || { echo "uncommitted changes in src/" >&2; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT/project" "$OUT/app"
# app zip
A="$OUT/app"
cp -R docs "$A/viewer"
git archive HEAD src requirements.txt kaggle/gradio_live demo/uploads | tar -x -C "$A"
find "$A/src" -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$A/rr_data/rescuenet_r768/test/images"
for i in $(cat splits/rescuenet_test_eval100.txt); do
  [[ "$i" == 12195 || "$i" == 15833 ]] && continue                   # these two are in the data zip
  cp "$DATA/rescuenet_r768/test/images/$i.jpg" "$A/rr_data/rescuenet_r768/test/images/"
done
cp release/README_app.md slides/demo_route.gif "$A/"
(cd "$A" && COPYFILE_DISABLE=1 zip -qrX "$OUT/project/${P}_app.zip" . -x '*.DS_Store' -x '__MACOSX/*')
# the other four
cp "$BUNDLE_ZIP" "$OUT/project/${P}_data.zip"
cp notebook/rescue_router_executed.ipynb "$OUT/project/${P}_code.ipynb"
cp "writeup/${P}_writeup.pdf" "$OUT/project/"
cp "slides/${P}_slides.pptx" "$OUT/project/"
(cd "$OUT/project" && COPYFILE_DISABLE=1 zip -qX0 "$OUT/${P}_project.zip" "${P}_code.ipynb" "${P}_writeup.pdf" \
   "${P}_slides.pptx" "${P}_data.zip" "${P}_app.zip")

# checks: names, exactly one .ipynb in all zips (nested included), no macOS junk
"$HOME/.venvs/rr/bin/python" - "$OUT/${P}_project.zip" <<'PY'
import io, sys, zipfile
def walk(z, pre=''):
    for n in z.namelist():
        yield pre + n
        if n.endswith('.zip'):
            yield from walk(zipfile.ZipFile(io.BytesIO(z.read(n))), pre + n + '!/')
top = zipfile.ZipFile(sys.argv[1])
names = list(walk(top))
P = '20260876_MarlenMelis'
want = {f'{P}_code.ipynb', f'{P}_writeup.pdf', f'{P}_slides.pptx', f'{P}_data.zip', f'{P}_app.zip'}
assert set(top.namelist()) == want, top.namelist()
nb = [n for n in names if n.lower().endswith('.ipynb')]
assert nb == [f'{P}_code.ipynb'], nb
junk = [n for n in names if '__MACOSX' in n or '/._' in n or n.endswith('.DS_Store')]
assert not junk, junk[:5]
for i in top.infolist():
    print(f'{i.file_size / 1e6:8.1f} MB  {i.filename}')
print(f'{sum(1 for _ in names)} entries, .ipynb: {nb}')
PY
ls -l "$OUT/${P}_project.zip"
