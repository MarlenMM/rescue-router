#!/bin/zsh
# Build the code tarball for the private Kaggle dataset marlenmelis/rr-code (PLAN §15 "Kaggle setup").
#   kaggle/build_rr_code.sh            # writes ~/rr_push/ds/rr_code.tar.gz, prints the upload command
# Contents: src/ splits/ supports/ kaggle/live/ VERSION (git HEAD), plus the small result files the GPU notebooks need at the
# same relative paths (results/ is not in git): the tuned decodes, the tier table (route-card sentences) and the
# viewer index (Gradio dropdown labels). Built with macOS metadata off (Kaggle unpacks the archive; see memory).
set -e
ROOT=${0:A:h:h}
OUT=~/rr_push/ds
TMP=$(mktemp -d)
cd $ROOT
git diff --quiet HEAD -- src splits supports || echo "warning: uncommitted changes in src/splits/supports are NOT included (git HEAD)"
git archive HEAD src splits supports kaggle/live | tar -x -C $TMP
git rev-parse --short HEAD > $TMP/VERSION
mkdir -p $TMP/results/kaggle_v2/rr/decode $TMP/results/routing/tiers $TMP/app/scenes $TMP/results_decode
cp results/kaggle_v2/rr/decode/*.json $TMP/results/kaggle_v2/rr/decode/
cp results/kaggle_v2/rr/decode/*.json $TMP/results_decode/          # old path, used by kaggle/live/live_check.py
cp results/routing/tiers/tiers.json $TMP/results/routing/tiers/
cp app/scenes/scenes.json $TMP/app/scenes/
mkdir -p $OUT
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf $OUT/rr_code.tar.gz -C $TMP .
rm -rf $TMP
tar -tzf $OUT/rr_code.tar.gz | grep -c . | xargs echo "files:"
ls -la $OUT/rr_code.tar.gz
echo "upload (needs Marlen's OK for the GPU run that uses it): cd $OUT && ~/.venvs/kaggle/bin/kaggle datasets version -p . -m \"code $(git -C $ROOT rev-parse --short HEAD)\""
