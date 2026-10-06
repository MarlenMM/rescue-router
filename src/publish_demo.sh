#!/usr/bin/env bash
# Publish the built web viewer (docs/) to the PUBLIC demo repo MarlenMM/rescue-router-demo (GitHub Pages).
# The main repo rescue-router stays private until grading (PLAN §15). Needs Marlen's explicit OK each time.
#
#   bash src/publish_demo.sh            # update an existing demo repo
#   bash src/publish_demo.sh --create   # first time: create the public repo and enable Pages
set -euo pipefail
REPO=MarlenMM/rescue-router-demo
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$HOME/rr_push/rescue-router-demo"
PY="$HOME/.venvs/rr/bin/python"

node --test "$ROOT/app/test/router.test.mjs"            # JS router must still match Python
"$PY" "$ROOT/src/build_viewer.py"

if [[ "${1:-}" == "--create" ]]; then
  gh repo create "$REPO" --public --description "Rescue Router: few-shot flood mapping (Chameleon, 50 labels) + truck/boat rescue routing, interactive demo" 
fi
rm -rf "$WORK"; mkdir -p "$(dirname "$WORK")"
gh repo clone "$REPO" "$WORK" -- --depth 1 2>/dev/null || { mkdir -p "$WORK"; git -C "$WORK" init -q -b main; git -C "$WORK" remote add origin "https://github.com/$REPO.git"; }
find "$WORK" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$ROOT/docs/." "$WORK/"
cat > "$WORK/README.md" <<'MD'
# Rescue Router: interactive demo

Live: https://marlenmm.github.io/rescue-router-demo/

A drone photo after a flood is mapped into clear road, flooded road, flooded/dry building and water by
[Chameleon](https://github.com/GitGyun/chameleon) (Kim et al., ECCV 2024), adapted from only 50 labelled photos.
A router on the predicted map finds how a rescue team reaches each house: by truck, then boat, then the last stretch
on foot (never back into a vehicle). Click a house to see the route, choose where the team enters, and compare with
the labelled map (the same router on the human labels) or swipe between them.

KAIST AIC200 mini-project (Track A + B). This repository holds only the static viewer; the code repository
will be made public after grading.

Data: [FloodNet](https://github.com/BinaLab/FloodNet-Supervised_v1.0) (CDLA-Permissive-1.0);
[RescueNet](https://springernature.figshare.com/collections/RescueNet_A_High_Resolution_UAV_Semantic_Segmentation_Benchmark_Dataset_for_Natural_Disaster_Damage_Assessment/6647354/1)
(figshare release CC0; the GitHub release states CC BY-NC-ND 4.0). Map: [Leaflet](https://leafletjs.com).
MD
git -C "$WORK" add -A
git -C "$WORK" commit -qm "Viewer build $(git -C "$ROOT" rev-parse --short HEAD)" || echo "nothing to commit"
git -C "$WORK" push -q -u origin main
if [[ "${1:-}" == "--create" ]]; then
  gh api -X POST "repos/$REPO/pages" -f "source[branch]=main" -f "source[path]=/" >/dev/null
fi
echo "Pushed. Pages: https://marlenmm.github.io/rescue-router-demo/ (first build takes ~1 min)"
