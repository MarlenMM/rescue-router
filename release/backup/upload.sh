#!/usr/bin/env bash
# Upload the parts made by `python src/backup.py pack` to two independent PRIVATE places (PLAN §16, fix F4):
#   1. Kaggle dataset marlenmelis/rr-backup (private; a new dataset version per backup)
#   2. a release on the private code repo MarlenMM/rescue-router (tag backup-YYYY-MM-DD; assets < 2 GiB each)
#
#   bash release/backup/upload.sh [~/rr_backup]          # both
#   bash release/backup/upload.sh [~/rr_backup] kaggle   # only one of them (kaggle | github)
#
# Restore: download the parts (kaggle datasets download marlenmelis/rr-backup -p <dir> --unzip, or
# gh release download backup-YYYY-MM-DD -R MarlenMM/rescue-router -D <dir>), then
#   python src/backup.py restore --src <dir> --dest <restore dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="${1:-$HOME/rr_backup}"
ONLY="${2:-both}"
KAGGLE="$HOME/.venvs/kaggle/bin/kaggle"
DATASET=marlenmelis/rr-backup
TAG="backup-$(date +%Y-%m-%d)"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"

cd "$SRC"                                   # short relative paths: the Kaggle CLI fails on long ones
(cd "$SRC" && shasum -a 256 -c "$ROOT/release/backup/PARTS.sha256")     # never upload a part that does not match
cp "$ROOT/release/backup/"{MANIFEST.tsv,PARTS.sha256,EXTERNAL.tsv} .

if [[ "$ONLY" != github ]]; then
  cat > dataset-metadata.json <<JSON
{"title": "rr-backup", "id": "$DATASET", "licenses": [{"name": "other"}],
 "description": "Private backup of Rescue Router results and data (src/backup.py, commit $COMMIT)."}
JSON
  if "$KAGGLE" datasets status "$DATASET" >/dev/null 2>&1; then
    "$KAGGLE" datasets version -p . -m "backup $TAG (commit $COMMIT)"
  else
    "$KAGGLE" datasets create -p .           # private unless --public is given
  fi
fi

if [[ "$ONLY" != kaggle ]]; then
  gh release view "$TAG" -R MarlenMM/rescue-router >/dev/null 2>&1 || \
    gh release create "$TAG" -R MarlenMM/rescue-router --target main --title "Backup $TAG" \
      --notes "Results and data not in git (src/backup.py, commit $COMMIT). Restore: release/backup/upload.sh header."
  gh release upload "$TAG" -R MarlenMM/rescue-router --clobber ./*.tar.part-* MANIFEST.tsv PARTS.sha256 EXTERNAL.tsv
fi
echo "uploaded $(ls ./*.tar.part-* | wc -l | tr -d ' ') parts ($TAG)"
