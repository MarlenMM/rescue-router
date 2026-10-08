#!/bin/zsh
# One-shot (re)start of the live Gradio link on a fresh Colab CLI session (8 Oct presentation).
# Usage: zsh kaggle/gradio_live/colab_cli_up.sh [session]   -> prints SHARE LINK: ... at the end
# colab upload fails (HTTP 500) on files of a few hundred MB, so the zips go up in 40 MB parts, 4 at a time.
set -u
setopt null_glob
export PATH="$HOME/.local/bin:$PATH"
SES=${1:-live}
HERE=${0:A:h}
D=~/rr_push/submission/project
TMP=${TMPDIR:-/tmp}/rr_chunks; mkdir -p $TMP
echo "$(date +%T) session $SES"
colab status 2>&1 | grep -q "$SES" || colab new -s $SES --gpu T4 || exit 1
cat > $TMP/gd.py <<'PY'
import os, subprocess
os.makedirs("/content/parts", exist_ok=True)
print(subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv', shell=True, capture_output=True, text=True).stdout)
if not (os.path.exists('/content/generalist.ckpt') and os.path.getsize('/content/generalist.ckpt') == 2967114775):
    subprocess.Popen('pip install -q gdown >/dev/null 2>&1; gdown --fuzzy 1_UrVMFy18kEcrxwD1b_ohxuTfJIsOWsG '
                     '-O /content/generalist.ckpt > /content/gdown.log 2>&1; echo DONE >> /content/gdown.log',
                     shell=True, start_new_session=True)
    print('gdown started')
PY
colab exec -s $SES -f $TMP/gd.py
for z in data app; do
  f=$D/20260876_MarlenMelis_$z.zip
  [[ -f $TMP/${z}_parts.ok ]] || { rm -f $TMP/$z.part.*; split -b 40m $f $TMP/$z.part. && touch $TMP/${z}_parts.ok; }
done
up() { for i in 1 2 3; do colab upload -s $SES $1 /content/parts/${1:t} >/dev/null 2>&1 && return 0; sleep 3; done; echo "FAILED $1"; }
n=0
for f in $TMP/*.part.*; do up $f & (( ++n % 4 == 0 )) && wait; done; wait
echo "$(date +%T) uploads done"
cat > $TMP/join.py <<PY
import os, glob, subprocess
for z, n in (('data', $(stat -f%z $D/20260876_MarlenMelis_data.zip)), ('app', $(stat -f%z $D/20260876_MarlenMelis_app.zip))):
    parts = sorted(glob.glob(f'/content/parts/{z}.part.*'))
    out = f'/content/20260876_MarlenMelis_{z}.zip'
    subprocess.run(f'cat {" ".join(parts)} > {out}', shell=True)
    print(z, len(parts), 'parts', os.path.getsize(out), 'bytes, expected', n, 'OK' if os.path.getsize(out) == n else 'MISMATCH')
print(subprocess.run('md5sum /content/20260876_*.zip; ls -l /content/generalist.ckpt; tail -c 300 /content/gdown.log',
                     shell=True, capture_output=True, text=True).stdout)
PY
colab exec -s $SES -f $TMP/join.py
md5 -q $D/20260876_MarlenMelis_data.zip $D/20260876_MarlenMelis_app.zip
echo "$(date +%T) waiting for the checkpoint"
cat > $TMP/wait.py <<'PY'
import os, time
t = time.time()
while time.time() - t < 1500 and not (os.path.exists('/content/generalist.ckpt') and os.path.getsize('/content/generalist.ckpt') == 2967114775):
    time.sleep(5)
print('ckpt', os.path.getsize('/content/generalist.ckpt') if os.path.exists('/content/generalist.ckpt') else None)
PY
colab exec -s $SES --timeout 1600 -f $TMP/wait.py
echo "$(date +%T) setup + start app"
colab exec -s $SES --timeout 1800 -f $HERE/colab_cli_live.py
echo "$(date +%T) done"
