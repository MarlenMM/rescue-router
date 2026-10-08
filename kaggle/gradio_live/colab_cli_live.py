"""Rescue Router live Gradio app on a Colab T4 driven by the official Colab CLI (no browser, no Drive): the CLI copy
of colab_gradio_live.py (8 Oct presentation link). Run from the Mac with kaggle/gradio_live/colab_cli_up.sh, or:
    colab exec -s live --timeout 1800 -f kaggle/gradio_live/colab_cli_live.py
Needs on the VM: /content/20260876_*_{data,app}.zip (colab upload) and /content/generalist.ckpt (gdown).
Setup is cached in /content/rr, so running it again only restarts the app (kills the old one; new share link).
The app runs detached (setsid + timeout) and stops by itself at END_UTC; log /content/gradio_live.log.
"""
import datetime, glob, os, re, subprocess, sys, time
END_UTC = datetime.datetime(2026, 10, 8, 10, 5, tzinfo=datetime.timezone.utc)   # 19:05 KST
GRADIO = 'gradio==6.29.1'
T0 = time.time()


def sh(cmd, tail=1500):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f'$ {cmd[:110]} [rc={r.returncode}]\n' + (r.stdout + r.stderr)[-tail:], flush=True)
    return r


def find(p):
    hits = sorted(glob.glob(p))
    return hits[0] if hits else None


DATA_ZIP, APP_ZIP, META = find('/content/20260876_*_data.zip'), find('/content/20260876_*_app.zip'), '/content/generalist.ckpt'
assert DATA_ZIP, 'upload 20260876_MarlenMelis_data.zip to /content'
assert os.path.exists(META) and os.path.getsize(META) == 2967114775, 'generalist.ckpt missing or incomplete'
RR = '/content/rr'
if not os.path.exists(f'{RR}/BUNDLE.json'):
    sh(f'unzip -q -o "{DATA_ZIP}" -d {RR}')
if APP_ZIP and not os.path.exists(f'{RR}/viewer/index.html'):
    sh(f'unzip -q -o "{APP_ZIP}" "rr_data/*" "viewer/*" "src/gradio_viewer.py" -d {RR}')
APP = os.environ.get('RR_APP') or (f'{RR}/src/gradio_viewer.py' if os.path.exists(f'{RR}/viewer/index.html')
                                   else f'{RR}/src/gradio_app.py')
if not os.path.exists(f'{RR}/results/kaggle_v2/rr/preds/floodnet_50shot_s0_stratified_floodnet_test_eval100_fp16.npz'):
    sh(f'cd {RR} && {sys.executable} src/cache_pack.py unpack --src results/kaggle_v2/rr_packed --workers {os.cpu_count()}',
       tail=300)
if not os.path.isdir('/content/starter'):
    sh('git clone -q --depth 1 https://github.com/GitGyun/AIC200-Project-StructuredPrediction.git /content/starter')
if not os.path.exists('/content/.pip_done'):
    import numpy
    r = sh(f'{sys.executable} -m pip install -q pytorch_lightning==2.4.0 timm==0.6.13 easydict einops scikit-image '
           f'pycocotools fastremap numba flow_vis segment-anything "{GRADIO}" "numpy=={numpy.__version__}"', tail=400)
    if r.returncode == 0:
        open('/content/.pip_done', 'w').close()
SAM = '/content/sam_vit_b_01ec64.pth'
if not os.path.exists(SAM):
    import urllib.request
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', SAM + '.part')
    os.rename(SAM + '.part', SAM)
print(f'setup done in {(time.time() - T0) / 60:.1f} min', flush=True)

# (re)start the app, detached from the kernel
sh('pkill -f "src/gradio_(viewer|app).py" ; sleep 2; rm -f /content/share_url.txt', tail=200)
left = int(END_UTC.timestamp() - time.time())
assert left > 60, 'past the end time'
LOG = '/content/gradio_live.log'
args = (f'--live --share --ckpt {RR}/results/kaggle_v2/rr/ckpt/floodnet_50shot_s0_stratified.ckpt --mt_ckpt {META} '
        f'--chameleon /content/starter/chameleon --data {RR}/rr_data --runs /content/runs_data --amp fp16')
with open(LOG, 'a') as f:
    f.write(f'=== start {time.strftime("%Y-%m-%d %H:%M:%S")} UTC {os.path.basename(APP)} for {left} s\n')
subprocess.Popen(f'cd {RR} && RR_SAM_WEIGHTS={SAM} PYTHONUNBUFFERED=1 exec timeout {left} {sys.executable} -u {APP} '
                 f'{args} >> {LOG} 2>&1', shell=True, start_new_session=True,
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
start = os.path.getsize(LOG)
url = None
while time.time() - T0 < 1700:
    time.sleep(3)
    m = re.search(r'https://[\w.-]+\.gradio\.live', open(LOG).read()[start:])
    if m:
        url = m.group(0)
        break
    if not subprocess.run('pgrep -f "src/gradio_(viewer|app).py"', shell=True, capture_output=True).stdout:
        break
if url:
    open('/content/share_url.txt', 'w').write(url + '\n')
    print('=' * 70 + f'\nSHARE LINK: {url}  ({os.path.basename(APP)})\nserving until {END_UTC:%H:%M} UTC\n' + '=' * 70)
else:
    print('NO SHARE LINK; log tail:\n' + open(LOG).read()[-3000:])
