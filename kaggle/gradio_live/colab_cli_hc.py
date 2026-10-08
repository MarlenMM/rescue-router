"""VM-side health check for the live link: app process, GPU memory, log tail, uploads served (new scene folders)."""
import os, subprocess
up = '/tmp/rr_viewer/scenes/up'
toks = sorted(os.listdir(up), key=lambda t: os.path.getmtime(f'{up}/{t}')) if os.path.isdir(up) else []
seen = set(open('/content/uploads_seen.txt').read().split()) if os.path.exists('/content/uploads_seen.txt') else set()
new = [t for t in toks if t not in seen]
if new:
    open('/content/uploads_seen.txt', 'a').write(''.join(t + '\n' for t in new))
print('uploads total', len(seen) + len(new), 'new', len(new), 'latest', toks[-1:] )
print(subprocess.run('pgrep -af "src/gradio_(viewer|app).py" | grep -v pgrep | cut -c1-80; nvidia-smi --query-gpu=memory.used --format=csv,noheader; tail -n 4 /content/gradio_live.log | cut -c1-200', shell=True, capture_output=True, text=True).stdout)
