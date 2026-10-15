"""After a Phase 2 session is pushed: wait for it, download its outputs, run every CPU step (read-only on Kaggle).

    python kaggle/phase2/finish.py curve|var [--every 600] [--no_wait] [--dry]

1. polls `kaggle kernels status marlenmelis/rr-p2-<session>` every --every seconds until it is complete, errored or
   cancelled (network errors are retried: the Mac may sleep);
2. downloads the outputs with kaggle/fetch_output.py into results/phase2 (resumable; also on error, as a session
   stopped by the environment check still leaves logs/envcheck.txt);
3. stops with the environment check's verdict if it failed, else runs kaggle/phase2/post.py <session>.
It pushes, uploads and changes nothing on Kaggle. Log: ~/rr_local/phase2_finish_<session>.log (tee'd to stdout).
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KAGGLE = os.path.expanduser('~/.venvs/kaggle/bin/kaggle')
KPY = os.path.expanduser('~/.venvs/kaggle/bin/python')
DONE = ('complete', 'error', 'cancel')                   # KernelWorkerStatus values (lower case) that end a session


def status(slug):
    """the kernel's status word ('running', 'complete', ...); None on a network / API error"""
    r = subprocess.run([KAGGLE, 'kernels', 'status', slug], capture_output=True, text=True)
    if r.returncode:
        return None
    low = r.stdout.lower()
    for word in ('complete', 'error', 'cancelacknowledged', 'cancelrequested', 'cancel', 'running', 'queued'):
        if word in low:
            return word
    return low.strip()[-40:] or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', choices=['curve', 'var', 'var_full'])
    ap.add_argument('--every', type=int, default=600, help='seconds between status checks')
    ap.add_argument('--no_wait', action='store_true', help='do not poll: download and post-process now')
    ap.add_argument('--dry', action='store_true', help='print the steps only')
    a = ap.parse_args()
    slug = f'marlenmelis/rr-p2-{a.session}'
    log = open(os.path.expanduser(f'~/rr_local/phase2_finish_{a.session}.log'), 'a')

    def say(msg):
        line = f'{time.strftime("%m-%d %H:%M")} {msg}'
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()

    def run(cmd):
        say('$ ' + ' '.join(cmd))
        if not a.dry:
            subprocess.run(cmd, check=True, cwd=ROOT)

    if not (a.no_wait or a.dry):
        while True:
            st = status(slug)
            say(f'{slug}: {st or "status unavailable (network?), retrying"}')
            if st and any(st.startswith(d) for d in DONE):
                break
            time.sleep(a.every)
    out = os.path.join(ROOT, 'results', 'phase2')
    run([KPY, os.path.join(ROOT, 'kaggle', 'fetch_output.py'), slug, out])
    env = os.path.join(out, 'rr', 'logs', 'envcheck.txt')
    lines = open(env).read().splitlines() if os.path.exists(env) else []
    verdict = next((x for x in lines if x.startswith('ENVCHECK')), None)
    say(verdict or 'no environment check verdict in the outputs')
    if verdict and 'FAIL' in verdict and not a.dry:
        sys.exit('environment check failed: no post-processing (PROTOCOL amendment 2: an infrastructure failure; '
                 'find the cause, then re-run the session once)')
    run([sys.executable, os.path.join(HERE, 'post.py'), a.session])
    say('done: read results/phase2/report/phase2.md and results/baselines/report/; refresh the backup')


if __name__ == '__main__':
    main()
