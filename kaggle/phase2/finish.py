"""After a Phase 2 session is pushed: wait for it, download its outputs, run every CPU step (read-only on Kaggle).

    python kaggle/phase2/finish.py curve|var [--every 600] [--no_wait] [--skip_download] [--dry]

1. polls `kaggle kernels status marlenmelis/rr-p2-<session>` every --every seconds until it is complete, errored or
   cancelled (network errors are retried: the Mac may sleep);
2. downloads the outputs with kaggle/fetch_output.py into results/phase2/download_<session> (resumable; also on
   error, as a session stopped by the environment check still leaves logs/envcheck.txt) and moves them into
   results/phase2/rr, where post.py reads both sessions' runs; the files both sessions write under the same name (the
   environment check, the two lane logs) go to rr/logs/<session>/ instead, so neither overwrites the other's;
3. stops with the environment check's verdict if it failed, else runs kaggle/phase2/post.py <session>.
Steps 2-3 hold a lock (~/rr_local/phase2_finish.lock): with curve and var running at once, their post-processing
(same report folders) never overlaps; the polling does not wait for it.
It pushes, uploads and changes nothing on Kaggle. Log: ~/rr_local/phase2_finish_<session>.log (tee'd to stdout).
"""
import argparse
import fcntl
import os
import re
import shutil
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


SESSION_FILES = re.compile(r'^(envcheck\.txt|b13_gpu\d+\.out)$')     # same name in every session's logs/


def merge(stage, dest, session):
    """move a downloaded session (stage/rr/..., stage/*.log) into the shared results folder dest (= results/phase2/rr);
    rr/logs/<file> that every session writes under one name -> dest/logs/<session>/<file>; returns the moved paths"""
    moved = []
    src_rr = os.path.join(stage, 'rr')
    for d, _, files in os.walk(src_rr):
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), src_rr)
            if os.path.dirname(rel) == 'logs' and SESSION_FILES.match(f):
                rel = os.path.join('logs', session, f)
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.replace(os.path.join(d, f), target)
            moved.append(rel)
    for f in os.listdir(stage):                                # the kernel's own log, <slug>.log
        if f.endswith('.log'):
            os.replace(os.path.join(stage, f), os.path.join(os.path.dirname(dest), f))
    shutil.rmtree(stage, ignore_errors=True)
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', choices=['curve', 'var', 'var_full'])
    ap.add_argument('--every', type=int, default=600, help='seconds between status checks')
    ap.add_argument('--no_wait', action='store_true', help='do not poll: download and post-process now')
    ap.add_argument('--skip_download', action='store_true',
                    help='the outputs are in results/phase2/rr already: only post-process (under the lock)')
    ap.add_argument('--dry', action='store_true', help='print the steps only')
    a = ap.parse_args()
    slug = f'marlenmelis/rr-p2-{a.session}'
    if not a.dry:
        sys.path.insert(0, os.path.join(ROOT, 'src'))
        import memguard
        memguard.ensure_running()                              # the post-processing must not freeze the Mac (§16.8)
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

    if not (a.no_wait or a.dry or a.skip_download):
        while True:
            st = status(slug)
            say(f'{slug}: {st or "status unavailable (network?), retrying"}')
            if st and any(st.startswith(d) for d in DONE):
                break
            time.sleep(a.every)
    out = os.path.join(ROOT, 'results', 'phase2')
    os.makedirs(out, exist_ok=True)
    lock = open(os.path.expanduser('~/rr_local/phase2_finish.lock'), 'w')
    say('waiting for the post-processing lock' if not a.dry else 'lock (skipped: dry run)')
    if not a.dry:
        fcntl.flock(lock, fcntl.LOCK_EX)                       # one session's download + post-processing at a time
    stage = os.path.join(out, f'download_{a.session}')
    if not a.skip_download:
        run([KPY, os.path.join(ROOT, 'kaggle', 'fetch_output.py'), slug, stage])
    if not (a.dry or a.skip_download):
        say(f'moved {len(merge(stage, os.path.join(out, "rr"), a.session))} files into results/phase2/rr')
    env = os.path.join(out, 'rr', 'logs', a.session, 'envcheck.txt')
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
