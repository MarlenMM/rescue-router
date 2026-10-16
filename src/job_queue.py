"""Restartable job queue for long local runs (the Mac's baseline jobs, PLAN §16.6).

The queue is a text file, one job per line: `name  command` (command run from the repository root; a command that
starts with `src/` runs with this Python). Blank lines and `#` comments are skipped. The file is re-read after every
job, so jobs can be appended while the queue runs. State lives outside the repository (--state, default
~/rr_local/queue/<queue name>/): `done` and `failed` list finished job names, `<name>.log` holds each job's output,
`status.log` the start/end times. A job that was killed (reboot, crash) is neither done nor failed and runs again on
the next start; baseline_seg.py resumes it from its last saved state. A failed job is not retried until its name is
removed from `failed`.

    caffeinate -is ~/.venvs/rr-cham/bin/python src/job_queue.py experiments/phase2/mac_queue.txt   # run / continue
    python src/job_queue.py experiments/phase2/mac_queue.txt --status                              # what is left
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jobs(path):
    out = []
    for line in open(path):
        line = line.split('#', 1)[0].strip()
        if line:
            name, cmd = line.split(None, 1)
            out.append((name, cmd))
    names = [n for n, _ in out]
    assert len(names) == len(set(names)), f'duplicate job names in {path}'
    return out


def listed(path):
    return set(open(path).read().split()) if os.path.exists(path) else set()


def stamp():
    return time.strftime('%m-%d %H:%M')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('queue')
    ap.add_argument('--state', default=None, help='state folder (default ~/rr_local/queue/<queue name>)')
    ap.add_argument('--status', action='store_true', help='print done / failed / pending and exit')
    a = ap.parse_args()
    state = a.state or os.path.expanduser(os.path.join('~/rr_local/queue', os.path.splitext(os.path.basename(a.queue))[0]))
    os.makedirs(state, exist_ok=True)
    f_done, f_failed, f_status, f_lock = (os.path.join(state, n) for n in ('done', 'failed', 'status.log', 'lock'))
    if a.status:
        done, failed = listed(f_done), listed(f_failed)
        for n, cmd in jobs(a.queue):
            print(f"{'done' if n in done else 'FAILED' if n in failed else 'pending':8s} {n:12s} {cmd}")
        if os.path.exists(f_lock):
            print(f'runner pid {open(f_lock).read().strip()} (lock {f_lock})')
        return
    if os.path.exists(f_lock):
        pid = int(open(f_lock).read() or 0)
        try:
            os.kill(pid, 0)
            sys.exit(f'another runner (pid {pid}) holds {f_lock}')
        except (ProcessLookupError, ValueError):
            pass                                    # stale lock from a reboot
    open(f_lock, 'w').write(str(os.getpid()))
    sys.path.insert(0, os.path.join(REPO, 'src'))
    import memguard
    try:
        while True:
            memguard.ensure_running()               # pauses the youngest heavy job when memory runs out (PLAN §16.8)
            done, failed = listed(f_done), listed(f_failed)
            todo = [(n, c) for n, c in jobs(a.queue) if n not in done and n not in failed]
            if not todo:
                open(f_status, 'a').write(f'{stamp()} queue empty\n')
                break
            name, cmd = todo[0]
            argv = shlex.split(os.path.expanduser(cmd))
            if argv[0].startswith('src/'):
                argv = [sys.executable] + argv
            open(f_status, 'a').write(f'{stamp()} start {name}\n')
            with open(os.path.join(state, f'{name}.log'), 'a') as log:
                log.write(f'== {stamp()} {" ".join(argv)}\n')
                log.flush()
                rc = subprocess.call(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            open(f_done if rc == 0 else f_failed, 'a').write(name + '\n')
            open(f_status, 'a').write(f'{stamp()} end {name} rc={rc}\n')
    finally:
        os.remove(f_lock)


if __name__ == '__main__':
    main()
