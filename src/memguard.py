"""Memory guard for the Mac's long jobs (PLAN §16.8): pauses the youngest heavy job when memory runs out, resumes it
when memory is back, so parallel work can never again push the machine into swap until it freezes.

16 Oct: router_sweep's 8 route_eval workers (~4.6 GB each) next to the Mac queue's 7 GB baseline training filled the
24 GB of memory and 33 GB of swap; the desktop stopped responding and the Mac was restarted. The pools now size
themselves to the free memory (rr.mem.workers); this guard catches whatever still adds up (two sessions' jobs, the
browser, a new script).

A job is a running Python process of this repository's scripts (src/, kaggle/, experiments/) that starts no other such
script, together with its own child processes (pool workers); it is heavy from HEAVY_GB of resident memory. Every
--every seconds: when the available memory (rr.mem.available_gb) is below --pause_gb while swap grows, or below
--critical_gb, the youngest heavy job is stopped (SIGSTOP; never the last running one); when it is above --resume_gb
(and the paused job's own size + 2 GB), the oldest paused job continues (SIGCONT), one at a time, --settle seconds
apart. Stopping loses nothing: the job continues where it was. On exit (Ctrl-C, SIGTERM, an error) every paused job
is continued; a new guard continues the jobs a killed one left paused.

    python src/memguard.py                # run in the foreground (one instance; log ~/rr_local/memguard.log)
    python src/memguard.py --status       # readings, jobs, paused jobs
    python src/memguard.py --resume_all   # continue every job a guard paused
finish.py, post.py and job_queue.py start it in the background (ensure_running) when none runs.
"""
import argparse
import atexit
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from rr import mem  # noqa: E402

STATE_DIR = os.path.expanduser(os.environ.get('RR_MEMGUARD_DIR', '~/rr_local'))
LOCK, STATE, LOG = (os.path.join(STATE_DIR, f) for f in ('memguard.lock', 'memguard.json', 'memguard.log'))
HEAVY_GB = 1.0
NOT_JOBS = ('memguard.py',)


def etime_s(s):
    """ps etime ([[dd-]hh:]mm:ss) in seconds"""
    days, _, rest = s.rpartition('-')
    parts = [int(x) for x in rest.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def footprints():
    """pid -> physical footprint in GB of the 80 largest processes (top's MEM). Unlike ps's resident size it counts
    what a process holds on the GPU (MPS memory is wired: a baseline training showed 0.3 GB resident, 7.5 GB here)"""
    try:
        out = subprocess.run(['top', '-l', '1', '-o', 'mem', '-n', '80', '-stats', 'pid,mem'], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    unit = {'B': 1 / 1024 ** 3, 'K': 1 / 1024 ** 2, 'M': 1 / 1024, 'G': 1, 'T': 1024}
    fp = {}
    for pid, num, u in re.findall(r'^\s*(\d+)\s+([\d.]+)([BKMGT])[+-]?\s*$', out, re.M):
        fp[int(pid)] = float(num) * unit[u]
    return fp


def ps():
    out = subprocess.run(['ps', '-axww', '-o', 'pid=,ppid=,etime=,rss=,stat=,command='], capture_output=True,
                         text=True).stdout
    fp = footprints()
    rows = []
    for line in out.splitlines():
        f = line.split(None, 5)
        if len(f) == 6:
            pid = int(f[0])
            rows.append(dict(pid=pid, ppid=int(f[1]), age=etime_s(f[2]), gb=fp.get(pid, int(f[3]) / 1024 ** 2),
                             stat=f[4], cmd=f[5]))
    return rows


def cwd_of(pid):
    try:
        out = subprocess.run(['lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn'], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return next((x[1:] for x in out.splitlines() if x.startswith('n')), None)


def script_of(cmd, repo=REPO):
    """(the repository script a Python command line runs, relative to repo; True when the path was relative), or None.
    Only Python executables count (not the caffeinate / nice / uv wrappers around them); the repository path may
    contain spaces, so an absolute script path is found by its prefix"""
    exe = cmd.split()[0] if cmd.split() else ''
    if 'python' not in os.path.basename(exe).lower():
        return None
    m = re.search(re.escape(repo.rstrip('/') + '/') + r'(\S+?\.py)(?:\s|$)', cmd)
    cands = [(m.group(1), False)] if m else [(t, True) for t in cmd.split()[1:] if t.endswith('.py')
                                             and not os.path.isabs(t)]
    for rel, relative in cands:
        if rel.split('/')[0] in ('src', 'kaggle', 'experiments') and os.path.basename(rel) not in NOT_JOBS:
            return rel, relative
    return None


def jobs(rows, repo=REPO, exists=os.path.exists, cwd=cwd_of):
    """the jobs among ps rows: a repository script process without such a script below it, with all its descendants:
    [dict(pid, pids, age, gb, script, stopped)]. A relative script path counts only when the process runs in the
    repository (another project's src/ is never touched)"""
    kids = {}
    for r in rows:
        kids.setdefault(r['ppid'], []).append(r)
    by_pid = {r['pid']: r for r in rows}

    def below(pid):
        out, todo = [], list(kids.get(pid, []))
        while todo:
            r = todo.pop()
            out.append(r)
            todo += kids.get(r['pid'], [])
        return out

    scripts = {}
    for r in rows:
        hit = script_of(r['cmd'], repo)
        if hit and exists(os.path.join(repo, hit[0])) and (not hit[1] or cwd(r['pid']) == repo):
            scripts[r['pid']] = hit[0]
    out = []
    for pid, s in scripts.items():
        sub = below(pid)
        if any(r['pid'] in scripts for r in sub):              # an orchestrator (post.py, job_queue.py, router_sweep)
            continue
        r = by_pid[pid]
        out.append(dict(pid=pid, pids=[pid] + [x['pid'] for x in sub], age=r['age'], script=s,
                        gb=r['gb'] + sum(x['gb'] for x in sub), stopped=r['stat'].startswith('T')))
    return out


def decide(running, paused, avail, swap_growth, since_last, a):
    """one step of the policy: ('pause', job) | ('resume', job) | None
    running: heavy jobs not paused by the guard; paused: jobs it paused (dicts with gb = size when paused)"""
    if avail is None or since_last < a.settle:
        return None
    short = avail < a.critical_gb or (avail < a.pause_gb and (swap_growth or 0) > 0.05)
    if short and len(running) > 1:
        return 'pause', min(running, key=lambda j: j['age'])       # the youngest: it has done the least work
    if paused and not short:
        j = max(paused, key=lambda j: j['age'])                     # the oldest first
        if avail >= max(a.resume_gb, j['gb'] + 2):
            return 'resume', j
    return None


def others_busy(rows, me, heavy_gb=HEAVY_GB, repo=REPO, exists=os.path.exists, cwd=cwd_of):
    """what a new heavy job would compete with: this repository's other heavy jobs that are running, and the Phase 2
    post-processing (post.py) while it lives (its steps start and end; the queue should not slip in between them)"""
    mine = {me}
    by_pid = {r['pid']: r for r in rows}
    p = by_pid.get(me, {}).get('ppid')
    while p in by_pid:                                           # the waiting job's own parents (job_queue.py)
        mine.add(p)
        p = by_pid[p]['ppid']
    busy = [j['script'] for j in jobs(rows, repo, exists=exists, cwd=cwd) if j['pid'] not in mine and j['gb'] >= heavy_gb
            and not j['stopped']]
    busy += ['kaggle/phase2/post.py' for r in rows if r['pid'] not in mine
             and (script_of(r['cmd'], repo) or ('',))[0] == 'kaggle/phase2/post.py']
    return busy


def admit(need_gb, what, every=30, max_wait_h=12):
    """before a heavy job allocates (a GPU training: MPS memory is wired, so pausing it later frees nothing): wait while
    less than need_gb is available AND another heavy job of this repository runs; never waits for other programs"""
    t0, said = time.time(), False
    while time.time() - t0 < max_wait_h * 3600:
        avail = mem.available_gb()
        busy = [] if avail is None or avail >= need_gb else others_busy(ps(), os.getpid())
        if not busy:
            break
        if not said:
            print(f'[mem] {what} waits: {avail:.1f} GB available, needs {need_gb:.0f} GB; running: '
                  f'{", ".join(sorted(set(busy)))}', flush=True)
            said = True
        time.sleep(every)
    if said:
        print(f'[mem] {what} starts after {(time.time() - t0) / 60:.0f} min', flush=True)


def signal_all(pids, sig):
    for p in pids:
        try:
            os.kill(p, sig)
        except (ProcessLookupError, PermissionError):
            pass


def load_state():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {}


def save_state(st):
    tmp = STATE + '.tmp'
    json.dump(st, open(tmp, 'w'), indent=1)
    os.replace(tmp, STATE)


def ensure_running():
    """start the guard in the background unless one runs (called by finish.py, post.py, job_queue.py)"""
    os.makedirs(STATE_DIR, exist_ok=True)
    fd = open(LOCK, 'a')
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False                                             # a guard holds the lock
    fcntl.flock(fd, fcntl.LOCK_UN)
    fd.close()
    subprocess.Popen([sys.executable, os.path.abspath(__file__)], cwd=REPO, start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=open(LOG, 'a'))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every', type=float, default=5, help='seconds between readings')
    ap.add_argument('--pause_gb', type=float, default=2.5, help='pause below this while swap grows')
    ap.add_argument('--critical_gb', type=float, default=1.0, help='pause below this in any case')
    ap.add_argument('--resume_gb', type=float, default=8.0, help='resume above this (and the job size + 2 GB)')
    ap.add_argument('--settle', type=float, default=20, help='seconds between two actions')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--resume_all', action='store_true')
    a = ap.parse_args()
    os.makedirs(STATE_DIR, exist_ok=True)
    st = load_state()
    if a.status:
        rows = ps()
        print(f'available {mem.available_gb():.1f} GB, swap used {mem.swap_used_gb() or 0:.1f} GB')
        for j in sorted(jobs(rows), key=lambda j: -j['gb']):
            print(f"  job {j['pid']:>6} {j['gb']:5.1f} GB  age {j['age'] / 60:6.1f} min  {j['script']}"
                  + ('  (stopped)' if j['stopped'] else ''))
        print('paused by the guard:', [(p['pid'], p['script']) for p in st.get('paused', [])] or 'none')
        return
    if a.resume_all:
        for p in st.get('paused', []):
            signal_all(p['pids'], signal.SIGCONT)
            print('continued', p['pid'], p['script'])
        save_state({'paused': []})
        return
    lock = open(LOCK, 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit('a memory guard is running already')
    log = open(LOG, 'a')

    def say(msg):
        log.write(f'{time.strftime("%m-%d %H:%M:%S")} {msg}\n')
        log.flush()

    paused = st.get('paused', [])
    for p in paused:                                             # left paused by a guard that was killed
        signal_all(p['pids'], signal.SIGCONT)
        say(f"continued {p['pid']} {p['script']} (left paused by an earlier guard)")
    paused = []

    def resume_all(*_):
        for p in paused:
            signal_all(p['pids'], signal.SIGCONT)
            say(f"continued {p['pid']} {p['script']} (guard exits)")
        paused.clear()
        save_state({'paused': []})

    atexit.register(resume_all)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, lambda *_: sys.exit(0))
    say(f'started (pid {os.getpid()}; pause < {a.pause_gb} GB while swapping or < {a.critical_gb} GB, '
        f'resume > {a.resume_gb} GB)')
    last_swap, last_action = mem.swap_used_gb(), 0.0
    while True:
        rows = ps()
        alive = {r['pid'] for r in rows}
        paused[:] = [p for p in paused if p['pid'] in alive]
        all_jobs = jobs(rows)
        paused_pids = {p['pid'] for p in paused}
        running = [j for j in all_jobs if j['gb'] >= HEAVY_GB and j['pid'] not in paused_pids and not j['stopped']]
        for p in paused:                                         # a paused job's new children belong to it as well
            p['pids'] = sorted(set(p['pids']) | set(next((j['pids'] for j in all_jobs if j['pid'] == p['pid']), [])))
        avail, swap = mem.available_gb(), mem.swap_used_gb()
        growth = None if swap is None or last_swap is None else swap - last_swap
        last_swap = swap
        act = decide(running, paused, avail, growth, time.time() - last_action, a)
        if act:
            kind, j = act
            if kind == 'pause':
                signal_all(j['pids'], signal.SIGSTOP)
                paused.append(dict(pid=j['pid'], pids=j['pids'], age=j['age'], gb=j['gb'], script=j['script']))
            else:
                signal_all(j['pids'], signal.SIGCONT)
                paused.remove(j)
            last_action = time.time()
            say(f"{kind} {j['pid']} {j['script']} ({j['gb']:.1f} GB; available {avail:.1f} GB, swap "
                f"{swap or 0:.1f} GB{'' if growth is None else f' {growth:+.2f}'}; {len(running)} heavy jobs running)")
            save_state({'paused': paused})
        time.sleep(a.every)


if __name__ == '__main__':
    main()
