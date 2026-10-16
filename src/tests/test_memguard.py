"""The memory guard (src/memguard.py) and the worker sizing (rr.mem): 16 Oct, 8 route_eval workers froze the Mac."""
import argparse
import os
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import memguard as G  # noqa: E402
from rr import mem  # noqa: E402

REPO = '/Users/x/2nd semester/AIC200 project'
A = argparse.Namespace(pause_gb=2.5, critical_gb=1.0, resume_gb=8.0, settle=20)


def row(pid, ppid, cmd, gb=0.1, age=60, stat='SN'):
    return dict(pid=pid, ppid=ppid, cmd=cmd, gb=gb, age=age, stat=stat)


def test_fit_keeps_the_reserve_free():
    assert mem.fit(8, 5.0, avail=20.0, reserve=6) == 2
    assert mem.fit(8, 5.0, avail=4.0, reserve=6) == 1          # never 0: one worker always runs
    assert mem.fit(8, 0.5, avail=40.0, reserve=6) == 8
    assert mem.fit(8, 5.0, avail=None) == 8                    # unknown memory: as asked


def test_etime():
    assert G.etime_s('05') == 5 and G.etime_s('01:02') == 62 and G.etime_s('1:00:00') == 3600
    assert G.etime_s('2-01:00:00') == 2 * 86400 + 3600


def test_script_of_python_only_and_paths_with_spaces():
    py = '/opt/homebrew/Python.app/Contents/MacOS/Python'
    assert G.script_of(f'{py} {REPO}/src/run_decode.py --run x', REPO) == ('src/run_decode.py', False)
    assert G.script_of(f'{py} src/route_eval.py --dataset floodnet', REPO) == ('src/route_eval.py', True)
    assert G.script_of(f'caffeinate -is {py} src/job_queue.py q.txt', REPO) is None      # a wrapper, not the job
    assert G.script_of(f'{py} scripts/evaluate.py', REPO) is None                         # not our folders
    assert G.script_of(f'{py} src/memguard.py', REPO) is None                             # never itself
    assert G.script_of('uv run python src/x.py', REPO) is None


def test_jobs_leaf_scripts_with_their_workers_and_only_this_repository():
    py = '/opt/homebrew/Python'
    rows = [row(1, 0, 'launchd'),
            row(10, 1, f'{py} src/job_queue.py q.txt'),                                    # orchestrator
            row(11, 10, f'{py} src/baseline_seg.py --shot 35', gb=7.3, age=300),
            row(20, 1, f'{py} {REPO}/kaggle/phase2/post.py curve'),                        # orchestrator
            row(21, 20, f'{py} {REPO}/src/run_decode.py --sam', gb=1.0, age=100),
            row(22, 21, f'{py} -c from multiprocessing.spawn import spawn_main', gb=0.8),  # its pool worker
            row(30, 1, f'{py} src/train.py', gb=6.0),                                      # another project's src/
            row(40, 1, f'{py} scripts/evaluate.py', gb=5.7)]
    exists = lambda p: not p.endswith('train.py')                                         # noqa: E731
    cwd = {10: REPO, 11: REPO, 30: '/Users/x/other'}.get
    js = {j['pid']: j for j in G.jobs(rows, REPO, exists=exists, cwd=cwd)}
    assert set(js) == {11, 21}
    assert js[21]['pids'] == [21, 22] and abs(js[21]['gb'] - 1.8) < 1e-9
    exists_all = lambda p: True                                                            # noqa: E731
    assert 30 not in {j['pid'] for j in G.jobs(rows, REPO, exists=exists_all, cwd=cwd)}   # its cwd is elsewhere


def test_decide_pauses_the_youngest_never_the_last_and_resumes_the_oldest():
    old, young = dict(pid=1, age=900, gb=7.0), dict(pid=2, age=30, gb=4.6)
    assert G.decide([old, young], [], avail=2.0, swap_growth=0.5, since_last=60, a=A) == ('pause', young)
    assert G.decide([old, young], [], avail=2.0, swap_growth=0.0, since_last=60, a=A) is None    # low, not swapping
    assert G.decide([old, young], [], avail=0.5, swap_growth=0.0, since_last=60, a=A) == ('pause', young)
    assert G.decide([old], [], avail=0.5, swap_growth=1.0, since_last=60, a=A) is None           # the last one runs
    assert G.decide([old, young], [], avail=0.5, swap_growth=1.0, since_last=5, a=A) is None     # settling
    p1, p2 = dict(pid=3, age=500, gb=4.6), dict(pid=4, age=40, gb=4.6)
    assert G.decide([old], [p1, p2], avail=9.0, swap_growth=0.0, since_last=60, a=A) == ('resume', p1)
    assert G.decide([old], [p1, p2], avail=6.0, swap_growth=0.0, since_last=60, a=A) is None     # not enough yet
    big = dict(pid=5, age=500, gb=9.0)
    assert G.decide([old], [big], avail=9.0, swap_growth=0.0, since_last=60, a=A) is None        # needs its size + 2


def test_a_new_heavy_job_waits_for_the_others_and_the_post_processing():
    py = '/opt/homebrew/Python'
    rows = [row(1, 0, 'launchd'),
            row(10, 1, f'{py} src/job_queue.py q.txt'),
            row(11, 10, f'{py} src/baseline_seg.py --shot 40', gb=0.3, age=5),               # the one asking
            row(20, 1, f'{py} {REPO}/kaggle/phase2/post.py curve'),
            row(21, 20, f'{py} {REPO}/src/router_sweep.py', gb=0.1),
            row(22, 21, f'{py} src/route_eval.py', gb=4.4),
            row(30, 1, f'{py} scripts/evaluate.py', gb=5.7)]                                  # another project
    exists, cwd = (lambda p: True), (lambda pid: REPO)                                        # noqa: E731
    busy = G.others_busy(rows, 11, repo=REPO, exists=exists, cwd=cwd)
    assert sorted(busy) == ['kaggle/phase2/post.py', 'src/route_eval.py']
    idle = [r for r in rows if r['pid'] not in (20, 21, 22)]
    assert G.others_busy(idle, 11, repo=REPO, exists=exists, cwd=cwd) == []                               # never waits for others
