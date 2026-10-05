"""Resumable run queue (PLAN §0.4): fine-tunes and prediction caches, one job at a time, one GPU.

    python src/run_queue.py --queue queue_gpu0.json --log <drive>/logs/queue_gpu0.log [--budget_h 11]

The queue file is a JSON list of jobs, each ["finetune" | "predict", [args...]]. Every job runs as its
own process (an OOM or crash cannot poison the next one); finetune.py skips finished runs and resumes
from _last.pt, predict.py skips existing caches, so re-running a queue only does what is missing.
A failed job is logged and the queue moves on. No new job starts after --budget_h hours.
Common arguments (--drive, --chameleon, ...) can be given after `--` and are appended to every job.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--budget_h', type=float, default=1e9)
    a, common = ap.parse_known_args()
    common = [c for c in common if c != '--']
    jobs = json.load(open(a.queue))
    t0 = time.time()
    with open(a.log, 'a') as log:
        def say(msg):
            log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}\n')
            log.flush()
            print(msg, flush=True)

        say(f'queue {a.queue}: {len(jobs)} jobs, common args {common}')
        for k, (script, args) in enumerate(jobs):
            if (time.time() - t0) / 3600 > a.budget_h:
                say(f'budget of {a.budget_h} h reached; stopping before job {k}')
                break
            cmd = [sys.executable, os.path.join(HERE, f'{script}.py'), *map(str, args), *common]
            say(f'[{k + 1}/{len(jobs)}] START {script} {" ".join(map(str, args))}')
            t = time.time()
            rc = subprocess.run(cmd, cwd=HERE).returncode
            say(f'[{k + 1}/{len(jobs)}] {"OK" if rc == 0 else f"FAILED rc={rc}"} after {(time.time() - t) / 60:.1f} min')
        say(f'DONE after {(time.time() - t0) / 3600:.2f} h')


if __name__ == '__main__':
    main()
