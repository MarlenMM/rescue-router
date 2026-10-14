"""Read-only launch check for a Phase 2 session (experiments/phase2/PROTOCOL.md). Pushes and uploads nothing.

    python kaggle/phase2/launch_check.py curve|var

Checks, in the checklist's order: the code is committed and pushed (the kernel embeds git HEAD); the session's
pre-flight log ends with PREFLIGHT OK (~/rr_local/preflight_<session>*.log, newer than the last change to src/);
the kernel folder was built from HEAD with the current plan; the private checkpoint dataset is staged (curve).
Then prints the remaining manual steps (Kaggle quota in the browser, dataset upload, kernel push) as commands.
"""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from make_plan import lanes, session  # noqa: E402

PUSH = os.path.expanduser('~/rr_push')
CKPT_DS = os.path.join(PUSH, 'rr-final-ckpt')


def git(*args):
    return subprocess.run(['git', '-C', ROOT, *args], capture_output=True, text=True).stdout.strip()


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'curve'
    ok, notes = True, []

    def check(cond, msg, fix=''):
        nonlocal ok
        print(('  OK   ' if cond else '  FAIL ') + msg + ('' if cond or not fix else f'\n         -> {fix}'))
        ok &= bool(cond)

    head = git('rev-parse', '--short', 'HEAD')
    dirty = git('status', '--porcelain', '--', 'src', 'splits', 'supports', 'kaggle')
    check(not dirty, f'src/, splits/, supports/, kaggle/ committed (HEAD {head})', 'commit first: the kernel embeds HEAD')
    git('fetch', '-q', 'origin')
    check(git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main'), 'HEAD pushed to origin/main', 'git push')
    last_src = max(os.path.getmtime(f) for f in glob.glob(os.path.join(ROOT, 'src', '**', '*.py'), recursive=True))
    logs = sorted(glob.glob(os.path.expanduser(f'~/rr_local/preflight_{name}*.log')), key=os.path.getmtime)
    good = [f for f in logs if 'PREFLIGHT OK' in open(f).read() and os.path.getmtime(f) > last_src]
    check(good, f'pre-flight OK after the last change to src/ ({os.path.basename(good[-1]) if good else "none"})',
          f'RR_DEVICE=mps python kaggle/phase2/make_plan.py preflight {name} --jobs {len(session(name)) - (name == "curve")}'
          f' > ~/rr_local/preflight_{name}_<date>.log' + ('  (curve: the last group, the 50-support prediction, '
                                                        'swaps on a 24-GB Mac and is left out)' if name == 'curve' else ''))
    kdir = os.path.join(PUSH, f'rr-p2-{name}')
    built = os.path.exists(os.path.join(kdir, 'run_b13_kaggle.py'))
    if built:
        s = open(os.path.join(kdir, 'run_b13_kaggle.py')).read()
        import base64
        import io
        import tarfile
        t = tarfile.open(fileobj=io.BytesIO(base64.b64decode(re.search(r"CODE_B64 = '([^']+)'", s).group(1))))
        version = t.extractfile('./VERSION').read().decode().strip()
        m = re.search(r'json\.loads\(("(?:[^"\\]|\\.)*")\)\.items\(\)', s)
        plan = json.loads(json.loads(m.group(1)))
        want, _ = lanes(session(name))
        norm = lambda pl: {str(k): [[[sc, [str(x) for x in a]] for sc, a in g] for g in v] for k, v in pl.items()}  # noqa: E731
        same = norm(plan) == norm(want)
        check(version == head, f'kernel built from HEAD (embedded {version})',
              f'python kaggle/phase2/make_plan.py kernel {name}')
        check(same, 'kernel plan = the current session plan', f'python kaggle/phase2/make_plan.py kernel {name}')
        meta = json.load(open(os.path.join(kdir, 'kernel-metadata.json')))
        check(meta.get('is_private') and meta.get('enable_gpu') and meta.get('machine_shape') == 'NvidiaTeslaT4',
              'kernel metadata: private, GPU, T4')
    else:
        check(False, f'kernel folder {kdir}', f'python kaggle/phase2/make_plan.py kernel {name}')
    if name == 'curve':
        ck = glob.glob(os.path.join(CKPT_DS, '**', 'floodnet_50shot_s0_stratified.ckpt'), recursive=True)
        check(ck and os.path.exists(os.path.join(CKPT_DS, 'dataset-metadata.json')),
              f'checkpoint dataset staged in {CKPT_DS}', 'see PLAN §16.4 cycle 2 (stage the submitted checkpoint)')
        notes.append(f'cd {CKPT_DS} && ~/.venvs/kaggle/bin/kaggle datasets create -p .   # once (private dataset)')
    _, load = lanes(session(name))
    print(f'\n{"READY" if ok else "NOT READY"}: {name}, {len(session(name))} job groups, lanes {[round(x) for x in load]} min, '
          f'~{(max(load) + 15) / 60:.1f} h of quota')
    print('Manual steps (Marlen\'s go first; memory phase2-gate):\n  1. Kaggle GPU quota in the browser (memory '
          'kaggle-operations)\n' + ''.join(f'  {k}. {n}\n' for k, n in enumerate(notes, 2))
          + f'  {len(notes) + 2}. cd {kdir} && ~/.venvs/kaggle/bin/kaggle kernels push -p .\n'
          f'  then: kaggle/fetch_output.py marlenmelis/rr-p2-{name} results/phase2; python kaggle/phase2/post.py {name}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
