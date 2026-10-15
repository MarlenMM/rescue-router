"""Build a Batch 13 Kaggle kernel folder from run_b13_kaggle.py with the code tarball embedded (no rr-code dataset:
its new versions failed to mount on 7 Oct) and the lane plan given as JSON.

    python kaggle/b13/make_kernel.py --slug rr-b13-r2 --plan plan_r2.json [--sources marlenmelis/rr-b13-r1a,...] [--start_limit 2.5]

plan JSON: {"0": [job_group, ...], "1": [...]}; a job group = list of [script, [args...]] run in order (the next group
starts only before start_limit hours). Placeholders in args: {OUT}. Writes ~/rr_push/<slug>/.
"""
import argparse, base64, json, os, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slug', required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--sources', default='', help='kernel outputs to mount')
    ap.add_argument('--datasets', default='', help='datasets to mount (small private ones mount reliably)')
    ap.add_argument('--start_limit', type=float, default=3.0)
    ap.add_argument('--setup', default='', help='python lines run after setup (e.g. copy inputs)')
    ap.add_argument('--docker_image', default='', help='pin the Kaggle image (digest), e.g. the one of the 8 Oct runs')
    a = ap.parse_args()
    subprocess.run(['zsh', os.path.join(ROOT, 'kaggle/build_rr_code.sh')], check=True, capture_output=True)
    b64 = base64.b64encode(open(os.path.expanduser('~/rr_push/ds/rr_code.tar.gz'), 'rb').read()).decode()
    s = open(os.path.join(HERE, 'run_b13_kaggle.py')).read()
    old = s[s.index('try:\n    src = os.path.dirname(find'):s.index("VERSION = open")]
    s = s.replace(old, "import base64, io\nroot = '/content/rr_code'\nwith tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64))) as t:\n"
                       "    t.extractall(root, filter='data')\n")
    plan = json.load(open(a.plan))
    s = re.sub(r"SESSION = '\w+'", f"SESSION = 'custom'\nCODE_B64 = '{b64}'", s)
    s = re.sub(r"START_LIMIT_H = [\d.]+", f"START_LIMIT_H = {a.start_limit}", s)
    s = s.replace("PLAN = PLANS[SESSION]", "PLANS['custom'] = {int(k): [[[sc, [str(x).replace('{OUT}', OUT) for x in args]] for sc, args in grp] for grp in v]\n"
                  f"                   for k, v in json.loads({json.dumps(json.dumps(plan))}).items()}}\n" + a.setup + "\nPLAN = PLANS[SESSION]")
    out = os.path.expanduser(f'~/rr_push/{a.slug}')
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'run_b13_kaggle.py'), 'w').write(s)
    meta = dict(id=f'marlenmelis/{a.slug}', title=a.slug, code_file='run_b13_kaggle.py', language='python', kernel_type='script',
                is_private=True, enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4', dataset_sources=[x for x in a.datasets.split(',') if x],
                competition_sources=[], kernel_sources=['marlenmelis/rr-fetch-data'] + [x for x in a.sources.split(',') if x],
                model_sources=[])
    if a.docker_image:                  # the published runs' image: same torch / CUDA / cuDNN (kernels pull -m shows it)
        meta.update(docker_image_pinning_type='original', docker_image=a.docker_image)
    json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
    compile(s, 'run_b13_kaggle.py', 'exec')
    print('wrote', out, len(s), 'bytes; push: cd', out, '&& ~/.venvs/kaggle/bin/kaggle kernels push -p .')


if __name__ == '__main__':
    main()
