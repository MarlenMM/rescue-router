"""Batch 13.5: plan for the final predictions of the chosen model (full test in two halves on the two T4s + the
zero-label RescueNet eval100 transfer), then kaggle/b13/make_kernel.py builds the kernel.

    python kaggle/b13/make_final.py --ckpts floodnet_50shot_s0_stratified_b13ctl_step1000 [more for I22 averaging]
        [--opts '--crop_weights 1,3,3,3,1'] --dataset_src marlenmelis/rr-b13-ckpt

Each checkpoint is predicted separately (I22 averages on CPU afterwards). val30 caches already exist from rounds 1-2.
"""
import argparse, json, os, shlex, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def pr(ck, split_file, opts, query='floodnet'):
    return ['predict', ['--ckpt', f'{{OUT}}/ckpt/{ck}.ckpt', '--dataset', 'floodnet', '--query_dataset', query,
                        '--split_file', split_file, '--split', 'test', '--amp', 'fp16', '--chunk', 1, '--channel_chunk', 1, *opts]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--opts', default='')
    ap.add_argument('--dataset_src', required=True)
    ap.add_argument('--slug', default='rr-b13-final')
    a = ap.parse_args()
    cks, opts = a.ckpts.split(','), shlex.split(a.opts)
    lane0 = [[['predict', ['--ckpt', f'{{OUT}}/ckpt/{c}.ckpt', '--dataset', 'floodnet', '--split_file', 'floodnet_val_tune30',
                           '--split', 'val', '--amp', 'fp16', '--chunk', 1, '--channel_chunk', 1, *opts]]] for c in cks]   # consistency check
    lane0 += [[pr(c, 'floodnet_test_full_part1', opts)] for c in cks]
    lane1 = [[pr(c, 'floodnet_test_full_part2', opts)] for c in cks]
    for k, c in enumerate(cks):                 # transfer caches alternate between the lanes
        (lane0 if k % 2 else lane1).append([pr(c, 'rescuenet_test_eval100', opts, 'rescuenet')])
    plan = {'0': lane0, '1': lane1}
    f = os.path.join(HERE, f'plan_{a.slug}.json')
    json.dump(plan, open(f, 'w'), indent=1)
    setup = ("with tarfile.open(find('rescuenet_r768.tar')) as t:\n    t.extractall('/content/rr_data', filter='data')\n"
             "for _f in glob.glob('/kaggle/input/**/*_b13*.ckpt', recursive=True):\n    shutil.copy(_f, f'{OUT}/ckpt/')\n"
             "sh(f'ls -la {OUT}/ckpt; ls /content/rr_data')")
    subprocess.run([sys.executable, os.path.join(HERE, 'make_kernel.py'), '--slug', a.slug, '--plan', f, '--start_limit', '9',
                    '--setup', setup], check=True)
    p = os.path.expanduser(f'~/rr_push/{a.slug}/kernel-metadata.json')
    m = json.load(open(p))
    m['dataset_sources'] = [a.dataset_src]
    json.dump(m, open(p, 'w'), indent=1)
    print(json.dumps(plan)[:600], '\n', m['dataset_sources'], m['kernel_sources'])


if __name__ == '__main__':
    main()
