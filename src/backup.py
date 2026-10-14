"""Off-site backup of everything that is not in git (PLAN §16, fix F4).

Groups (each one archive, split into parts of < 2 GB so a GitHub release accepts them; the part names end in
`.part-aa`, `.part-ab`, ... so Kaggle does not unpack them):
  results    results/ (checkpoints, prediction caches, reports; 4 GB, untracked, inside iCloud Drive)
  rr_data    the three preprocessed data tars in ~/rr_data (the folders there are these tars unpacked)
  submission the KLMS zip as submitted on 8 Oct (~/rr_push/final)
  local      untracked project files: patched starter (chameleon/, chameleon-main/), course PDFs, slides PPTX, ...
Public downloads are only recorded (URL + SHA-256), not copied: SAM ViT-B weights, generalist.ckpt.

  python src/backup.py pack   [--out ~/rr_backup]      # archives + parts + release/backup/MANIFEST.tsv, PARTS.sha256
  python src/backup.py verify --root <dir> --group results   # files under <dir> against the manifest
  python src/backup.py restore --src <downloaded parts> --dest <dir>  # join, check, unpack, verify every file

MANIFEST.tsv (committed) lists every backed-up file with its size and SHA-256; PARTS.sha256 the parts themselves.
"""
import argparse
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser('~')
RR_DATA = os.environ.get('RR_DATA', os.path.join(HOME, 'rr_data'))
MAN_DIR = os.path.join(ROOT, 'release', 'backup')
PART = '1900m'
TAR_ENV = dict(os.environ, COPYFILE_DISABLE='1')
SKIP = {'.DS_Store', '__pycache__', '.pytest_cache', '.ruff_cache'}

# group -> (base directory the archive paths are relative to, member paths)
LOCAL_FILES = ['chameleon', 'chameleon-main', 'assets', 'tools', 'chameleon.ipynb', 'README_starter.md',
               '.gitignore.starter', '20260876_MarlenMelis_slides.pptx', 'AIC200_TA_Session_MiniProject_Chameleon.pdf',
               'AIC200_presentation_order.pdf', 'NEW_CHAT_PROMPT.md', 'NEW_CHAT_PROMPT_GRADIO.md',
               'NEW_CHAT_PROMPT_VIDEO.md', 'compass_artifact_wf-280fc84c-4184-8f72-b128-f5ffb289d4f8_text_markdown.md',
               'SUMMARY_FOR_NEW_CHAT.md']
GROUPS = {
    'results': (ROOT, ['results']),
    'rr_data': (RR_DATA, ['floodnet_r768.tar', 'rescuenet_r768.tar', 'floodnet_r1024_subset.tar']),
    'submission': (os.path.join(HOME, 'rr_push', 'final'), ['20260876_MarlenMelis_project.zip']),
    'local': (ROOT, LOCAL_FILES),
}
# not copied (public); restore by download and check the hash
EXTERNAL = [
    ('sam_vit_b_01ec64.pth', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
     os.path.join(RR_DATA, 'weights', 'sam_vit_b_01ec64.pth')),
]


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def walk(base, members):
    """Relative paths of every regular file under base/members (sorted, junk skipped)."""
    out = []
    for m in members:
        p = os.path.join(base, m)
        if not os.path.exists(p):
            sys.exit(f'missing: {p}')
        if os.path.isfile(p):
            out.append(m)
            continue
        for d, dirs, files in os.walk(p):
            dirs[:] = sorted(x for x in dirs if x not in SKIP)
            out += [os.path.relpath(os.path.join(d, f), base) for f in sorted(files) if f not in SKIP]
    return sorted(out)


def read_manifest():
    rows = {}
    with open(os.path.join(MAN_DIR, 'MANIFEST.tsv')) as f:
        next(f)
        for line in f:
            g, rel, size, h = line.rstrip('\n').split('\t')
            rows.setdefault(g, {})[rel] = (int(size), h)
    return rows


def pack(a):
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(MAN_DIR, exist_ok=True)
    groups = a.groups or list(GROUPS)
    old = read_manifest() if os.path.exists(os.path.join(MAN_DIR, 'MANIFEST.tsv')) else {}
    man = {g: v for g, v in old.items() if g not in groups}
    for g in groups:
        base, members = GROUPS[g]
        files = walk(base, members)
        man[g] = {rel: (os.path.getsize(os.path.join(base, rel)), sha256(os.path.join(base, rel))) for rel in files}
        print(f'{g}: {len(files)} files, {sum(s for s, _ in man[g].values()) / 1e9:.2f} GB', flush=True)
        for f in os.listdir(a.out):
            if f.startswith(f'{g}.tar.part-'):
                os.remove(os.path.join(a.out, f))
        lst = os.path.join(a.out, f'{g}.files')
        with open(lst, 'w') as f:
            f.write('\n'.join(files) + '\n')
        # bsdtar: no xattrs / AppleDouble files; the file list keeps junk (.DS_Store, caches) out
        tar = subprocess.Popen(['tar', '--no-xattrs', '--no-mac-metadata', '-cf', '-', '-C', base, '-T', lst],
                               stdout=subprocess.PIPE, env=TAR_ENV)
        subprocess.run(['split', '-b', PART, '-', os.path.join(a.out, f'{g}.tar.part-')], stdin=tar.stdout, check=True)
        tar.stdout.close()
        if tar.wait():
            sys.exit(f'tar failed for {g}')
        os.remove(lst)
    with open(os.path.join(MAN_DIR, 'MANIFEST.tsv'), 'w') as f:
        f.write('group\tpath\tsize\tsha256\n')
        for g in sorted(man):
            for rel in sorted(man[g]):
                f.write(f'{g}\t{rel}\t{man[g][rel][0]}\t{man[g][rel][1]}\n')
    parts = sorted(p for p in os.listdir(a.out) if '.tar.part-' in p)
    with open(os.path.join(MAN_DIR, 'PARTS.sha256'), 'w') as f:
        for p in parts:
            f.write(f'{sha256(os.path.join(a.out, p))}  {p}\n')
    with open(os.path.join(MAN_DIR, 'EXTERNAL.tsv'), 'w') as f:
        f.write('name\turl\tsha256\n')
        for name, url, path in EXTERNAL:
            f.write(f'{name}\t{url}\t{sha256(path) if os.path.exists(path) else "unknown"}\n')
    print(f'{len(parts)} parts in {a.out}; manifest in {MAN_DIR}')


def verify_tree(root, group, man):
    bad = 0
    for rel, (size, h) in man[group].items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p) or os.path.getsize(p) != size or sha256(p) != h:
            print(f'BAD {group}: {rel}')
            bad += 1
    print(f'{group}: {len(man[group]) - bad}/{len(man[group])} files match')
    return bad


def verify(a):
    sys.exit(1 if verify_tree(a.root, a.group, read_manifest()) else 0)


def restore(a):
    man = read_manifest()
    want = {}
    with open(os.path.join(MAN_DIR, 'PARTS.sha256')) as f:
        for line in f:
            h, p = line.split()
            want[p] = h
    bad = 0
    for g in a.groups or sorted(man):
        parts = sorted(p for p in want if p.startswith(f'{g}.tar.part-'))
        for p in parts:
            if sha256(os.path.join(a.src, p)) != want[p]:
                sys.exit(f'part checksum mismatch: {p}')
        dest = os.path.join(a.dest, g)
        os.makedirs(dest, exist_ok=True)
        cat = subprocess.Popen(['cat'] + [os.path.join(a.src, p) for p in parts], stdout=subprocess.PIPE)
        subprocess.run(['tar', '-xf', '-', '-C', dest], stdin=cat.stdout, check=True)
        cat.stdout.close()
        cat.wait()
        bad += verify_tree(dest, g, man)
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('pack')
    p.add_argument('--out', default=os.path.join(HOME, 'rr_backup'))
    p.add_argument('--groups', nargs='*', choices=list(GROUPS))
    p = sp.add_parser('verify')
    p.add_argument('--root', required=True)
    p.add_argument('--group', required=True, choices=list(GROUPS))
    p = sp.add_parser('restore')
    p.add_argument('--src', required=True)
    p.add_argument('--dest', required=True)
    p.add_argument('--groups', nargs='*', choices=list(GROUPS))
    a = ap.parse_args()
    {'pack': pack, 'verify': verify, 'restore': restore}[a.cmd](a)


if __name__ == '__main__':
    main()
