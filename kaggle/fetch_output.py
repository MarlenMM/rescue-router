"""Download a Kaggle notebook's output files with resume + retries (the CLI's single streamed GET
breaks on large files with IncompleteRead). Run with the Kaggle venv:

    ~/.venvs/kaggle/bin/python kaggle/fetch_output.py marlenmelis/rr-gpu-queue results/kaggle_v2 [regex]

Lists the output (all pages), then fetches each file with `curl --http1.1 -C -`, one fresh curl per attempt
(each resumes from the bytes already on disk; HTTP/2 streams kept dying). Also saves the notebook log as <slug>.log. Signed URLs are never printed.
"""

import os
import re
import subprocess
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def list_output(api, owner, slug):
    files, log, token = [], None, None
    with api.build_kaggle_client() as k:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug, req.page_size = owner, slug, 200
            if token:
                req.page_token = token
            resp = k.kernels.kernels_api_client.list_kernel_session_output(req)
            files += list(resp.files or [])
            log = log or resp.log
            token = resp.next_page_token
            if not token:
                return files, log


def main():
    kernel, out = sys.argv[1], sys.argv[2]
    pat = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
    owner, slug = kernel.split('/')[:2]
    api = KaggleApi()
    api.authenticate()
    files, log = list_output(api, owner, slug)
    os.makedirs(out, exist_ok=True)
    if log:
        open(os.path.join(out, f'{slug}.log'), 'w').write(log)
    bad = []
    for f in files:
        if pat and not pat.search(f.file_name):
            continue
        dst = os.path.join(out, f.file_name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        size = getattr(f, 'size', None) or getattr(f, 'total_bytes', None)
        if size and os.path.exists(dst) and os.path.getsize(dst) == size:
            print(f'= {f.file_name}', flush=True)
            continue
        for attempt in range(60):      # a fresh curl per attempt resumes from the current file size
            rc = subprocess.run(['curl', '-sS', '-L', '--http1.1', '-C', '-', '-o', dst, f.url]).returncode
            if rc in (0, 33):              # 33 = range not satisfiable: the file is already complete
                rc = 0
                break
            time.sleep(3)
        got = os.path.getsize(dst) if os.path.exists(dst) else -1
        ok = rc == 0 and (not size or got == size)
        print(f'{"+" if ok else "!"} {f.file_name} {got} B' + ('' if ok else f' (rc={rc}, expected {size})'), flush=True)
        if not ok:
            bad.append(f.file_name)
    print(f'{len(files)} files listed, {len(bad)} failed: {bad}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
