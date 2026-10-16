"""Memory readings and worker-pool sizing (light: standard library only; PLAN §16.8).

16 Oct: router_sweep's 8 route_eval workers (~4.6 GB each on confirm100) next to a 7 GB baseline training filled the
Mac's 24 GB and 33 GB of swap; the machine froze and was restarted. Every pool of CPU workers now asks workers() how
many fit in the memory that is free, and src/memguard.py pauses the youngest heavy job when memory runs out anyway.

    available_gb()      memory that can be used without paging (macOS: free + inactive + speculative + purgeable
                        pages; Linux: MemAvailable); None when unknown
    workers(n, per_gb)  min(n, what fits in available_gb() - RESERVE_GB at per_gb each), at least 1
"""
import os
import re
import subprocess
import sys

RESERVE_GB = float(os.environ.get('RR_MEM_RESERVE_GB', 6))   # left for the desktop, the browser and the Claude app


def _sysctl(name):
    try:
        return subprocess.run(['sysctl', '-n', name], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def total_gb():
    v = _sysctl('hw.memsize')
    if v.isdigit():
        return int(v) / 1024 ** 3
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024 ** 3
    except (ValueError, OSError):
        return None


def available_gb():
    if sys.platform == 'darwin':
        try:
            out = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        page = int(re.search(r'page size of (\d+) bytes', out).group(1)) if 'page size of' in out else 16384
        pages = {k.strip(): int(v.strip().rstrip('.')) for k, v in re.findall(r'^([^:"]+):\s+(\d+)\.?$', out, re.M)}
        n = sum(pages.get(k, 0) for k in ('Pages free', 'Pages inactive', 'Pages speculative', 'Pages purgeable'))
        return n * page / 1024 ** 3 if pages else None
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def swap_used_gb():
    m = re.search(r'used = ([\d.]+)M', _sysctl('vm.swapusage'))
    return float(m.group(1)) / 1024 if m else None


def fit(n, per_gb, avail, reserve=RESERVE_GB):
    """how many of n workers needing per_gb each fit in avail GB, keeping reserve GB free (at least 1)"""
    if avail is None or per_gb <= 0:
        return max(1, n)
    return max(1, min(n, int((avail - reserve) // per_gb)))


def workers(n, per_gb, what='workers'):
    """n, or fewer when n workers of per_gb GB each would not fit in the memory that is free now"""
    avail = available_gb()
    k = fit(n, per_gb, avail)
    if k < n:
        print(f'[mem] {what}: {k} instead of {n} (~{per_gb:.1f} GB each, {avail:.1f} GB available, '
              f'{RESERVE_GB:.0f} GB kept free)', file=sys.stderr, flush=True)
    return k
