"""Compare two per-frame 2 KB RAM traces: best frame shift, first differing frame, share of identical frames.

usage: python rl/scripts/ram_compare.py A.ram.bin B.ram.bin [--max-shift 3]
"""
import argparse
import numpy as np


def load(p):
    return np.fromfile(p, np.uint8).reshape(-1, 2048)


def compare(a, b, max_shift=3):
    best = None
    for s in range(-max_shift, max_shift + 1):
        x, y = (a[s:], b) if s >= 0 else (a, b[-s:])
        n = min(len(x), len(y))
        same = np.all(x[:n] == y[:n], axis=1)
        first = int(np.argmin(same)) if not same.all() else n
        r = dict(shift=s, frames=n, first_diff=first, identical_frac=float(same.mean()))
        if best is None or (r['first_diff'], r['identical_frac']) > (best['first_diff'], best['identical_frac']):
            best = r
    return best


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('--max-shift', type=int, default=3)
    a = ap.parse_args()
    print(compare(load(a.a), load(a.b), a.max_shift))
