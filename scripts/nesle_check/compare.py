"""Compare fceumm and NeSLE RAM traces frame by frame.

usage: python compare.py TRACE_DIR GAME.. [--fill zero|ff|fceu] [--start-dot D] [--boundary wrap|vblank]
                         [--skip-addr A,B,..]
Prints the first differing frame and bytes, the share of identical frames, and the best frame offset.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
TRACE_BIN = Path(os.environ.get('NESLE_TRACE_BIN', HERE.parents[1] / '.cache/nesle-check/nesle_trace'))


def run(trace_dir, game, fill, start_dot=0, boundary='wrap'):
    meta = json.loads((trace_dir / f'{game}.json').read_text())
    out = trace_dir / f'{game}.nesle.ram.bin'
    p = subprocess.run([str(TRACE_BIN), meta['rom'], str(trace_dir / f'{game}.buttons.bin'), str(out), fill,
                        str(start_dot), boundary],
                       capture_output=True, text=True)
    ref = np.fromfile(trace_dir / f'{game}.ram.bin', np.uint8).reshape(-1, 2048)
    got = np.fromfile(out, np.uint8).reshape(-1, 2048) if out.exists() else np.zeros((0, 2048), np.uint8)
    return meta, p, ref, got


def summary(meta, p, ref, got, skip=()):
    n = min(len(ref), len(got))
    mask = np.ones(2048, bool)
    mask[list(skip)] = False
    diff = (ref[:n] != got[:n]) & mask
    bad = diff.any(1)
    first = int(np.argmax(bad)) if bad.any() else None
    res = dict(game=meta['game'], frames_ref=len(ref), frames_nesle=len(got), exit=p.returncode,
               nesle_msg=p.stderr.strip().splitlines()[-1] if p.stderr.strip() else '',
               identical_frames=float(1 - bad.mean()) if n else 0.0, first_diff_frame=first,
               boot_frames=meta['boot_frames'])
    if first is not None:
        addrs = np.flatnonzero(diff[first])
        res['first_diff'] = [(f'${a:04X}', int(ref[first, a]), int(got[first, a])) for a in addrs[:12]]
        res['n_diff_bytes_first'] = int(len(addrs))
        res['last_frame_diff_bytes'] = int(diff[n - 1].sum())
        # does NeSLE run a frame ahead or behind? compare a window around the first difference
        lo, hi = max(0, first - 5), min(n - 3, first + 60)
        best = {}
        for d in (-2, -1, 0, 1, 2):
            a0, a1 = max(lo, lo + d), min(hi, hi + d)
            if a1 > a0:
                best[d] = int(((ref[a0:a1] != got[a0 - d:a1 - d]) & mask).sum())
        res['offset_window_diff_bytes'] = best
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace_dir')
    ap.add_argument('games', nargs='+')
    ap.add_argument('--fill', default='zero')
    ap.add_argument('--start-dot', type=int, default=0)
    ap.add_argument('--boundary', default='wrap')
    ap.add_argument('--skip-addr', default='')
    a = ap.parse_args()
    skip = [int(x, 16) for x in a.skip_addr.split(',') if x]
    for g in a.games:
        meta, p, ref, got = run(Path(a.trace_dir), g, a.fill, a.start_dot, a.boundary)
        print(json.dumps(summary(meta, p, ref, got, skip)))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
