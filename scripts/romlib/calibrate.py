"""Is a NeSLE/fceumm divergence a NeSLE bug, or does the game just amplify tiny timing differences?

For each ROM, plays gpu_check's buttons on three emulators: fceumm (stable_retro), FCEUX 2.6 (headless, Lua RAM dump,
rl/scripts/fceux_trace.lua) and NeSLE. Reports when each pair first diverges (> DIVERGED differing bytes outside the
stack page). If fceumm and FCEUX (two mature emulators) also diverge early, the game is timing sensitive and NeSLE's
divergence alone says nothing; if NeSLE matches either reference, it plays like a real emulator.

usage: python calibrate.py OUT.jsonl --list ROMS.txt [--workers 12] [--frames 3600]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import gpu_check as g

ROOT = Path(__file__).resolve().parents[2]


def fceux_ram(rom, pad):
    d = tempfile.mkdtemp()
    try:
        pad.tofile(f'{d}/b.bin')
        env = dict(os.environ, HOME=d, TRACE_BUTTONS=f'{d}/b.bin', TRACE_OUT=f'{d}/r.bin', TRACE_FRAMES=str(len(pad)))
        subprocess.run(['xvfb-run', '-a', '/usr/games/fceux', '--sound', '0', '--ramInit', '1', '--loadlua',
                        str(ROOT / 'rl/scripts/fceux_trace.lua'), rom], env=env, capture_output=True, timeout=600)
        p = Path(f'{d}/r.bin')
        return np.fromfile(p, np.uint8).reshape(-1, 2048) if p.exists() else np.zeros((0, 2048), np.uint8)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def nesle_ram(rom, pad):
    d = tempfile.mkdtemp()
    try:
        pad.tofile(f'{d}/b.bin')
        subprocess.run([str(g.TRACE), rom, f'{d}/b.bin', f'{d}/r.bin', 'ff', '0', 'vblank'], capture_output=True,
                       timeout=600)
        p = Path(f'{d}/r.bin')
        return np.fromfile(p, np.uint8).reshape(-1, 2048) if p.exists() else np.zeros((0, 2048), np.uint8)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def diverge(a, b, shifts=(-3, -2, -1, 0, 1, 2, 3)):
    """First frame where the differing bytes (stack page left out) stay above the pair's noise floor + DIVERGED for
    30 frames; the noise floor is the median over frames 60-360 (emulators differ in sound/timing scratch bytes even
    on correct games). Best frame shift wins. None = never diverged; -1 = no frames to compare."""
    cols = list(range(0x100)) + list(range(0x200, 0x800))
    best = -1
    for s in shifts:
        x, y = (a[s:], b) if s >= 0 else (a, b[-s:])
        n = min(len(x), len(y))
        if n < 400:
            continue
        per = (x[:n, cols] != y[:n, cols]).sum(1)
        bad = per > np.median(per[60:360]) + g.DIVERGED
        run = np.convolve(bad.astype(int), np.ones(30, int), 'valid') == 30
        d = int(np.argmax(run)) if run.any() else None
        if d is None:
            return None
        best = max(best, d)
    return best


def run(rom, frames):
    pad = g.buttons(rom, frames)
    ref, fx, ne = g.fceumm_ram(rom, pad), fceux_ram(rom, pad), nesle_ram(rom, pad)
    return dict(rom=Path(rom).name, fceumm_fceux=diverge(ref, fx), nesle_fceumm=diverge(ne, ref),
                nesle_fceux=diverge(ne, fx), fceux_frames=len(fx), nesle_frames=len(ne))


def _w(a):
    try:
        return run(*a)
    except Exception as e:  # noqa: BLE001
        return dict(rom=Path(a[0]).name, error=repr(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--list', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--frames', type=int, default=3600)
    a = ap.parse_args()
    roms = [r for r in Path(a.list).read_text().splitlines() if r]
    done = {json.loads(l)['rom'] for l in open(a.out)} if Path(a.out).exists() else set()
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, 'a') as fh:
        for r in pool.imap_unordered(_w, [(r, a.frames) for r in roms if Path(r).name not in done]):
            fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
