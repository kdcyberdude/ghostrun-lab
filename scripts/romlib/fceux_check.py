"""Does FCEUX run each ROM? Headless FCEUX (xvfb) plays fceux_boot_check.lua: 30 s from power-on with Start/A taps.

usage: python fceux_check.py OUT.jsonl --list FILE_OF_ROM_PATHS [--workers 16] [--frames 1800]
One JSON line per ROM: rom, ran (the Lua result was written), screens (distinct screens sampled every 0.5 s),
rams (distinct RAM states), colours (on the last screen), log tail. A ROM FCEUX cannot load writes no result;
one that loads but never draws shows screens <= 1 / colours <= 1.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
FCEUX = os.environ.get('FCEUX', '/usr/games/fceux')


def check(rom, frames):
    d = tempfile.mkdtemp()
    try:
        env = dict(os.environ, HOME=d, BOOT_OUT=f'{d}/r.txt', BOOT_FRAMES=str(frames))
        try:
            p = subprocess.run(['xvfb-run', '-a', FCEUX, '--sound', '0', '--loadlua', str(HERE / 'fceux_boot_check.lua'),
                                rom], env=env, capture_output=True, text=True, timeout=300)
            log = (p.stdout + p.stderr).strip().splitlines()
        except subprocess.TimeoutExpired:
            log = ['timeout']
        r = dict(rom=Path(rom).name, ran=False, log=' | '.join(log[-3:])[:300])
        if Path(f'{d}/r.txt').exists():
            n, s, m, c = map(int, Path(f'{d}/r.txt').read_text().split())
            r.update(ran=True, screens=s, rams=m, colours=c)
        return r
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--list', required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--frames', type=int, default=1800)
    a = ap.parse_args()
    roms = [r for r in Path(a.list).read_text().splitlines() if r]
    done = {json.loads(l)['rom'] for l in open(a.out)} if Path(a.out).exists() else set()
    todo = [r for r in roms if Path(r).name not in done]
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, 'a') as fh:
        for r in ex.map(lambda x: check(x, a.frames), todo):
            fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
