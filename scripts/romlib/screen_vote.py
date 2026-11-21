"""Third opinion for screen_check.py failures: which of fceumm and NeSLE draws the frame like FCEUX does?

For each ROM screen_check flagged, FCEUX (2.6, --ramInit 1, rl/scripts/fceux_trace.lua) plays the same buttons and
takes screenshots at the flagged frame +-6; the shot whose RAM is closest to fceumm's RAM at that frame is used
(FCEUX runs a frame or two off). Emulators use different RGB palettes, so pictures are compared by layout: the share of
pixels that one colour-to-colour mapping explains (1.0 = the same picture in other colours).

verdict: nesle_ok   FCEUX agrees with NeSLE (>= 0.97) and less with fceumm: the reference drew it wrong
         nesle_bad  FCEUX agrees with fceumm and not with NeSLE
         unclear    neither, or both (a frame in motion)

usage: python screen_vote.py OUT.jsonl --checks screen_check.jsonl --png-dir DIR [--only NAMES.txt] [--workers 16]
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/romlib'))
ROM_DIR = Path('/home/kd/Desktop/roms')
KEEP = np.ones(2048, bool)
KEEP[0x100:0x200] = False


def layout_match(a, b):
    """Share of pixels explained by the best colour -> colour mapping from a to b (palette-independent)."""
    ka = (a[..., 0].astype(np.int64) << 16 | a[..., 1].astype(np.int64) << 8 | a[..., 2]).ravel()
    kb = (b[..., 0].astype(np.int64) << 16 | b[..., 1].astype(np.int64) << 8 | b[..., 2]).ravel()
    best = collections.defaultdict(int)
    for (x, y), n in collections.Counter(zip(ka.tolist(), kb.tolist())).items():
        best[x] = max(best[x], n)
    return sum(best.values()) / len(ka)


def fceux_shots(rom, pad, frames):
    d = tempfile.mkdtemp()
    try:
        pad.tofile(f'{d}/b.bin')
        env = dict(os.environ, HOME=d, TRACE_BUTTONS=f'{d}/b.bin', TRACE_OUT=f'{d}/r.bin',
                   TRACE_FRAMES=str(max(frames) + 1), TRACE_SHOTS=','.join(map(str, frames)),
                   TRACE_SHOT_OUT=f'{d}/s.bin')
        subprocess.run(['xvfb-run', '-a', '/usr/games/fceux', '--sound', '0', '--ramInit', '1', '--loadlua',
                        str(ROOT / 'rl/scripts/fceux_trace.lua'), rom], env=env, capture_output=True, timeout=900)
        ram = np.fromfile(f'{d}/r.bin', np.uint8).reshape(-1, 2048)
        raw = np.fromfile(f'{d}/s.bin', np.uint8)
        per = 256 * 240 * 4
        shots = {}
        for k, f in enumerate(sorted(frames)):
            if (k + 1) * per <= len(raw):
                shots[f] = raw[k * per + 256 * 240:(k + 1) * per].reshape(240, 256, 3)
        return ram, shots
    finally:
        shutil.rmtree(d, ignore_errors=True)


def vote(row, png_dir):
    from gpu_check import buttons, fceumm_ram
    from PIL import Image
    rom = str(ROM_DIR / row['rom'])
    f = row['worst_frame']
    im = np.asarray(Image.open(Path(png_dir) / (Path(rom).stem + '.png')).convert('RGB'))
    ref, nes = im[:, :256], im[:, 260:516]
    pad = buttons(rom, f + 8)
    ref_ram = fceumm_ram(rom, pad)[f]
    frames = [g for g in range(f - 6, f + 7) if g >= 0]
    ram, shots = fceux_shots(rom, pad, frames)
    if not shots:
        return dict(rom=row['rom'], error='no FCEUX shots')
    g = min(shots, key=lambda g: int(((ram[g] != ref_ram) & KEEP).sum()) if g < len(ram) else 10**6)
    fx = shots[g]
    fx_ref, fx_nes, ref_nes = layout_match(fx, ref), layout_match(fx, nes), layout_match(ref, nes)
    verdict = ('nesle_ok' if fx_nes >= 0.97 and fx_nes > fx_ref + 0.01 else
               'nesle_bad' if fx_ref >= 0.97 and fx_ref > fx_nes + 0.01 else 'unclear')
    return dict(rom=row['rom'], frame=f, fceux_frame=g, ram_diff=int(((ram[g] != ref_ram) & KEEP).sum()),
                fceux_vs_fceumm=round(fx_ref, 4), fceux_vs_nesle=round(fx_nes, 4), fceumm_vs_nesle=round(ref_nes, 4),
                verdict=verdict)


def _w(args):
    try:
        return vote(*args)
    except Exception as e:  # noqa: BLE001
        return dict(rom=args[0]['rom'], error=repr(e)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--checks', required=True)
    ap.add_argument('--png-dir', required=True)
    ap.add_argument('--only')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    only = set(Path(a.only).read_text().splitlines()) if a.only else None
    rows = [json.loads(l) for l in open(a.checks)]
    rows = [r for r in rows if r.get('pass') is False and (only is None or r['rom'] in only)]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, 'w') as fh:
        for r in pool.imap_unordered(_w, [(r, a.png_dir) for r in rows]):
            fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
