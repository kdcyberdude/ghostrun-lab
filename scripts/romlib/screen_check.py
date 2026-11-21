"""Does NeSLE draw a ROM's picture like fceumm? gpu_check.py compares CPU RAM only; a wrong CHR bank, mirroring or
nametable mapping leaves RAM right and the picture wrong. One ROM per worker process.

Both emulators play gpu_check.buttons(rom) from power-on (NeSLE with the same frame shift gpu_check found best, from
its jsonl, else 0). fceumm runs through fceumm_lr (stable_retro's core, uncropped 256x240); NeSLE through
scripts/nesle_check/nesle_capi (the CUDA renderer compiled for the CPU), built into --lib.

Every EVERY frames, when both RAMs (stack page excluded) differ in at most SCRATCH bytes at this frame and the previous
one (the same game state, as gpu_check counts it), the pictures are compared. Per ROM: frames compared, the share that are pixel-identical,
the median and 95th percentile share of differing pixels. Pass: >= 90% of compared frames identical, or the 95th
percentile below 0.5% of the pixels. The worst frame is saved as PNG (fceumm | NeSLE | difference) for failures.

usage: FCEUMM_CORE=... python screen_check.py OUT.jsonl --list ROMS.txt --lib LIB.so [--shifts gpu_check.jsonl...]
       [--frames 3600] [--every 10] [--png-dir DIR] [--workers 20]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/romlib'))
sys.path.insert(0, str(ROOT / 'scripts/nesle_check'))
sys.path.insert(0, str(ROOT / 'scripts/reward10'))
KEEP = np.ones(2048, bool)
KEEP[0x100:0x200] = False
SCRATCH = 8          # as gpu_check: a few interrupt-timing scratch bytes differ on correctly run games


def check(rom, frames, every, shift, png_dir):
    from gpu_check import buttons, shifted
    from fceumm_lr import Fceumm
    import neslenes
    res = dict(rom=Path(rom).name, shift=shift)
    pad = buttons(rom, frames)
    ref = Fceumm(rom)
    ref_ram, ref_pic = np.empty((frames, 2048), np.uint8), {}
    for f, p in enumerate(pad):
        ref.step_byte(int(p), 1)
        ref_ram[f] = ref.ram()
        if f % every == 0:
            ref_pic[f] = ref.screen()
    nes = neslenes.NesleNes(rom)
    names = list(neslenes.NES_BITS)
    npad = shifted(pad, shift)
    got_ram = np.empty((len(npad), 2048), np.uint8)
    pics = {}
    try:
        for g, p in enumerate(npad):
            nes.step(tuple(n for n in names if int(p) >> neslenes.NES_BITS[n] & 1), 1)
            got_ram[g] = nes.ram()
            f = g - shift                       # NeSLE frame g shows the reference's frame g - shift
            if 0 <= f < frames and f % every == 0:
                pics[f] = nes.screen()
    except neslenes.EmulatorFault as e:
        res['fault'] = str(e)[:200]
        got_ram = got_ram[:g]
    shares, worst = [], (-1.0, None)
    for f, rp in ref_pic.items():
        g = f + shift
        if f < 1 or g < 1 or g >= len(got_ram) or f not in pics:
            continue
        if (((ref_ram[f] != got_ram[g]) & KEEP).sum() > SCRATCH or
                ((ref_ram[f - 1] != got_ram[g - 1]) & KEEP).sum() > SCRATCH):
            continue
        diff = (rp != pics[f]).any(2)
        s = float(diff.mean())
        shares.append(s)
        if s > worst[0]:
            worst = (s, f)
    res['compared'] = len(shares)
    if shares:
        a = np.array(shares)
        res.update(identical=round(float((a == 0).mean()), 4), median=round(float(np.median(a)), 5),
                   p95=round(float(np.percentile(a, 95)), 5), worst=round(worst[0], 4), worst_frame=worst[1])
        res['pass'] = bool(res['identical'] >= 0.9 or res['p95'] < 0.005)
        if not res['pass'] and png_dir:
            from PIL import Image
            f = worst[1]
            rp, gp = ref_pic[f], pics[f]
            mask = np.zeros_like(rp)
            mask[(rp != gp).any(2)] = (255, 0, 255)
            gap = np.full((240, 4, 3), 128, np.uint8)
            Image.fromarray(np.hstack([rp, gap, gp, gap, mask])).save(Path(png_dir) / (Path(rom).stem + '.png'))
    else:
        res['pass'] = None                    # never in the same state: nothing to compare (see gpu_check)
    return res


def _w(args):
    rom = args[0]
    try:
        return check(*args)
    except Exception as e:  # noqa: BLE001
        return dict(rom=Path(rom).name, error=repr(e)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--list', required=True)
    ap.add_argument('--lib', required=True)
    ap.add_argument('--shifts', nargs='*', default=[])
    ap.add_argument('--frames', type=int, default=3600)
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--png-dir')
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()
    os.environ['NESLE_HOST_LIB'] = str(Path(a.lib).resolve())
    if a.png_dir:
        Path(a.png_dir).mkdir(parents=True, exist_ok=True)
    shift = {}
    for p in a.shifts:
        for line in open(p):
            r = json.loads(line)
            if r.get('shift') is not None:
                shift[r['rom']] = r['shift']
    roms = [r for r in Path(a.list).read_text().splitlines() if r]
    done = {json.loads(l)['rom'] for l in open(a.out)} if Path(a.out).exists() else set()
    jobs = [(r, a.frames, a.every, shift.get(Path(r).name, 0), a.png_dir) for r in roms if Path(r).name not in done]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, 'a') as fh:
        for r in pool.imap_unordered(_w, jobs):
            fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
