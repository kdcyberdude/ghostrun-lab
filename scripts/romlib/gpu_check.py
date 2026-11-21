"""Does NeSLE (the GPU emulator's core, CPU build) run a ROM exactly like fceumm? One ROM per worker process.

The same per-frame buttons (power-on, then Start/A taps and random held d-pad + A/B, seeded by the file name) are played
on fceumm (stable_retro, the reference NeSLE was matched to) and on .cache/romlib/nesle_trace; the 2 KB CPU RAM after
every frame is compared. Exact equality is not expected: interrupt-timing scratch bytes differ by a few per frame on games NeSLE
is known to run correctly (SMB, Contra, Mega Man 2, Punch-Out!!: 1-6 bytes, stack page excluded). A ROM whose play goes
another way differs in dozens to hundreds of bytes within seconds of random play. Pass: the 95th percentile and the
last 5 s mean of differing bytes (outside the stack page) are both <= SCRATCH.

usage: python gpu_check.py OUT.jsonl ROM... [--frames 3600] [--workers 24]
       python gpu_check.py OUT.jsonl --list FILE_OF_ROM_PATHS
Writes one JSON line per ROM: rom, mapper, nesle_ok (loaded), identical_frac, first_diff, fault.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACE = Path(os.environ.get('ROMLIB_NESLE_TRACE', ROOT / '.cache/romlib/nesle_trace'))
sys.path.insert(0, str(ROOT / 'scripts/reward10'))
# NES order: bit0 A, B, Select, Start, Up, Down, Left, bit7 Right
SCRATCH = 8        # differing bytes per frame still counted as the same play (timing scratch)
DIVERGED = 64      # this many differing bytes: the play went another way
A, B, START, UP, DOWN, LEFT, RIGHT = 1, 2, 8, 16, 32, 64, 128


def buttons(rom, frames):
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(Path(rom).name.encode()).digest()[:8], 'little'))
    out = np.zeros(frames, np.uint8)
    f = 60
    # menus: Start and A taps for the first ~20 s
    while f < min(frames, 1200):
        out[f:f + 4] = START if (f // 90) % 2 == 0 else A
        f += 90
    while f < frames:
        n = int(rng.integers(8, 60))
        d = rng.choice([0, RIGHT, RIGHT, RIGHT, LEFT, UP, DOWN, RIGHT | UP, RIGHT | DOWN])
        ab = (A if rng.random() < 0.35 else 0) | (B if rng.random() < 0.5 else 0)
        out[f:f + n] = d | ab
        f += n
    return out


def fceumm_ram(rom, pad):
    from nesram import Nes, BUTTONS
    names = ['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']
    nes = Nes(rom)
    rams = np.empty((len(pad), 2048), np.uint8)
    for i, p in enumerate(pad):
        nes.step(tuple(n for k, n in enumerate(names) if p >> k & 1), 1)
        rams[i] = nes.ram()
    return rams


def mapper(rom):
    h = Path(rom).read_bytes()[:16]
    m = (h[6] >> 4) | (h[7] & 0xF0)
    if h[12:16] != b'\0\0\0\0' and (h[7] & 0x0C) != 0x08:
        m = h[6] >> 4
    return m


def nesle_run(rom, pad):
    with tempfile.TemporaryDirectory() as d:
        bp, op = Path(d) / 'b.bin', Path(d) / 'r.bin'
        pad.tofile(bp)
        p = subprocess.run([str(TRACE), rom, str(bp), str(op), 'ff', '0', 'vblank'], capture_output=True, text=True,
                           timeout=600)
        got = np.fromfile(op, np.uint8).reshape(-1, 2048) if op.exists() else np.zeros((0, 2048), np.uint8)
    return p, got


def shifted(pad, sh):
    """Buttons for an emulator whose game runs `sh` frames behind (sh > 0) or ahead (sh < 0) of the reference."""
    if sh > 0:
        return np.concatenate([np.zeros(sh, np.uint8), pad])
    if sh < 0:
        return np.concatenate([pad[-sh:], np.zeros(-sh, np.uint8)])
    return pad


def check(rom, frames):
    """Boot offsets of a frame or two between emulators are normal (power-on PPU timing). NeSLE's frame f + sh is
    compared with the reference's frame f, and NeSLE gets the buttons shifted by sh, so both games see each press at
    the same moment of their own timeline. The best shift wins."""
    res = dict(rom=Path(rom).name, mapper=mapper(rom))
    pad = buttons(rom, frames)
    try:
        ref = fceumm_ram(rom, pad)
    except Exception as e:  # noqa: BLE001 - report any reference failure
        res['ref_error'] = repr(e)[:200]
        return res
    res['ref_moves'] = int(len(np.unique(ref[::30], axis=0)))
    keep = np.ones(2048, bool)
    keep[0x100:0x200] = False                      # stack page: interrupt timing scratch
    best = None
    for sh in (0, 1, -1, 2, -2):
        p, got = nesle_run(rom, shifted(pad, sh))
        ok = p.returncode == 0 and len(got) > 0
        if best is None:
            res['nesle_ok'], res['nesle_msg'] = ok, (p.stderr.strip().splitlines() or [''])[-1][:200]
        if not ok:
            if sh == 0:
                break
            continue
        x, y = (ref, got[sh:]) if sh >= 0 else (ref[-sh:], got)
        n = min(len(x), len(y), frames - 2)
        per = ((x[:n] != y[:n]) & keep).sum(1)
        r = dict(shift=sh, frames=int(n), identical_frac=round(float((per == 0).mean()), 4),
                 diff_p95=float(np.percentile(per, 95)), diff_last=float(per[-300:].mean()),
                 diverged_at=int(np.argmax(per > DIVERGED)) if (per > DIVERGED).any() else None)
        if best is None or (r['diff_p95'], r['diff_last']) < (best['diff_p95'], best['diff_last']):
            best = r
        if r['diff_p95'] <= SCRATCH and r['diff_last'] <= SCRATCH:
            break
    res.update(best or dict(frames=0, diff_p95=1e9, diff_last=1e9, diverged_at=0))
    res['pass'] = bool(res.get('nesle_ok') and res['frames'] >= frames - 4 and res['diff_p95'] <= SCRATCH
                       and res['diff_last'] <= SCRATCH)
    return res


def _worker(args):
    rom, frames = args
    try:
        return check(rom, frames)
    except Exception as e:  # noqa: BLE001
        return dict(rom=Path(rom).name, error=repr(e)[:300], **{'pass': False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('roms', nargs='*')
    ap.add_argument('--list')
    ap.add_argument('--frames', type=int, default=3600)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    roms = list(a.roms) + (Path(a.list).read_text().splitlines() if a.list else [])
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)['rom'] for l in open(a.out) if l.strip()}
    todo = [(r, a.frames) for r in roms if r and Path(r).name not in done]
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, 'a') as fh:
        for r in pool.imap_unordered(_worker, todo):
            fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
