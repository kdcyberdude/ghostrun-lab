"""Find a game's boot sequence and the RAM bytes the player moves (stage 1 of a per-game RAM map), on fceumm.

Boot: from power-on, wait, then alternate menu presses (Start, A, Start, ...) with waits; before each press, probe
whether the game is controllable. Probe = from a savestate, play PROBE frames of: Right held, Left held, nothing.
A byte "follows the d-pad" when Right moves it one way, Left the other way (or not at all), and nothing leaves it
almost still; bytes equal to the raw pad (games keep the pad in RAM, menus included) are left out because they also
follow the pad. Controllable = at least one such byte. The press sequence up to that point is the boot script.

Position candidates are ranked by how cleanly they separate Right from Left; 16-bit pairs (a high byte that carries
when the low byte wraps) are joined. The same probe with Up/Down gives vertical candidates.

RAMFIND_STABLE=k: a boot counts only when control holds k of 3 later checks (1, 3, 6 s of no input); see stable().

usage: [RAMFIND_MAX_PRESSES=40] [RAMFIND_SEED=1] [RAMFIND_STABLE=2] python ramfind.py OUT.jsonl ROM... [--workers 24]   or   --list FILE
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
sys.path.insert(0, str(ROOT / 'scripts/reward10'))

PROBE = 45
WAIT = 180          # after a press: games play a jingle or intro before the player can move (a Start then pauses)
MAX_PRESSES = int(os.environ.get('RAMFIND_MAX_PRESSES', 16))
SEED = int(os.environ.get('RAMFIND_SEED', 0))   # another seed: another random press tail (deeper retries)
PRESS = 8
# menus: Start/A advance, a d-pad tap moves a cursor off a locked choice (Mega Man 2's stage select starts on Wily)
MENU = ['START', 'A', 'START', 'RIGHT', 'START', 'DOWN', 'A', 'START', 'B', 'START']
# after MENU: seeded random presses (menus that need a cursor walk, e.g. 1943's power-up screen wants Down to END)
TAIL = ['START', 'A', 'DOWN', 'DOWN', 'UP', 'RIGHT', 'LEFT', 'B', 'SELECT', 'START']


def pad_byte(buttons):
    bits = {'A': 1, 'B': 2, 'SELECT': 4, 'START': 8, 'UP': 16, 'DOWN': 32, 'LEFT': 64, 'RIGHT': 128}
    return sum(bits[b] for b in buttons)


class Rec:
    """fceumm with a per-frame pad log (the boot script)."""

    def __init__(self, rom):
        from nesram import Nes
        self.nes = Nes(rom)
        self.pads = []

    def step(self, buttons=(), frames=1):
        self.nes.step(tuple(buttons), frames)
        self.pads += [pad_byte(buttons)] * frames

    def run(self, buttons, frames):
        """Play from the current state without logging; returns RAM after every frame [frames, 2048]."""
        out = []
        self.nes.step(tuple(buttons), frames, record=out)
        return np.stack(out)


def follows(rec, a, b, frames=PROBE):
    """Bytes that buttons a and b push apart. Returns (score per byte [2048], traces)."""
    s = rec.nes.save()
    start = rec.nes.ram().astype(np.int32)
    ta = rec.run(a, frames).astype(np.int32)
    rec.nes.load(s)
    tb = rec.run(b, frames).astype(np.int32)
    rec.nes.load(s)
    tn = rec.run((), frames).astype(np.int32)
    rec.nes.load(s)
    # signed change from the start, mod 256
    da = ((ta[-1] - start + 128) % 256) - 128
    db = ((tb[-1] - start + 128) % 256) - 128
    dn = ((tn[-1] - start + 128) % 256) - 128
    sep = np.abs(da - db).astype(np.float64)
    noise = np.abs(dn) + np.abs(np.diff(tn, axis=0)).sum(0) * 0.0
    pad_a, pad_b = pad_byte(a), pad_byte(b)
    # raw pad copies: the byte equals the pad (or its low/high nibble view) during the probe
    rawpad = ((ta[5:] == pad_a).mean(0) > 0.8) | ((tb[5:] == pad_b).mean(0) > 0.8)
    # a moving position changes step by step: many small changes, no wild jumps
    steps_a = np.abs(((np.diff(ta, axis=0) + 128) % 256) - 128)
    smooth = (steps_a.max(0) <= 16) & ((steps_a > 0).sum(0) >= frames // 3)
    smooth[0x100:0x200] = False    # stack page: return addresses and temporaries, never game state
    score = np.where(~rawpad & smooth & (sep >= 6) & (np.abs(dn) <= 2), sep, 0.0)
    return score, dict(da=da, db=db, dn=dn)


PAIRS = [(('RIGHT',), ('LEFT',)), (('UP',), ('DOWN',)), (('A',), ()), (('B',), ())]
PROBE_EVERY = 25


def controllable(rec, pairs):
    """Probe the given button pairs; returns {pair index: score array} for the pairs that move something."""
    out = {}
    for i in pairs:
        score, _ = follows(rec, *PAIRS[i])
        if score.any():
            out[i] = score
    return out


STABLE = (60, 180, 360)   # idle frames after a found boot where control is checked again
STABLE_NEED = int(os.environ.get('RAMFIND_STABLE', 0))   # how many of those must still have control (0 = off, as v1)


def stable(rec, found):
    """Control that holds: after waiting STABLE frames (no input) the same pair still moves bytes, at least
    STABLE_NEED times. A title-screen cursor, a demo or a one-off animation passes a single probe, not this."""
    if STABLE_NEED <= 0:
        return True
    s = rec.nes.save()
    ok, t = 0, 0
    for w in STABLE:
        rec.nes.step((), w - t)
        t = w
        if any(follows(rec, *PAIRS[i])[0].any() for i in found):
            ok += 1
    rec.nes.load(s)
    return ok >= STABLE_NEED


def press(rom, i):
    if i < len(MENU) or MAX_PRESSES <= 16:
        return MENU[i % len(MENU)]
    rng = np.random.default_rng([i, SEED, *Path(rom).name.encode()])
    return TAIL[int(rng.integers(len(TAIL)))]


def boot(rec, power_on, pairs, rom=''):
    rec.nes.load(power_on)
    rec.pads = []
    rec.step((), 60)
    presses, found = 0, {}
    while presses <= MAX_PRESSES:
        # probe through the wait too: some games become controllable between presses (and a Start in play pauses)
        for _ in range(WAIT // PROBE_EVERY + 1):
            found = controllable(rec, pairs)
            if found and not stable(rec, found):
                found = {}
            if found:
                break
            rec.step((), PROBE_EVERY)
        if found or presses == MAX_PRESSES:
            break
        rec.step((press(rom, presses),), PRESS)
        presses += 1
    return rec, presses, found


def find(rom):
    # Right/Left first: Up/Down and A also move menu cursors, so they only count when Right/Left never moves anything
    rec = Rec(rom)
    power_on = rec.nes.save()          # one emulator per process (stable-retro): restart = load this state
    rec, presses, found = boot(rec, power_on, [0], rom)
    if not found:
        rec, presses, found = boot(rec, power_on, [1, 2, 3], rom)
    res = dict(rom=Path(rom).name, controllable=bool(found), boot_frames=len(rec.pads), presses=presses,
               moved_by=[' / '.join('+'.join(x) or 'none' for x in PAIRS[i]) for i in found])
    if not found:
        return res
    for i, key in ((0, 'x_candidates'), (1, 'y_candidates'), (2, 'a_candidates')):
        score = found.get(i)
        if score is None:
            score, _ = follows(rec, *PAIRS[i])
        good = np.flatnonzero(score)
        res[key] = [(f'${a:04X}', float(score[a])) for a in good[np.argsort(-score[good])][:8]]
    res['boot_pads'] = bytes(rec.pads).hex()
    return res


def _w(rom):
    try:
        return find(rom)
    except Exception as e:  # noqa: BLE001
        return dict(rom=Path(rom).name, error=repr(e)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('roms', nargs='*')
    ap.add_argument('--list')
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    roms = list(a.roms) + (Path(a.list).read_text().splitlines() if a.list else [])
    done = {json.loads(l)['rom'] for l in open(a.out)} if Path(a.out).exists() else set()
    todo = [r for r in roms if r and Path(r).name not in done]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, 'a') as fh:
        for r in pool.imap_unordered(_w, todo):
            fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
