"""Stage 2 of a per-game RAM map: lives and score, from random play after the stage-1 boot (ramfind.py), on fceumm.

Play: replay the boot pads, then EPISODES x FRAMES of seeded random play (mostly forward with jumps and fire, some
random directions; no Start/Select). RAM after every frame.
Lives: a byte holding a small number (0..9, or BCD <= $09) that changes rarely, almost always by exactly -1, and starts
the play at 1..9. Several bytes may hold copies (one for the HUD): all are reported, best first.
Score: a run of 2..8 bytes read as a decimal number (one digit per byte or packed BCD, either order; or 16-bit binary)
that starts below START_MAX, never goes down within an episode while lives do not reset it, rises at least 3 times,
rises at most IDLE_RISES times with no input and not at the same frames in every episode, and whose least significant
byte moves. The longest, most often rising wins. readers.py (stage 3) re-ranks both lists by replayed pokes.

usage: python ramfind2.py OUT.jsonl --stage1 ramfind_v1.jsonl [--workers 20]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/reward10'))
ROM_DIR = Path('/home/kd/Desktop/roms')
EPISODES, FRAMES = 3, 6000
IDLE = 1500
NAMES = ['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']


def play(rom, boot_pads, seed):
    from nesram import Nes
    nes = Nes(rom)
    for p in bytes.fromhex(boot_pads):
        nes.step(tuple(n for k, n in enumerate(NAMES) if p >> k & 1), 1)
    start = nes.save()
    rng = np.random.default_rng(seed)
    eps, snaps = [], []
    for e in range(EPISODES):
        nes.load(start)
        out, f = [], 0
        while f < FRAMES:
            if f >= 1500 * (len(snaps) + 1) and len(snaps) < 3:
                snaps.append(nes.save())         # mid-play states for the poke test
            n = int(rng.integers(6, 40))
            d = rng.choice(['RIGHT', 'RIGHT', 'RIGHT', 'LEFT', 'UP', 'DOWN', ''])
            btn = tuple(b for b in (d, 'A' if rng.random() < 0.4 else '', 'B' if rng.random() < 0.5 else '') if b)
            nes.step(btn, n, record=out)
            f += n
        eps.append(np.stack(out[:FRAMES]))
    nes.load(start)
    idle = []
    nes.step((), IDLE, record=idle)                 # no input: a score does not rise, timers and counters do
    return nes, eps, np.stack(idle), snaps


def respawns(eps, xaddr):
    """Frames where the stage-1 position byte jumps (a respawn or a screen change), per episode."""
    if xaddr is None:
        return [np.zeros(0, int) for _ in eps]
    out = []
    for r in eps:
        v = r[:, xaddr].astype(np.int32)
        jump = np.abs(((np.diff(v) + 128) % 256) - 128) > 24
        out.append(np.flatnonzero(jump))
    return out


def lives(eps, xaddr=None):
    """Lives byte candidates. A death is a -1 at least 120 frames into play and 60 frames after the byte's previous
    change (countdown timers tick in bursts); after 0 (or $FF) the byte may jump back up (game over, attract mode),
    any other rise counts against it. Ranked by deaths minus other changes, then by deaths a respawn confirms."""
    jumps = respawns(eps, xaddr)
    cands = []
    for a in list(range(0, 0x100)) + list(range(0x200, 0x800)):
        ok, dec, other, conf, starts = True, 0, 0, 0, []
        for r, jp in zip(eps, jumps):
            v = r[:, a].astype(np.int32)
            body = v[v != 0xFF]                       # $FF: "no lives left" marker after 0 (SMB)
            if len(body) == 0 or (body.max() > 9 and not (body.max() <= 0x99 and ((body & 0xF) <= 9).all())):
                ok = False
                break
            dv = ((np.diff(v) + 128) % 256) - 128     # 0 -> $FF counts as -1
            idx = np.flatnonzero(dv)
            if len(idx) > 12 or (len(idx) >= 3 and (idx[2:] - idx[:-2] < 30).any()):
                ok = False
                break
            prev, run = -10**6, 0
            for i in idx:
                if dv[i] == -1 and i >= 120 and i - prev >= 60:
                    run += 1
                    if run > v[0] + 1:
                        ok = False
                        break
                    dec += 1
                    conf += int(len(jp) > 0 and (np.abs(jp - i) <= 240).any())
                elif dv[i] > 0 and v[i] in (0, 0xFF):
                    run = 0                           # refill after game over
                else:
                    other += 1
                prev = i
            if not ok:
                break
            starts.append(int(v[0]))
        if ok and dec >= 1 and dec > other and all(1 <= s_ <= 9 for s_ in starts):
            cands.append((f'${a:04X}', dec, other, conf, starts[0]))
    cands.sort(key=lambda c: (-(c[1] - c[2]), -c[3], c[0]))
    return cands[:20]


def number(r, a, n, kind):
    b = r[:, a:a + n].astype(np.int64)
    if kind == 'digits_hi':     # most significant digit first
        return (b * (10 ** np.arange(n - 1, -1, -1))).sum(1), (b <= 9).all()
    if kind == 'digits_lo':
        return (b * (10 ** np.arange(n))).sum(1), (b <= 9).all()
    if kind == 'bin_lo':        # binary, low byte first
        return (b * (256 ** np.arange(n))).sum(1), True
    if kind == 'bcd_lo':        # packed BCD, least significant byte first (Castlevania, Gradius)
        b = b[:, ::-1]
    hi, lo = b >> 4, b & 15
    dig = np.stack([hi, lo], 2).reshape(len(b), 2 * n)
    return (dig * (10 ** np.arange(2 * n - 1, -1, -1))).sum(1), ((hi <= 9) & (lo <= 9)).all()


def round_step(d):
    """A score step: k * 10^m with one nonzero digit (10, 200, 1000 points; or 1, 2, 5 in units of 100)."""
    while d % 10 == 0:
        d //= 10
    return 1 <= d <= 9


KINDS = (('digits_hi', range(3, 9)), ('digits_lo', range(3, 9)), ('bcd', range(2, 5)), ('bcd_lo', range(2, 5)),
         ('bin_lo', (2,)))


def low_end(a, n, kind):
    """Address of the least significant byte of a span."""
    return a + n - 1 if kind in ('digits_hi', 'bcd') else a


def lsb_moves(eps, a, n, kind):
    """A score's lowest byte changes: a packed/binary span whose low byte never moves while a higher one does reads
    the bytes in the wrong order or is too wide (one constant low digit is allowed for digit spans: SMB keeps a units
    digit that 100-point steps never touch)."""
    lo = low_end(a, n, kind)
    if kind.startswith('digits'):
        lo2 = lo - 1 if kind == 'digits_hi' else lo + 1
        return any((r[:, lo] != r[0, lo]).any() or (r[:, lo2] != r[0, lo2]).any() for r in eps)
    return any((r[:, lo] != r[0, lo]).any() for r in eps)


def score_events(eps, a, n, kind):
    """(good, bad) changes of the number at a. Good: a rise by a round step, 4+ frames after the previous change
    (one write, not a byte-by-byte fill). A drop is allowed only to 0 (a reset); any other drop disqualifies."""
    good, bad = 0, 0
    for r in eps:
        v, valid = number(r, a, n, kind)
        if not valid or v[0] >= START_MAX:            # a new game starts with (almost) no points: the boot may
            # end a little after play began (Pac-Man eats a dot before the first probe sees control)
            return None
        d = np.diff(v)
        idx = np.flatnonzero(d)
        prev = -10**6
        for i in idx:
            if d[i] < 0:
                if v[i + 1] != 0:
                    return None
                bad += 0
            elif i - prev >= 4 and round_step(int(d[i])):
                good += 1
            else:
                bad += 1
            prev = i
    return good, bad


MIN_RISES = 3
START_MAX = 100
IDLE_RISES = 5


def same_frames(eps, a, n, kind):
    ch = [tuple(np.flatnonzero(np.diff(number(r, a, n, kind)[0]))) for r in eps]
    return len(eps) > 1 and all(c == ch[0] for c in ch)


def score(eps, idle):
    best = []
    for kind, sizes in KINDS:
        for n in sizes:
            for a in list(range(0, 0x100 - n)) + list(range(0x200, 0x800 - n)):
                ev = score_events(eps, a, n, kind)
                if not ev or ev[0] < MIN_RISES or ev[1] * 3 > ev[0] or ev[0] > 100 * len(eps) \
                        or not lsb_moves(eps, a, n, kind):
                    continue
                iv, _ = number(idle, a, n, kind)
                # idle: a timer or counter moves on its own; a score may rise a few times (enemies that run into
                # the player, Final Mission). Rises at the same frames in every episode do not depend on play
                # (an intro or a tally), so they are not a score either.
                if (np.diff(iv) != 0).sum() <= IDLE_RISES and not same_frames(eps, a, n, kind):
                    best.append((f'${a:04X}', n, kind, ev[0], ev[1]))
    # overlapping spans read the same digits: keep the best per region (more good steps, fewer bad, then longer)
    best.sort(key=lambda c: (-(c[3] - 2 * c[4]), -c[1]))
    out, used = [], set()
    for c in best:
        a = int(c[0][1:], 16)
        span = set(range(a, a + c[1]))
        if span & used:
            continue
        used |= span
        out.append(c)
    return out[:40]


POKE_FRAMES = 60


def poke_test(nes, snaps, edits, keeps):
    """Write edits {addr: value} into mid-play states, idle POKE_FRAMES frames. Returns (states where keeps(ram before,
    ram after) holds, states where the screen differs from the same idle without the write: the HUD shows it)."""
    kept = shown = 0
    for snap in snaps:
        nes.load(snap)
        nes.step((), POKE_FRAMES)
        control = nes.screen()
        nes.load(snap)
        before = nes.ram()
        for a, v in edits(before).items():
            nes.poke(a, v)
        poked = nes.ram()
        nes.step((), POKE_FRAMES)
        kept += bool(keeps(poked, nes.ram()))
        diff = (nes.screen() != control).any(2)
        shown += bool(0 < diff.mean() < 0.02)   # a few HUD digits, not a different scene
    return kept, shown


def check_lives(nes, snaps, cands):
    out = []
    for c in cands:
        a = int(c[0][1:], 16)
        new = lambda r: {a: int(r[a]) + 2 if r[a] <= 7 else int(r[a]) - 2}
        keep = lambda p, f: int(f[a]) in (int(p[a]), int(p[a]) - 1)
        out.append(tuple(c) + poke_test(nes, snaps, new, keep))
    out.sort(key=lambda c: (-c[6], -c[5]))       # stable: the play ranking breaks ties
    return out[:6]


def check_score(nes, snaps, cands):
    out = []
    for c in cands:
        a, n, kind = int(c[0][1:], 16), c[1], c[2]
        b = a + n - 3 if kind in ('digits_hi',) else a + 2 if kind == 'digits_lo' else a + n - 2 if kind == 'bcd' \
            else a + 1
        b = min(max(b, a), a + n - 1)
        def new(r, b=b, kind=kind):
            x = int(r[b])
            if kind in ('digits_hi', 'digits_lo'):
                return {b: x + 1 if x < 9 else x - 1}
            if kind == 'bcd':
                return {b: x + 0x10 if x < 0x90 else x - 0x10}
            return {b: x + 1 if x < 255 else x - 1}
        def keep(p, f, a=a, n=n, kind=kind):
            vp, _ = number(p[None], a, n, kind)
            vf, ok = number(f[None], a, n, kind)
            return ok and vf[0] >= vp[0] and f[b] == p[b] if vf[0] == vp[0] else ok and vf[0] > vp[0]
        out.append(tuple(c) + poke_test(nes, snaps, new, keep))
    out.sort(key=lambda c: (-c[6], -c[5]))
    return out[:4]


def run(row):
    rom = str(ROM_DIR / row['rom'])
    nes, eps, idle, snaps = play(rom, row['boot_pads'], 11)
    xs = row.get('x_candidates') or row.get('y_candidates') or []
    xaddr = int(xs[0][0][1:], 16) if xs else None
    return dict(rom=row['rom'], lives=check_lives(nes, snaps, lives(eps, xaddr)),
                score=check_score(nes, snaps, score(eps, idle)))


def _w(row):
    try:
        return run(row)
    except Exception as e:  # noqa: BLE001
        return dict(rom=row['rom'], error=repr(e)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--stage1', required=True)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.stage1)]
    rows = [r for r in rows if r.get('boot_pads') and (not a.only or r['rom'] in a.only)]
    done = {json.loads(l)['rom'] for l in open(a.out)} if Path(a.out).exists() else set()
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, 'a') as fh:
        for r in pool.imap_unordered(_w, [r for r in rows if r['rom'] not in done]):
            fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
