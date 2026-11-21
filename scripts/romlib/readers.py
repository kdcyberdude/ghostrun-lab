"""Stage 3 of a per-game RAM map: one reader per ROM (position, lives, score, level, death, in-play, game over).

Play (collect): replay the stage-1 boot pads (ramfind.py), then EPISODES x FRAMES of seeded random play whose pads are
generated up front, so any stretch can be replayed exactly from a savestate (with or without a RAM poke). Every
PROBE_EVERY frames a control probe runs from a savestate: the stage-1 pair (Right vs Left, ...) for PROBE frames; the
frame counts as controllable when the stage-1 position bytes end up apart. RAM after every frame, savestates every
SNAP_EVERY frames, and an idle run (no input) from the boot state.

Fields (each from ramfind2.py's candidates plus checks that use the probes and replays):
  lives      ramfind2.lives candidates; a death must be confirmed by a loss of control around the decrement
             (respawn), then a replay check: poke lives+1 before a real death, the difference must survive it.
  score      ramfind2.score candidates; replay check: poke a digit before a real rise, the difference must survive
             the rise (the game adds to the stored number) and the screen must differ after the rise (HUD redraw).
  death      the (byte, value) whose rising edge comes just before each lives decrement and seldom otherwise.
  in_play    the (byte, values) that best separates controllable from uncontrollable probes (held-out episode).
  game_over  the (byte, value) that turns on after the last life is lost and never during play.
  level      a byte constant during play whose +1 poke before a respawn reloads a different scene and stays.

usage: python readers.py OUT.jsonl --stage1 ramfind_v1.jsonl [--stage1b ramfind_v2.jsonl] [--workers 16] [--only ROM...]
       python readers.py --truth OUT.jsonl     (the reward10 truth games; adds per-field checks against games/*.py)
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
sys.path.insert(0, str(ROOT / 'scripts/romlib'))
ROM_DIR = Path('/home/kd/Desktop/roms')
CACHE = ROOT / '.cache/romlib'
EPISODES, FRAMES = 3, 6000
IDLE = 1500
PROBE_EVERY, PROBE = 40, 16
SNAP_EVERY = 200
NAMES = ['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']
BIT = {n: 1 << k for k, n in enumerate(NAMES)}
PAIRS = {'RIGHT / LEFT': (('RIGHT',), ('LEFT',)), 'UP / DOWN': (('UP',), ('DOWN',)), 'A / none': (('A',), ()),
         'B / none': (('B',), ())}


def buttons(p):
    return tuple(n for k, n in enumerate(NAMES) if p >> k & 1)


def random_pads(rng, frames):
    """Mostly forward with jumps and fire, some other directions; no Start/Select (a Start would pause)."""
    out = []
    while len(out) < frames:
        n = int(rng.integers(6, 40))
        d = rng.choice(['RIGHT', 'RIGHT', 'RIGHT', 'LEFT', 'UP', 'DOWN', ''])
        p = (BIT[d] if d else 0) | (BIT['A'] if rng.random() < 0.4 else 0) | (BIT['B'] if rng.random() < 0.5 else 0)
        out += [p] * n
    return np.array(out[:frames], np.uint8)


def lit_frac(img):
    """Fraction of the screen that is not the backdrop (the most common colour)."""
    flat = img.reshape(-1, 3).astype(np.int32)
    key = flat[:, 0] << 16 | flat[:, 1] << 8 | flat[:, 2]
    vals, cnt = np.unique(key, return_counts=True)
    return float(1 - cnt.max() / len(key))


class Play:
    """The recorded play of one ROM, replayable."""

    def __init__(self, rom, row, seed=11):
        from nesram import Nes
        self.nes = nes = Nes(rom)
        self.power_on = nes.save()
        self._at = None
        self.boot_pads = list(bytes.fromhex(row['boot_pads']))
        for p in self.boot_pads:
            nes.step(buttons(p), 1)
        self.start = nes.save()
        self.pair = PAIRS[(row.get('moved_by') or ['RIGHT / LEFT'])[0]]
        key = {'RIGHT / LEFT': 'x_candidates', 'UP / DOWN': 'y_candidates'}.get((row.get('moved_by') or [''])[0],
                                                                                'a_candidates')
        cands = row.get(key) or row.get('x_candidates') or row.get('y_candidates') or []
        self.pos = [int(c[0][1:], 16) for c in cands[:3]]
        self.collect(seed)

    def collect(self, seed):
        nes = self.nes
        rng = np.random.default_rng(seed)
        self.pads, self.ram, self.snaps, self.probe_t, self.probe_ok = [], [], [], [], []
        self._at, lit = None, []
        for e in range(EPISODES):
            nes.load(self.start)
            pads = random_pads(rng, FRAMES)
            out, snaps, pt, pok = [], {}, [], []
            for f in range(FRAMES):
                if f % SNAP_EVERY == 0:
                    snaps[f] = nes.save()
                if f % PROBE_EVERY == 0 and self.pos:
                    pt.append(f)
                    pok.append(self.probe())
                if f % 1000 == 500:
                    lit.append(lit_frac(nes.screen()))
                nes.step(buttons(pads[f]), 1, record=out)
            self.pads.append(pads)
            self.ram.append(np.stack(out))
            self.snaps.append(snaps)
            self.probe_t.append(np.array(pt, int))
            self.probe_ok.append(np.array(pok, bool))
        self.lit = float(np.max(lit))
        nes.load(self.start)
        idle = []
        nes.step((), IDLE, record=idle)
        self.idle = np.stack(idle)

    def probe(self):
        nes, s = self.nes, self.nes.save()
        a, b = self.pair
        nes.step(a, PROBE)
        ra = nes.ram()
        nes.load(s)
        nes.step(b, PROBE)
        rb = nes.ram()
        nes.load(s)
        return bool(any(abs(int(ra[p]) - int(rb[p])) >= 2 for p in self.pos))

    def replay(self, e, f0, f1, pokes=None, record=None):
        """Load the snapshot at or before f0 of episode e, poke {addr: fn(ram) -> value} at f0, play the recorded pads
        to f1. Returns (ram, screen) at f1; record (a list) gets RAM after every frame from f0."""
        nes = self.nes
        s = max(k for k in self.snaps[e] if k <= f0)
        nes.load(self.snaps[e][s])
        for f in range(s, f0):
            nes.step(buttons(self.pads[e][f]), 1)
        if pokes:
            r = nes.ram()
            for a, fn in pokes.items():
                nes.poke(a, fn(r))
        for f in range(f0, f1):
            nes.step(buttons(self.pads[e][f]), 1, record=record)
        return nes.ram(), nes.screen()

    def save_npz(self, path):
        np.savez_compressed(path, ram=np.stack(self.ram), pads=np.stack(self.pads), idle=self.idle,
                            probe_t=np.stack(self.probe_t) if self.pos else np.zeros((EPISODES, 0), int),
                            probe_ok=np.stack(self.probe_ok) if self.pos else np.zeros((EPISODES, 0), bool))

    def run_screens(self, e, f0, f1, at, pokes=None):
        """replay() that also returns the screens after the frames in `at` (f0 < t <= f1) and RAM at f1."""
        nes = self.nes
        if self._at != (e, f0):              # many replays start from the same frame: keep that state
            s = max(k for k in self.snaps[e] if k <= f0)
            nes.load(self.snaps[e][s])
            for f in range(s, f0):
                nes.step(buttons(self.pads[e][f]), 1)
            self._at, self._at_state = (e, f0), nes.save()
        nes.load(self._at_state)
        if pokes:
            r = nes.ram()
            for a, fn in pokes.items():
                nes.poke(a, fn(r))
        shots = {}
        for f in range(f0, f1):
            nes.step(buttons(self.pads[e][f]), 1)
            if f + 1 in at:
                shots[f + 1] = nes.screen()
        return nes.ram(), shots


# ---------------------------------------------------------------- helpers

def probes_near(play, e, lo, hi):
    """Probe results (controllable?) of episode e with lo <= t < hi."""
    t, ok = play.probe_t[e], play.probe_ok[e]
    return ok[(t >= lo) & (t < hi)]


def screen_diff(a, b):
    return float((a != b).any(2).mean())


def lives_steps(v):
    """Death decrements of a lives trace (ramfind2.lives rules): -1 at least 120 frames in, 60 after the last change."""
    dv = ((np.diff(v.astype(np.int32)) + 128) % 256) - 128
    out, prev = [], -10**6
    for i in np.flatnonzero(dv):
        if dv[i] == -1 and i >= 120 and i - prev >= 60:
            out.append(int(i) + 1)          # first frame with the new value
        prev = i
    return out


# ---------------------------------------------------------------- lives

def uncontrolled(play, e, lo, hi):
    """Longest stretch of uncontrollable probes of episode e that overlaps [lo, hi): (frames, control came back)."""
    t, ok = play.probe_t[e], play.probe_ok[e]
    best, k = (0, False), 0
    while k < len(t):
        if ok[k]:
            k += 1
            continue
        j = k
        while j < len(t) and not ok[j]:
            j += 1
        s_, end = int(t[k]), int(t[j]) if j < len(t) else FRAMES
        if s_ < hi and end > lo and end - s_ > best[0]:
            best = (end - s_, j < len(t))
        k = j
    return best


def find_lives(play):
    """ramfind2.lives candidates, re-ranked: a death must show a loss of control (respawn) around the decrement, and
    a +1 poke shortly before a real decrement must survive it (the game subtracts from the stored count)."""
    import ramfind2
    cands = ramfind2.lives(play.ram)[:12]
    rows = []
    for c in cands:
        a = int(c[0][1:], 16)
        steps = [(e, i) for e in range(EPISODES) for i in lives_steps(play.ram[e][:, a])]
        runs = [uncontrolled(play, e, i - 300, i + 60) for e, i in steps]
        conf = sum(n >= 80 for n, _ in runs)
        rows.append(dict(addr=a, deaths=c[1], other=c[2], start=c[4], steps=steps, respawn=conf,
                         regained=sum(n >= 80 and back for n, back in runs)))
    rows.sort(key=lambda r: (-(r['respawn'] - r['other']), -(r['deaths'] - r['other'])))
    for r in rows[:6]:
        a, kept, shown, tried = r['addr'], 0, 0, 0
        for e, i in r['steps'][:3]:
            f0, f1 = max(i - 20, 0), min(i + 180, FRAMES)
            at = set(range(i + 20, f1 + 1, 20))
            rc, sc = play.run_screens(e, f0, f1, at)
            rp, sp = play.run_screens(e, f0, f1, at, {a: lambda m, a=a: int(m[a]) + 1})
            tried += 1
            kept += (int(rp[a]) - int(rc[a])) % 256 == 1     # mod 256: the last death goes 0 -> $FF (SMB)
            shown += any(0 < screen_diff(sc[t], sp[t]) < 0.02 for t in sc)
        r.update(kept=kept, shown=shown, tried=tried)
    rows = [r for r in rows if 'tried' in r]
    rows.sort(key=lambda r: (-(r['kept'] == r['tried'] and r['tried'] > 0), -(r['respawn'] - r['other']),
                             -r['shown'], -(r['deaths'] - r['other'])))
    return rows


# ---------------------------------------------------------------- score

def score_rises(play, a, n, kind):
    """(episode, frame) of every round-step rise of a score candidate (first frame with the new value)."""
    import ramfind2
    out = []
    for e in range(EPISODES):
        v, _ = ramfind2.number(play.ram[e], a, n, kind)
        d = np.diff(v)
        for i in np.flatnonzero(d > 0):
            if ramfind2.round_step(int(d[i])):
                out.append((e, int(i) + 1))
    return out


def poke_digit(kind, a, n):
    """(byte, fn(old) -> new): one change of a middle digit of the span (not the lowest, which the next rise moves)."""
    lo = ramfind2_low_end(a, n, kind)
    step = 1 if kind == 'digits_hi' or kind == 'bcd' else -1
    b = lo - step if n > 1 else lo                     # the second least significant byte
    if kind.startswith('digits'):
        return b, lambda x: x + 1 if x < 9 else x - 1
    if kind.startswith('bcd'):
        return b, lambda x: x + 0x10 if x < 0x90 else x - 0x10
    return b, lambda x: x + 1 if x < 255 else x - 1


def ramfind2_low_end(a, n, kind):
    import ramfind2
    return ramfind2.low_end(a, n, kind)


def find_score(play):
    """ramfind2.score candidates, checked by replay: a digit poked 30 frames before a real rise must still be there,
    on top of the rise, 60 frames later (the game adds to the stored number), and the screen after the rise must
    differ from the unpoked replay by a few HUD digits (the redraw that the rise forces)."""
    import ramfind2
    rows = []
    for c in ramfind2.score(play.ram, play.idle)[:30]:
        a, n, kind = int(c[0][1:], 16), c[1], c[2]
        rises = score_rises(play, a, n, kind)
        pick, seen = [], set()
        for e, i in rises:                               # spread the checks over episodes
            if e not in seen and i >= 31:
                pick.append((e, i))
                seen.add(e)
        pick += [x for x in rises if x not in pick and x[1] >= 31][:3 - len(pick)]
        b, fn = poke_digit(kind, a, n)
        kept = shown = 0
        for e, i in pick[:3]:
            f0, f1 = i - 30, min(i + 60, FRAMES)
            at = set(range(i + 5, f1 + 1, 5))
            r0 = play.ram[e][f0 - 1].copy()
            r1 = r0.copy()
            r1[b] = fn(int(r0[b])) & 0xFF
            delta = int(ramfind2.number(r1[None], a, n, kind)[0][0] - ramfind2.number(r0[None], a, n, kind)[0][0])
            rc, sc = play.run_screens(e, f0, f1, at)
            rp, sp = play.run_screens(e, f0, f1, at, {b: lambda m, b=b: fn(int(m[b])) & 0xFF})
            vc, okc = ramfind2.number(rc[None], a, n, kind)
            vp, okp = ramfind2.number(rp[None], a, n, kind)
            kept += bool(okc and okp and vc[0] != ramfind2.number(r0[None], a, n, kind)[0][0]
                         and vp[0] - vc[0] == delta)
            shown += any(0 < screen_diff(sc[t], sp[t]) < 0.02 for t in sc)
        rows.append(dict(addr=a, n=n, kind=kind, rises=c[3], bad=c[4], tried=len(pick[:3]), kept=kept, shown=shown))
    rows.sort(key=lambda r: (-(r['tried'] > 0 and r['kept'] == r['tried'] and r['shown'] >= min(2, r['tried'])),
                             -(r['kept'] + r['shown']), -(r['rises'] - 2 * r['bad']), -r['n']))
    for r in rows[:3]:
        if r['tried'] and r['kept'] == r['tried'] and r['shown'] >= min(2, r['tried']):
            orient_score(play, r)
    return rows


FLIP = {'digits_hi': 'digits_lo', 'digits_lo': 'digits_hi', 'bcd': 'bcd_lo', 'bcd_lo': 'bcd'}


def orient_score(play, r):
    """Byte order by a carry: before a real rise that changes one byte c of the span, set c to the top of its range
    (9, $99, $FF); the rise then carries into the next more significant byte, which tells the order. Low scores
    move one byte only, so both orders read the same numbers in play; a wrong order breaks at the first carry.
    Rewrites r (kind, addr) when the carry goes the other way; r['carry'] = 'lower'/'higher' address or None."""
    a, n, kind = r['addr'], r['n'], r['kind']
    top = 9 if kind.startswith('digits') else 0x99 if kind.startswith('bcd') else 0xFF
    r['carry'] = None
    for e, i in score_rises(play, a, n, kind):
        if i < 31:
            continue
        ch = np.flatnonzero(play.ram[e][i - 1][a - 2:a + n + 2] != play.ram[e][i][a - 2:a + n + 2]) + a - 2
        if len(ch) != 1:
            continue
        c = int(ch[0])
        f1 = min(i + 60, FRAMES)
        rc, _ = play.run_screens(e, i - 30, f1, set())
        rp, _ = play.run_screens(e, i - 30, f1, set(), {c: lambda m: top})
        lo, hi = rp[c - 1] != rc[c - 1], rp[c + 1] != rc[c + 1]
        if lo != hi:
            r['carry'] = 'lower' if lo else 'higher'
            msb_first = kind in ('digits_hi', 'bcd')
            if kind in FLIP and lo != msb_first:
                kind = FLIP[kind]
                r.update(kind=kind, addr=c - n + 1 if kind in ('digits_hi', 'bcd') else c, flipped=True)
        break


# ---------------------------------------------------------------- in play

def probe_ram(play):
    """RAM at every probe (the state the probe started from) and its label, per episode."""
    X, y, ep = [], [], []
    for e in range(EPISODES):
        t, ok = play.probe_t[e], play.probe_ok[e].copy()
        idx = np.flatnonzero(ok)
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a - 1 <= GAP:                     # mid-jump, mid-attack: still playing
                ok[a:b] = True
        X.append(play.ram[e][np.maximum(t - 1, 0)])
        y.append(ok)
        ep.append(np.full(len(t), e))
    return np.concatenate(X), np.concatenate(y), np.concatenate(ep)


GAP = 2     # probes: an uncontrollable gap this short between controllable probes still counts as play


def value_rule(x, y):
    """Values of one byte whose probes are mostly controllable."""
    vals, inv = np.unique(x, return_inverse=True)
    pos = np.bincount(inv, weights=y.astype(float), minlength=len(vals))
    cnt = np.bincount(inv, minlength=len(vals))
    return vals[pos * 2 > cnt]


def balanced(pred, y):
    return 0.5 * (pred[y].mean() + (~pred[~y]).mean())


def find_in_play(play):
    """The byte whose values best tell controllable from uncontrollable probes, scored on a held-out episode (a byte
    with many values, a timer, cannot learn a rule that carries over). in_play = byte in values."""
    X, y, ep = probe_ram(play)
    if len(y) == 0 or y.mean() < 0.05 or y.mean() > 0.95:
        return []
    rows = []
    for a in [a for a in range(0x800) if not 0x100 <= a < 0x200]:
        x = X[:, a]
        accs = []
        for k in range(EPISODES):
            tr, te = ep != k, ep == k
            if y[te].all() or not y[te].any():
                continue
            vals = value_rule(x[tr], y[tr])
            accs.append(balanced(np.isin(x[te], vals), y[te]))
        if accs:
            vals = value_rule(x, y)
            rows.append(dict(addr=a, values=[int(v) for v in vals], cv=float(np.mean(accs)),
                             fit=float(balanced(np.isin(x, vals), y))))
    nv = n_values(play)
    for r in rows:
        r['nvals'] = int(nv[r['addr']])
    # a mode/state byte (few values) over a byte that happens to correlate (a timer, a sprite); cv in 0.02 steps
    rows.sort(key=lambda r: (-round(r['cv'] / 0.02), r['nvals'] > STATE_VALUES, len(r['values']), -r['fit']))
    return rows[:8]


STATE_VALUES = 16


def play_frames(play):
    ends = [lost_control(play, k) for k in range(EPISODES)]
    return np.concatenate([play.ram[k][:ends[k] or FRAMES] for k in range(EPISODES)])


def n_values(play):
    """Distinct values of every byte during play (before any game end)."""
    r = play_frames(play)
    srt = np.sort(r, 0)
    return 1 + (srt[1:] != srt[:-1]).sum(0)


def dwell(play, a, v):
    """Mean length (frames) of the stretches where byte a == v."""
    runs = []
    for e in range(EPISODES):
        x = np.concatenate([[False], play.ram[e][:, a] == v, [False]])
        d = np.flatnonzero(x[1:] != x[:-1])
        runs += list(d[1::2] - d[::2])
    return float(np.mean(runs)) if runs else 0.0


# ---------------------------------------------------------------- control runs, game over, death

def lost_control(play, e):
    """Start frame of the first uncontrollable stretch of episode e that lasts >= GO_RUN frames (or runs to the end of
    the episode, then at least GO_RUN / 2), after control was seen: the game ended (game over, then the title or the
    attract demo, where a stray probe may still move something). None when the episode never ends."""
    t, ok = play.probe_t[e], play.probe_ok[e].copy()
    lone = ok.copy()
    lone[1:-1] &= ~ok[:-2] & ~ok[2:]                 # one controllable probe between uncontrollable ones: a stray
    ok[1:-1] &= ~lone[1:-1]                          # (a title or demo that reacts to the d-pad once)
    seen = False
    k = 0
    while k < len(t):
        if ok[k]:
            seen, k = True, k + 1
            continue
        j = k
        while j < len(t) and not ok[j]:
            j += 1
        end = int(t[j]) if j < len(t) else FRAMES
        if seen and (end - t[k] >= GO_RUN or (j == len(t) and end - t[k] >= GO_RUN // 2)):
            return int(t[k] - PROBE_EVERY // 2)
        k = j
    return None


GO_RUN = 800


def edges(r):
    """(frame, new value) of every change of each byte: dict addr -> (frames, values)."""
    ch = r[1:] != r[:-1]
    f, a = np.nonzero(ch)
    out = {}
    order = np.argsort(a, kind='stable')
    f, a = f[order] + 1, a[order]
    bounds = np.searchsorted(a, np.arange(0x801))
    for addr in range(0x800):
        s, t = bounds[addr], bounds[addr + 1]
        if s < t:
            out[addr] = (f[s:t], r[f[s:t], addr])
    return out


def find_game_over(play):
    """The (byte, value) that first appears just after the game ends (control lost for good) in every episode that
    ends, and never before that in any episode."""
    ends = [lost_control(play, e) for e in range(EPISODES)]
    if not any(x is not None for x in ends):
        return []
    stats = {}
    for e in range(EPISODES):
        end = ends[e]
        cut = FRAMES if end is None else end - 100
        for a, (f, v) in edges(play.ram[e]).items():
            if 0x100 <= a < 0x200:
                continue
            before = set(v[f < cut].tolist()) | {int(play.ram[e][0, a])}
            for val in set(v.tolist()) | before:
                s = stats.setdefault((a, int(val)), [0, 0, []])
                if val in before:
                    s[1] += 1                     # seen during play: not a game-over value
                elif end is not None:
                    hit = f[(v == val) & (f >= cut)]
                    if len(hit) and hit[0] <= end + 600:
                        s[0] += 1
                        s[2].append(int(hit[0]) - end)
    n_end = sum(x is not None for x in ends)
    rows = [dict(addr=a, value=v, hits=s[0], ends=n_end, delay=float(np.mean(s[2])))
            for (a, v), s in stats.items() if s[1] == 0 and s[0] == n_end]
    nv = n_values(play)
    for r in rows:
        r.update(nvals=int(nv[r['addr']]), dwell=dwell(play, r['addr'], r['value']))
    # a state byte (few values in play) that holds the new value (the GAME OVER screen), soonest after the end
    rows.sort(key=lambda r: (not (r['nvals'] <= STATE_VALUES and r['dwell'] >= 30), abs(r['delay'] - 120), r['addr']))
    return rows[:8]


def find_death(play, lives):
    """The (byte, value) whose appearance comes shortly (<= 400 frames) before each lives decrement and seldom
    anywhere else during play (death animation, 'player dying' state)."""
    ends = [lost_control(play, e) for e in range(EPISODES)]
    steps = [(e, i) for e, i in lives['steps'] if ends[e] is None or i <= ends[e] + 400]
    if not steps:
        return []
    tot = {}
    for e in range(EPISODES):
        di = np.array([i for ee, i in steps if ee == e], int)
        stop = FRAMES if ends[e] is None else ends[e] + 400
        for a, (f, v) in edges(play.ram[e]).items():
            if 0x100 <= a < 0x200 or a == lives['addr']:
                continue
            keep = f < stop
            f, v = f[keep], v[keep]
            for val in np.unique(v):
                fv = f[v == val]
                s = tot.setdefault((a, int(val)), [0, 0, []])
                for i in di:
                    h = fv[(fv <= i) & (fv >= i - 400)]
                    if len(h):
                        s[0] += 1
                        s[2].append(int(i - h[0]))
                near = np.zeros(len(fv), bool)
                for i in di:
                    near |= (fv <= i) & (fv >= i - 400)
                if ends[e] is not None:
                    near |= (fv >= ends[e] - 400)
                s[1] += int((~near).sum())
    rows = [dict(addr=a, value=v, hits=s[0], false=s[1], lead=float(np.mean(s[2])) if s[2] else 0.0)
            for (a, v), s in tot.items() if s[0] > 0]
    nv = n_values(play)
    rows.sort(key=lambda r: (-(r['hits'] - r['false']), -r['lead']))
    rows = rows[:60]
    for r in rows:
        r.update(nvals=int(nv[r['addr']]), dwell=dwell(play, r['addr'], r['value']), deaths=len(steps))
    # a state byte that holds the value for a while (death animation), the earliest such edge before the decrement
    rows.sort(key=lambda r: (-(r['hits'] - r['false']), not (r['nvals'] <= STATE_VALUES and r['dwell'] >= 8),
                             -r['lead']))
    return rows[:8]


# ---------------------------------------------------------------- level

def find_level(play, lives, skip=()):
    """A byte that holds a small number through play (or steps +1 once: a stage cleared), and whose +1 poke a little
    before a respawn stays, and changes the scene the game rebuilds while still giving control back (a crash, a
    game-over or pause flag does not). Phase 1 (to 150 frames after the decrement) keeps bytes that change the
    screen at all; phase 2 plays on until 90 frames after control returned and probes control."""
    steps = []
    for e, i in lives['steps']:
        t, ok = play.probe_t[e], play.probe_ok[e]
        back = t[(t > i) & ok]
        if i >= 60 and len(back) and back[0] + 90 <= FRAMES:
            steps.append((e, i, int(max(back[0], i + 150) + 90)))
    if not steps:
        return []
    e, i, fb = steps[0]
    ends = [lost_control(play, k) for k in range(EPISODES)]
    r0 = play.ram[0][0]
    const = (r0 <= 31) & (r0 == play.ram[1][0]) & (r0 == play.ram[2][0])
    stepped = np.zeros(0x800, bool)
    for k in range(EPISODES):
        d = np.diff(play.ram[k][:ends[k] or FRAMES].astype(np.int16), axis=0)
        n = (d != 0).sum(0)
        const &= (n == 0) | ((n == 1) & ((d == 0) | (d == 1)).all(0))
        stepped |= n == 1
    const[0x100:0x200] = False
    for a in skip:
        const[a] = False
    f0, f1 = i - 60, i + 150
    _, ctrl = play.run_screens(e, f0, f1, {f1})
    rows = []
    for a in np.flatnonzero(const):
        a = int(a)
        rp, sp = play.run_screens(e, f0, f1, {f1}, {a: lambda m, a=a: int(m[a]) + 1})
        if rp[a] == int(play.ram[e][f0 - 1][a]) + 1 and screen_diff(ctrl[f1], sp[f1]) > 0:
            rows.append(dict(addr=a, value=int(r0[a]), step=bool(stepped[a]), diff150=screen_diff(ctrl[f1], sp[f1])))
    # a stage number is read when the stage is (re)built: poked in the middle of play it changes nothing on screen
    # but a HUD digit; a PPU shadow, a palette or an object byte acts at once
    mid = [int(t) for t, ok in zip(play.probe_t[e], play.probe_ok[e]) if ok and 60 <= t <= f0 - 300]
    if mid:
        tm = mid[len(mid) // 2]
        _, ctrl = play.run_screens(e, tm, tm + 60, {tm + 60})
        for row in rows:
            a = row['addr']
            _, sp = play.run_screens(e, tm, tm + 60, {tm + 60}, {a: lambda m, a=a: int(m[a]) + 1})
            row['diff_mid'] = screen_diff(ctrl[tm + 60], sp[tm + 60])
        rows = [row for row in rows if row['diff_mid'] < 0.02]
    _, ctrl = play.run_screens(e, f0, fb, {fb})
    for row in rows:
        a = row['addr']
        rp, sp = play.run_screens(e, f0, fb, {fb}, {a: lambda m, a=a: int(m[a]) + 1})
        row.update(kept=bool(rp[a] == int(play.ram[e][f0 - 1][a]) + 1), diff=screen_diff(ctrl[fb], sp[fb]))
        # control: a new stage may open with an intro, so wait up to 600 frames more (pads: none)
        row['control'] = False
        if play.pos and row['kept'] and row['diff'] >= 0.2:
            for _ in range(16):
                if play.probe():
                    row['control'] = True
                    break
                play.nes.step((), 40)
    # a stage rebuilt at the respawn changes the screen early (its intro) and late (its layout): both count
    # a new game starts at stage 0 (SMB's world $075F); a scene-type byte that also rebuilds the screen does not
    # (SMB's area type $074E is 1 in 1-1)
    rows.sort(key=lambda r: (-(r['kept'] and r['control'] and r['diff'] >= 0.2), -(r['value'] == 0), -r['step'],
                             -(r['diff150'] >= 0.2), -r['diff']))
    return rows[:10]


def level_candidates(play, skip=()):
    """Bytes that look like a stage number: <= 31 at the start of every episode, then constant or one +1 step
    until control is lost for good."""
    ends = [lost_control(play, k) for k in range(EPISODES)]
    r0 = play.ram[0][0]
    const = (r0 <= 31) & (r0 == play.ram[1][0]) & (r0 == play.ram[2][0])
    for k in range(EPISODES):
        d = np.diff(play.ram[k][:ends[k] or FRAMES].astype(np.int16), axis=0)
        n = (d != 0).sum(0)
        const &= (n == 0) | ((n == 1) & ((d == 0) | (d == 1)).all(0))
    const[0x100:0x200] = False
    for a in skip:
        const[a] = False
    return [int(a) for a in np.flatnonzero(const)]


BOOT_AFTER = 300      # frames played after the boot pads (the new stage's intro, then its layout)


def find_level_boot(play, skip=()):
    """Stage number without a death: a stage is built from its number between the last menu press and control.
    Poke +1 right after the last press of the boot, play the rest of the boot and BOOT_AFTER idle frames: a stage
    byte keeps the +1, the new stage looks different (> 20% of the screen) and the player gets control again.
    A byte the game reads every frame (scroll, palette, object) also changes the picture, so the same poke made in
    the middle of play must change at most a few HUD pixels."""
    pads = play.boot_pads
    pressed = [i for i, p in enumerate(pads) if p]
    if not pressed or not play.pos:
        return []
    t = pressed[-1] + 1
    nes = play.nes
    nes.load(play.power_on)
    for p in pads[:t]:
        nes.step(buttons(p), 1)
    at_t = nes.save()

    def run(poke=None):
        nes.load(at_t)
        if poke is not None:
            nes.poke(poke, int(nes.ram()[poke]) + 1)
        for p in pads[t:]:
            nes.step(buttons(p), 1)
        nes.step((), BOOT_AFTER)
        return nes.ram(), nes.screen()

    _, ctrl = run()
    rows = []
    mid = [int(x) for x, ok in zip(play.probe_t[0], play.probe_ok[0]) if ok and x >= 300]
    for a in level_candidates(play, skip):
        before = int(play.ram[0][0][a])
        ram, pic = run(a)
        d = screen_diff(ctrl, pic)
        if int(ram[a]) != before + 1 or d < 0.2:
            continue
        row = dict(addr=a, value=before, diff=d, control=False)
        for _ in range(16):                         # a stage may open with an intro: wait up to 640 frames more
            if play.probe():
                row['control'] = True
                break
            nes.step((), 40)
        if mid:
            tm = mid[len(mid) // 2]
            _, c = play.run_screens(0, tm, tm + 60, {tm + 60})
            _, q = play.run_screens(0, tm, tm + 60, {tm + 60}, {a: lambda m, a=a: int(m[a]) + 1})
            row['diff_mid'] = screen_diff(c[tm + 60], q[tm + 60])
        if row['control'] and row.get('diff_mid', 0) < 0.02:
            rows.append(row)
    rows.sort(key=lambda r: (-(r['value'] == 0), -r['diff']))
    return rows


# ---------------------------------------------------------------- the reader

def hexa(a):
    return f'${a:04X}'


def analyze(play, row):
    """All fields of one ROM; each is a dict with a confidence and evidence, or 'unknown'."""
    res = dict(rom=row['rom'], boot_pads=bytes(play.boot_pads).hex(), boot_frames=len(play.boot_pads),
               moved_by=(row.get('moved_by') or [None])[0],
               lit=round(play.lit, 3))
    ok_frac = float(np.mean(np.concatenate(play.probe_ok))) if play.pos else 0.0
    ok_early = float(np.mean([ok[:15].mean() for ok in play.probe_ok])) if play.pos else 0.0
    res['controllable_frac'] = round(ok_frac, 3)
    res['controllable_first600'] = round(ok_early, 3)
    pos = {}
    for k in ('x', 'y'):
        c = row.get(f'{k}_candidates') or []
        if c:
            pos[k] = c[0][0]
            pos[f'{k}_alt'] = [x[0] for x in c[1:3]]
            pos[f'{k}_score'] = c[0][1]
    res['position'] = dict(pos, confidence='high' if ok_early >= 0.3 else 'low',
                           evidence='stage 1: the d-pad pushes the byte apart (ramfind.py), '
                                    f'control probes ok {ok_early:.0%} of the first 600 frames') if pos else 'unknown'
    # lives
    lv = find_lives(play)
    top = lv[0] if lv else None
    good = (top and top['tried'] and top['kept'] == top['tried'] and top['regained'] >= 1
            and top['respawn'] >= 0.75 * len(top['steps']))
    res['lives'] = dict(addr=hexa(top['addr']), start=top['start'], deaths=top['deaths'], other=top['other'],
                        respawn=top['respawn'], poke_kept=f"{top['kept']}/{top['tried']}",
                        poke_shown=f"{top['shown']}/{top['tried']}",
                        confidence='high' if top['tried'] >= 2 and top['shown'] else 'medium',
                        evidence='-1 steps at deaths (control lost around each), a +1 poke before a death survives it'
                        ) if good else 'unknown'
    # score
    sc = find_score(play)
    top = sc[0] if sc else None
    # the HUD must redraw after at least 2 of 3 rises: 1 of 3 was always a wrong byte on the SMB1 family ($00F2)
    good = top and top['tried'] and top['kept'] == top['tried'] and top['shown'] >= min(2, top['tried'])
    res['score'] = dict(addr=hexa(top['addr']), n=top['n'], kind=top['kind'], rises=top['rises'],
                        poke_kept=f"{top['kept']}/{top['tried']}", poke_shown=f"{top['shown']}/{top['tried']}",
                        carry=top.get('carry'),
                        confidence='high' if top['tried'] >= 2 and top['shown'] >= 2 else 'medium',
                        evidence='round-step rises; a digit poked before a real rise is added to and redrawn'
                        ) if good else 'unknown'
    # in play
    ip = find_in_play(play)
    top = ip[0] if ip else None
    res['in_play'] = dict(addr=hexa(top['addr']), values=top['values'], cv=round(top['cv'], 3),
                          confidence='high' if top['cv'] >= 0.95 else 'medium' if top['cv'] >= 0.88 else 'low',
                          evidence='byte values that separate controllable from uncontrollable probes '
                                   '(balanced accuracy on a held-out episode)'
                          ) if top and top['cv'] >= 0.78 else 'unknown'
    # death (needs lives)
    res['death'] = 'unknown'
    if isinstance(res['lives'], dict):
        lrow = lv[0]
        dt = find_death(play, lrow)
        top = dt[0] if dt else None
        if top and top['hits'] >= 0.8 * top['deaths'] and top['false'] <= top['hits'] // 4:
            state = top['nvals'] <= STATE_VALUES and top['dwell'] >= 8
            res['death'] = dict(addr=hexa(top['addr']), value=top['value'], hits=top['hits'], deaths=top['deaths'],
                                false=top['false'], lead=round(top['lead']), nvals=top['nvals'],
                                dwell=round(top['dwell'], 1),
                                confidence='high' if state and top['hits'] == top['deaths'] and not top['false']
                                else 'medium' if state else 'low',
                                evidence='byte == value starts shortly before every lives decrement, seldom else')
    # game over
    go = [r for r in find_game_over(play) if r['nvals'] <= STATE_VALUES and r['dwell'] >= 30]   # a state, held
    top = go[0] if go else None
    res['game_over'] = 'unknown'
    if top:
        conf = 'low' if not isinstance(res['lives'], dict) else 'high' if top['ends'] >= 2 else 'medium'
        res['game_over'] = dict(addr=hexa(top['addr']), value=top['value'], ends=top['ends'],
                                delay=round(top['delay']), nvals=top['nvals'], dwell=round(top['dwell'], 1),
                                confidence=conf, evidence='byte == value first appears when control is lost for good '
                                '(every episode that ends), never during play')
    # level (needs a respawn)
    res['level'] = 'unknown'
    if isinstance(res['lives'], dict):
        skip = [int(res[k]['addr'][1:], 16) for k in ('lives', 'score', 'in_play', 'death', 'game_over')
                if isinstance(res[k], dict)]
        lvl = find_level(play, lv[0], skip)
        top = lvl[0] if lvl else None
        if top and top['kept'] and top['diff'] >= 0.2:
            res['level'] = dict(addr=hexa(top['addr']), value=top['value'], diff=round(top['diff'], 3),
                                confidence='medium', evidence='constant in play; +1 before a respawn stays and '
                                'reloads a different scene', alt=[hexa(r['addr']) for r in lvl[1:4]
                                                                 if r['kept'] and r['diff'] >= 0.2])
    if res['level'] == 'unknown':
        skip = [int(res[k]['addr'][1:], 16) for k in ('lives', 'score', 'in_play', 'death', 'game_over')
                if isinstance(res[k], dict)]
        lvl = find_level_boot(play, skip)
        if lvl:
            top = lvl[0]
            res['level'] = dict(addr=hexa(top['addr']), value=top['value'], diff=round(top['diff'], 3),
                                confidence='low',     # a scene-type byte passes too (SMB's area type $074E)
                                evidence='+1 right after the last boot press builds a different stage the player '
                                         'controls; the same poke in play changes at most a HUD digit (may be a '
                                         'stage-type byte, not the stage number)',
                                alt=[hexa(r['addr']) for r in lvl[1:4]])
    return res


def _w(row):
    try:
        play = Play(str(ROM_DIR / row['rom']), row)
        out = analyze(play, row)
        if row.get('truth'):
            import truth
            ram = np.stack(play.ram)
            out['truth'] = dict(slug=row['truth'], checks=truth.evaluate(row['truth'], out, ram,
                                                                           truth.truth_signals(row['truth'], ram)))
        return out
    except Exception as e:  # noqa: BLE001
        return dict(rom=row['rom'], error=repr(e)[:300])


def stage1(paths):
    """Stage-1 rows by ROM; a later file's controllable row replaces an earlier one (the 40-press retry)."""
    rows = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r.get('controllable') or r['rom'] not in rows:
                rows[r['rom']] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--stage1', nargs='+', default=[str(ROOT / 'reports/romlib/ramfind_v1.jsonl'),
                                                    str(ROOT / 'reports/romlib/ramfind_v2.jsonl')])
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--truth', action='store_true', help='only the reward10 truth games, with checks')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    rows = stage1(a.stage1)
    if a.truth:
        import truth
        todo = []
        for slug in truth.SLUGS:
            r = rows.get(truth.rom_of(slug))
            if r and r.get('boot_pads'):
                todo.append(dict(r, truth=slug))
    else:
        todo = [r for r in rows.values() if r.get('boot_pads') and (not a.only or r['rom'] in a.only)]
    done = {json.loads(l)['rom'] for l in open(a.out)} if Path(a.out).exists() else set()
    todo = [r for r in todo if r['rom'] not in done]
    todo.sort(key=lambda r: r['rom'])
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, 'a') as fh:
        for r in pool.imap_unordered(_w, todo):
            fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
