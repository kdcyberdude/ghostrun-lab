"""Gate 3: action masking and the controller follow the contract (rl/csrc/controller.cuh, host build).

Checked against an independent Python reading of the spec (DECISIONS.md 2026-11-10):
  * 81 / 162 / 324 moves for games with neither / Start / Start+Select; opposite directions impossible;
  * Start/Select only where START_SELECT allows (from scripts/reward10 modules);
  * turbo pressed on even frames only, hold = pressed every frame, A+TurboA never both;
  * a dropped input masks exactly the moves that use it; sticky repeats release only the dropped part;
  * the dropout schedule: at most one input off, runs of 8..22 decisions, healthy gaps >= 1 with mean ~35,
    ~30% of decisions with an input off, each input ~3.75%, timeline = the next 45 decisions exactly,
    same seed -> same schedule, disabled -> never off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rl' / 'tests'))
sys.path.insert(0, str(ROOT / 'scripts' / 'reward10'))
sys.path.insert(0, str(ROOT / 'rl' / 'scripts'))
import hostlib  # noqa: E402
import contract  # noqa: E402
from gen_reward_consts import GAMES  # noqa: E402

UP, DOWN, LEFT, RIGHT, A, B, SELECT, START = 16, 32, 64, 128, 1, 2, 4, 8
DPAD = [0, UP, UP | RIGHT, RIGHT, DOWN | RIGHT, DOWN, DOWN | LEFT, LEFT, UP | LEFT]
DROP_NAMES = ['Up', 'Down', 'Left', 'Right', 'A-hold', 'B-hold', 'TurboA', 'TurboB']
NM = 324


def ref_unpack(i):
    d, i = i % 9, i // 9
    a, i = i % 3, i // 3
    b, i = i % 3, i // 3
    s, i = i % 2, i // 2
    return d, a, b, s, i % 2


def ref_pad(i, frame):
    d, a, b, s, sel = ref_unpack(i)
    even = frame % 2 == 0
    return (DPAD[d] | (A if a == 1 or (a == 2 and even) else 0) | (B if b == 1 or (b == 2 and even) else 0)
            | (START if s else 0) | (SELECT if sel else 0))


def ref_uses(i, drop):
    d, a, b, s, sel = ref_unpack(i)
    return {0: DPAD[d] & UP, 1: DPAD[d] & DOWN, 2: DPAD[d] & LEFT, 3: DPAD[d] & RIGHT,
            4: a == 1, 5: b == 1, 6: a == 2, 7: b == 2}[drop]


def tables(lib, start_ok, select_ok):
    legal = np.zeros(9 * NM, np.uint8)
    rest = np.zeros(9 * NM, np.int32)
    lib.rlc_legal_table(start_ok, select_ok, hostlib.ptr(legal), hostlib.ptr(rest))
    return legal.reshape(9, NM), rest.reshape(9, NM)


def test_pad_bytes():
    lib = hostlib.load()
    out = np.zeros(2 * NM, np.uint8)
    lib.rlc_pad_table(hostlib.ptr(out))
    out = out.reshape(NM, 2)
    for i in range(NM):
        for f in (0, 1):
            assert out[i, f] == ref_pad(i, f), (i, f)
        p = out[i, 0] | out[i, 1]
        assert not (p & UP and p & DOWN) and not (p & LEFT and p & RIGHT)


def test_moves_per_game_and_masks():
    lib = hostlib.load()
    want_count = {}
    for slug in GAMES:
        ss = getattr(contract.load(slug), 'START_SELECT', {})
        s_ok, sel_ok = int('START' in ss), int('SELECT' in ss)
        legal, rest = tables(lib, s_ok, sel_ok)
        want_count[slug] = int(legal[0].sum())
        for i in range(NM):
            d, a, b, s, sel = ref_unpack(i)
            allowed_ss = (not s or s_ok) and (not sel or sel_ok)
            assert legal[0, i] == allowed_ss, (slug, i)
            for drop in range(8):
                assert legal[drop + 1, i] == (allowed_ss and not ref_uses(i, drop)), (slug, i, DROP_NAMES[drop])
                # sticky repeat: release only the dropped part (and forbidden Start/Select)
                r = rest[drop + 1, i]
                assert legal[drop + 1, r], (slug, i, drop, r)
                pr, pi = ref_pad(r, 0) | ref_pad(r, 1), ref_pad(i, 0) | ref_pad(i, 1)
                assert pr & ~pi == 0, 'a repeat never adds a button'
    assert want_count['punchout'] == 324 and want_count['zelda'] == 162
    assert all(want_count[g] == 81 for g in GAMES if g not in ('punchout', 'zelda')), want_count


def schedule(lib, seed, n, enabled=True, with_timeline=False):
    dropped = np.zeros(n, np.int8)
    tl = np.zeros((n, 8, 45), np.uint8) if with_timeline else None
    lib.rlc_schedule(seed, int(enabled), n, hostlib.ptr(dropped), hostlib.ptr(tl) if with_timeline else None)
    return dropped, tl


def runs(x):
    """(value, length) of the runs of an int sequence."""
    edges = np.flatnonzero(np.diff(x)) + 1
    starts = np.r_[0, edges]
    lens = np.diff(np.r_[starts, len(x)])
    return x[starts], lens


def test_dropout_schedule_statistics():
    lib = hostlib.load()
    n = 15 * 3600 * 20  # 20 hours of decisions
    d, _ = schedule(lib, lib.rlc_episode_seed(7, 3, 11), n)
    vals, lens = runs(d)
    inner_v, inner_l = vals[1:-1], lens[1:-1]  # ignore the cut runs at both ends
    assert set(np.unique(d)) <= set(range(-1, 8))
    on = inner_v >= 0
    # consecutive dropouts may touch only through a gap of >= 1 healthy decision
    assert not np.any(on[1:] & on[:-1]), 'two dropouts without a healthy gap'
    assert inner_l[on].min() >= 8 and inner_l[on].max() <= 22
    assert inner_l[~on].min() >= 1
    assert abs(inner_l[~on].mean() - 35) < 1.0, inner_l[~on].mean()
    frac = (d >= 0).mean()
    assert 0.29 < frac < 0.31, frac
    share = np.bincount(d[d >= 0], minlength=8) / n
    assert np.all(np.abs(share - 0.0375) < 0.003), share


def test_timeline_is_the_future():
    lib = hostlib.load()
    n = 3000
    d, tl = schedule(lib, 12345, n, with_timeline=True)
    for t in range(n - 45):
        want = np.zeros((8, 45), np.uint8)
        for j in range(45):
            if d[t + j] >= 0:
                want[d[t + j], j] = 1
        assert np.array_equal(tl[t], want), t
        assert tl[t].sum(0).max() <= 1, 'at most one input off per column'


def test_seeds_and_disable():
    lib = hostlib.load()
    a, _ = schedule(lib, 99, 5000)
    b, _ = schedule(lib, 99, 5000)
    c, _ = schedule(lib, 100, 5000)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    off, tl = schedule(lib, 99, 500, enabled=False, with_timeline=True)
    assert (off == -1).all() and tl.sum() == 0
    s = {lib.rlc_episode_seed(1, e, k) for e in range(64) for k in range(64)}
    assert len(s) == 64 * 64


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
