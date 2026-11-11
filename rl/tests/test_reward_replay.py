"""Gate 2: the ported reward (rl/csrc/reward10.cuh) equals scripts/reward10 on the same RAM, bit for bit.

Replays the saved per-frame RAM traces (.cache/nesle-check/traces/<game>.ram.bin from fceumm and
<game>.nesle.ram.bin from NeSLE) through Python RewardTracker and the C++ port, sampling one snapshot every
`stride` frames from several start offsets. When an episode ends, both restart a fresh tracker on the next
snapshot, so one trace exercises many episodes. Rewards must be equal as doubles (no tolerance); done flags,
events, event levels, death counts and end reasons must be identical.

The C++ build here is the host compile of the same header the GPU kernel includes (-ffp-contract=off). The GPU
build is checked by rl/tests/test_reward_gpu.py (needs the GPU; ask the user first).

usage: python -m pytest rl/tests/test_reward_replay.py -q      (or run directly for a per-game summary)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'reward10'))
sys.path.insert(0, str(ROOT / 'rl' / 'scripts'))
import contract  # noqa: E402
from reward import RewardTracker  # noqa: E402
from gen_reward_consts import GAMES  # noqa: E402
sys.path.insert(0, str(ROOT / 'rl' / 'tests'))
import hostlib  # noqa: E402

TRACES = ROOT / '.cache' / 'nesle-check' / 'traces'
# fceumm random play (fceumm_trace.py GAME DIR --random 6000 --seed S --max-frames 36000), seeds 1 and 2
RANDOM_TRACES = [ROOT / '.cache' / 'rl' / 'traces-random' / f'seed{s}' for s in (1, 2)]
EV_BITS = {'skip': 1, 'level': 2, 'death': 4, 'won': 8, 'game_over': 16, 'stall': 32, 'rollback': 64}
END = {None: 0, 'game_over': 1, 'won': 2, 'skip': 3, 'stall': 4}


def build_lib():
    return hostlib.load()


def c_replay(lib, gid, rams, frames):
    n = len(rams)
    rams = np.ascontiguousarray(rams, np.uint8)
    frames = np.ascontiguousarray(frames, np.int32)
    out = dict(reward=np.zeros(n, np.float64), done=np.zeros(n, np.uint8), events=np.zeros(n, np.uint32),
               level=np.zeros(n, np.int32), deaths=np.zeros(n, np.int32), end=np.zeros(n, np.uint8))
    p = hostlib.ptr
    overflow = lib.rl10_replay(gid, p(rams), n, p(frames), p(out['reward']), p(out['done']), p(out['events']),
                               p(out['level']), p(out['deaths']), p(out['end']))
    assert overflow == 0, 'key set overflow'
    return out


def py_episode(g, rams, frames):
    """Python RewardTracker over snapshots until done; returns per-step rows and the index after the episode."""
    tr = RewardTracker(g)
    rows = []
    for i in range(len(rams)):
        r, done, ev = tr.update(rams[i], i, frames=int(frames[i]))
        bits, lvl = 0, 0
        for e in ev:
            bits |= EV_BITS[e[0]]
            if e[0] in ('level', 'skip'):
                lvl = e[1]
        rows.append((r, done, bits, lvl, tr.deaths, END[tr.end_reason]))
        if done:
            return rows, i + 1
    return rows, len(rams)


def compare_trace(lib, slug, ram_path, stride, offset):
    g = contract.load(slug)
    gid = GAMES.index(slug)
    ram = np.fromfile(ram_path, np.uint8).reshape(-1, 2048)
    snaps = ram[offset::stride]
    frames = np.full(len(snaps), stride, np.int32)
    stats = dict(steps=0, episodes=0, reward=0.0, events=0)
    i = 0
    while i < len(snaps):
        rows, j = py_episode(g, snaps[i:], frames[i:])
        c = c_replay(lib, gid, snaps[i:i + j], frames[i:i + j])
        for k, (r, done, bits, lvl, deaths, end) in enumerate(rows):
            got = (c['reward'][k], bool(c['done'][k]), int(c['events'][k]), int(c['level'][k]), int(c['deaths'][k]),
                   int(c['end'][k]))
            want = (r, bool(done), bits, lvl, deaths, end)
            assert got == want, (f'{slug} {ram_path.name} stride {stride} offset {offset}: snapshot {i + k} '
                                 f'(frame {offset + (i + k) * stride}) python {want} != port {got}')
            stats['reward'] += r
            stats['events'] += bin(bits).count('1')
        stats['steps'] += len(rows)
        stats['episodes'] += 1
        i += j
    return stats


def trace_files():
    out = []
    for slug in GAMES:
        for name in (f'{slug}.ram.bin', f'{slug}.nesle.ram.bin'):
            p = TRACES / name
            if p.exists():
                out.append((slug, p))
        for d in RANDOM_TRACES:
            if (d / f'{slug}.ram.bin').exists():
                out.append((slug, d / f'{slug}.ram.bin'))
    return out


CASES = [(4, 0), (4, 1), (4, 2), (4, 3), (1, 0), (8, 5)]


def test_consts_current():
    subprocess.run([sys.executable, str(ROOT / 'rl' / 'scripts' / 'gen_reward_consts.py'), '--check'], check=True)


def test_all_games_have_traces():
    have = {s for s, _ in trace_files()}
    assert have == set(GAMES), f'missing traces: {sorted(set(GAMES) - have)}'


def test_rollback_boundaries_match_python_and_cpp():
    lib = build_lib()
    g = contract.load('smb')

    def ram(x, routine=8, level=15, mode=1):
        r = np.zeros(2048, np.uint8)
        r[0x770], r[0x772], r[0x0E], r[0x75A] = mode, 3, routine, 2
        r[0x75F], r[0x75C] = level // 4, level % 4
        r[0x6D], r[0x86] = (x // 256) % 256, x % 256
        return r

    cases = [
        ([ram(750), ram(0)], True),             # at threshold
        ([ram(749), ram(0)], False),            # just below threshold
        ([ram(1013), ram(-8)], True),           # signed maze rewind
        ([ram(1013), ram(0, routine=6)], False), # death/respawn
        ([ram(1013), ram(0, routine=7)], False), # pipe/invalid position
        ([ram(1013, routine=7), ram(0)], False),
        ([ram(1013), ram(0, level=16)], False),  # next level
        ([ram(1013), ram(0, mode=0)], False),   # not in play
        ([ram(1013, level=31), ram(0, level=31, mode=2)], False),  # native win
    ]
    for snapshots, rollback in cases:
        frames = np.full(len(snapshots), 4, np.int32)
        rows, _ = py_episode(g, snapshots, frames)
        c = c_replay(lib, GAMES.index('smb'), snapshots, frames)
        assert bool(c['events'][1] & EV_BITS['rollback']) == rollback
        for i, row in enumerate(rows):
            actual = (c['reward'][i], bool(c['done'][i]), int(c['events'][i]), int(c['level'][i]),
                      int(c['deaths'][i]), int(c['end'][i]))
            assert actual == row


def test_reward_replay_bit_exact():
    lib = build_lib()
    for slug, path in trace_files():
        for stride, offset in CASES:
            compare_trace(lib, slug, path, stride, offset)


if __name__ == '__main__':
    lib = build_lib()
    for slug, path in trace_files():
        tot = dict(steps=0, episodes=0, reward=0.0, events=0)
        for stride, offset in CASES:
            s = compare_trace(lib, slug, path, stride, offset)
            for k in tot:
                tot[k] += s[k]
        print(f'{slug:13s} {path.parent.name[:12]:12s} {path.name:28s} identical: {tot["steps"]:7d} updates, {tot["episodes"]:4d} episodes, '
              f'{tot["events"]:5d} events, reward sum {tot["reward"]:.3f}')
