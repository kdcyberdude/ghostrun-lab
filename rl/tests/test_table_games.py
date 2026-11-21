"""Table games (rl/games_table.json, rl/csrc/reward10.cuh observe_table) in the environment's CPU build against an
independent path: NeSLE's CPU harness (scripts/nesle_check) driven with the env's logged per-frame pad bytes, and the
Python RewardTracker on scripts/reward10/table.py (the Python twin of the table reader) on that harness's RAM.

Per decision the reward (as float32) and done must be identical, and every env colour number must map to one harness
RGB (the pictures agree). Also: the 12 hand-read games keep ids 0..11, table games get 12+.

Needs RL_TABLE_GAMES (default rl/games_table.json), RL_CPU_LIB or a buildable rl/csrc + NeSLE (NESLE_INC,
NESLE_ROM_CPP), and NESLE_HOST_LIB for the harness, all built from the same NeSLE.
usage: python rl/tests/test_table_games.py [steps] [slug,slug,...]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('RL_TABLE_GAMES', str(ROOT / 'rl/games_table.json'))
sys.path[:0] = [str(ROOT / 'rl'), str(ROOT / 'scripts/nesle_check'), str(ROOT / 'scripts/reward10'), str(ROOT / 'rl/tests')]
import neslenes  # noqa: E402
from reward import RewardTracker  # noqa: E402
import table  # noqa: E402
from rlenv import RlEnv, GAMES, MANIFEST, TABLE, rom_path, boot_pads  # noqa: E402
from actions import random_legal  # noqa: E402
from test_env_cpu import check_pictures  # noqa: E402





def pick():
    full = [g for g, v in TABLE.items() if min(v['table'][k] for k in ('lives_addr', 'game_over_addr', 'score_addr')) >= 0]
    bare = [g for g, v in TABLE.items() if v['table']['lives_addr'] < 0 and v['table']['score_addr'] < 0]
    return [full[0], full[len(full) // 2], bare[0]]


def run(steps=300, seed=7, games=None):
    games = games or pick()
    assert GAMES[:len(MANIFEST)] == list(MANIFEST) and all(GAMES.index(g) >= len(MANIFEST) for g in games)
    env = RlEnv({g: 2 for g in games}, device='cpu', seed=seed, sticky=True, idle_max=0, dropout=True)
    env.start()
    gen = torch.Generator().manual_seed(seed)
    ref = {}
    for e in range(env.n):
        slug = env.games[e // 2]
        n = neslenes.NesleNes(rom_path(slug))
        for b in boot_pads(slug):
            neslenes.lib().nh_step(n.h, b, 1)
        for _ in range(4):
            neslenes.lib().nh_step(n.h, 0, 1)
        tr = RewardTracker(table.game(TABLE[slug]))
        tr.update(n.ram(), 0, frames=4)
        ref[e] = dict(slug=slug, n=n, tr=tr, pairs={}, alive=True, steps=0, ret=0.0)
        check_pictures(ref[e]['pairs'], env.obs[e, 3].numpy(), n.screen())
    for t in range(steps):
        env.actions.copy_(random_legal(env.masks.clone(), gen))
        env.step()
        for e, r in ref.items():
            if not r['alive']:
                continue
            for f in range(4):
                neslenes.lib().nh_step(r['n'].h, int(env.pads[e, f]), 1)
                if not env.done[e]:
                    check_pictures(r['pairs'], env.obs[e, f].numpy(), r['n'].screen())
            rew, done, _ = r['tr'].update(r['n'].ram(), t + 1, frames=4)
            assert float(np.float32(rew)) == float(env.reward[e]), (r['slug'], e, t, rew, float(env.reward[e]))
            assert bool(done) == bool(env.done[e]), (r['slug'], e, t, done, int(env.done[e]))
            r['steps'] += 1
            r['ret'] += rew
            if done:
                r['alive'] = False
    out = {}
    for e, r in ref.items():
        bad = {c: s for c, s in r['pairs'].items() if len(s) != 1}
        assert not bad, (r['slug'], 'a colour number shows as several harness colours', list(bad.items())[:3])
        out.setdefault(r['slug'], []).append((r['steps'], round(r['ret'], 3), len(r['tr'].reader.cells)))
    return out


def test_table_games_match_harness_and_python_reward():
    run(150)


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    games = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    for slug, s in run(steps, games=games).items():
        print(f'{slug:40s} identical; (decisions, return, cells) per env: {s}')
