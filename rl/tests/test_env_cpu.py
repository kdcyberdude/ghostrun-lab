"""The environment (CPU reference build of rl/csrc/rlenv_core.cuh) against an independent path: NeSLE's verified CPU
harness (scripts/nesle_check, nh::Console) driven with the env's logged per-frame pad bytes, and the Python reward10
RewardTracker on that harness's RAM.

Checked for every game, with sticky actions and button dropout on (idle 0 so the start is known):
  * boot + 4 neutral frames give the same first picture;
  * every decision: reward (as float32) and done are identical; the 4 pictures are the same picture (each env
    colour number maps to exactly one harness RGB);
  * pad bytes never press a dropped input or a forbidden Start/Select.
usage: python rl/tests/test_env_cpu.py [steps] [game,game,...]
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'rl'), str(ROOT / 'scripts/nesle_check'), str(ROOT / 'scripts/reward10')]
os.environ.setdefault('NESLE_HOST_LIB', str(ROOT / '.cache/nesle-check/libnesle_host.so'))
import neslenes  # noqa: E402
from reward import RewardTracker  # noqa: E402
import contract  # noqa: E402
from rlenv import RlEnv, GAMES, boot_pads  # noqa: E402
from actions import random_legal  # noqa: E402

DROP_BITS = [16, 32, 64, 128]  # Up, Down, Left, Right


def harness(slug):
    g = contract.load(slug)
    n = neslenes.NesleNes(g.ROM)
    for b in boot_pads(slug):
        neslenes.lib().nh_step(n.h, b, 1)
    return g, n


def check_pictures(pairs, idx, rgb):
    """pairs: dict colour number -> set of harness RGB; adds this frame's pixels."""
    k = idx.reshape(-1)
    v = rgb[8:232, 8:248].reshape(-1, 3)
    key = (v[:, 0].astype(np.int64) << 16) | (v[:, 1].astype(np.int64) << 8) | v[:, 2]
    for c in np.unique(k):
        pairs.setdefault(int(c), set()).update(np.unique(key[k == c]).tolist())


def run(steps=300, seed=5, games=None):
    env = RlEnv({g: 2 for g in (games or GAMES)}, device='cpu', seed=seed, sticky=True, idle_max=0, dropout=True)
    env.start()
    gen = torch.Generator().manual_seed(seed)
    ref = {}
    for e in range(env.n):
        slug = env.games[e // 2]
        g, n = harness(slug)
        for _ in range(4):
            neslenes.lib().nh_step(n.h, 0, 1)
        tr = RewardTracker(g)
        tr.update(n.ram(), 0, frames=4)
        ref[e] = dict(slug=slug, g=g, n=n, tr=tr, pairs={}, alive=True, steps=0)
        check_pictures(ref[e]['pairs'], env.obs[e, 3].numpy(), n.screen())
    for t in range(steps):
        masks = env.masks.clone()
        env.actions.copy_(random_legal(masks, gen))
        env.step()
        for e, r in ref.items():
            if not r['alive']:
                continue
            m = masks[e].numpy()
            for f in range(4):
                pad = int(env.pads[e, f])
                # a pressed direction must be legal for some d-pad choice still allowed, Start/Select only if allowed
                if pad & 8:
                    assert m[16], (r['slug'], 'Start pressed while not allowed')
                if pad & 4:
                    assert m[18], (r['slug'], 'Select pressed while not allowed')
                for bit, cols in zip(DROP_BITS, ([1, 2, 8], [4, 5, 6], [6, 7, 8], [2, 3, 4])):
                    if pad & bit:
                        assert any(m[c] for c in cols), (r['slug'], 'dropped direction pressed', pad)
                neslenes.lib().nh_step(r['n'].h, pad, 1)
                check_pictures(r['pairs'], env.obs[e, f].numpy(), r['n'].screen()) if not env.done[e] else None
            rew, done, _ = r['tr'].update(r['n'].ram(), t + 1, frames=4)
            assert float(np.float32(rew)) == float(env.reward[e]), (r['slug'], e, t, rew, float(env.reward[e]))
            assert bool(done) == bool(env.done[e]), (r['slug'], e, t, done, int(env.done[e]))
            r['steps'] += 1
            if done:
                r['alive'] = False  # the env restarted; this check follows each env's first episode
    out = {}
    for e, r in ref.items():
        bad = {c: s for c, s in r['pairs'].items() if len(s) != 1}
        assert not bad, (r['slug'], 'a colour number shows as several harness colours', list(bad.items())[:3])
        out.setdefault(r['slug'], []).append(r['steps'])
    return out


def test_env_matches_harness_and_python_reward():
    run(150)


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    games = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    for slug, s in run(steps, games=games).items():
        print(f'{slug:12s} identical for {s} decisions')
