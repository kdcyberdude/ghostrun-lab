"""Attempts (rlenv_core.cuh attempt_limit, used by rl/routes.py and rl/explore.py): GPU build == CPU build with
training settings (sticky, dropout, idle frames) while every episode starts from a level entrance or a route state,
and the attempt rules hold: an episode from a saved slot ends at its level's clear, its first death, the decision
limit or a kill; a route-state start has no idle frames.

usage: python rl/tests/test_attempts.py ROUTES_DIR [--steps 1500]     (a folder written by rl/explore.py)
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlenv import RlEnv  # noqa: E402
from routes import Routes  # noqa: E402
import slots as slotlib  # noqa: E402

LIMIT, SLOTS = 300, 512


def main(directory, steps, n=16):
    torch.set_num_threads(16)
    kw = dict(seed=11, sticky=True, idle_max=30, dropout=True, level_cap=64 + SLOTS, level_slots=64)
    envs = [RlEnv({'smb': n}, device=d, **kw) for d in ('cuda', 'cpu')]
    for e in envs:
        e.set_attempts(LIMIT, kill=True)
        slotlib.load_entrances(e, 0, Path(directory) / 'entrances.npy')
        routes = Routes.from_dir(e, 0, directory, SLOTS)
        routes.p[:] = np.maximum(routes.K // 2, 0)        # windows in the middle: entrance, window and back starts
        e.set_level_plan(0, 1.0, routes.weights(sorted(e.saved_levels()['smb'])))
        e.start()
    g, c = envs
    gen = torch.Generator().manual_seed(3)
    ends, kills, route_starts = {}, 0, 0
    for t in range(steps):
        assert torch.equal(g.cur_start.cpu(), c.cur_start), f'step {t}: cur_start differs'
        m = c.masks.float()
        parts = [torch.multinomial(m[:, a:b], 1, generator=gen).squeeze(1) for a, b in ((0, 9), (9, 12), (12, 15))]
        mv = (parts[0] + 9 * (parts[1] + 3 * parts[2])).to(torch.int16)
        kill = torch.zeros(n, dtype=torch.uint8)
        if t % 97 == 50:
            kill[t % n] = 1
        started = c.cur_start.clone()
        for e in envs:
            e.actions.copy_(mv.to(e.device))
            e.kill.copy_(kill.to(e.device))
            e.step()
        for k in ('reward', 'done', 'events', 'masks', 'timeline', 'pads', 'parts'):
            if not torch.equal(getattr(g, k).cpu(), getattr(c, k)):
                raise AssertionError(f'step {t}: {k} differs')
        d = c.done.bool()
        if kill.any():
            assert bool(d[kill.bool()].all()), f'step {t}: a killed attempt went on'
            kills += 1
        if d.any():
            assert torch.equal(g.info.cpu()[d], c.info[d]) and torch.equal(g.start_level.cpu()[d], c.start_level[d])
            assert torch.equal(c.start_level[d], started[d]), 'start slot of a finished episode'
            for row, s in zip(c.info[d].tolist(), c.start_level[d].tolist()):
                end, length, idle = int(row[2]), int(row[1]), int(row[6])
                assert s >= 0, 'every episode starts from a saved slot here'
                assert end != 1 and length <= LIMIT, (end, length)    # the first death ends it: never a game over
                if s >= 64:
                    assert idle == 0, 'a route-state start has no idle frames'
                    route_starts += 1
                ends[end] = ends.get(end, 0) + 1
    a, b = g.state_dict()['env'], c.state_dict()['env']
    assert (a == b).all(), 'saved env states differ'
    assert ends.get(7, 0) > 0 and ends.get(8, 0) > 0 and kills > 0 and route_starts > 0, (ends, kills, route_starts)
    print(f'ok: GPU == CPU for {steps} decisions x {n} envs of attempts; ends {ends} (6 clear, 7 death, 8 limit/kill), '
          f'{route_starts} episodes from route states')


def guide_payback(directory, n=4):
    """reward10's maze guide is a potential that is zero at every episode end. An attempt that ends without a clear
    must give it back too, or dying in a maze castle pays better than clearing it (7-4, 2026-11-14). Killed at its
    first decision from the last state of a maze castle's route, an attempt pays back at least one guide segment
    (0.25); in a castle without a guide it pays nothing."""
    out = {}
    for d in ('cuda', 'cpu'):
        e = RlEnv({'smb': n}, device=d, seed=1, sticky=False, idle_max=0, dropout=False, level_cap=64 + SLOTS,
                  level_slots=64)
        e.set_attempts(LIMIT, kill=True)
        slotlib.load_entrances(e, 0, Path(directory) / 'entrances.npy')
        routes = Routes.from_dir(e, 0, directory, SLOTS)
        for lv, low, high in ((15, -9.0, -0.24), (27, -9.0, -0.24), (3, -0.01, 0.01)):   # 4-4, 7-4; 1-4 has no guide
            w = np.zeros(routes.cap)
            w[routes.slot(lv, int(routes.K[lv]))] = 1.0
            e.set_level_plan(0, 1.0, w)
            e.start()
            e.actions.zero_()
            e.kill.fill_(1)
            e.step()
            r = e.reward.cpu().clone()
            assert bool(e.done.bool().all()) and bool((r >= low).all()) and bool((r <= high).all()), (d, lv, r.tolist())
            assert torch.allclose(e.parts.cpu().sum(1), r, atol=1e-6), 'the reward parts sum to the reward'
            out[d, lv] = r
    assert all(torch.equal(out['cuda', lv], out['cpu', lv]) for lv in (15, 27, 3))
    print('ok: a killed attempt pays the maze guide back: 4-4 %.3f, 7-4 %.3f, 1-4 %.3f' %
          tuple(float(out['cpu', lv][0]) for lv in (15, 27, 3)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('routes')
    ap.add_argument('--steps', type=int, default=1500)
    a = ap.parse_args()
    main(a.routes, a.steps)
    guide_payback(a.routes)
