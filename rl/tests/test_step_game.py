"""GPU: stepping each game on its own stream (RlEnv.step_game, used by the per-game rollout) gives exactly the same
outputs as the whole-batch step, with sticky + idle + dropout on and random legal moves.
usage: python rl/tests/test_step_game.py [STEPS] [PER_GAME]"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlenv import RlEnv, GAMES  # noqa: E402
from actions import random_legal  # noqa: E402

OUTS = ['obs', 'emph', 'reward', 'done', 'masks', 'timeline', 'pads', 'events']


def main(steps=200, per_game=8):
    kw = dict(seed=5, sticky=True, idle_max=30, dropout=True)
    a = RlEnv({g: per_game for g in GAMES}, device='cuda', **kw)
    b = RlEnv({g: per_game for g in GAMES}, device='cuda', **kw)
    a.start(); b.start()
    streams = [torch.cuda.Stream() for _ in b.games]
    gen = torch.Generator(device='cuda').manual_seed(5)
    dones = 0
    for t in range(steps):
        act = random_legal(a.masks, gen)
        a.actions.copy_(act); b.actions.copy_(act)
        a.step()
        main_s = torch.cuda.current_stream()
        for gi, s in enumerate(streams):
            s.wait_stream(main_s)
            with torch.cuda.stream(s):
                b.step_game(gi, b.obs, b.emph)
        for s in streams:
            main_s.wait_stream(s)
        for k in OUTS:
            if not torch.equal(getattr(a, k), getattr(b, k)):
                print(f'DIFFERENT at step {t} in {k}')
                return 1
        d = a.done.bool()
        dones += int(d.sum())
        if d.any() and not torch.equal(a.info[d], b.info[d]):
            print(f'DIFFERENT info at step {t}')
            return 1
    print(f'identical for {steps} steps x {a.n} envs ({dones} finished episodes)')
    return 0


if __name__ == '__main__':
    sys.exit(main(*(int(x) for x in sys.argv[1:])))
