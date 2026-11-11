"""GPU env speed: decisions/s for several env counts, optionally without drawing (to split the cost)."""
import ctypes, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlenv import RlEnv, GAMES
from actions import random_legal


def run(per_game, no_render=0, steps=32, games=None):
    env = RlEnv({g: per_game for g in (games or GAMES)}, device='cuda', seed=1)
    if no_render:
        env.lib.rlenv_set_no_render.argtypes = [ctypes.c_void_p, ctypes.c_int]
        env.lib.rlenv_set_no_render(env.h, 1)
    env.start()
    gen = torch.Generator(device='cuda').manual_seed(0)
    for _ in range(4):
        env.actions.copy_(random_legal(env.masks, gen)); env.step()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(steps):
        env.actions.copy_(random_legal(env.masks, gen)); env.step()
    torch.cuda.synchronize(); dt = time.time() - t
    return dict(envs=env.n, no_render=no_render, decisions_per_s=round(steps * env.n / dt), ms_per_step=round(1000 * dt / steps, 1))


if __name__ == '__main__':
    games = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    for pg in [int(x) for x in sys.argv[1].split(',')]:
        for nr in (0, 1):
            print(run(pg, nr, games=games), flush=True)
