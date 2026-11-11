"""The env's reward split (rlenv.parts) sums to its reward on every step, all 12 games (CPU build).
usage: python rl/tests/test_reward_parts.py [steps] [per_game]"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlenv import RlEnv, GAMES, PART_NAMES  # noqa: E402
from actions import random_legal  # noqa: E402

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
per = int(sys.argv[2]) if len(sys.argv) > 2 else 4
env = RlEnv({g: per for g in GAMES}, device='cpu', seed=3, sticky=True, idle_max=30, dropout=True)
env.start()
gen = torch.Generator().manual_seed(0)
worst, seen = 0.0, torch.zeros(len(PART_NAMES))
for _ in range(steps):
    env.actions.copy_(random_legal(env.masks, gen)); env.step()
    worst = max(worst, float((env.parts.sum(1) - env.reward).abs().max()))
    seen += (env.parts != 0).sum(0)
print('max |sum(parts) - reward|', worst)
print('nonzero steps per part', dict(zip(PART_NAMES, seen.long().tolist())))
assert worst < 1e-5, 'parts do not sum to the reward'
print('reward parts: OK')
