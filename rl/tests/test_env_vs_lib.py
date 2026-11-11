"""Two builds of the env must give identical outputs for the same seeds and moves (e.g. before/after an optimisation).
usage: python rl/tests/test_env_vs_lib.py OLD_LIB NEW_LIB [steps] [per_game] [device]"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlenv import RlEnv, GAMES  # noqa: E402
from actions import random_legal  # noqa: E402

old, new = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
per = int(sys.argv[4]) if len(sys.argv) > 4 else 8
dev = sys.argv[5] if len(sys.argv) > 5 else 'cpu'
kw = dict(seed=11, sticky=True, idle_max=30, dropout=True, device=dev)
a = RlEnv({g: per for g in GAMES}, lib_path=old, **kw)
b = RlEnv({g: per for g in GAMES}, lib_path=new, **kw)
a.start(); b.start()
OUTS = ['obs', 'emph', 'reward', 'done', 'masks', 'timeline', 'pads', 'events', 'info']
gen = torch.Generator(device=dev).manual_seed(5)
ends = 0
for t in range(steps + 1):
    for k in OUTS:
        if not torch.equal(getattr(a, k), getattr(b, k)):
            bad = (getattr(a, k) != getattr(b, k)).reshape(a.n, -1).any(1).nonzero().flatten().tolist()
            print(f'DIFFERENT at step {t}: {k} envs {bad[:10]}'); sys.exit(1)
    if t == steps:
        break
    ends += int(a.done.sum())
    act = random_legal(a.masks, gen)
    a.actions.copy_(act); b.actions.copy_(act)
    a.step(); b.step()
print(f'identical: {steps} steps x {a.n} envs, {ends} episode restarts (plus {a.n} starts)')
