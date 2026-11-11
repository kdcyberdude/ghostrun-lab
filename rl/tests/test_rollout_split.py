"""Replay check of the split rollout: collect one rollout with --rollout-split S (pictures kept on the GPU), then reload
the env's start state and replay the recorded moves with whole-batch steps: pictures, rewards, dones, masks must match.
usage (GPU): python rl/tests/test_rollout_split.py [split] [envs_per_game] [games]"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
split = sys.argv[1] if len(sys.argv) > 1 else '2'
per = sys.argv[2] if len(sys.argv) > 2 else '1024'
games = sys.argv[3] if len(sys.argv) > 3 else 'smb'
sys.argv = ['x', '--run', '.cache/rl/test-split', '--games', games, '--envs-per-game', per, '--steps', '16',
            '--minibatch-envs', '16', '--obs-host', '0', '--compile', '0', '--eval-every-minutes', '0',
            '--rollout-split', split, '--moods', '8']
import train  # noqa: E402
from model import pack  # noqa: E402
t = train.Trainer(train.parse())
t.start()
for _ in range(3):   # some steps first, so episodes and restarts are in flight
    t.refresh_actor(); b = t.collect()
env_state = t.env.state_dict()
t.refresh_actor(); b = t.collect()
print('units', [(u[0], u[1], u[2]) for u in t.units])
env = t.env
env.load_state_dict(env_state)
obs = torch.empty_like(t.obs.slot(b.r, 0)); emph = torch.empty_like(b.b_emph[0])
env.bind_obs(obs, emph)
bad = []
for s in range(t.T):
    env.actions.copy_(pack([b.b_parts[s, k] for k in range(5)]))
    env.step(); torch.cuda.synchronize()
    for name, x, y in (('obs', obs, t.obs.slot(b.r, s + 1)), ('emph', emph, b.b_emph[s + 1]), ('reward', env.reward, b.b_rew[s]),
                       ('done', env.done.bool(), b.b_done[s]), ('masks', env.masks, b.b_masks[s + 1])):
        if not torch.equal(x, y):
            bad.append((s, name))
print('replay identical' if not bad else f'DIFFERENT: {bad[:8]}')
assert not bad
